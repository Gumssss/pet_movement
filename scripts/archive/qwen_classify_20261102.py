"""Frozen, literature-assisted zero-shot IMU-image classification and evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/vlm'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def inputs():
    with (ROOT / 'dataset/splits/windows.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    records = [{'window_id': w['window_id'], 'image_file': 'dataset/vlm/images/sample_' +
                hashlib.sha256(w['window_id'].encode()).hexdigest()[:20] + '.webp'}
               for w in rows if w['split'] == 'test']
    assert len(records) == 3708 and len({r['window_id'] for r in records}) == 3708
    return records

def read_rendered():
    path = DATA / 'images.csv'
    if not path.exists(): return {}
    # A trailing row being written is ignored until it is complete.
    text = path.read_text()
    if not text.endswith('\n'): text = text[:text.rfind('\n')+1]
    return {r['window_id']: r for r in csv.DictReader(text.splitlines()) if all(v is not None for v in r.values())}

def parse(response, classes, truncated=False):
    if truncated: return None, 'truncated'
    text = response.strip()
    if not text: return None, 'empty'
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
        text = re.sub(r'\s*```$', '', text)
    try:
        item = json.loads(text)
        assert isinstance(item, dict) and set(item) == {'label'} and isinstance(item['label'], str)
        labels = {c.casefold(): c for c in classes}
        return labels[item['label'].strip().casefold()], 'ok'
    except (ValueError, AssertionError, KeyError, TypeError):
        refused = any(s in text.lower() for s in ['无法判断','无法确定','不能判断','无法识别','cannot determine','unable to determine'])
        return None, 'refusal' if refused else 'parse_error'

def load_model(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(8)
    torch.manual_seed(20261101)
    path = ROOT / 'models' / f'Qwen3.5-{args.model}'
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    model = AutoModelForImageTextToText.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        device_map=args.device, attn_implementation='sdpa',
    ).eval()
    torch.cuda.synchronize(args.device)
    return model, processor, time.perf_counter() - started

def generate(model, processor, paths, prompt, device):
    import torch
    from PIL import Image
    began = time.perf_counter()
    conversations = []
    for path in paths:
        with Image.open(path) as im:
            image = im.convert('RGB').copy()
        image.info.clear()
        assert image.size == (1280,1152)
        conversations.append([{'role':'user','content':[{'type':'image','image':image}, {'type':'text','text':prompt}]}])
    batch = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
              return_tensors='pt', return_dict=True, padding=True, enable_thinking=False)
    counts = batch['attention_mask'].sum(1).tolist()
    grids = batch['image_grid_thw'].tolist()
    assert all(g == [1,72,80] for g in grids), grids
    batch = batch.to(device)
    torch.cuda.synchronize(device)
    preprocessed = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=32, do_sample=False,
                                logits_to_keep=1, use_cache=True)
    torch.cuda.synchronize(device)
    ended = time.perf_counter()
    tokens = output[:,batch['input_ids'].shape[1]:].cpu()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos,int) else eos
    decoded = processor.batch_decode(tokens,skip_special_tokens=True)
    lengths, truncated = [], []
    for row in tokens.tolist():
        end = next((i+1 for i,v in enumerate(row) if v in eos), len(row))
        lengths.append(end)
        truncated.append(end == 32 and not any(v in eos for v in row))
    del batch, output
    return decoded, counts, lengths, truncated, preprocessed-began, ended-preprocessed, ended-began

def run(args):
    import torch
    import importlib.metadata as metadata
    dest = ROOT / 'results' / ('qwen_4b' if args.model=='4B' else 'qwen_27b')
    dest.mkdir(exist_ok=True)
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    prompt = (DATA / 'prompt.txt').read_text()
    render_protocol = json.loads((DATA/'render_protocol.json').read_text())
    assert render_protocol['window_manifest_sha256'] == sha(ROOT/'dataset/splits/windows.csv')
    assert render_protocol['split_sha256'] == sha(ROOT/'dataset/splits/dog_splits.json')
    assert render_protocol['width'] == 1280 and render_protocol['height'] == 1152
    processor_check = json.loads((DATA/'processor_check.json').read_text())
    assert processor_check[args.model]['image_grid_thw'] == [1,72,80]
    planned = inputs()
    manifest = json.loads((ROOT/'models/model_manifest.json').read_text())[f'Qwen3.5-{args.model}']
    frozen = {
        'experiment': 'imu_2s_v1_literature_zero_shot', 'model': f'Qwen3.5-{args.model}',
        'model_revision': manifest['revision'], 'classes': classes,
        'prompt_sha256': sha(DATA/'prompt.txt'), 'render_protocol_sha256': sha(DATA/'render_protocol.json'),
        'research_sha256': sha(DATA/'research.md'), 'windows_sha256': sha(ROOT/'dataset/splits/windows.csv'),
        'planned_inputs_sha256': hashlib.sha256(json.dumps(planned,sort_keys=True).encode()).hexdigest(),
        'evaluation_split': 'test', 'expected_windows': 3708, 'training_examples': 0,
        'generation': {'do_sample':False,'enable_thinking':False,'max_new_tokens':32,'dtype':'bfloat16','attention':'sdpa'},
        'parsing': 'exact label-only JSON; optional enclosing fence; casefold label matching; failures counted wrong',
        'out_of_range_policy': render_protocol['out_of_range'],
    }
    pp = dest/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == frozen, 'Frozen inputs/protocol changed.'
    else: write_json(pp,frozen)
    (dest/'prompt.txt').write_text(prompt)
    run_start = time.perf_counter()
    model, processor, load_seconds = load_model(args)
    torch.cuda.reset_peak_memory_stats(args.device)
    print(f'Loaded {args.model} in {load_seconds:.2f}s on {args.device}',flush=True)
    if not (dest/'preflight.json').exists():
        result = generate(model,processor,[DATA/'preview.webp'],prompt,args.device)
        raw, counts, lengths, truncated, pre, gen, wall = result
        label,status = parse(raw[0],classes,truncated[0])
        write_json(dest/'preflight.json',{'split':'train','purpose':'technical check only; not included in evaluation',
                  'response':raw[0],'parsed_label':label,'status':status,'input_tokens':counts[0],
                  'generated_tokens':lengths[0], 'preprocess_seconds':pre,'generation_seconds':gen,
                  'image_to_response_seconds':wall,'load_seconds':load_seconds,'device':args.device})
        print(f'Preflight {args.model}: status={status}, input_tokens={counts[0]}, generated_tokens={lengths[0]}, seconds={wall:.2f}',flush=True)
    if args.preflight_only: return
    raw_file = dest/'responses.jsonl'
    done = {}
    if raw_file.exists():
        for line in raw_file.read_text().splitlines():
            item=json.loads(line)
            assert item['window_id'] not in done
            done[item['window_id']]=item
    assert set(done).issubset({r['window_id'] for r in planned})
    pending = [r for r in planned if r['window_id'] not in done]
    batch_size = args.batch_size
    started = time.perf_counter()
    waited = 0.; batches = 0; oom_events = []
    timings=[]
    with raw_file.open('a') as stream:
        while pending:
            current=pending[:batch_size]
            waiting=time.perf_counter()
            rendered=read_rendered()
            while any(r['window_id'] not in rendered for r in current):
                time.sleep(1)
                rendered=read_rendered()
                if time.perf_counter()-waiting>1800: raise RuntimeError('Timed out waiting for rendered images.')
            waited += time.perf_counter()-waiting
            paths=[]
            for r in current:
                saved=rendered[r['window_id']]
                assert saved['image_file']==r['image_file']
                path=ROOT/r['image_file']
                assert sha(path)==saved['image_sha256']
                paths.append(path)
            try:
                raw, counts, lengths, truncations, pre, gen, wall = generate(model,processor,paths,prompt,args.device)
            except torch.OutOfMemoryError:
                oom_events.append({'completed':len(done),'batch_size':batch_size})
                if batch_size==1: raise
                batch_size=max(1,batch_size//2)
                torch.cuda.empty_cache()
                print(f'OOM; retry identical inputs with batch_size={batch_size}',flush=True)
                continue
            batches+=1
            timings.append({'n':len(current),'preprocess_seconds':pre,'generation_seconds':gen,'image_to_response_seconds':wall})
            for i,r in enumerate(current):
                label,status=parse(raw[i],classes,truncations[i])
                item={**r,'image_sha256':rendered[r['window_id']]['image_sha256'],
                      'response':raw[i],'prediction':label,'status':status,'input_tokens':counts[i],
                      'generated_tokens':lengths[i], 'batch_size':len(current),
                      'batch_preprocess_seconds':pre,'batch_generation_seconds':gen,'batch_image_to_response_seconds':wall,
                      'amortized_image_to_response_seconds':wall/len(current)}
                stream.write(json.dumps(item,ensure_ascii=False)+'\n');done[r['window_id']]=item
            stream.flush();os.fsync(stream.fileno())
            pending=pending[len(current):]
            elapsed=time.perf_counter()-started
            stats={'completed':len(done),'expected':3708,'session_completed':sum(t['n'] for t in timings),
                   'remaining':len(pending),'batch_size':batch_size,'session_wall_seconds':elapsed,
                   'status_counts':dict(__import__('collections').Counter(r['status'] for r in done.values()))}
            write_json(dest/'progress.json',stats)
            if batches%5==0 or not pending:
                print(f'{args.model}: {len(done)}/3708; session {elapsed:.1f}s; {stats["status_counts"]}',flush=True)
    assert len(done)==3708
    session={
        'model':args.model,'device':args.device,'gpu':torch.cuda.get_device_name(args.device),
        'load_seconds':load_seconds,'session_model_run_wall_seconds':time.perf_counter()-run_start,
        'formal_session_wall_seconds':time.perf_counter()-started,'render_wait_and_manifest_read_seconds':waited,
        'peak_cuda_allocated_GiB':torch.cuda.max_memory_allocated(args.device)/2**30,
        'peak_cuda_reserved_GiB':torch.cuda.max_memory_reserved(args.device)/2**30,
        'batch_timings':timings,'oom_events':oom_events,
        'versions':{name:metadata.version(name) for name in ['torch','transformers','accelerate','numpy','pillow']},
        'code_sha256':sha(Path(__file__)), 'images_manifest_sha256':sha(DATA/'images.csv'),
    }
    sp=dest/'runtime.json'
    previous=json.loads(sp.read_text()) if sp.exists() else {'sessions':[]}
    previous['sessions'].append(session);write_json(sp,previous)
    print(f'COMPLETE {args.model}: 3708 recorded responses',flush=True)

def evaluate(args):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
    dest=ROOT/'results'/('qwen_4b' if args.model=='4B' else 'qwen_27b')
    p=json.loads((dest/'protocol.json').read_text());classes=p['classes']
    rows=[json.loads(line) for line in (dest/'responses.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['window_id'] for r in rows})==3708
    windows=pd.read_csv(ROOT/'dataset/splits/windows.csv');w=windows[windows.split.eq('test')].copy()
    raw=pd.DataFrame(rows)
    assert set(raw.window_id)==set(w.window_id)
    joined=w.merge(raw,on='window_id',validate='one_to_one',sort=False)
    invalid='__INVALID__'
    truth=joined.label.to_numpy();pred=joined.prediction.fillna(invalid).to_numpy()
    dogs=joined.dog_id.to_numpy();all_labels=classes+[invalid]
    cm=confusion_matrix(truth,pred,labels=all_labels)[:7,:]
    def scores(matrix):
        support=matrix.sum(1);true_positive=np.diag(matrix[:,:7]);predicted=matrix[:,:7].sum(0)
        f1=np.divide(2*true_positive,support+predicted,out=np.zeros(7),where=(support+predicted)>0)
        recall=np.divide(true_positive,support,out=np.zeros(7),where=support>0)
        return {'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall[support>0].mean()),
                'accuracy':float(true_positive.sum()/support.sum())}
    metrics=scores(cm)
    assert abs(metrics['macro_f1']-f1_score(truth,pred,labels=classes,average='macro',zero_division=0))<1e-12
    assert abs(metrics['accuracy']-accuracy_score(truth,pred))<1e-12
    unique=np.unique(dogs)
    dog_cms=np.stack([confusion_matrix(truth[dogs==d],pred[dogs==d],labels=all_labels)[:7,:] for d in unique])
    rng=np.random.default_rng(20261101)
    boots=[scores(dog_cms[rng.integers(0,len(unique),len(unique))].sum(0)) for _ in range(2000)]
    metrics.update({
        'model':p['model'],'experiment':p['experiment'],'test_windows':3708,
        'status_counts':{str(k):int(v) for k,v in joined.status.value_counts().items()},'parse_success_rate':float(joined.status.eq('ok').mean()),
        'failures_counted_as_incorrect':True,'classification_report':classification_report(truth,pred,labels=classes,target_names=classes,output_dict=True,zero_division=0),
        'confusion_matrix':cm.tolist(),'confusion_rows':classes,'confusion_columns':all_labels,
        'per_dog':{str(d):{**scores(dog_cms[i]),'windows':int((dogs==d).sum()),'class_support':dog_cms[i].sum(1).tolist()} for i,d in enumerate(unique)},
        'dog_bootstrap_95ci':{key:np.quantile([b[key] for b in boots],[.025,.975]).tolist() for key in ['macro_f1','balanced_accuracy','accuracy']},
        'mean_input_tokens':float(joined.input_tokens.mean()),'total_input_tokens':int(joined.input_tokens.sum()),
        'total_generated_tokens':int(joined.generated_tokens.sum()),
        'amortized_image_to_response_seconds_p50':float(joined.amortized_image_to_response_seconds.median()),
        'amortized_image_to_response_seconds_p95':float(joined.amortized_image_to_response_seconds.quantile(.95)),
        'latency_note':'batch wall time per item is amortized throughput cost, not single-request latency; preflight.json is separate',
    })
    images=pd.read_csv(DATA/'images.csv')
    image_info=w[['window_id']].merge(images,on='window_id',validate='one_to_one')
    counts=[c for c in image_info if c.endswith(('_above','_below'))]
    metrics['rendering']={'total_render_seconds':float(image_info.render_seconds.sum()),
                          'windows_with_out_of_range':int(image_info[counts].sum(1).gt(0).sum()),
                          'out_of_range_points':{c:int(image_info[c].sum()) for c in counts},
                          'total_channel_samples':3708*200*6}
    runtimes=json.loads((dest/'runtime.json').read_text())
    metrics['runtime']={
        'load_seconds_total':sum(s['load_seconds'] for s in runtimes['sessions']),
        'model_run_wall_seconds_total':sum(s['session_model_run_wall_seconds'] for s in runtimes['sessions']),
        'peak_cuda_allocated_GiB':max(s['peak_cuda_allocated_GiB'] for s in runtimes['sessions']),
        'formal_preprocess_seconds_total':sum(b['preprocess_seconds'] for s in runtimes['sessions'] for b in s['batch_timings']),
        'formal_generation_seconds_total':sum(b['generation_seconds'] for s in runtimes['sessions'] for b in s['batch_timings']),
        'formal_image_to_response_seconds_total':sum(b['image_to_response_seconds'] for s in runtimes['sessions'] for b in s['batch_timings']),
    }
    metrics['runtime']['render_plus_inference_seconds_per_window']=(metrics['rendering']['total_render_seconds']+metrics['runtime']['formal_image_to_response_seconds_total'])/3708
    metrics['runtime']['cost_note']='render time + batch preprocessing/generation, amortized per window; excludes load, preflight, manifest I/O, waits and 2s acquisition; components are not measured whole-pipeline latency'
    write_json(dest/'metrics.json',metrics)
    columns=['window_id','dog_id','session','start_sec','end_exclusive_sec','label','prediction','status','response','image_file','image_sha256','input_tokens','generated_tokens','batch_size','amortized_image_to_response_seconds']
    joined[columns].to_csv(dest/'predictions.csv',index=False)
    print(json.dumps({k:metrics[k] for k in ['model','test_windows','macro_f1','balanced_accuracy','accuracy','status_counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','evaluate'])
    parser.add_argument('--model',choices=['4B','27B'],required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch size must be positive')
    run(args) if args.command=='run' else evaluate(args)
