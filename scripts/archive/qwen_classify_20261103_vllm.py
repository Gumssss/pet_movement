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
os.environ.setdefault('VLLM_CACHE_ROOT', '/tmp/pet-vllm-cache')
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/pet-vllm-triton')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/pet-vllm-inductor')
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/vlm'
MODEL_SPECS = {
    '4B': {'name':'Qwen3.5-4B','folder':'qwen_4b','check':'processor_check.json','research':'research.md'},
    '27B': {'name':'Qwen3.5-27B','folder':'qwen_27b','check':'processor_check.json','research':'research.md'},
    '3.8-27B': {'name':'Qwen3.8-27B','folder':'qwen3_8_27b','check':'processor_check_qwen38.json','research':'research_en.md'},
}

def result_dir(args):
    if getattr(args,'output_dir',None):
        target=Path(args.output_dir).resolve()
        assert target.is_relative_to(ROOT/'results') and target != ROOT/'results'
        return target
    return ROOT / 'results' / MODEL_SPECS[args.model]['folder']

def process_gpu_memory(device):
    import os,psutil,pynvml
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByIndex(int(device.split(':')[1]))
    pids={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
    total=pynvml.nvmlDeviceGetMemoryInfo(handle).total
    processes=pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    used=sum(p.usedGpuMemory for p in processes if p.pid in pids and 0<=p.usedGpuMemory<=total)
    assert used>0, 'No GPU memory found for the vLLM process tree.'
    name=pynvml.nvmlDeviceGetName(handle)
    return {'observed_process_memory_GiB':used/2**30,'gpu':name.decode() if isinstance(name,bytes) else name}

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
    assert re.fullmatch(r'cuda:\d+',args.device), 'A CUDA device is required for this vLLM runner.'
    os.environ['CUDA_VISIBLE_DEVICES']=args.device.split(':')[1]
    import torch
    from transformers import AutoProcessor
    from vllm import LLM
    torch.set_num_threads(8)
    torch.manual_seed(20261101)
    path = ROOT / 'models' / MODEL_SPECS[args.model]['name']
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    model = LLM(model=str(path),dtype='bfloat16',tensor_parallel_size=1,
                trust_remote_code=False,seed=20261101,gpu_memory_utilization=.85,
                max_model_len=4096,max_num_seqs=32,max_num_batched_tokens=8192,
                limit_mm_per_prompt={'image':1,'video':0},
                mm_processor_kwargs={'min_pixels':65536,'max_pixels':1280*1152},
                generation_config='vllm',enable_prefix_caching=False,
                disable_log_stats=True)
    return model, processor, time.perf_counter() - started

def generate(model, processor, paths, prompt, device):
    from PIL import Image
    from vllm import SamplingParams
    began = time.perf_counter()
    requests = []
    for path in paths:
        with Image.open(path) as im:
            image = im.convert('RGB').copy()
        image.info.clear()
        assert image.size == (1280,1152)
        conversation=[{'role':'user','content':[{'type':'image','image':image}, {'type':'text','text':prompt}]}]
        text=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        requests.append({'prompt':text,'multi_modal_data':{'image':image}})
    preprocessed = time.perf_counter()
    output=model.generate(requests,sampling_params=SamplingParams(
            temperature=0.,max_tokens=32,seed=20261101,top_p=1.,top_k=0,
            presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.),use_tqdm=False)
    ended = time.perf_counter()
    assert len(output)==len(paths)
    decoded=[item.outputs[0].text for item in output]
    counts=[len(item.prompt_token_ids) for item in output]
    lengths=[len(item.outputs[0].token_ids) for item in output]
    truncated=[item.outputs[0].finish_reason=='length' for item in output]
    return decoded, counts, lengths, truncated, preprocessed-began, ended-preprocessed, ended-began

def run(args):
    import torch
    import importlib.metadata as metadata
    spec = MODEL_SPECS[args.model]
    dest = result_dir(args)
    dest.mkdir(exist_ok=True)
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    prompt_path = Path(args.prompt).resolve() if args.prompt else DATA/'prompt.txt'
    prompt = prompt_path.read_text()
    render_protocol = json.loads((DATA/'render_protocol.json').read_text())
    assert render_protocol['window_manifest_sha256'] == sha(ROOT/'dataset/splits/windows.csv')
    assert render_protocol['split_sha256'] == sha(ROOT/'dataset/splits/dog_splits.json')
    assert render_protocol['width'] == 1280 and render_protocol['height'] == 1152
    processor_check = json.loads((DATA/spec['check']).read_text())
    assert processor_check[args.model]['image_grid_thw'] == [1,72,80]
    planned = inputs()
    manifest = json.loads((ROOT/'models/model_manifest.json').read_text())[spec['name']]
    frozen = {
        'experiment': 'imu_2s_v1_literature_zero_shot_en' if args.model=='3.8-27B' else 'imu_2s_v1_literature_zero_shot', 'model': spec['name'],
        'model_revision': manifest['revision'], 'classes': classes,
        'prompt_sha256': sha(prompt_path), 'render_protocol_sha256': sha(DATA/'render_protocol.json'),
        'research_sha256': sha(DATA/spec['research']), 'windows_sha256': sha(ROOT/'dataset/splits/windows.csv'),
        'planned_inputs_sha256': hashlib.sha256(json.dumps(planned,sort_keys=True).encode()).hexdigest(),
        'evaluation_split': 'test', 'expected_windows': 3708, 'training_examples': 0,
        'generation': {'do_sample':False,'enable_thinking':False,'max_new_tokens':32,'dtype':'bfloat16','attention':'sdpa'},
        'parsing': 'exact label-only JSON; optional enclosing fence; casefold label matching; failures counted wrong',
        'out_of_range_policy': render_protocol['out_of_range'],
    }
    frozen['generation'].pop('attention')
    frozen.update(inference_backend='vllm',backend_version=metadata.version('vllm'),
                  engine={'tensor_parallel_size':1,'max_model_len':4096,'max_num_seqs':32,
                          'max_num_batched_tokens':8192,'gpu_memory_utilization':.85,
                          'enable_prefix_caching':False,'quantization':None,
                          'speculative_decoding':False,'guided_decoding':False,
                          'flashinfer_sampler':False,'sampler':'vllm_native'},
                  timing_scope='vLLM engine call includes internal preprocessing, scheduling, GPU execution and detokenization')
    if args.model=='3.8-27B':
        assert prompt.isascii(), 'New experiment requires the English-only prompt.'
        frozen.update(prompt_language='en',model_repo_id=manifest['repo_id'],
                      prompt_source_file=str(prompt_path.relative_to(ROOT)),
                      research_source_file=str((DATA/spec['research']).relative_to(ROOT)),
                      processor_check_sha256=sha(DATA/spec['check']),
                      images_manifest_sha256=sha(DATA/'images.csv'),
                      comparison_note='Model version and prompt language both differ from the 2026-11-02 Chinese experiments.')
    pp = dest/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == frozen, 'Historical/frozen result protected: use a new --output-dir for changed backend or prompt.'
    else: write_json(pp,frozen)
    (dest/'prompt.txt').write_text(prompt)
    run_start = time.perf_counter()
    model, processor, load_seconds = load_model(args)
    memory=process_gpu_memory(args.device)
    observed_memory_peak=memory['observed_process_memory_GiB']
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
            observed_memory_peak=max(observed_memory_peak,process_gpu_memory(args.device)['observed_process_memory_GiB'])
            timings.append({'n':len(current),'preprocess_seconds':pre,'generation_seconds':gen,'image_to_response_seconds':wall})
            for i,r in enumerate(current):
                label,status=parse(raw[i],classes,truncations[i])
                item={**r,'image_sha256':rendered[r['window_id']]['image_sha256'],
                      'inference_backend':'vllm',
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
    memory=process_gpu_memory(args.device)
    session={
        'model':args.model,'device':args.device,'gpu':memory['gpu'],'inference_backend':'vllm',
        'load_seconds':load_seconds,'session_model_run_wall_seconds':time.perf_counter()-run_start,
        'formal_session_wall_seconds':time.perf_counter()-started,'render_wait_and_manifest_read_seconds':waited,
        'peak_cuda_allocated_GiB':None,'peak_cuda_reserved_GiB':None,
        'observed_gpu_process_memory_GiB_peak':max(observed_memory_peak,memory['observed_process_memory_GiB']),
        'memory_measurement':'NVML sum of this process and its vLLM children on the selected GPU, sampled after engine init and each batch; includes cache reservation; not a PyTorch allocator peak',
        'batch_timings':timings,'oom_events':oom_events,
        'versions':{name:metadata.version(name) for name in ['vllm','torch','transformers','accelerate','numpy','pillow']},
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
    dest=result_dir(args)
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
        'inference_backend':p.get('inference_backend','transformers'),'prompt_language':p.get('prompt_language','zh'),
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
        'peak_cuda_allocated_GiB':max((s['peak_cuda_allocated_GiB'] for s in runtimes['sessions'] if s['peak_cuda_allocated_GiB'] is not None),default=None),
        'formal_preprocess_seconds_total':sum(b['preprocess_seconds'] for s in runtimes['sessions'] for b in s['batch_timings']),
        'formal_generation_seconds_total':sum(b['generation_seconds'] for s in runtimes['sessions'] for b in s['batch_timings']),
        'formal_image_to_response_seconds_total':sum(b['image_to_response_seconds'] for s in runtimes['sessions'] for b in s['batch_timings']),
    }
    metrics['runtime']['render_plus_inference_seconds_per_window']=(metrics['rendering']['total_render_seconds']+metrics['runtime']['formal_image_to_response_seconds_total'])/3708
    metrics['runtime']['cost_note']='render time + batch preprocessing/generation, amortized per window; excludes load, preflight, CPU text decoding/parsing, manifest I/O, waits and 2s acquisition; components are not measured whole-pipeline latency'
    if p.get('inference_backend')=='vllm':
        metrics['runtime']['observed_gpu_process_memory_GiB_peak']=max(s['observed_gpu_process_memory_GiB_peak'] for s in runtimes['sessions'])
        metrics['runtime']['memory_measurement']=runtimes['sessions'][0]['memory_measurement']
        metrics['runtime']['vllm_engine_call_seconds_total']=metrics['runtime']['formal_generation_seconds_total']
        metrics['runtime']['timing_scope']='vLLM engine call includes internal tokenization/image processing, scheduling, GPU execution and output detokenization; it is not GPU kernel time alone.'
        metrics['runtime']['cost_note']='rendering + parent image/template preparation + vLLM engine request wall time, amortized per window; excludes engine initialization/compile/warmup, preflight, external JSON parsing, manifest I/O, waits and 2s acquisition; whole-run wall time is separate.'
    write_json(dest/'metrics.json',metrics)
    columns=['window_id','dog_id','session','start_sec','end_exclusive_sec','label','prediction','status','response','image_file','image_sha256','input_tokens','generated_tokens','batch_size','amortized_image_to_response_seconds']
    joined[columns].to_csv(dest/'predictions.csv',index=False)
    print(json.dumps({k:metrics[k] for k in ['model','test_windows','macro_f1','balanced_accuracy','accuracy','status_counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','evaluate'])
    parser.add_argument('--model',choices=list(MODEL_SPECS),required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--prompt',type=Path,help='Explicit prompt file; use the stored result prompt to reproduce historical Chinese runs')
    parser.add_argument('--output-dir',type=Path,help='New result directory under the project results directory')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch size must be positive')
    run(args) if args.command=='run' else evaluate(args)
