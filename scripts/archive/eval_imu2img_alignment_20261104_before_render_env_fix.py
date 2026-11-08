"""vLLM validation/prompt controls and matched tests for frozen-Qwen image alignment."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import time

import qwen_classify as q
from eval_qwen35_images import load_model

ROOT=q.ROOT
SIMPLE=ROOT/'dataset/vlm_imu2img_simple'
ALIGNED=ROOT/'dataset/vlm_imu2img_qwen'
TRAIN=ROOT/'results/imu2img_qwen_alignment'
MODEL=ROOT/'models/IMU2IMG-Qwen3.5-4B-aligned'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def inputs(split,folder):
    with (ROOT/'dataset/splits/windows.csv').open() as f:windows=[w for w in csv.DictReader(f) if w['split']==split]
    with (folder/'images.csv').open() as f:images=list(csv.DictReader(f))
    assert [r['window_id'] for r in images]==[r['window_id'] for r in windows]
    assert len(images)==(3708 if split=='test' else 4727)
    for row in images:assert sha(ROOT/row['image_file'])==row['image_sha256']
    return windows,images


def frozen_protocol(folder,prompt,split,kind,epoch=None):
    model=json.loads((ROOT/'models/model_manifest.json').read_text())['Qwen3.5-4B']
    return {'model':'Qwen3.5-4B','model_revision':model['revision'],'qwen_finetuned':False,'qwen_trainable_parameters':0,
        'experiment':'imu2img_frozen_qwen_'+kind,'kind':kind,'split':split,'expected_windows':3708 if split=='test' else 4727,
        'classes':json.loads((ROOT/'results/protocol.json').read_text())['classes'],
        'inference_backend':'vllm','backend_version':md.version('vllm'),'prompt_language':'en',
        'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'image_manifest_sha256':sha(folder/'images.csv'),'data_dir':str(folder.relative_to(ROOT)),
        'windows_sha256':sha(ROOT/'dataset/splits/windows.csv'),'split_sha256':sha(ROOT/'dataset/splits/dog_splits.json'),
        'image_size':[224,224],'visual_tokens':49,'converter_epoch':epoch,
        'generation':{'temperature':0,'enable_thinking':False,'max_new_tokens':32,'seed':20261101,'dtype':'bfloat16',
                      'guided_decoding':False,'speculative_decoding':False},
        'engine':{'tensor_parallel_size':1,'max_model_len':4096,'max_num_seqs':32,'max_num_batched_tokens':8192,
                  'gpu_memory_utilization':.85,'enable_prefix_caching':False,'flashinfer_sampler':False,
                  'min_pixels':224*224,'max_pixels':1280*1152},
        'supervision_note':'Qwen remains frozen. Images are supervised converter representations; this is not zero-label system evaluation.'}


def run_dataset(args,model,processor,folder,dest,prompt,split,kind,epoch=None):
    dest.mkdir(parents=True,exist_ok=True)
    windows,images=inputs(split,folder);classes=json.loads((ROOT/'results/protocol.json').read_text())['classes']
    p=frozen_protocol(folder,prompt,split,kind,epoch);file=dest/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==p,'Frozen evaluation protocol changed.'
    else:q.write_json(file,p)
    (dest/'prompt.txt').write_text(prompt)
    raw=dest/'responses.jsonl';rows=[json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
    done={r['window_id']:r for r in rows};assert len(done)==len(rows)
    pending=[r for r in images if r['window_id'] not in done]
    began=time.perf_counter();timings=[]
    with raw.open('a') as stream:
        for start in range(0,len(pending),32):
            current=pending[start:start+32]
            texts,counts,lengths,trunc,pre,gen,wall=q.generate(model,processor,[ROOT/r['image_file'] for r in current],prompt,args.device,(224,224))
            timings.append({'n':len(current),'preprocess_seconds':pre,'generation_seconds':gen,'request_wall_seconds':wall})
            for i,item in enumerate(current):
                label,status=q.parse(texts[i],classes,trunc[i])
                row={k:item[k] for k in ['window_id','image_file','image_sha256']}
                row.update(response=texts[i],prediction=label,status=status,input_tokens=counts[i],generated_tokens=lengths[i],
                           batch_size=len(current),amortized_request_seconds=wall/len(current),inference_backend='vllm')
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');done[row['window_id']]=row
            stream.flush()
            progress={'completed':len(done),'expected':len(images),'remaining':len(images)-len(done),
                      'status_counts':dict(Counter(r['status'] for r in done.values())),'wall_seconds':time.perf_counter()-began}
            q.write_json(dest/'progress.json',progress)
            if len(done)%320==0 or len(done)==len(images):print(kind+' '+json.dumps(progress),flush=True)
    runtime_path=dest/'runtime.json';runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {'sessions':[]}
    if timings:
        runtime['sessions'].append({'wall_seconds':time.perf_counter()-began,'batch_timings':timings,'inference_backend':'vllm',
            'device':args.device,'code_sha256':sha(Path(__file__)),'shared_generation_code_sha256':sha(ROOT/'scripts/qwen_classify.py')})
        q.write_json(runtime_path,runtime)
    return evaluate(dest)


def evaluate(dest):
    import numpy as np,pandas as pd
    from sklearn.metrics import confusion_matrix,classification_report,f1_score,recall_score,accuracy_score
    p=json.loads((dest/'protocol.json').read_text());classes=p['classes']
    raw=[json.loads(s) for s in (dest/'responses.jsonl').read_text().splitlines()]
    w=pd.read_csv(ROOT/'dataset/splits/windows.csv');w=w[w.split.eq(p['split'])].copy()
    assert len(raw)==len(w)==len({r['window_id'] for r in raw})==p['expected_windows']
    joined=w.merge(pd.DataFrame(raw),on='window_id',validate='one_to_one',sort=False)
    assert joined.window_id.tolist()==w.window_id.tolist()
    truth=joined.label.to_numpy();pred=joined.prediction.fillna('__INVALID__').to_numpy()
    cm=confusion_matrix(truth,pred,labels=classes+['__INVALID__'])[:7,:]
    def scores(matrix):
        support=matrix.sum(1);tp=np.diag(matrix[:,:7]);counted=matrix[:,:7].sum(0)
        f1=np.divide(2*tp,support+counted,out=np.zeros(7),where=(support+counted)>0)
        recall=np.divide(tp,support,out=np.zeros(7),where=support>0)
        return {'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall[support>0].mean()),'accuracy':float(tp.sum()/support.sum())}
    m=scores(cm)
    assert abs(m['macro_f1']-f1_score(truth,pred,labels=classes,average='macro',zero_division=0))<1e-12
    assert abs(m['balanced_accuracy']-recall_score(truth,pred,labels=classes,average='macro',zero_division=0))<1e-12
    assert abs(m['accuracy']-accuracy_score(truth,pred))<1e-12
    m.update(model=p['model'],split=p['split'],windows=len(raw),inference_backend='vllm',prompt_language='en',
        status_counts=dict(Counter(joined.status)),parse_success_rate=float(joined.status.eq('ok').mean()),
        failures_counted_as_incorrect=True,prediction_counts=dict(Counter(joined.prediction.fillna('__INVALID__'))),
        classification_report=classification_report(truth,pred,labels=classes,output_dict=True,zero_division=0),
        confusion_matrix=cm.tolist(),confusion_rows=classes,confusion_columns=classes+['__INVALID__'],
        mean_input_tokens=float(joined.input_tokens.mean()),total_input_tokens=int(joined.input_tokens.sum()),
        total_generated_tokens=int(joined.generated_tokens.sum()))
    if p['split']=='test':
        dogs=joined.dog_id.to_numpy();unique=np.unique(dogs)
        cms=np.stack([confusion_matrix(truth[dogs==d],pred[dogs==d],labels=classes+['__INVALID__'])[:7,:] for d in unique])
        m['per_dog']={str(d):{**scores(cms[i]),'windows':int((dogs==d).sum())} for i,d in enumerate(unique)}
        rng=np.random.default_rng(20261101);boots=[scores(cms[rng.integers(0,len(unique),len(unique))].sum(0)) for _ in range(2000)]
        m['dog_bootstrap_95ci']={k:np.quantile([b[k] for b in boots],[.025,.975]).tolist() for k in ['macro_f1','balanced_accuracy','accuracy']}
    q.write_json(dest/'metrics.json',m)
    columns=['window_id','dog_id','session','start_sec','end_exclusive_sec','label','prediction','status','response',
             'image_file','image_sha256','input_tokens','generated_tokens','batch_size','amortized_request_seconds']
    joined[columns].to_csv(dest/'predictions.csv',index=False)
    print('SCORES '+str(dest.relative_to(ROOT))+' '+json.dumps({k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy','status_counts']}),flush=True)
    return m


def baseline_all(args):
    args.model='4B';args.method='imu2img'
    began=time.perf_counter();model,processor,load_seconds=load_model(args)
    root=ROOT/'results/qwen3.5_4B_imu2img_promptval'
    old_prompt=(ROOT/'dataset/vlm_imu2img/prompt.txt').read_text();simple=(SIMPLE/'prompt.txt').read_text()
    old=run_dataset(args,model,processor,SIMPLE/'validation',root/'old_prompt',old_prompt,'validation','old_prompt_validation')
    new=run_dataset(args,model,processor,SIMPLE/'validation',root/'simple_prompt',simple,'validation','simple_prompt_validation')
    q.write_json(root/'comparison.json',{'old_prompt_validation':{k:old[k] for k in ['macro_f1','balanced_accuracy','accuracy']},
         'simple_prompt_validation':{k:new[k] for k in ['macro_f1','balanced_accuracy','accuracy']},
         'selection_note':'Simplified prompt was fixed before both validation results; no prompt search or test-dependent changes.'})
    run_dataset(args,model,processor,SIMPLE,ROOT/'results/qwen3.5_4B_imu2img_simple_en',simple,'test','old_converter_simple_prompt_test')
    q.write_json(root/'runtime.json',{'model_load_seconds':load_seconds,'whole_session_wall_seconds':time.perf_counter()-began,
                                     'stages':['old_validation','simple_validation','simple_test'],'qwen_finetuned':False})
    print('BASELINE ALL COMPLETE',flush=True)


def aligned_all(args):
    args.model='4B';args.method='imu2img'
    assert (TRAIN/'training_complete.json').exists()
    model,processor,load_seconds=load_model(args);prompt=(ALIGNED/'prompt.txt').read_text();scores=[]
    for epoch in [1,2,3]:
        metric=run_dataset(args,model,processor,ALIGNED/'validation'/f'epoch_{epoch}',TRAIN/'validation'/f'epoch_{epoch}',
                           prompt,'validation','aligned_checkpoint_validation',epoch)
        scores.append({'epoch':epoch,'macro_f1':metric['macro_f1'],'balanced_accuracy':metric['balanced_accuracy'],'accuracy':metric['accuracy']})
    best=max(scores,key=lambda s:s['macro_f1'])
    selection={'metric':'full validation vLLM greedy macro-F1','best_epoch':best['epoch'],'best_macro_f1':best['macro_f1'],
               'scores':scores,'test_used_for_selection':False,'tie_break':'earliest epoch'}
    q.write_json(TRAIN/'checkpoint_selection.json',selection)
    from shutil import copyfile
    copyfile(MODEL/'checkpoints'/f"epoch_{best['epoch']}"/'converter.safetensors',MODEL/'converter.safetensors')
    with (TRAIN/'render_test.log').open('w') as f:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'scripts/align_imu2img_qwen.py'),'render',
                        '--epoch',str(best['epoch']),'--split','test','--device','cuda:1'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    test=run_dataset(args,model,processor,ALIGNED,ROOT/'results/qwen3.5_4B_imu2img_aligned_en',prompt,'test','qwen_aligned_converter_test',best['epoch'])
    q.write_json(TRAIN/'evaluation_complete.json',{'checkpoint_selection':selection,'test_scores':{k:test[k] for k in ['macro_f1','balanced_accuracy','accuracy']},
                                                'qwen_finetuned':False,'vllm_load_seconds':load_seconds})
    print('ALIGNED ALL COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['baseline-all','aligned-all','evaluate'])
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--result-dir',type=Path)
    args=parser.parse_args()
    if args.command=='baseline-all':baseline_all(args)
    elif args.command=='aligned-all':aligned_all(args)
    else:
        assert args.result_dir and args.result_dir.resolve().is_relative_to(ROOT/'results')
        evaluate(args.result_dir.resolve())
