"""Audit raw generation, LR selection, checkpoints, frozen bases and merged weights."""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import json
from pathlib import Path

import sixplot_qwen38_common as c
import sixplot_qwen38_lora as run
ROOT=c.ROOT


def read(path):return json.loads(Path(path).read_text())


def check_result(key,stage,tag,split):
    import pandas as pd
    from PIL import Image
    import numpy as np
    from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score
    import qwen_classify as q
    dest=run.result_folder(key,stage,tag,split);p=read(dest/'protocol.json');m=read(dest/'metrics.json')
    records=[json.loads(line) for line in (dest/'responses.jsonl').read_text().splitlines()]
    df=pd.read_csv(dest/'predictions.csv');w=pd.read_csv(ROOT/'dataset/splits/windows.csv');w=w[w.split.eq(split)]
    folder=run.image_folder(key,stage,tag,split);im=pd.read_csv(folder/'images.csv');im=im[im.window_id.isin(w.window_id)]
    assert len(records)==len(df)==df.window_id.nunique()==len(w)==p['expected_windows']
    assert df.window_id.tolist()==w.window_id.tolist()==im.window_id.tolist()==[r['window_id'] for r in records]
    assert df.label.tolist()==w.label.tolist() and df.dog_id.tolist()==w.dog_id.tolist()
    ckpt=c.checkpoint(key,stage,tag)
    assert p['key']==key and p['stage']==stage and p['split']==split
    assert p['adapter_sha256']==c.sha(ckpt/'adapter_model.safetensors')
    train_dir=c.out(key)/('lr_search/lr_'+c.lr_tag(float(tag))) if stage=='pilot' else c.out(key)
    tp=read(train_dir/'training_protocol.json')
    assert p['training_protocol_sha256']==c.sha(train_dir/'training_protocol.json')
    assert p['windows_sha256']==c.sha(ROOT/'dataset/splits/windows.csv') and p['splits_sha256']==c.sha(ROOT/'dataset/splits/dog_splits.json')
    assert p['image_manifest_sha256']==c.sha(folder/'images.csv') and p['render_protocol_sha256']==c.sha(c.DATA/'render_protocol.json')
    assert p['prompt_sha256']==c.sha(dest/'prompt.txt')==c.sha(c.out(key)/'prompt.txt')==c.sha(c.PROMPT)
    joined=df.merge(pd.DataFrame(records),on='window_id',validate='one_to_one',suffixes=('_csv','_raw'))
    for name in ['prediction','status','response','input_tokens','generated_tokens','image_file','image_sha256']:
        assert joined[name+'_csv'].fillna('').tolist()==joined[name+'_raw'].fillna('').tolist(),name
    for name in ['image_file','image_sha256']:assert df[name].tolist()==im[name].tolist()
    for row in im.itertuples():
        assert c.sha(ROOT/row.image_file)==row.image_sha256
        assert Path(row.image_file).name.startswith('sample_')
    for index in [0,len(im)//2,len(im)-1]:
        with Image.open(ROOT/im.iloc[index].image_file) as image:
            assert image.size==(1280,1152) and image.mode=='RGB' and image.format=='WEBP'
            assert not any(k in image.info for k in ['exif','xmp','icc_profile'])
    for r in records:
        prediction,status=q.parse(r['response'],c.classes(),r['status']=='truncated')
        assert prediction==r['prediction'] and status==r['status'] and r['inference_backend']=='vllm'
        assert r['input_tokens']==tp['processor']['prefix_tokens']
    truth=df.label;pred=df.prediction.fillna('__INVALID__');cs=c.classes()
    scores={'macro_f1':float(f1_score(truth,pred,labels=cs,average='macro',zero_division=0)),
            'balanced_accuracy':float(recall_score(truth,pred,labels=cs,average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(truth,pred))}
    assert all(abs(m[k]-v)<1e-12 for k,v in scores.items())
    assert m['confusion_matrix']==confusion_matrix(truth,pred,labels=cs+['__INVALID__'])[:7,:].tolist()
    assert m['status_counts']==dict(Counter(r['status'] for r in records))
    rt=read(dest/'runtime.json')
    assert sum(b['n'] for session in rt['sessions'] for b in session['batch_timings'])==len(records)
    assert all(s['code_sha256']==c.sha(Path(run.__file__)) and s['common_code_sha256']==c.sha(Path(c.__file__))
               and s['generation_code_sha256']==c.sha(ROOT/'scripts/qwen_classify.py') for s in rt['sessions'])
    return m,df


def check_weights(key,epoch):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    target=c.model_dir(key);a=load_file(str(target/'adapter/adapter_model.safetensors'))
    assert c.sha(target/'adapter/adapter_model.safetensors')==c.sha(c.checkpoint(key,'formal',epoch)/'adapter_model.safetensors')
    assert len(a)==828 and sum(t.numel() for t in a.values())==13881216
    ac=read(target/'adapter/adapter_config.json')
    assert ac['r']==4 and ac['lora_alpha']==8 and ac['lora_dropout']==.05
    bm=read(c.base_dir(key)/'model.safetensors.index.json')['weight_map'];mm=read(target/'model.safetensors.index.json')['weight_map']
    omitted=set(bm)-set(mm);assert not set(mm)-set(bm)
    assert len(omitted)==15 and all(n.startswith('mtp.') for n in omitted)
    pairs={}
    for name in a:
        if name.endswith('.lora_A.weight'):
            stem=name.removesuffix('.lora_A.weight');pairs[stem.removeprefix('base_model.model.')+'.weight']=(name,stem+'.lora_B.weight')
    assert len(pairs)==414
    scopes=Counter();unchanged=0
    with ExitStack() as stack:
        base={n:stack.enter_context(safe_open(str(c.base_dir(key)/n),framework='pt')) for n in set(bm.values())}
        merged={n:stack.enter_context(safe_open(str(target/n),framework='pt')) for n in set(mm.values())}
        for name in mm:
            source=base[bm[name]].get_tensor(name);b=merged[mm[name]].get_tensor(name);wanted=source.to(b.dtype)
            if name in pairs:
                an,bn=pairs[name];wanted=wanted+(a[bn]@a[an]*2).to(wanted.dtype);scopes[c.scope(name)]+=1
            else:unchanged+=1
            assert torch.equal(wanted,b),name
    assert dict(scopes)=={'ViT':108,'visual_projection':2,'language_attention':304}
    expected=read(ROOT/'models/model_manifest.json')[c.NAMES[key]]
    official=[]
    for f in expected['files']:
        if f.get('sha256'):
            assert c.sha(c.base_dir(key)/f['path'])==f['sha256'];official.append(f['path'])
    assert sum(n.endswith('.safetensors') for n in official)==len(set(bm.values()))
    return {'merged_lora_modules_verified':dict(scopes),'all_other_exported_weights_match_base':unchanged,
            'unused_mtp_keys_omitted':sorted(omitted),
            'official_source_file_sha256_verified':official,'merge_rounding':'FP32 LoRA delta cast to BF16, then added to BF16 base'}


def check_tracking(directory,steps,peak,warmup):
    import math
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    train=[row for row in rows if 'train/loss' in row]
    assert len(train)==steps and [r['optimizer_step'] for r in train]==list(range(1,steps+1))
    for r in train:
        t=r['optimizer_step']-1
        factor=t/max(1,warmup) if t<warmup else .5*(1+math.cos(math.pi*(t-warmup)/(steps-warmup)))
        assert math.isfinite(r['train/loss']) and r['train/loss']>=0
        assert abs(r['train/learning_rate']-peak*factor)<1e-12
    assert read(directory/'wandb_run.json')['mode']=='offline' and (directory/'loss_curve.png').exists()
    assert len(list((directory/'wandb').glob('offline-run-*/run-*.wandb')))==1
    return {'recorded_updates':len(train),'all_losses_finite':True,'ten_percent_linear_then_cosine_lr_verified':True,'offline_wandb_and_png_present':True}


def main():
    import numpy as np
    import torch
    from sklearn.metrics import confusion_matrix
    torch.set_num_threads(8)
    assert (c.SUITE/'generation_complete.json').exists() and c.KEYS==['qwen38_27b']
    protocol=read(c.SUITE/'experiment_protocol.json');batch=read(c.SUITE/'batch_selection.json');key='qwen38_27b';path=c.SUITE
    assert protocol['mode']=='sixplot' and protocol['gradient_checkpointing'] and protocol['formal_epochs']==3
    assert protocol['world_size']==batch['world_size']==1 and protocol['gradient_accumulation']==batch['gradient_accumulation']==1
    assert protocol['input_protocol_sha256']==c.sha(path/'input_protocol.json')
    assert read(path/'source_sha256.json')==run.sources()
    pilot=read(path/'pilot_training_complete.json');formal=read(path/'formal_training_verified.json');cfg=read(path/'training_protocol.json');done=read(path/'training_complete.json')
    technical=read(path/'single_gpu_check.json');gradient=read(path/'single_gpu_gradient_check.json')
    assert technical['microbatch_per_gpu']==batch['microbatch_per_gpu'] and technical['gradient_accumulation']==1 and technical['optimizer_steps']==3
    assert all(v>0 for v in gradient['lora'].values()) and gradient['qwen_base_gradients_none']
    assert pilot['qwen_base_sha256_before']==pilot['qwen_base_sha256_after']==formal['qwen_base_sha256_before']==formal['qwen_base_sha256_after']
    assert pilot['initial_lora_sha256']==formal['initial_lora_sha256']==technical['initial_lora_sha256']
    assert pilot['source_sha256']==formal['source_sha256']==cfg['source_sha256']==run.sources()
    micro=batch['microbatch_per_gpu'];assert cfg['microbatch_per_gpu']==micro and cfg['effective_batch_size']==micro
    assert cfg['cuda_visible_device_count']==1 and cfg['gpu_uuid']==batch['gpu_uuid']==c.GPU_UUID
    assert cfg['lora']['gradient_checkpointing'] and cfg['lora']['trainable_parameters']==13881216
    steps_epoch=(16461+micro-1)//micro;steps=steps_epoch*3
    assert done['optimizer_steps']==cfg['optimizer_steps']==steps and cfg['updates_per_epoch']==steps_epoch
    assert done['sample_presentations']==16461*3 and cfg['warmup_steps']==int(np.ceil(.1*steps))
    assert len({tuple(t['input_order_sha256_per_rank']) for t in pilot['trials']})==1
    tracking={};scores=[]
    for lr in c.LRS:
        dest=path/'lr_search'/('lr_'+c.lr_tag(lr));tp=read(dest/'training_protocol.json');td=read(dest/'training_complete.json')
        assert td['optimizer_steps']==tp['optimizer_steps']==100 and tp['warmup_steps']==10
        assert tp['initial_lora_sha256']==cfg['initial_lora_sha256'] and tp['source_sha256']==run.sources() and tp['peak_lora_lr']==lr
        assert tp['microbatch_per_gpu']==micro and tp['gradient_accumulation']==1 and td['sample_presentations']==100*micro
        assert tp['world_size']==1 and tp['cuda_visible_device_count']==1 and tp['gpu_uuid']==c.GPU_UUID
        tracking[c.lr_tag(lr)]=check_tracking(dest/'tracking',100,lr,10)
        m,_=check_result(key,'pilot',lr,'validation');scores.append({'lr':lr,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
    selection=read(path/'lr_selection.json');best=max(scores,key=lambda r:(r['macro_f1'],-r['lr']))
    assert scores==selection['scores'] and best['lr']==selection['best_lr']==cfg['peak_lora_lr'] and not selection['test_used_for_selection']
    tracking['formal']=check_tracking(path/'tracking',steps,cfg['peak_lora_lr'],cfg['warmup_steps']);vals=[]
    for epoch in [1,2,3]:
        m,_=check_result(key,'formal',epoch,'validation');vals.append({'epoch':epoch,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
    cs=read(path/'checkpoint_selection.json');best=max(vals,key=lambda r:r['macro_f1'])
    assert cs['scores']==vals and best['epoch']==cs['best_epoch'] and not cs['test_used_for_selection']
    m,df=check_result(key,'formal',best['epoch'],'test');weight_proof=check_weights(key,best['epoch'])
    prior=read(path/'prior_files_sha256.json')
    for name,digest in prior.items():assert c.sha(ROOT/name)==digest,name
    for split in ['train','validation','test']:
        for r in c.load_rows(split).itertuples():assert c.sha(ROOT/r.image_file)==r.image_sha256
    dogs=df.dog_id.to_numpy();unique=np.unique(dogs);y=df.label.to_numpy();pred=df.prediction.fillna('__INVALID__').to_numpy()
    cms=np.stack([confusion_matrix(y[dogs==d],pred[dogs==d],labels=c.classes()+['__INVALID__'])[:7,:] for d in unique])
    def metrics(a):
        tp=np.diag(a[:,:7]);support=a.sum(1);den=support+a[:,:7].sum(0)
        return {'macro_f1':float(np.divide(2*tp,den,out=np.zeros(7),where=den>0).mean()),
                'balanced_accuracy':float(np.divide(tp,support,out=np.zeros(7),where=support>0)[support>0].mean()),'accuracy':float(tp.sum()/support.sum())}
    rng=np.random.default_rng(c.SEED);boot=[metrics(cms[rng.integers(0,len(unique),len(unique))].sum(0)) for _ in range(2000)]
    for name in ['macro_f1','balanced_accuracy','accuracy']:assert np.allclose(m['dog_bootstrap_95ci'][name],np.quantile([b[name] for b in boot],[.025,.975]),atol=1e-12,rtol=0)
    report={'model':c.NAMES[key],'test':m,'lr_search':scores,'formal_validation':vals,'selected_lr':selection['best_lr'],'selected_epoch':best['epoch'],
             'audit':{'weights':weight_proof,'raw_csv_metrics_and_protocols':True,'tracking':tracking,'identical_candidate_initializations_and_input_orders':True,
                      'single_gpu_uuid':c.GPU_UUID,'frozen_base_memory_hash':formal['qwen_base_sha256_after'],'historical_files_unchanged':len(prior)}}
    c.write(path/'comparison.json',report)
    text=f"""# Qwen3.8-27B sixplot + LoRA 结果

最佳LR {selection['best_lr']:g}；选中epoch {best['epoch']}。

|测试指标|结果|
|---|---:|
|Macro-F1|{m['macro_f1']:.6f}|
|Balanced accuracy|{m['balanced_accuracy']:.6f}|
|Accuracy|{m['accuracy']:.6f}|

3,708测试窗，vLLM英文贪心生成；完整验证选LR及epoch，测试不参与选优。单物理GPU0，batch{micro}，梯度累积1，3epoch，r4/alpha8/dropout0.05，ViT/投影/语言混合注意力LoRA，基础模型冻结，GC开启，10%linear warmup+90%cosine。W&B离线记录及各阶段loss_curve.png均保留。普通图保持原1280×1152六子图与长英文波形提示。

独立验证全部414目标矩阵合并、非目标权重、官方基础文件SHA、原回答/CSV/指标、历史文件、候选初始状态与输入顺序、每步LR/损失日志。详细 comparison.json。
"""
    p=path/'README.md';s=p.read_text().split('<!-- final-results -->')[0]
    s='\n'.join('状态：学习率筛选、正式训练、测试及独立检查全部完成。' if line.startswith('状态：') else line for line in s.split('\n'))
    p.write_text(s+'\n<!-- final-results -->\n'+text);(c.model_dir(key)/'README.md').write_text(text+'\nRoot model weights merged for vLLM; adapter/ preserves selected LoRA.\n')
    p=ROOT/'HANDOFF.md';s=p.read_text().replace('## 2026-11-05 Qwen3.8-27B sixplot单卡LoRA（进行中）','## 2026-11-05 Qwen3.8-27B sixplot单卡LoRA（已完成）',1)
    s=s.replace('<!-- sixplot38-status -->',f"已完成：LR={selection['best_lr']:g}，epoch={best['epoch']}，测试macro-F1={m['macro_f1']:.6f}。详细results/qwen3.8_27B_lora_sixplot_single_gpu/comparison.json。",1);p.write_text(s)
    c.write(path/'complete.json',{'state':'complete','independent_validation_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':run.sources()})
    c.write(path/'progress.json',{'state':'complete','completed_utc':datetime.now(timezone.utc).isoformat()})
    print('QWEN3.8 SIXPLOT SINGLE-GPU EXPERIMENT VALIDATED AND COMPLETE',flush=True)


if __name__=='__main__':main()
