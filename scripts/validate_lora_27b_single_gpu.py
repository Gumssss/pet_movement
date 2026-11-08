"""Audit raw generation, LR selection, checkpoints, frozen bases and merged weights."""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import json
from pathlib import Path

import imu2img_lora27_single_common as c
import lora_27b_single_gpu as run
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
    folder=run.image_folder(key,stage,tag,split);im=pd.read_csv(folder/'images.csv')
    assert len(records)==len(df)==df.window_id.nunique()==len(w)==p['expected_windows']
    assert df.window_id.tolist()==w.window_id.tolist()==im.window_id.tolist()==[r['window_id'] for r in records]
    assert df.label.tolist()==w.label.tolist() and df.dog_id.tolist()==w.dog_id.tolist()
    ckpt=c.checkpoint(key,stage,tag)
    assert p['key']==key and p['stage']==stage and p['split']==split
    assert p['adapter_sha256']==c.sha(ckpt/'adapter_model.safetensors')
    assert p['converter_sha256']==c.sha(ckpt/'converter.safetensors')==read(folder/'render_protocol.json')['converter_sha256']
    train_dir=c.out(key)/('lr_search/lr_'+c.lr_tag(float(tag))) if stage=='pilot' else c.out(key)
    tp=read(train_dir/'training_protocol.json')
    assert p['training_protocol_sha256']==c.sha(train_dir/'training_protocol.json')
    assert p['windows_sha256']==c.sha(ROOT/'dataset/splits/windows.csv') and p['splits_sha256']==c.sha(ROOT/'dataset/splits/dog_splits.json')
    assert p['image_manifest_sha256']==c.sha(folder/'images.csv') and p['render_protocol_sha256']==c.sha(folder/'render_protocol.json')
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
            assert image.size==(224,224) and image.mode=='RGB' and image.format=='WEBP'
            assert (np.asarray(image)[:,:,2]==128).all() and not any(k in image.info for k in ['exif','xmp','icc_profile'])
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
    converter=target/'converter.safetensors'
    assert c.sha(converter)==c.sha(c.checkpoint(key,'formal',epoch)/'converter.safetensors')
    warm=load_file(str(c.WARM));new=load_file(str(converter));assert warm.keys()==new.keys()
    changes={s:sum(float((new[k]-warm[k]).abs().sum()) for k in new if k.startswith(s)) for s in ['encoders','decoders']}
    assert all(v>0 for v in changes.values())
    expected=read(ROOT/'models/model_manifest.json')[c.NAMES[key]]
    official=[]
    for f in expected['files']:
        if f.get('sha256'):
            assert c.sha(c.base_dir(key)/f['path'])==f['sha256'];official.append(f['path'])
    assert sum(n.endswith('.safetensors') for n in official)==len(set(bm.values()))
    return {'merged_lora_modules_verified':dict(scopes),'all_other_exported_weights_match_base':unchanged,
            'unused_mtp_keys_omitted':sorted(omitted),'converter_changes_l1':changes,
            'official_source_file_sha256_verified':official,'merge_rounding':'FP32 LoRA delta cast to BF16, then added to BF16 base'}


def main():
    import numpy as np
    import torch
    from sklearn.metrics import confusion_matrix
    torch.set_num_threads(8)
    assert (c.SUITE/'generation_complete.json').exists() and c.KEYS==['qwen38_27b']
    protocol=read(c.SUITE/'protocol.json');batch=read(c.SUITE/'batch_selection.json')
    assert protocol['mode']=='joint' and protocol['gradient_checkpointing'] and protocol['formal_epochs']==3
    assert protocol['world_size']==batch['world_size']==1 and protocol['gradient_accumulation']==batch['gradient_accumulation']==2
    assert protocol['warm_converter_sha256']==c.sha(c.WARM) and protocol['cache_sha256']==c.sha(ROOT/'dataset/cache/pet_windows.npz')
    assert read(c.SUITE/'source_sha256.json')==run.sources()
    key='qwen38_27b';path=c.out(key);pilot=read(path/'pilot_training_complete.json');formal=read(path/'formal_training_verified.json')
    cfg=read(path/'training_protocol.json');done=read(path/'training_complete.json');technical=read(path/'single_gpu_check.json')
    assert technical['microbatch_per_gpu']==96 and technical['gradient_accumulation']==2 and technical['optimizer_steps']==3
    gradient=read(path/'single_gpu_gradient_check.json')
    assert all(v>0 for v in [*gradient['lora'].values(),*gradient['converter'].values()]) and gradient['qwen_base_gradients_none']
    assert pilot['qwen_base_sha256_before']==pilot['qwen_base_sha256_after']==formal['qwen_base_sha256_before']==formal['qwen_base_sha256_after']
    assert pilot['initial_lora_sha256']==formal['initial_lora_sha256']==technical['initial_lora_sha256']
    assert pilot['source_sha256']==formal['source_sha256']==cfg['source_sha256']==run.sources()
    assert cfg['microbatch_per_gpu']==batch['microbatch_per_gpu']==96 and cfg['effective_batch_size']==batch['effective_batch_size']==192
    assert cfg['cuda_visible_device_count']==1 and cfg['gpu_uuid']==batch['gpu_uuid']=='GPU-0ded27d3-e266-f20b-2476-f2980be2980e'
    assert cfg['gradient_checkpointing'] and cfg['lora']['gradient_checkpointing'] and cfg['lora']['trainable_parameters']==14716800
    assert done['optimizer_steps']==cfg['optimizer_steps']==258 and cfg['updates_per_epoch']==86 and cfg['microbatches_per_epoch']==172
    assert done['sample_presentations']==16461*3 and cfg['warmup_steps']==26
    assert len({tuple(t['input_order_sha256_per_rank']) for t in pilot['trials']})==1
    scores=[]
    for lr in c.LRS:
        dest=path/'lr_search'/('lr_'+c.lr_tag(lr));tp=read(dest/'training_protocol.json');td=read(dest/'training_complete.json')
        assert td['optimizer_steps']==tp['optimizer_steps']==100 and tp['warmup_steps']==10
        assert tp['initial_lora_sha256']==cfg['initial_lora_sha256'] and tp['warm_converter_sha256']==c.sha(c.WARM)
        assert tp['source_sha256']==run.sources() and tp['peak_lora_lr']==lr and tp['peak_converter_lr']==lr*.05
        assert tp['microbatch_per_gpu']==96 and tp['gradient_accumulation']==2 and td['sample_presentations']==19149
        assert tp['world_size']==1 and tp['cuda_visible_device_count']==1 and tp['gpu_uuid']==cfg['gpu_uuid']
        m,_=check_result(key,'pilot',lr,'validation');scores.append({'lr':lr,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
    selection=read(path/'lr_selection.json');best=max(scores,key=lambda r:(r['macro_f1'],-r['lr']))
    assert scores==selection['scores'] and best['lr']==selection['best_lr']==cfg['peak_lora_lr'] and not selection['test_used_for_selection']
    vals=[]
    for epoch in [1,2,3]:
        m,_=check_result(key,'formal',epoch,'validation');vals.append({'epoch':epoch,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
    cs=read(path/'checkpoint_selection.json');best=max(vals,key=lambda r:r['macro_f1'])
    assert cs['scores']==vals and best['epoch']==cs['best_epoch'] and not cs['test_used_for_selection']
    m,df=check_result(key,'formal',best['epoch'],'test');weight_proof=check_weights(key,best['epoch'])
    prior=read(c.SUITE/'prior_files_sha256.json')
    for name,digest in prior.items():assert c.sha(ROOT/name)==digest,name
    dogs=df.dog_id.to_numpy();unique=np.unique(dogs);y=df.label.to_numpy();pred=df.prediction.fillna('__INVALID__').to_numpy()
    cms=np.stack([confusion_matrix(y[dogs==d],pred[dogs==d],labels=c.classes()+['__INVALID__'])[:7,:] for d in unique])
    def metrics(a):
        tp=np.diag(a[:,:7]);support=a.sum(1);den=support+a[:,:7].sum(0)
        return {'macro_f1':float(np.divide(2*tp,den,out=np.zeros(7),where=den>0).mean()),
                'balanced_accuracy':float(np.divide(tp,support,out=np.zeros(7),where=support>0)[support>0].mean()),
                'accuracy':float(tp.sum()/support.sum())}
    rng=np.random.default_rng(c.SEED);boot=[metrics(cms[rng.integers(0,len(unique),len(unique))].sum(0)) for _ in range(2000)]
    for name in ['macro_f1','balanced_accuracy','accuracy']:
        assert np.allclose(m['dog_bootstrap_95ci'][name],np.quantile([b[name] for b in boot],[.025,.975]),atol=1e-12,rtol=0)
    comparison={'model':c.NAMES[key],'test':m,'lr_search':scores,'formal_validation':vals,'selected_lr':selection['best_lr'],
                'selected_epoch':best['epoch'],'audit':{'weights':weight_proof,'raw_csv_metrics_and_protocols':True,
                'identical_candidate_initializations_and_input_orders':True,'single_physical_gpu':cfg['gpu_uuid'],
                'single_gpu_gradient_accumulation_verified':True,'frozen_base_memory_hash':formal['qwen_base_sha256_after'],
                'historical_files_unchanged':len(prior)}}
    c.write(c.SUITE/'comparison.json',comparison)
    report=f"""# Qwen3.8-27B 单卡联合LoRA实验结果

- LoRA LR：{selection['best_lr']:g}；转换器 LR：{selection['best_lr']*.05:g}；选中 epoch：{best['epoch']}。
- 测试 Macro-F1：{m['macro_f1']:.6f}；Balanced accuracy：{m['balanced_accuracy']:.6f}；Accuracy：{m['accuracy']:.6f}；窗口数：3708。
- 单物理GPU0，microbatch96、梯度累积2、有效batch192、gradient checkpointing。ViT/visual projection/language attention LoRA r4 alpha8 dropout0.05，转换器联合训练，基础模型冻结。
- 三个LR各100步，完整验证Macro-F1选LR，重新初始化正式3epoch后再次用验证集选轮，测试不参与选优。
- 独立核验通过：414个合并矩阵、非目标权重、原回答/预测/指标、犬只bootstrap、初始化/输入顺序、历史文件及官方原权重SHA。

完整记录见 comparison.json，所有阶段均只使用同一物理GPU。双卡历史结果独立保留，不混入本次指标。
"""
    (path/'README.md').write_text(report)
    p=c.SUITE/'README.md';text=p.read_text().split('<!-- final-results -->')[0]
    text='\n'.join('状态：单卡实验、测试与独立核验全部完成。' if line.startswith('状态：') else line for line in text.split('\n'))
    p.write_text(text+'\n<!-- final-results -->\n'+report)
    (c.model_dir(key)/'README.md').write_text(report+'\nRoot weights merged for vLLM; adapter/ retains selected LoRA; converter.safetensors is the matching FP32 converter.\n')
    (c.data_dir(key)/'prompt.txt').write_bytes(c.PROMPT.read_bytes())
    (c.data_dir(key)/'method.md').write_text('224×224 losslessRGB WebP, 49 chronological IMU patches; triaxial RMS acceleration and gyro, center196 of original200 points; fixed blue128.\n'+report)
    p=ROOT/'HANDOFF.md';s=p.read_text().replace('## 2026-11-05 Qwen3.8-27B单卡联合LoRA（进行中）','## 2026-11-05 Qwen3.8-27B单卡联合LoRA（已完成）',1)
    s=s.replace('<!-- single-gpu-status -->',f"已完成：LR={selection['best_lr']:g}，epoch={best['epoch']}，测试macro-F1={m['macro_f1']:.6f}；详见results/lora_27b_single_gpu/。",1);p.write_text(s)
    for name in ['README.md','results/README.md']:
        p=ROOT/name;s=p.read_text()
        if '<!-- single-gpu-completed -->' not in s:
            link='results/lora_27b_single_gpu/README.md' if name=='README.md' else 'lora_27b_single_gpu/README.md'
            p.write_text(s+f'\n<!-- single-gpu-completed -->\nQwen3.8-27B单卡联合LoRA已完成，见 [实验报告]({link})。\n')
    c.write(c.SUITE/'complete.json',{'state':'complete','keys':c.KEYS,'independent_validation_passed':True,
                 'completed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':run.sources()})
    c.write(c.SUITE/'progress.json',{'state':'complete','completed_utc':datetime.now(timezone.utc).isoformat()})
    print('SINGLE-GPU QWEN3.8 EXPERIMENT VALIDATED AND COMPLETE',flush=True)


if __name__=='__main__':main()
