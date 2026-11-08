"""Independently audit the three LoRA experiments and write final comparison reports."""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.venv/lora-packages'))
import lora_comparison as run


def check_result(key,epoch,split):
    import pandas as pd
    from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score
    import qwen_classify as q
    dest=run.out(key) if split=='test' else run.out(key)/'validation'/f'epoch_{epoch}'
    protocol=json.loads((dest/'protocol.json').read_text());m=json.loads((dest/'metrics.json').read_text())
    records=[json.loads(line) for line in (dest/'responses.jsonl').read_text().splitlines()]
    df=pd.read_csv(dest/'predictions.csv');windows=pd.read_csv(ROOT/'dataset/splits/windows.csv')
    windows=windows[windows.split.eq(split)];im=pd.read_csv(ROOT/protocol['image_dir']/'images.csv')
    im=im[im.window_id.isin(windows.window_id)]
    assert len(records)==len(df)==df.window_id.nunique()==len(windows)==protocol['expected_windows']
    assert df.window_id.tolist()==windows.window_id.tolist()==im.window_id.tolist()==[r['window_id'] for r in records]
    assert df.label.tolist()==windows.label.tolist() and df.dog_id.tolist()==windows.dog_id.tolist()
    for name in ['image_file','image_sha256']:assert df[name].tolist()==im[name].tolist()
    assert protocol['key']==key and protocol['epoch']==epoch and protocol['split']==split
    assert protocol['adapter_sha256']==run.sha(run.model_dir(key)/'checkpoints'/f'epoch_{epoch}'/'adapter_model.safetensors')
    assert protocol['training_protocol_sha256']==run.sha(run.out(key)/'training_protocol.json')
    assert protocol['windows_sha256']==run.sha(ROOT/'dataset/splits/windows.csv')
    assert protocol['split_sha256']==run.sha(ROOT/'dataset/splits/dog_splits.json')
    assert protocol['image_manifest_sha256']==run.sha(ROOT/protocol['image_dir']/'images.csv')
    assert protocol['prompt_sha256']==run.sha(dest/'prompt.txt')==run.sha(run.out(key)/'prompt.txt')
    combined=df.merge(pd.DataFrame(records),on='window_id',validate='one_to_one',suffixes=('_csv','_raw'))
    for name in ['prediction','status','response','input_tokens','generated_tokens','image_sha256']:
        assert combined[name+'_csv'].fillna('').tolist()==combined[name+'_raw'].fillna('').tolist(),name
    for r in records:
        pred,status=q.parse(r['response'],run.classes(),r['status']=='truncated')
        assert pred==r['prediction'] and status==r['status']
        assert r['inference_backend']=='vllm'
    truth=df.label;pred=df.prediction.fillna('__INVALID__');cs=run.classes()
    scores={'macro_f1':f1_score(truth,pred,labels=cs,average='macro',zero_division=0),
            'balanced_accuracy':recall_score(truth,pred,labels=cs,average='macro',zero_division=0),
            'accuracy':accuracy_score(truth,pred)}
    assert all(abs(m[k]-v)<1e-12 for k,v in scores.items())
    assert m['confusion_matrix']==confusion_matrix(truth,pred,labels=cs+['__INVALID__'])[:7,:].tolist()
    assert m['status_counts']==dict(Counter(r['status'] for r in records))
    train_cfg=json.loads((run.out(key)/'training_protocol.json').read_text())
    assert all(r['input_tokens']==train_cfg['processor']['prefix_tokens'] for r in records),'Train/deployment token geometry differs.'
    runtime=json.loads((dest/'runtime.json').read_text())
    assert sum(b['n'] for s in runtime['sessions'] for b in s['batch_timings'])==len(records)
    assert all(s['code_sha256']==run.sha(Path(run.__file__)) for s in runtime['sessions'])
    assert all(s['shared_generation_code_sha256']==run.sha(ROOT/'scripts/qwen_classify.py') for s in runtime['sessions'])
    return m,df,{'raw_csv_metrics_consistent':True,'inputs_and_labels_verified':True,'raw_parser_verified':True,
                  'all_failures_counted_in_denominator':True,'windows':len(records),'scores':scores}


def check_model(key,epoch):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    target=run.model_dir(key);adapter=load_file(str(target/'adapter/adapter_model.safetensors'))
    assert run.sha(target/'adapter/adapter_model.safetensors')==run.sha(target/'checkpoints'/f'epoch_{epoch}'/'adapter_model.safetensors')
    assert len(adapter)==500 and sum(p.numel() for p in adapter.values())==5226496
    cfg=json.loads((target/'adapter/adapter_config.json').read_text())
    assert cfg['r']==4 and cfg['lora_alpha']==8 and cfg['lora_dropout']==.05
    base_map=json.loads((run.BASE/'model.safetensors.index.json').read_text())['weight_map']
    merged_map=json.loads((target/'model.safetensors.index.json').read_text())['weight_map']
    omitted=set(base_map)-set(merged_map)
    assert not set(merged_map)-set(base_map)
    assert len(omitted)==15 and all(n.startswith('mtp.') for n in omitted),omitted
    expected={}
    for name in adapter:
        if not name.endswith('.lora_A.weight'):continue
        stem=name.removesuffix('.lora_A.weight')
        base_name=stem.removeprefix('base_model.model.')+'.weight'
        expected[base_name]=adapter[stem+'.lora_B.weight']@adapter[name]*2
    assert len(expected)==250
    unchanged=0;changed=0;changed_scopes=Counter()
    with ExitStack() as stack:
        base={name:stack.enter_context(safe_open(str(run.BASE/name),framework='pt')) for name in set(base_map.values())}
        merged={name:stack.enter_context(safe_open(str(target/name),framework='pt')) for name in set(merged_map.values())}
        for name in merged_map:
            source=base[base_map[name]].get_tensor(name);b=merged[merged_map[name]].get_tensor(name)
            a=source.to(b.dtype)
            if name in expected:
                # Exact PEFT0.21.2 safe_merge arithmetic: cast delta to BF16, then add.
                wanted=a+expected[name].to(a.dtype)
                assert torch.equal(b,wanted),name
                changed+=1;changed_scopes[run.old.scope(name)]+=1
            else:
                assert torch.equal(a,b),name;unchanged+=1
    converter={}
    if key!='sixplot':
        converter_sha=run.sha(target/'converter.safetensors')
        assert converter_sha==run.sha(target/'checkpoints'/f'epoch_{epoch}'/'converter.safetensors')
        if key=='imu2img_joint':
            for split in ['validation','test']:
                rendered=json.loads((run.image_folder(key,epoch,split)/'render_protocol.json').read_text())
                assert rendered['converter_sha256']==converter_sha
        else:
            assert converter_sha==run.sha(run.WARM)
            rendered=json.loads((run.data_dir(key)/'render_protocol.json').read_text())
            assert rendered['converter_weights_sha256']==converter_sha
        warm=load_file(str(run.WARM));new=load_file(str(target/'converter.safetensors'))
        assert warm.keys()==new.keys()
        converter={s:sum(float((new[k]-warm[k]).abs().sum()) for k in new if k.startswith(s)) for s in ['encoders','decoders']}
        if key=='imu2img_frozen':assert all(v==0 for v in converter.values())
        else:assert all(v>0 for v in converter.values())
    return {'all_250_merged_weights_equal_base_plus_lora':True,'non_target_weights_match_base_in_export_dtype':unchanged,
            'base_only_unused_mtp_weights':sorted(omitted),
            'merged_projection_counts':dict(changed_scopes),'converter_weight_changes_l1':converter,
            'selected_adapter_parameters':5226496,'rounding':'Source norms may be FP32; cast source to export BF16. Cast FP32 LoRA delta to BF16 before BF16 addition, matching PEFT safe_merge.'}


def main():
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from sklearn.metrics import confusion_matrix
    torch.set_num_threads(8)
    assert (run.SUITE/'generation_complete.json').exists()
    protocol=json.loads((run.SUITE/'protocol.json').read_text());results={};training={};selections={};frames={};validations={}
    initial_adapters=[];initial_bases=[]
    for key in run.KEYS:
        path=run.out(key);train=json.loads((path/'training_complete.json').read_text());cfg=json.loads((path/'training_protocol.json').read_text())
        inputs=json.loads((path/'input_protocol.json').read_text())
        assert inputs['cache_sha256']==run.sha(ROOT/'dataset/cache/pet_windows.npz')
        assert inputs['prompt_sha256']==run.sha(path/'prompt.txt')
        assert inputs['render_protocol_sha256']==run.sha(run.data_dir(key)/'render_protocol.json')
        assert inputs['test_manifest_sha256']==run.sha(run.data_dir(key)/'images.csv')
        if key=='sixplot':
            manifest=run.data_dir(key)/'supervised/images.csv'
            assert inputs['train_validation_image_manifest_sha256']==run.sha(manifest)
            for row in pd.read_csv(manifest).itertuples():assert run.sha(ROOT/row.image_file)==row.image_sha256
        assert train['epochs']==3 and train['optimizer_steps']==1545
        assert train['qwen_base_versions_unchanged'] and train['qwen_base_frozen_hash_before']==train['qwen_base_frozen_hash_after']
        assert train['trainable_parameters']==(6062080 if key=='imu2img_joint' else 5226496)
        assert train['code_sha256']==run.sha(Path(run.__file__))==cfg['code_sha256']
        initial_adapters.append(train['initial_lora_sha256']);initial_bases.append(train['qwen_base_frozen_hash_before'])
        assert cfg['warm_converter_sha256']==protocol['warm_converter_sha256']==run.sha(run.WARM)
        assert cfg['gradient_accumulation']==4 and cfg['effective_batch_size']==32 and cfg['microbatch_per_gpu']==4
        smoke=json.loads((path/'backward_check.json').read_text());assert smoke['qwen_base_frozen_no_grad_and_versions_unchanged']
        assert all(v>0 for v in smoke['lora_grad_norms'].values())
        if key=='imu2img_joint':assert all(v>0 for v in smoke['converter_grad_norms'].values())
        if key=='imu2img_frozen':assert all(v==0 for v in smoke['converter_grad_norms'].values())
        selection=json.loads((path/'checkpoint_selection.json').read_text());scores=[]
        for epoch in [1,2,3]:
            m,df,_=check_result(key,epoch,'validation');scores.append({'epoch':epoch,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
        assert scores==selection['scores'] and not selection['test_used_for_selection']
        best=max(scores,key=lambda x:x['macro_f1']);assert best['epoch']==selection['best_epoch']
        m,df,check=check_result(key,best['epoch'],'test');check.update(check_model(key,best['epoch']))
        p=json.loads((path/'protocol.json').read_text())
        for row in df.itertuples():
            file=ROOT/row.image_file;assert run.sha(file)==row.image_sha256
            with Image.open(file) as image:
                assert list(image.size)==p['image_size']
                assert not any(k in image.info for k in ['exif','xmp','icc_profile'])
                if key!='sixplot':assert image.getchannel('B').getextrema()==(128,128)
        write_check={**check,'same_fresh_lora_initialization_checked_in_suite':True,'qwen_base_frozen':True,
                     'checkpoint_selected_on_full_validation':True,'selected_epoch':best['epoch']}
        run.write(path/'validation.json',write_check)
        results[key]=m;training[key]=train;selections[key]=selection;frames[key]=df;validations[key]=write_check
        print('VERIFIED '+key,flush=True)
    assert len(set(initial_adapters))==len(set(initial_bases))==1
    assert (run.out('imu2img_frozen')/'prompt.txt').read_bytes()==(run.out('imu2img_joint')/'prompt.txt').read_bytes()
    prior=json.loads((run.SUITE/'prior_files_sha256.json').read_text())
    for path,digest in prior.items():assert run.sha(ROOT/path)==digest,path
    manifest=json.loads((ROOT/'models/model_manifest.json').read_text())['Qwen3.5-4B']
    for file in manifest['files']:
        if file['path'].endswith('.safetensors'):assert run.sha(run.BASE/file['path'])==file['sha256']
    # Paired resampling of the same nine test dogs, fixed7class macro-F1.
    dogs=np.unique(frames['sixplot'].dog_id);assert len(dogs)==9
    cms={}
    for key,df in frames.items():
        cms[key]=np.stack([confusion_matrix(df[df.dog_id.eq(d)].label,df[df.dog_id.eq(d)].prediction.fillna('__INVALID__'),
                     labels=run.classes()+['__INVALID__'])[:7,:] for d in dogs])
    def score(cm):
        tp=np.diag(cm[:,:7]);den=cm.sum(1)+cm[:,:7].sum(0)
        return np.divide(2*tp,den,out=np.zeros(7),where=den>0).mean()
    rng=np.random.default_rng(run.SEED);boot=[]
    for _ in range(2000):
        pick=rng.integers(0,9,9);boot.append({k:float(score(v[pick].sum(0))) for k,v in cms.items()})
    differences={}
    for a,b in [('imu2img_frozen','sixplot'),('imu2img_joint','imu2img_frozen')]:
        differences[a+'_minus_'+b]={'macro_f1_difference':results[a]['macro_f1']-results[b]['macro_f1'],
            'paired_dog_bootstrap_95ci':np.quantile([r[a]-r[b] for r in boot],[.025,.975]).tolist()}
    summary={'experiments':{k:{'scores':{m:results[k][m] for m in ['macro_f1','balanced_accuracy','accuracy']},
                              'selected_epoch':selections[k]['best_epoch'],'status_counts':results[k]['status_counts'],
                              'prediction_counts':results[k]['prediction_counts'],
                              'training_session_wall_seconds':training[k]['session_wall_seconds'],
                              'training_peak_allocated_GiB_rank0':json.loads((run.out(k)/'training_progress.json').read_text())['peak_allocated_GiB_rank0']} for k in run.KEYS},
             'paired_differences':differences,'same_initial_lora_sha256':initial_adapters[0],
             'same_frozen_qwen_base_sha256':initial_bases[0],'historical_files_unchanged':len(prior),
             'qwen_official_base_files_unchanged':True,'test_windows':3708,'test_dogs':9,'single_training_seed':run.SEED,
             'completed_utc':datetime.now(timezone.utc).isoformat()}
    run.write(run.SUITE/'comparison.json',summary)
    joint_data=ROOT/'dataset/vlm_imu2img_qwen_lora_joint'
    (joint_data/'prompt.txt').write_bytes((run.out('imu2img_joint')/'prompt.txt').read_bytes())
    (joint_data/'method.md').write_text(f'''# 与LoRA联合训练的IMU2IMG图片

此目录只保存第三组联合转换器的验证和测试图片。转换器从原冻结Qwen对齐epoch3初始化，与全新r4/alpha8 LoRA联合训练；Qwen基础权重冻结。输入仍为原200点窗口经两模态三轴RMS、中心196点、49×4点patch，输出224×224 RGB无损WebP，B固定128；BF16前向、固定8位round，与训练可微桥接的前向一致。

validation/epoch_1至epoch_3分别对应三个checkpoint的全部4727验证窗。按完整验证macro-F1选epoch{selections['imu2img_joint']['best_epoch']}，test/images/保存其3708测试图片。各images.csv记录窗口来源与SHA，各render_protocol.json记录转换器SHA。匿名文件名不含标签、犬只或Task，元数据及清单不会传入VLM。prompt.txt是实际使用的简短英文文本，与第二组完全一致。

[训练和完整结果](../../results/qwen3.5_4B_lora_imu2img_joint/README.md)、[三组比较](../../results/lora_comparison/README.md)、[最终模型](../../models/Qwen3.5-4B-lora-imu2img-joint/README.md)。训练图片实时生成，不另存全部训练图；验证和测试图完整保存。此表示仍为学习纹理，图片亮度不能直接当作原波形幅度。
''')
    names={'sixplot':'①普通六子图＋LoRA','imu2img_frozen':'②固定IMU2IMG_qwen＋LoRA','imu2img_joint':'③联合IMU2IMG_qwen＋LoRA'}
    table='|实验|测试Macro-F1|Balanced accuracy|Accuracy|选中epoch|\n|---|---:|---:|---:|---:|\n'
    for key in run.KEYS:
        m=results[key];table+=f"|{names[key]}|{m['macro_f1']:.6f}|{m['balanced_accuracy']:.6f}|{m['accuracy']:.6f}|{selections[key]['best_epoch']}|\n"
    for key in run.KEYS:
        path=run.out(key);m=results[key];s=selections[key];t=training[key]
        text=f'''# {names[key]}

本轮始于2026-11-04，现已完成3epoch、1545步、完整验证选模与全部3708测试窗口。正式生成使用vLLM0.26.0，全部测试预测/原回答/指标已独立核验。

## 三组对照

'''+table+f'''
本组状态{json.dumps(m['status_counts'],ensure_ascii=False)}，解析成功率{m['parse_success_rate']:.2%}，预测分布{json.dumps(m['prediction_counts'],ensure_ascii=False)}。所有解析失败均计入3708窗分母，没有按标签重试。

## 训练与输入

三组均从同一原始Qwen3.5-4B初始化相同seed的新LoRA，不加载旧ACL LoRA。250个ViT/视觉投影/语言混合注意力线性模块，r4、alpha8、dropout0.05，5226496个LoRA参数；Qwen所有基础权重冻结。仅联合组额外更新835584个转换器参数。两组IMU2IMG从同一已对齐epoch3转换器热启动，固定组参数全程不变。普通组是原六轴1280×1152波形图/1440视觉token＋原英文波形特征提示；IMU两组224×224/49视觉token＋完全相同简短英文提示。

相同16461训练、4727验证、3708测试窗，犬只隔离。3epoch，有效batch32（2GPU×micro4×累积4），AdamW、LoRA LR2e-4、联合转换器LR1e-5、wd0.01、5%warmup线性衰减、clip1。BF16基础模型、FP32LoRA/优化器；答案JSON＋EOS损失，输入和padding屏蔽，无类别重采样或增强。普通图开启非重入梯度检查点；IMU两组不开。训练前普通图不启用检查点时OOM，发生在任何优化前，诊断日志保留。

IMU训练图片从RMS196点实时生成，两组前向处理一致。联合图像使用可微8位round直通估计及官方融合归一化/patch顺序，转换器BF16前向、FP32参数，保存无损WebP；冻结组没有转换器梯度。Qwen基础参数完整内存SHA前后相同，两个IMU组初始化、三组LoRA初始化指纹分别一致。选中adapter全部250处合并矩阵逐位等于base＋LoRA，其余参与生成的基础权重与原权重按BF16导出精度逐位一致；未使用的15个MTP张量不在HF导出中。原官方权重文件SHA未变。

每epoch验证答案NLL仅作监控，最后对三个checkpoint分别进行4727窗完整vLLM生成，以macro-F1最高选epoch{s['best_epoch']}；同分选较早者。测试没有参与选择。温度0、关闭思考、最多32输出token，使用相同解析函数。训练墙钟{t['session_wall_seconds']:.2f}秒；输入token和梯度检查点不同，不能将训练速度差异只归因于可训练参数量。各阶段计时保存在runtime及训练日志中。

|epoch|验证Macro-F1|验证Balanced accuracy|验证Accuracy|
|---|---:|---:|---:|
'''
        for row in s['scores']:text+=f"|{row['epoch']}|{row['macro_f1']:.6f}|{row['balanced_accuracy']:.6f}|{row['accuracy']:.6f}|\n"
        text+='\n## 每类测试结果\n\n|类别|Precision|Recall|F1|窗口数|\n|---|---:|---:|---:|---:|\n'
        for cls in run.classes():
            c=m['classification_report'][cls];text+=f"|{cls}|{c['precision']:.4f}|{c['recall']:.4f}|{c['f1-score']:.4f}|{int(c['support'])}|\n"
        text+='\n按9只测试犬整体bootstrap2000次的95%区间：\n\n'
        for name,ci in m['dog_bootstrap_95ci'].items():text+=f'- {name}：[{ci[0]:.6f}, {ci[1]:.6f}]\n'
        text+='''
## 文件与边界

- [predictions.csv](predictions.csv)、[responses.jsonl](responses.jsonl)、[metrics.json](metrics.json)、[validation.json](validation.json)：完整预测、原输出、各类/犬指标/区间与独立校验。
- [training_protocol.json](training_protocol.json)、[protocol.json](protocol.json)、[prompt.txt](prompt.txt)：固定训练/生成配置与实际英文输入。
- validation/：三个epoch各4727条完整验证输出；checkpoint_selection.json：选模；training_complete.json：冻结基础权重证明。
- [三组汇总](../lora_comparison/README.md)：配对犬只bootstrap差值、初始化公平性和计算边界。

①②同时改变图片和对应提示，是整体方案比较；②③相同初始化、提示和新增训练预算，仅转换器是否更新不同。IMU转换器此前使用同一批训练标签做过额外监督预训练，标签种类/窗口预算不增加，但总训练计算更多。固定测试集此前已被查看，本轮没有按新测试输出调参。仅一个训练随机种子，犬只bootstrap不覆盖训练随机性。
'''
        text+=f'\n[模型与选中adapter](../../models/{run.model_dir(key).name}/README.md)。\n'
        (path/'README.md').write_text(text)
        model=run.model_dir(key)
        (model/'README.md').write_text(f'''# {model.name}

已完成{names[key]}，选中epoch{s['best_epoch']}。根目录为合并LoRA的完整BF16 Qwen权重，可用vLLM直接加载；adapter/保存所选PEFT适配器；checkpoints/保留三轮adapter；last_state.pt为恢复状态。IMU两组另存converter.safetensors，联合组使用自己的选中转换器生成图片，固定组转换器与原对齐版本一致。

原始Qwen3.5-4B基础权重未更新，r4/alpha8/dropout0.05，ViT/投影/语言混合注意力共250模块、5226496个LoRA参数。联合组再更新835584转换器参数。合并250矩阵和其余基础权重按BF16导出精度独立逐位核验。未使用的15个MTP张量不在HF导出中。

推理使用结果目录内的实际英文prompt.txt。IMU两组设置mm_processor_kwargs的min_pixels=50176、max_pixels=1474560，保持224×224图片和49视觉token；普通组保持原1280×1152图片和1440视觉token。生成配置固定max_model_len=4096、关闭思考、温度0，详见对应结果protocol.json及scripts/lora_comparison.py evaluate入口。

测试Macro-F1={m['macro_f1']:.6f}、BA={m['balanced_accuracy']:.6f}、accuracy={m['accuracy']:.6f}。[完整报告](../../results/{path.name}/README.md)。
''')
        if key!='sixplot':
            for name in ['LICENSE','NOTICE']:(model/name).write_bytes((ROOT/'scripts/vendor/mantis_tgu'/name).read_bytes())
        run.write(path/'complete.json',{'state':'complete','training_validation_selection_test_reports_verified':True,'best_epoch':s['best_epoch']})
    text=(run.SUITE/'README.md').read_text()
    text=text.replace('当前进行中，固定配置见protocol.json；运行日志pipeline.log。用户已授权全部训练、验证、测试，勿重复启动。','已完成全部三组训练、完整验证选模、3708窗测试及独立校验。固定配置见protocol.json；运行日志pipeline.log。实验已完成，不必重跑。')
    text+='\n## 完整测试结果\n\n'+table+'\n三组LoRA初始参数与Qwen基础参数指纹相同，原始基础权重官方SHA未变，'+str(len(prior))+'个历史文件指纹未变。\n\n'
    text+='|配对差值|Macro-F1差值|按犬只配对bootstrap95%区间|\n|---|---:|---|\n'
    for name,d in differences.items():
        ci=d['paired_dog_bootstrap_95ci'];text+=f"|{name}|{d['macro_f1_difference']:.6f}|[{ci[0]:.6f}, {ci[1]:.6f}]|\n"
    text+='\n[①普通图报告](../qwen3.5_4B_lora_sixplot/README.md)、[②冻结转换器报告](../qwen3.5_4B_lora_imu2img_frozen/README.md)、[③联合报告](../qwen3.5_4B_lora_imu2img_joint/README.md)。comparison.json保存完整汇总，单组目录保留原回答、预测、每类/每犬指标和三轮验证输出。\n'
    (run.SUITE/'README.md').write_text(text)
    for dest in [run.SUITE,*[run.out(k) for k in run.KEYS],*[run.model_dir(k) for k in run.KEYS]]:
        for target in re.findall(r'\]\(([^)]+)\)',(dest/'README.md').read_text()):
            if not target.startswith('http'):assert (dest/target).exists(),target
    shared=ROOT/'results/README.md';shared.write_text(shared.read_text()+'\n## 三组相同LoRA配置实验\n\n'+table+'\n[完整比较与配对区间](lora_comparison/README.md)。三组新LoRA配置相同，原始Qwen权重冻结，旧结果保留。\n')
    readme=ROOT/'README.md';readme.write_text(readme.read_text()+'\n## 三组相同LoRA配置实验\n\n已完成普通六子图、冻结IMU2IMG与联合IMU2IMG三组LoRA训练及相同3708窗测试。[完整比较](results/lora_comparison/README.md)。\n')
    handoff=ROOT/'HANDOFF.md';text=handoff.read_text();a=text.index('## 2026-11-04 三组相同LoRA配置实验');b=text.index('\n## 2026-11-04 冻结Qwen的IMU2IMG对齐',a)
    section='## 2026-11-04 三组相同LoRA配置实验（已完成）\n\n'+table+'\n三组从原始Qwen3.5-4B初始化相同新LoRA（r4/alpha8/dropout0.05、ViT/投影/语言混合注意力250模块5226496参数），Qwen基础参数全程冻结且内存SHA前后不变。两组IMU从原对齐epoch3转换器初始化，固定组converter不变；联合组追加835584转换器参数LR1e-5，LoRA LR均2e-4。3epoch、有效batch32、双卡micro4累积4，按完整4727验证窗vLLM宏F1选epoch再测试3708窗。普通波形长英文提示，IMU两组相同简短提示。普通图启用梯度检查点，IMU不开；效率需披露差异。\n\n结果与协议results/lora_comparison/及三份results/qwen3.5_4B_lora_sixplot、_imu2img_frozen、_imu2img_joint；模型models/Qwen3.5-4B-lora-sixplot、-imu2img-frozen、-imu2img-joint，各保留三个adaptercheckpoint及最终合并模型，IMU还存选中converter。普通新训练/验证图dataset/vlm/supervised/，联合图dataset/vlm_imu2img_qwen_lora_joint/；固定组复用原已对齐验证及测试图。用户已删除dataset/vlm_imu2img_simple，不再使用或重建。\n\n脚本lora_comparison.py和validate_lora_comparison.py；已逐条核验原回答/预测/指标，全部250矩阵合并正确且非目标权重逐位不变、基础官方SHA不变、历史文件未变、相同LoRA初始化已验证。comparison.json含配对犬只bootstrap差值。全部工作完成，不自动重新训练、测试或调参。\n'
    handoff.write_text(text[:a]+section+text[b:])
    run.write(run.SUITE/'complete.json',{'state':'complete','all_three_experiments_verified':True,'historical_files_unchanged':len(prior),
                                       'completed_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
