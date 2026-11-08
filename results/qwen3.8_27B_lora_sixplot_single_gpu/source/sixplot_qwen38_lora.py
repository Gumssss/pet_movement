"""Single-GPU native sixplot + fresh Qwen3.8-27B LoRA, LR search and offline W&B."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import sixplot_qwen38_common as c
ROOT=c.ROOT


def prepare(mode='sixplot'):
    import pandas as pd
    from PIL import Image
    c.SUITE.mkdir(exist_ok=True);c.model_dir().mkdir(exist_ok=True)
    if not (c.SUITE/'prior_files_sha256.json').exists():
        paths=[p for p in (ROOT/'results').rglob('*') if p.is_file() and p.parent!=ROOT/'results' and not p.is_relative_to(c.SUITE)]
        paths += [p for p in (ROOT/'dataset/splits').glob('*') if p.is_file()]
        c.write(c.SUITE/'prior_files_sha256.json',{str(p.relative_to(ROOT)):c.sha(p) for p in paths})
    dest=c.SUITE/'prompt.txt'
    if dest.exists():assert dest.read_bytes()==c.PROMPT.read_bytes()
    else:dest.write_bytes(c.PROMPT.read_bytes())
    evidence=[]
    for split in ['train','validation','test']:
        rows=c.load_rows(split)
        for r in rows.itertuples():assert c.sha(ROOT/r.image_file)==r.image_sha256
        with Image.open(ROOT/rows.iloc[0].image_file) as im:assert im.size==(1280,1152)
        evidence.append({'split':split,'windows':len(rows),'manifest_sha256':c.sha(c.manifest(split))})
    c.write(c.SUITE/'input_protocol.json',{'splits':evidence,'render_protocol_sha256':c.sha(c.DATA/'render_protocol.json'),
            'axis_limits_source':'dataset/vlm/render_protocol.json: training-derived fixed y_limits','prompt_sha256':c.sha(c.PROMPT),
            'renderer_sha256':c.sha(ROOT/'scripts/render_imu.py'),'all24896_image_hashes_verified':True})
    manifest=json.loads((ROOT/'models/model_manifest.json').read_text())['Qwen3.8-27B']
    protocol={'state':'fixed_before_new_validation_outputs','model':'Qwen3.8-27B','keys':c.KEYS,'mode':'sixplot',
       'base_revision':manifest['revision'],'seed':c.SEED,'world_size':1,'gpu_uuid':c.GPU_UUID,'physical_gpu':0,'reserved_for_others_gpu':1,
       'lora':{'r':4,'alpha':8,'dropout':.05,'target_scopes':['ViT','visual_projection','language_attention'],'trainable_parameters':13881216},
       'base_weights_frozen':True,'fresh_initialization':'Original Qwen3.8-27B + fresh identical-seed LoRA for each candidate and formal training; no old adapters',
       'lr_candidates':c.LRS,'pilot_optimizer_steps':100,'lr_selection':'Full4727 validation vLLM greedy Macro-F1; smallest LR on exact ties; test excluded',
       'formal_epochs':3,'checkpoint_selection':'Full validation vLLM Macro-F1 across3 epochs; earliest epoch on exact ties; test excluded',
       'scheduler':'10% linear warmup +90% cosine decay to zero, independently for each pilot and full formal run',
       'optimizer':'AdamW','betas':[.9,.999],'epsilon':1e-8,'weight_decay':.01,'max_grad_norm':1.,
       'gradient_checkpointing':True,'use_reentrant':False,'batch_rule':'Largest stable native1280x1152 single-GPU microbatch with2GiB reserved headroom; accumulation1',
       'gradient_accumulation':1,'precision':'BF16 original base and forward; FP32 LoRA and AdamW',
       'train_windows':16461,'validation_windows':4727,'test_windows':3708,
       'windows_sha256':c.sha(ROOT/'dataset/splits/windows.csv'),'splits_sha256':c.sha(ROOT/'dataset/splits/dog_splits.json'),
       'prompt_sha256':c.sha(c.PROMPT),'input_protocol_sha256':c.sha(c.SUITE/'input_protocol.json'),
       'image_size':[1280,1152],'visual_tokens':1440,'image_method':'Existing sixplot, all200points/all6channels; fixed training-derived axis limits; no IMU2IMG converter',
       'loss':'Answer JSON+EOS token CE; input/image/padding masked; no augmentation, resampling, loss weighting or dropped windows',
       'sampling':'Single-rank seeded sampler; repeat reshuffled training set if a100-update pilot exhausts the data; candidates get identical order',
       'tracking':{'backend':'wandb','mode':'offline','project':'pet-movement','group':'qwen38-27b-sixplot-lora','logs':'Every optimizer update loss/LR/gradnorm; per-epoch validation NLL; generated validation F1 stored separately'},
       'test_note':'Fixed dog-disjoint test set was viewed historically; no new test feedback used to select LR/epoch. Extra LR-search training compute is recorded.'}
    path=c.SUITE/'experiment_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:c.write(path,protocol)
    print('SIXPLOT27B SINGLE-GPU PROTOCOL FIXED',flush=True)


def optimizer_for(system,lr,mode=None):
    import torch
    parameters=[p for p in system.parameters() if p.requires_grad]
    assert all(not p.requires_grad or 'lora_' in n for n,p in system.named_parameters())
    return torch.optim.AdamW(parameters,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01),parameters


def profile(key='qwen38_27b',device='cuda:0'):
    import torch
    torch.set_num_threads(8);assert device=='cuda:0' and torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==c.GPU_UUID
    torch.cuda.set_device(0)
    frame=c.load_rows('train');representatives=[frame[frame.label.eq(label)].iloc[0].to_dict() for label in c.classes()]
    system,processor,info=c.make_system();system.to(device).train()
    frozen=[p for p in system.qwen.parameters() if not p.requires_grad];versions=[p._version for p in frozen]
    optimizer,parameters=optimizer_for(system,1e-5);total=torch.cuda.get_device_properties(0).total_memory
    records=[];safe=[]
    def attempt(batch_size):
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter();batch=c.to_device(system.answers([representatives[i%7] for i in range(batch_size)]),device)
        result=system(batch);assert torch.isfinite(result['loss']);result['loss'].backward()
        grads={s:sum(float(p.grad.float().square().sum()) for name,p in system.qwen.named_parameters()
                    if p.requires_grad and c.scope(name)==s and p.grad is not None)**.5 for s in info['modules_per_scope']}
        assert all(v>0 and math.isfinite(v) for v in grads.values()) and all(p.grad is None for p in frozen)
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad);optimizer.step();torch.cuda.synchronize()
        return {'microbatch':batch_size,'status':'ok','loss':float(result['loss'].detach()),'step_wall_seconds':time.perf_counter()-began,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'lora_grad_norms':grads}
    def record(batch):
        try:r=attempt(batch)
        except torch.OutOfMemoryError as error:r={'microbatch':batch,'status':'oom','message':str(error)}
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        assert versions==[p._version for p in frozen]
        if r['status']=='ok':
            r['reserved_headroom_GiB']=total/2**30-r['peak_reserved_GiB']
            if r['reserved_headroom_GiB']>=2:safe.append(batch)
        records.append(r)
        c.write(c.SUITE/'batch_profile.json',{'purpose':'Training-only, disposable weights; native sixplot and all LoRA scopes+AdamW+GC enabled',
            'gpu_uuid':c.GPU_UUID,'total_GiB':total/2**30,'records':records,'safe_max_microbatch':max(safe) if safe else None,
            'processor':system.answers.check,'lora':info,'common_code_sha256':c.sha(Path(c.__file__))})
        print('PROFILE '+json.dumps(r),flush=True)
        return r['status']=='ok' and r['reserved_headroom_GiB']>=2
    upper=None
    for batch in [1,2,4,6,8,10,12,14,16,20,24,32,48,64]:
        if not record(batch):upper=batch;break
    assert safe,'Native sixplot cannot fit even microbatch1.'
    if upper is not None:
        low=max(safe)
        for batch in range(low+1,upper):
            if not record(batch):break
    print('PROFILE COMPLETE safe_microbatch='+str(max(safe)),flush=True)



def sources():
    paths=[Path(__file__),Path(c.__file__),ROOT/'scripts/sixplot_tracking.py',ROOT/'scripts/train_qwen35_lora.py',
           ROOT/'scripts/lora_comparison.py',ROOT/'scripts/qwen_classify.py',ROOT/'scripts/eval_imu2img_alignment.py']
    return {str(p.relative_to(ROOT)):c.sha(p) for p in paths}


def adapter_fingerprint(system):
    from peft import get_peft_model_state_dict
    import hashlib
    digest=hashlib.sha256()
    for name,value in sorted(get_peft_model_state_dict(system.qwen).items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(system,key,stage,tag):
    path=c.checkpoint(key,stage,tag);system.qwen.save_pretrained(path,safe_serialization=True)
    return {'path':str(path.relative_to(ROOT)),'adapter_sha256':c.sha(path/'adapter_model.safetensors')}


def train(key,stage,check_only=False,micro_override=None):
    import hashlib
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,DistributedSampler
    from transformers import get_cosine_schedule_with_warmup
    assert key=='qwen38_27b'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-0ded27d3-e266-f20b-2476-f2980be2980e'
    assert torch.cuda.device_count()==1
    torch.set_num_threads(8);torch.cuda.set_device(0);device=torch.device('cuda:0')
    cfg=json.loads((c.SUITE/'experiment_protocol.json').read_text());mode=cfg['mode'];assert mode=='sixplot'
    bs=json.loads((c.SUITE/'batch_selection.json').read_text());micro=micro_override or bs['microbatch_per_gpu'];accum=bs['gradient_accumulation']
    assert micro>=1 and accum==1 and cfg['world_size']==1
    frame=c.load_rows();system,processor,info=c.make_system(key,mode);system.to(device)
    frozen=[p for p in system.qwen.parameters() if not p.requires_grad]
    base_hash=c.frozen_fingerprint(system) if not check_only else None
    versions=[p._version for p in frozen];initial_adapter=adapter_fingerprint(system)
    initial={n:p.detach().cpu().clone() for n,p in system.named_parameters() if p.requires_grad}
    train_rows=frame[frame.split.eq('train')].to_dict('records')
    sampler=DistributedSampler(train_rows,num_replicas=1,rank=0,shuffle=True,seed=c.SEED,drop_last=False)
    loader=DataLoader(train_rows,batch_size=micro,sampler=sampler,collate_fn=system.answers,num_workers=0)
    validation=DataLoader(frame[frame.split.eq('validation')].to_dict('records'),batch_size=micro,
                           shuffle=False,collate_fn=system.answers,num_workers=0)
    updates_per_epoch=math.ceil(len(loader)/accum)
    rates=c.LRS[:1] if check_only else (c.LRS if stage=='pilot' else [json.loads((c.out(key)/'lr_selection.json').read_text())['best_lr']])
    all_began=time.perf_counter();trials=[]
    for lr in rates:
        torch.manual_seed(c.SEED);random.seed(c.SEED);np.random.seed(c.SEED)
        with torch.no_grad():
            for name,p in system.named_parameters():
                if p.requires_grad:p.copy_(initial[name].to(device))
        assert adapter_fingerprint(system)==initial_adapter
        optimizer,parameters=optimizer_for(system,lr,mode)
        total_steps=3 if check_only else (100 if stage=='pilot' else updates_per_epoch*3)
        warmup_steps=math.ceil(.1*total_steps)
        schedule=get_cosine_schedule_with_warmup(optimizer,warmup_steps,total_steps,num_cycles=.5)
        trial_dir=c.out(key)/('lr_search/lr_'+c.lr_tag(lr)) if stage=='pilot' else c.out(key)
        if not check_only:
            protocol={**cfg,'key':key,'stage':stage,'peak_lora_lr':lr,
               'microbatch_per_gpu':micro,'effective_batch_size':micro*accum,'optimizer_steps':total_steps,
               'updates_per_epoch':updates_per_epoch,'microbatches_per_epoch':len(loader),'warmup_steps':warmup_steps,
               'initial_lora_sha256':initial_adapter,'qwen_base_sha256_before':base_hash,'lora':info,
               'processor':system.answers.check,'source_sha256':sources(),
               'accumulation_loss':'Sum answer-token CE over each update divided by all supervised tokens in that update, including the short epoch-tail batch',
               'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_visible_device_count':torch.cuda.device_count(),
               'versions':{p:__import__('importlib.metadata',fromlist=['version']).version(p) for p in ['torch','transformers','peft','vllm','wandb']}}
            dest=trial_dir/'training_protocol.json'
            if dest.exists():assert json.loads(dest.read_text())==protocol,'Existing training protocol differs; preserve old attempt before retry.'
            else:c.write(dest,protocol)
            c.write(trial_dir/'target_modules.json',info)
        from sixplot_tracking import Tracker
        tracker=None if check_only else Tracker(trial_dir/'tracking',protocol,f'qwen38-27b-sixplot-{stage}-lr{c.lr_tag(lr)}','qwen38-27b-sixplot-lora')
        began=time.perf_counter();step=0;cycle=0;cursor=0;sampler.set_epoch(cycle);positions=list(iter(sampler));iterator=iter(loader)
        nll=0.;tokens=0;sample_count=0;input_digest=hashlib.sha256();system.train();torch.cuda.reset_peak_memory_stats()
        while step<total_steps:
            batches=[]
            for _ in range(accum):
                try:batch=next(iterator)
                except StopIteration:break
                batches.append(batch)
                for position in positions[cursor:cursor+len(batch['input_ids'])]:input_digest.update((train_rows[position]['window_id']+'\n').encode())
                cursor+=len(batch['input_ids'])
            assert batches,'An optimizer update cannot be empty.'
            count=sum(int(batch['labels'].ne(-100).sum()) for batch in batches);assert count>0
            optimizer.zero_grad(set_to_none=True)
            step_nll=0.;step_tokens=0;presented=0
            for batch in batches:
                result=system(c.to_device(batch,device));assert torch.isfinite(result['loss'])
                weighted=result['loss']*(result['tokens'].to(result['loss'].dtype)/count)
                weighted.backward()
                step_nll+=float(result['nll_sum']);step_tokens+=int(result['tokens']);presented+=len(batch['input_ids'])
            assert step_tokens==count and all(p.grad is None for p in frozen)
            if step==0:
                grads={s:sum(float(p.grad.float().square().sum()) for name,p in system.qwen.named_parameters()
                       if p.requires_grad and c.scope(name)==s and p.grad is not None)**.5 for s in info['modules_per_scope']}
                assert all(v>0 and math.isfinite(v) for v in grads.values())
                if check_only:c.write(c.out(key)/'single_gpu_gradient_check.json',{'lora':grads,'qwen_base_gradients_none':True})
            grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad)
            used_lr=optimizer.param_groups[0]['lr'];optimizer.step();schedule.step();step+=1;nll+=step_nll;tokens+=step_tokens;sample_count+=presented
            row={'key':key,'stage':stage,'peak_lr':lr,'optimizer_step':step,'total_steps':total_steps,
                 'sampler_epoch':cycle+1,'train_answer_nll':nll/tokens,'learning_rates':schedule.get_last_lr(),
                 'wall_seconds':time.perf_counter()-began,'microbatch_per_gpu':micro,'gradient_accumulation':accum,
                 'effective_batch_size':micro*accum,'actual_samples_this_update':presented,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                 'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
            c.write(c.out(key)/'single_gpu_check_progress.json' if check_only else trial_dir/'training_progress.json',row)
            if tracker:tracker.log({'optimizer_step':step,'train/loss':step_nll/step_tokens,'train/running_loss':nll/tokens,'train/learning_rate':used_lr,'train/grad_norm':float(grad),'train/epoch':cycle+1,'memory/peak_allocated_GiB':row['peak_allocated_GiB']})
            if step%5==0 or step==total_steps:
                print('TRAIN '+json.dumps(row),flush=True)
                if not check_only:
                    with (trial_dir/'training_history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            epoch_finished=cursor==len(train_rows)
            if stage=='formal' and epoch_finished and not check_only:
                epoch=cycle+1;assert step==epoch*updates_per_epoch;system.eval();a=0.;b=0
                with torch.inference_mode():
                    for val_batch in validation:
                        value=system(c.to_device(val_batch,device));a+=float(value['nll_sum']);b+=int(value['tokens'])
                assert versions==[p._version for p in frozen]
                record={'epoch':epoch,'optimizer_step':step,'validation_answer_nll':a/b,'validation_tokens':b,
                        'checkpoint':save_checkpoint(system,key,'formal',epoch),'wall_seconds':time.perf_counter()-began}
                with (c.out(key)/'validation_nll_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('VALIDATION_NLL '+json.dumps(record),flush=True)
                if tracker:tracker.log({'optimizer_step':step,'validation/answer_nll':a/b,'validation/epoch':epoch})
                system.train()
            if epoch_finished and step<total_steps:
                cycle+=1;cursor=0;sampler.set_epoch(cycle);positions=list(iter(sampler));iterator=iter(loader)
        assert versions==[p._version for p in frozen]
        record={'stage':stage,'peak_lr':lr,'optimizer_steps':step,'sampler_epochs_started':cycle+1,
                'sample_presentations':sample_count,'input_order_sha256_per_rank':[input_digest.hexdigest()],
                'initial_lora_sha256':initial_adapter,
                'qwen_base_versions_unchanged':True,'qwen_base_gradients_always_none':True,
                'session_wall_seconds':time.perf_counter()-began,'microbatch_per_gpu':micro,'gradient_accumulation':accum,
                'effective_batch_size':micro*accum,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
                'common_code_sha256':c.sha(Path(c.__file__)),'code_sha256':c.sha(Path(__file__))}
        if check_only:
            record['free_device_GiB_after_training']=torch.cuda.mem_get_info()[0]/2**30
            assert record['free_device_GiB_after_training']>=2,'CAPACITY_HEADROOM_LT_2GIB'
            record['technical_only']=True;c.write(c.out(key)/'single_gpu_check.json',record)
        elif stage=='pilot':
            record['checkpoint']=save_checkpoint(system,key,'pilot',lr);c.write(trial_dir/'training_complete.json',record)
        else:
            assert sample_count==3*len(train_rows);c.write(c.out(key)/'training_complete.json',record)
        if tracker:tracker.finish({'optimizer_updates_completed':step,'peak_lora_lr':lr,'stage':stage})
        trials.append(record);del optimizer,schedule;gc.collect();torch.cuda.empty_cache()
    if not check_only:
        after=c.frozen_fingerprint(system);assert after==base_hash
        proof={'stage':stage,'qwen_base_sha256_before':base_hash,'qwen_base_sha256_after':after,
               'qwen_base_versions_unchanged':True,'qwen_base_gradients_always_none':True,
               'initial_lora_sha256':initial_adapter,'trials':trials,'whole_session_wall_seconds':time.perf_counter()-all_began,
               'completed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources()}
        if stage=='pilot':
            assert len({tuple(t['input_order_sha256_per_rank']) for t in trials})==1
            c.write(c.out(key)/'pilot_training_complete.json',proof)
        else:c.write(c.out(key)/'formal_training_verified.json',proof)


def export(key,stage,tag,final=False):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor
    from peft import PeftModel
    torch.set_num_threads(8)
    source=c.checkpoint(key,stage,tag)
    target=c.model_dir(key) if final else c.model_dir(key)/'evaluation_merged'
    base=Qwen3_5ForConditionalGeneration.from_pretrained(c.base_dir(key),local_files_only=True,dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(base,source)
    probes={}
    for name,layer in model.named_modules():
        scope=c.scope(name)
        if scope and scope not in probes and hasattr(layer,'lora_A'):
            delta=layer.get_delta_weight('default').detach().float();assert delta.abs().sum()>0
            probes[scope]={'module':name,'delta_l1':float(delta.abs().sum())}
    assert len(probes)==3
    model=model.merge_and_unload(safe_merge=True)
    assert not any('lora_' in n for n,_ in model.named_parameters())
    model.save_pretrained(target,safe_serialization=True,max_shard_size='5GB')
    AutoProcessor.from_pretrained(c.base_dir(key),local_files_only=True).save_pretrained(target)
    if final:
        adapter=target/'adapter';adapter.mkdir(exist_ok=True)
        for name in ['adapter_config.json','adapter_model.safetensors']:shutil.copyfile(source/name,adapter/name)
        c.write(c.out(key)/'export_validation.json',{'selected_epoch':int(tag),'adapter_sha256':c.sha(adapter/'adapter_model.safetensors'),
             'merged_for_vllm':True,'nonzero_scopes':probes,
             'merged_weight_files_sha256':{p.name:c.sha(p) for p in target.glob('model-*.safetensors')},
             'checkpoint':str(source.relative_to(ROOT))})
    print('EXPORT COMPLETE '+key+' '+stage+' '+str(tag)+' final='+str(final),flush=True)


def image_folder(key,stage,tag,split):return c.DATA if split=='test' else c.DATA/'supervised'


def result_folder(key,stage,tag,split):
    if split=='test':return c.SUITE
    return c.SUITE/('lr_search/lr_'+c.lr_tag(float(tag))+'/validation' if stage=='pilot' else 'validation/epoch_'+str(int(tag)))


def evaluate(key,stage,tag,split,device,final=False):
    assert device=='cuda:0' and os.environ.get('CUDA_VISIBLE_DEVICES')==c.GPU_UUID
    import torch
    import importlib.metadata as md
    from transformers import AutoProcessor
    from vllm import LLM
    import qwen_classify as q
    from eval_imu2img_alignment import evaluate as metrics
    torch.set_num_threads(8);assert torch.cuda.device_count()==1
    target=c.model_dir(key) if final else c.model_dir(key)/'evaluation_merged'
    dest=result_folder(key,stage,tag,split);dest.mkdir(parents=True,exist_ok=True)
    frame=c.load_rows(split);source=c.checkpoint(key,stage,tag);prompt=(c.SUITE/'prompt.txt').read_text()
    train_dir=c.SUITE/('lr_search/lr_'+c.lr_tag(float(tag))) if stage=='pilot' else c.SUITE
    protocol={'model':c.model_dir(key).name,'key':key,'stage':stage,'tag':tag,'split':split,'classes':c.classes(),
      'base_revision':json.loads((ROOT/'models/model_manifest.json').read_text())['Qwen3.8-27B']['revision'],
      'expected_windows':len(frame),'inference_backend':'vllm','vllm_version':md.version('vllm'),'qwen_base_weights_frozen':True,
      'lora_r':4,'lora_alpha':8,'image_dir':str(image_folder(key,stage,tag,split).relative_to(ROOT)),
      'image_manifest_sha256':c.sha(c.manifest(split)),'render_protocol_sha256':c.sha(c.DATA/'render_protocol.json'),
      'adapter_sha256':c.sha(source/'adapter_model.safetensors'),'training_protocol_sha256':c.sha(train_dir/'training_protocol.json'),
      'prompt_sha256':c.sha(c.SUITE/'prompt.txt'),'windows_sha256':c.sha(ROOT/'dataset/splits/windows.csv'),
      'splits_sha256':c.sha(ROOT/'dataset/splits/dog_splits.json'),'image_size':[1280,1152],'visual_tokens':1440,
      'gpu_uuid':c.GPU_UUID,'cuda_visible_device_count':1,
      'generation':{'temperature':0,'enable_thinking':False,'max_new_tokens':32,'seed':c.SEED,'guided_decoding':False},
      'engine':{'dtype':'bfloat16','tensor_parallel_size':1,'max_model_len':4096,'max_num_seqs':16,'max_num_batched_tokens':8192,
                'gpu_memory_utilization':.85,'enable_prefix_caching':False,'min_pixels':65536,'max_pixels':1280*1152,'flashinfer_sampler':False}}
    path=dest/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen evaluation protocol differs.'
    else:c.write(path,protocol)
    (dest/'prompt.txt').write_text(prompt)
    raw=dest/'responses.jsonl';done={}
    if raw.exists():
        for line in raw.read_text().splitlines():
            r=json.loads(line);assert r['window_id'] not in done;done[r['window_id']]=r
    assert set(done).issubset(set(frame.window_id))
    pending=[r for r in frame.to_dict('records') if r['window_id'] not in done]
    if pending:
        start=time.perf_counter();processor=AutoProcessor.from_pretrained(target,local_files_only=True)
        model=LLM(model=str(target),dtype='bfloat16',tensor_parallel_size=1,trust_remote_code=False,seed=c.SEED,
          gpu_memory_utilization=.85,max_model_len=4096,max_num_seqs=16,max_num_batched_tokens=8192,
          limit_mm_per_prompt={'image':1,'video':0},mm_processor_kwargs={'min_pixels':65536,'max_pixels':1280*1152},
          generation_config='vllm',enable_prefix_caching=False,disable_log_stats=True)
        load_seconds=time.perf_counter()-start;timings=[]
        with raw.open('a') as f:
            for offset in range(0,len(pending),16):
                group=pending[offset:offset+16]
                for r in group:assert c.sha(ROOT/r['image_file'])==r['image_sha256']
                texts,counts,lengths,truncated,pre,gen,wall=q.generate(model,processor,[ROOT/r['image_file'] for r in group],prompt,device,(1280,1152))
                timings.append({'n':len(group),'preprocess_seconds':pre,'generation_seconds':gen,'request_wall_seconds':wall})
                for i,item in enumerate(group):
                    prediction,status=q.parse(texts[i],c.classes(),truncated[i]);r={name:item[name] for name in ['window_id','image_file','image_sha256']}
                    r.update(response=texts[i],prediction=prediction,status=status,input_tokens=counts[i],generated_tokens=lengths[i],
                             batch_size=len(group),amortized_request_seconds=wall/len(group),inference_backend='vllm')
                    f.write(json.dumps(r,ensure_ascii=False)+'\n');done[r['window_id']]=r
                f.flush();state={'completed':len(done),'expected':len(frame),'remaining':len(frame)-len(done),
                                'status_counts':dict(Counter(r['status'] for r in done.values())),'wall_seconds':time.perf_counter()-start}
                c.write(dest/'progress.json',state)
                if len(done)%320==0 or len(done)==len(frame):print('GENERATE '+json.dumps(state),flush=True)
        file=dest/'runtime.json';runtime=json.loads(file.read_text()) if file.exists() else {'sessions':[]}
        runtime['sessions'].append({'load_seconds':load_seconds,'wall_seconds':time.perf_counter()-start,'batch_timings':timings,
             'device':device,'gpu_uuid':c.GPU_UUID,'code_sha256':c.sha(Path(__file__)),'common_code_sha256':c.sha(Path(c.__file__)),
             'generation_code_sha256':c.sha(ROOT/'scripts/qwen_classify.py')});c.write(file,runtime)
    m=metrics(dest)
    from sixplot_tracking import Tracker
    tracker=Tracker(dest/'evaluation_tracking',{'stage':stage,'tag':str(tag),'split':split,'windows':len(frame)},
                    f'qwen38-sixplot-{stage}-{tag}-{split}','qwen38-27b-sixplot-lora')
    tracker.log({'optimizer_step':0,**{split+'/'+k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
    tracker.finish({k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']})



def run_process(arguments,log,env=None):
    c.write(c.SUITE/'progress.json',{'command':arguments,'log':str(log.relative_to(ROOT)),
              'started_utc':datetime.now(timezone.utc).isoformat(),'state':'running'})
    print('RUN '+str(log.relative_to(ROOT)),flush=True)
    with log.open('a') as f:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-B',*arguments],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)


def select_lr(key):
    scores=[]
    for lr in c.LRS:
        m=json.loads((result_folder(key,'pilot',lr,'validation')/'metrics.json').read_text())
        scores.append({'lr':lr,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
    best=max(scores,key=lambda r:(r['macro_f1'],-r['lr']))
    result={'best_lr':best['lr'],'best_macro_f1':best['macro_f1'],'scores':scores,'metric':'Full validation vLLM macro-F1',
            'optimizer_steps_per_candidate':100,'ties':'lowest LR','test_used_for_selection':False}
    c.write(c.out(key)/'lr_selection.json',result)
    print('LR_SELECTED '+key+' '+json.dumps(result),flush=True)


def pipeline(preflight_only=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==c.GPU_UUID
    if (c.SUITE/'stopped_by_user.json').exists():raise RuntimeError('User stopped this experiment; explicit resume required before removing stop marker.')
    cfg=json.loads((c.SUITE/'experiment_protocol.json').read_text());script='scripts/sixplot_qwen38_lora.py';key='qwen38_27b'
    assert cfg['world_size']==1 and cfg['formal_epochs']==3
    if not (c.SUITE/'batch_profile.json').exists():run_process([script,'profile','--key',key],c.SUITE/'batch_profile.log')
    p=json.loads((c.SUITE/'batch_profile.json').read_text())
    dest=c.SUITE/'batch_selection.json'
    if (c.SUITE/'single_gpu_check.json').exists():
        selection=json.loads(dest.read_text());micro=selection['microbatch_per_gpu']
    else:
        attempts=[]
        for micro in range(p['safe_max_microbatch'],0,-1):
            selection={'microbatch_per_gpu':micro,'world_size':1,'gradient_accumulation':1,'effective_batch_size':micro,
                       'gpu_uuid':c.GPU_UUID,'physical_gpu':0,'free_physical_gpu':1,'profile_sha256':c.sha(c.SUITE/'batch_profile.json'),
                       'rule':'Largest batch passing3 consecutive native-image optimizer updates with>=2GiB actual free device memory'}
            c.write(dest,selection)
            log=c.SUITE/f'single_gpu_check_batch{micro}.log'
            try:
                run_process([script,'single-check','--key',key],log)
            except subprocess.CalledProcessError:
                contents=log.read_text()
                if not any(t in contents for t in ['CUDA out of memory','CAPACITY_HEADROOM_LT_2GIB']):raise
                attempts.append({'microbatch':micro,'status':'insufficient_capacity','log':str(log.relative_to(ROOT))})
                c.write(c.SUITE/'continuous_batch_attempts.json',attempts)
                continue
            attempts.append({'microbatch':micro,'status':'passed','log':str(log.relative_to(ROOT))})
            c.write(c.SUITE/'continuous_batch_attempts.json',attempts);break
        else:raise RuntimeError('No stable native-image single-GPU training batch.')
    if preflight_only:return
    archived=c.SUITE/'source';archived.mkdir(exist_ok=True)
    for name,digest in sources().items():
        target=archived/Path(name).name
        if target.exists():assert c.sha(target)==digest
        else:shutil.copyfile(ROOT/name,target)
    c.write(c.SUITE/'source_sha256.json',sources())
    if not (c.SUITE/'pilot_training_complete.json').exists():run_process([script,'pilot-train','--key',key],c.SUITE/'pilot_train.log')
    if not (c.SUITE/'lr_selection.json').exists():
        for lr in c.LRS:
            tag=c.lr_tag(lr);dest=result_folder(key,'pilot',lr,'validation')
            if (dest/'metrics.json').exists():continue
            run_process([script,'export','--key',key,'--stage','pilot','--tag',tag],c.SUITE/('export_pilot_'+tag+'.log'))
            run_process([script,'evaluate','--key',key,'--stage','pilot','--tag',tag],c.SUITE/('validation_pilot_'+tag+'.log'))
        select_lr(key)
    if not (c.SUITE/'formal_training_verified.json').exists():run_process([script,'train','--key',key],c.SUITE/'train.log')
    if not (c.SUITE/'generation_complete.json').exists():
        scores=[]
        for epoch in [1,2,3]:
            dest=result_folder(key,'formal',epoch,'validation')
            if not (dest/'metrics.json').exists():
                run_process([script,'export','--key',key,'--tag',str(epoch)],c.SUITE/f'export_epoch{epoch}.log')
                run_process([script,'evaluate','--key',key,'--tag',str(epoch)],c.SUITE/f'validation_epoch{epoch}.log')
            m=json.loads((dest/'metrics.json').read_text());scores.append({'epoch':epoch,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
        best=max(scores,key=lambda r:r['macro_f1']);epoch=best['epoch']
        c.write(c.SUITE/'checkpoint_selection.json',{'best_epoch':epoch,'best_macro_f1':best['macro_f1'],'scores':scores,
                'metric':'Full validation vLLM macro-F1','test_used_for_selection':False,'ties':'earliest epoch'})
        run_process([script,'export','--key',key,'--tag',str(epoch),'--final'],c.SUITE/'export_final.log')
        run_process([script,'evaluate','--key',key,'--tag',str(epoch),'--split','test','--final'],c.SUITE/'test.log')
        temp=c.model_dir(key)/'evaluation_merged'
        if temp.exists():shutil.rmtree(temp)
        c.write(c.SUITE/'generation_complete.json',{'state':'complete','selected_epoch':epoch,'reports_pending':True})
    run_process(['scripts/validate_sixplot_qwen38.py'],c.SUITE/'validation.log')



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','profile','single-check','pilot-train','train','export','evaluate','select-lr','pipeline','preflight'])
    parser.add_argument('--key',choices=c.KEYS);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--mode',choices=['sixplot'],default='sixplot')
    parser.add_argument('--stage',choices=['pilot','formal'],default='formal');parser.add_argument('--tag',default='1')
    parser.add_argument('--split',choices=['validation','test'],default='validation');parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.mode)
    elif args.command in ['pipeline','preflight']:
        try:pipeline(args.command=='preflight')
        except BaseException as error:
            c.write(c.SUITE/'failure.json',{'state':'failed','type':type(error).__name__,'error':str(error),'utc':datetime.now(timezone.utc).isoformat()})
            raise
    else:
        assert args.key
        if args.command=='profile':profile(args.key,args.device)
        elif args.command=='single-check':train(args.key,'pilot',check_only=True)
        elif args.command=='pilot-train':train(args.key,'pilot')
        elif args.command=='train':train(args.key,'formal')
        elif args.command=='select-lr':select_lr(args.key)
        elif args.command=='export':export(args.key,args.stage,args.tag,args.final)
        else:evaluate(args.key,args.stage,args.tag,args.split,args.device,args.final)
