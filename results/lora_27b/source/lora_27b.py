"""IMU2IMG + fresh 27B LoRA: memory fit, 100-step LR search and full experiments."""
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

import imu2img_lora27_common as c
ROOT=c.ROOT


def prepare(mode):
    c.SUITE.mkdir(exist_ok=True)
    destinations=[c.out(k) for k in c.KEYS]
    if not (c.SUITE/'prior_files_sha256.json').exists():
        paths=[p for p in (ROOT/'results').rglob('*') if p.is_file() and p.parent!=ROOT/'results'
               and not p.is_relative_to(c.SUITE) and not any(p.is_relative_to(d) for d in destinations)]
        paths += [p for p in (ROOT/'dataset/splits').glob('*') if p.is_file()]
        for name in ['vlm','vlm_acl','vlm_imu2img','vlm_imu2img_qwen']:
            paths += [p for p in (ROOT/'dataset'/name).glob('*') if p.is_file()]
        paths += [c.WARM]
        c.write(c.SUITE/'prior_files_sha256.json',{str(p.relative_to(ROOT)):c.sha(p) for p in paths})
    manifest=json.loads((ROOT/'models/model_manifest.json').read_text())
    for key in c.KEYS:
        c.out(key).mkdir(exist_ok=True);c.model_dir(key).mkdir(exist_ok=True)
        dest=c.out(key)/'prompt.txt'
        if dest.exists():assert dest.read_bytes()==c.PROMPT.read_bytes()
        else:dest.write_bytes(c.PROMPT.read_bytes())
    protocol={'state':'authorized_before_new_validation_outputs','start_date':'2026-11-05','keys':c.KEYS,'mode':mode,
        'base_models':{k:{'name':c.NAMES[k],'revision':manifest[c.NAMES[k]]['revision']} for k in c.KEYS},
        'lora':{'r':4,'alpha':8,'dropout':.05,'target_scopes':['ViT','visual_projection','language_attention'],
                'initialization':'Fresh from each original BF16 base, reseed before creating LoRA; no prior adapters'},
        'warm_converter':str(c.WARM.relative_to(ROOT)),'warm_converter_sha256':c.sha(c.WARM),
        'lr_candidates':c.LRS,'pilot_optimizer_steps':100,'pilot_sampling':'Cycle training-only DistributedSampler if exhausted; same seed/order/initialization for every LR',
        'lr_selection':'Full4727-window validation greedy vLLM macro-F1 after100 optimizer updates; lowest LR on exact ties; test excluded',
        'formal_epochs':3,'formal_initialization':'Restart from the original base + same fresh LoRA + same warm converter after LR selection',
        'checkpoint_selection':'Full validation greedy vLLM macro-F1 across3 formal epochs; earliest on ties; test excluded',
        'scheduler':'10% linear warmup +90% cosine decay to zero; applied separately to each100-step pilot and full formal run',
        'optimizer':'AdamW','betas':[.9,.999],'epsilon':1e-8,'weight_decay':.01,'max_grad_norm':1.,
        'converter_lr_ratio':.05,'gradient_checkpointing':True,'use_reentrant':False,
        'batch_rule':'Largest common stable per-GPU microbatch verified on both models; at least2GiB reserved headroom; two-GPU DDP, no extra accumulation',
        'world_size':2,'gradient_accumulation':1,'precision':'BF16 base/forward; FP32 LoRA/converter/AdamW; fixed8bit STE images',
        'seed':c.SEED,'train_windows':16461,'validation_windows':4727,'test_windows':3708,
        'windows_sha256':c.sha(ROOT/'dataset/splits/windows.csv'),'splits_sha256':c.sha(ROOT/'dataset/splits/dog_splits.json'),
        'cache_sha256':c.sha(ROOT/'dataset/cache/pet_windows.npz'),'prompt_sha256':c.sha(c.PROMPT),
        'label_budget':'Same16461 labeled training windows reused for pilots/full training and prior converter training; all extra optimization/validation compute is recorded',
        'test_note':'Fixed dog-disjoint test cohort has been viewed historically; no new test feedback used for LR or checkpoint selection'}
    path=c.SUITE/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen experiment protocol differs.'
    else:c.write(path,protocol)
    print('27B experiment protocol fixed.',flush=True)


def optimizer_for(system,lr,mode):
    import torch
    groups=[{'params':[p for p in system.qwen.parameters() if p.requires_grad],'lr':lr}]
    if mode=='joint':groups.append({'params':list(system.converter.parameters()),'lr':lr*.05})
    params=[p for g in groups for p in g['params']]
    assert {id(p) for p in params}=={id(p) for p in system.parameters() if p.requires_grad}
    return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.01),params


def profile(key,device):
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(8);torch.cuda.set_device(device)
    mode=json.loads((c.SUITE/'protocol.json').read_text())['mode']
    frame=c.load_rows().query("split=='train'")
    representatives=[frame[frame.label.eq(label)].iloc[0].to_dict() for label in c.classes()]
    system,processor,info=c.make_system(key,mode);system.to(device).train()
    frozen=[p for p in system.qwen.parameters() if not p.requires_grad];versions=[p._version for p in frozen]
    # Verify the real uint8 forward values with the independent official processor.
    probe=system.answers(representatives)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        unit=c.quantize_rgb(system.converter(probe['signals'].to(device)),False)
    rgb=torch.round(unit*255).byte().permute(0,2,3,1).cpu().numpy()
    reference=processor.image_processor(images=[Image.fromarray(a) for a in rgb],return_tensors='pt')['pixel_values']
    assert torch.equal(c.pack_qwen_pixels(unit).cpu().to(torch.bfloat16),reference.to(torch.bfloat16))
    del probe,unit,rgb,reference
    optimizer,parameters=optimizer_for(system,1e-5,mode)
    total=torch.cuda.get_device_properties(device).total_memory
    records=[];safe=[]
    def attempt(batch_size):
        torch.cuda.reset_peak_memory_stats(device);began=time.perf_counter()
        rows=[representatives[i%7] for i in range(batch_size)]
        batch=c.to_device(system.answers(rows),device)
        result=system(batch);assert torch.isfinite(result['loss'])
        result['loss'].backward()
        grads={s:sum(float(p.grad.float().square().sum()) for name,p in system.qwen.named_parameters()
                     if p.requires_grad and c.scope(name)==s and p.grad is not None)**.5 for s in info['modules_per_scope']}
        conv={s:sum(float(p.grad.float().square().sum()) for name,p in system.converter.named_parameters()
                    if name.startswith(s) and p.grad is not None)**.5 for s in ['encoders','decoders']}
        assert all(v>0 and math.isfinite(v) for v in grads.values())
        if mode=='joint':assert all(v>0 and math.isfinite(v) for v in conv.values())
        else:assert all(v==0 for v in conv.values())
        assert all(p.grad is None for p in frozen)
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad)
        optimizer.step();torch.cuda.synchronize(device)
        return {'microbatch':batch_size,'status':'ok','loss':float(result['loss'].detach()),
                'peak_allocated_GiB':torch.cuda.max_memory_allocated(device)/2**30,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved(device)/2**30,
                'step_wall_seconds':time.perf_counter()-began,'lora_grad_norms':grads,'converter_grad_norms':conv}
    for batch_size in [8,16,32,64,96,128,160,192,256,320,384,512]:
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        try:
            result=attempt(batch_size)
        except torch.OutOfMemoryError as exc:
            result={'microbatch':batch_size,'status':'oom','message':str(exc)}
        records.append(result)
        if result['status']=='ok':
            result['reserved_headroom_GiB']=total/2**30-result['peak_reserved_GiB']
            if result['reserved_headroom_GiB']>=2:safe.append(batch_size)
        c.write(c.out(key)/'batch_profile.json',{'key':key,'device':device,'gpu':torch.cuda.get_device_name(device),
            'total_GiB':total/2**30,'mode':mode,'purpose':'Training-only capacity/gradient check with disposable updates; no validation or test',
            'records':records,'safe_max_microbatch':max(safe) if safe else None,'official_processor_bf16_equal':True,
            'processor':system.answers.check,'lora':info,'common_code_sha256':c.sha(Path(c.__file__))})
        print('BATCH_PROFILE '+json.dumps(result),flush=True)
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        assert versions==[p._version for p in frozen]
        if result['status']=='oom' or (safe and batch_size not in safe):break
    assert safe,'No batch fits the BF16 base.'
    print('PROFILE COMPLETE '+key+' safe_microbatch='+str(max(safe)),flush=True)


def sources():
    paths=[Path(__file__),Path(c.__file__),ROOT/'scripts/train_qwen35_lora.py',
           ROOT/'scripts/train_imu2img.py',ROOT/'scripts/align_imu2img_qwen.py']
    return {str(p.relative_to(ROOT)):c.sha(p) for p in paths}


def adapter_fingerprint(system):
    from peft import get_peft_model_state_dict
    import hashlib
    digest=hashlib.sha256()
    for name,value in sorted(get_peft_model_state_dict(system.qwen).items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(system,key,stage,tag):
    import torch
    from safetensors.torch import save_file
    path=c.checkpoint(key,stage,tag)
    system.qwen.save_pretrained(path,safe_serialization=True)
    save_file({n:p.detach().cpu().contiguous() for n,p in system.converter.state_dict().items()},str(path/'converter.safetensors'))
    return {'path':str(path.relative_to(ROOT)),'adapter_sha256':c.sha(path/'adapter_model.safetensors'),
            'converter_sha256':c.sha(path/'converter.safetensors')}


def train(key,stage,check_only=False,micro_override=None):
    import hashlib
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader,DistributedSampler
    from transformers import get_cosine_schedule_with_warmup
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    assert world==2
    torch.set_num_threads(8);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device)
    cfg=json.loads((c.SUITE/'protocol.json').read_text());mode=cfg['mode']
    micro=micro_override or json.loads((c.SUITE/'batch_selection.json').read_text())['microbatch_per_gpu']
    frame=c.load_rows();system,processor,info=c.make_system(key,mode);system.to(device)
    frozen=[p for p in system.qwen.parameters() if not p.requires_grad]
    base_hash=c.frozen_fingerprint(system) if rank==0 and not check_only else None
    initial_adapter=adapter_fingerprint(system)
    initial={n:p.detach().cpu().clone() for n,p in system.named_parameters() if p.requires_grad}
    train_rows=frame[frame.split.eq('train')].to_dict('records')
    sampler=DistributedSampler(train_rows,num_replicas=world,rank=rank,shuffle=True,seed=c.SEED,drop_last=False)
    loader=DataLoader(train_rows,batch_size=micro,sampler=sampler,collate_fn=system.answers,num_workers=0)
    validation=DataLoader(frame[frame.split.eq('validation')].to_dict('records')[rank::world],batch_size=micro,
                          shuffle=False,collate_fn=system.answers,num_workers=0)
    if stage=='pilot':rates=c.LRS[:1] if check_only else c.LRS
    else:rates=[json.loads((c.out(key)/'lr_selection.json').read_text())['best_lr']]
    wrapped=DDP(system,device_ids=[local],broadcast_buffers=False)
    if rank==0 and not check_only:assert c.frozen_fingerprint(system)==base_hash
    versions=[p._version for p in frozen]
    all_began=time.perf_counter();trials=[]
    for lr in rates:
        torch.manual_seed(c.SEED+rank);random.seed(c.SEED+rank);np.random.seed(c.SEED+rank)
        with torch.no_grad():
            for name,p in system.named_parameters():
                if p.requires_grad:p.copy_(initial[name].to(device))
        assert adapter_fingerprint(system)==initial_adapter
        optimizer,parameters=optimizer_for(system,lr,mode)
        total_steps=3 if check_only else (100 if stage=='pilot' else len(loader)*3)
        warmup_steps=math.ceil(.1*total_steps)
        schedule=get_cosine_schedule_with_warmup(optimizer,warmup_steps,total_steps,num_cycles=.5)
        trial_dir=c.out(key)/('lr_search/lr_'+c.lr_tag(lr)) if stage=='pilot' else c.out(key)
        if rank==0 and not check_only:
            protocol={**cfg,'key':key,'stage':stage,'peak_lora_lr':lr,'peak_converter_lr':lr*.05 if mode=='joint' else 0.,
              'microbatch_per_gpu':micro,'effective_batch_size':micro*world,'optimizer_steps':total_steps,
              'warmup_steps':warmup_steps,'initial_lora_sha256':initial_adapter,'qwen_base_sha256_before':base_hash,
              'lora':info,'processor':system.answers.check,'source_sha256':sources(),
              'versions':{p:__import__('importlib.metadata',fromlist=['version']).version(p) for p in ['torch','transformers','peft','vllm']}}
            dest=trial_dir/'training_protocol.json'
            if dest.exists():assert json.loads(dest.read_text())==protocol,'Existing training protocol differs.'
            else:c.write(dest,protocol)
            c.write(trial_dir/'target_modules.json',info)
        began=time.perf_counter();step=0;cycle=0;sampler.set_epoch(0);iterator=iter(loader)
        nll=torch.zeros((),device=device,dtype=torch.float64);tokens=torch.zeros_like(nll)
        sample_count=0;input_digest=hashlib.sha256();wrapped.train()
        epoch_records=[]
        while step<total_steps:
            try:batch=next(iterator)
            except StopIteration:
                cycle+=1;sampler.set_epoch(cycle);iterator=iter(loader);batch=next(iterator)
            # The sampler order is independent of LR; hash input indices, not labels.
            # Track the exact sampler's per-rank source order for this cycle.
            offset=(step%len(loader))*micro
            positions=list(iter(sampler))[offset:offset+len(batch['input_ids'])]
            for position in positions:input_digest.update((train_rows[position]['window_id']+'\n').encode())
            optimizer.zero_grad(set_to_none=True)
            result=wrapped(c.to_device(batch,device));loss=result['loss'];assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is None for p in frozen)
            grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad)
            optimizer.step();schedule.step();step+=1
            nll+=result['nll_sum'].double();tokens+=result['tokens'].double();sample_count+=len(batch['input_ids'])
            totals=torch.stack([nll,tokens]);dist.all_reduce(totals)
            if rank==0:
                row={'key':key,'stage':stage,'peak_lr':lr,'optimizer_step':step,'total_steps':total_steps,
                     'sampler_epoch':cycle+1,'train_answer_nll':float(totals[0]/totals[1]),
                     'learning_rates':schedule.get_last_lr(),'wall_seconds':time.perf_counter()-began,
                     'microbatch_per_gpu':micro,'effective_batch_size':micro*world,
                     'peak_allocated_GiB_rank0':torch.cuda.max_memory_allocated(device)/2**30,
                     'peak_reserved_GiB_rank0':torch.cuda.max_memory_reserved(device)/2**30}
                c.write((c.out(key)/'ddp_check_progress.json') if check_only else trial_dir/'training_progress.json',row)
                if step%5==0 or step==total_steps:
                    print('TRAIN '+json.dumps(row),flush=True)
                    if not check_only:
                        with (trial_dir/'training_history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if stage=='formal' and step%len(loader)==0 and not check_only:
                epoch=step//len(loader);system.eval()
                a=torch.zeros((),device=device,dtype=torch.float64);b=torch.zeros_like(a)
                with torch.inference_mode():
                    for val_batch in validation:
                        value=system(c.to_device(val_batch,device));a+=value['nll_sum'].double();b+=value['tokens'].double()
                values=torch.stack([a,b]);dist.all_reduce(values)
                assert versions==[p._version for p in frozen]
                if rank==0:
                    saved=save_checkpoint(system,key,'formal',epoch)
                    record={'epoch':epoch,'optimizer_step':step,'validation_answer_nll':float(values[0]/values[1]),
                            'validation_tokens':int(values[1]),'checkpoint':saved,'wall_seconds':time.perf_counter()-began}
                    epoch_records.append(record)
                    with (c.out(key)/'validation_nll_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print('VALIDATION_NLL '+json.dumps(record),flush=True)
                wrapped.train();dist.barrier()
        assert versions==[p._version for p in frozen]
        hashes=[None]*world;dist.all_gather_object(hashes,input_digest.hexdigest())
        if rank==0:
            record={'stage':stage,'peak_lr':lr,'optimizer_steps':step,'sampler_epochs_started':cycle+1,
                    'sample_presentations':sample_count*world,'input_order_sha256_per_rank':hashes,
                    'initial_lora_sha256':initial_adapter,'warm_converter_sha256':cfg['warm_converter_sha256'],
                    'qwen_base_versions_unchanged':True,'qwen_base_gradients_always_none':True,
                    'session_wall_seconds':time.perf_counter()-began,
                    'peak_allocated_GiB_rank0':torch.cuda.max_memory_allocated(device)/2**30,
                    'peak_reserved_GiB_rank0':torch.cuda.max_memory_reserved(device)/2**30,
                    'common_code_sha256':c.sha(Path(c.__file__)),'code_sha256':c.sha(Path(__file__))}
            if check_only:
                record.update(microbatch_per_gpu=micro,effective_batch_size=micro*world,technical_only=True)
                c.write(c.out(key)/'ddp_check.json',record)
            elif stage=='pilot':
                record['checkpoint']=save_checkpoint(system,key,'pilot',lr)
                c.write(trial_dir/'training_complete.json',record)
            else:c.write(c.out(key)/'training_complete.json',record)
            trials.append(record)
        dist.barrier()
        del optimizer,schedule
        gc.collect();torch.cuda.empty_cache()
    if not check_only:
        if rank==0:
            after=c.frozen_fingerprint(system);assert after==base_hash
            proof={'stage':stage,'qwen_base_sha256_before':base_hash,'qwen_base_sha256_after':after,
                   'qwen_base_versions_unchanged':True,'qwen_base_gradients_always_none':True,
                   'initial_lora_sha256':initial_adapter,'trials':trials,'whole_session_wall_seconds':time.perf_counter()-all_began,
                   'completed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources()}
            if stage=='pilot':
                assert len({tuple(t['input_order_sha256_per_rank']) for t in trials})==1
                c.write(c.out(key)/'pilot_training_complete.json',proof)
            else:c.write(c.out(key)/'formal_training_verified.json',proof)
        dist.barrier()
    dist.destroy_process_group()


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
        shutil.copyfile(source/'converter.safetensors',target/'converter.safetensors')
        c.write(c.out(key)/'export_validation.json',{'selected_epoch':int(tag),'adapter_sha256':c.sha(adapter/'adapter_model.safetensors'),
             'converter_sha256':c.sha(target/'converter.safetensors'),'merged_for_vllm':True,'nonzero_scopes':probes,
             'merged_weight_files_sha256':{p.name:c.sha(p) for p in target.glob('model-*.safetensors')},
             'checkpoint':str(source.relative_to(ROOT))})
    print('EXPORT COMPLETE '+key+' '+stage+' '+str(tag)+' final='+str(final),flush=True)


def image_folder(key,stage,tag,split):
    if split=='test':return c.data_dir(key)/'test'
    return c.data_dir(key)/('lr_search/lr_'+c.lr_tag(float(tag)) if stage=='pilot' else 'validation/epoch_'+str(int(tag)))


def render(key,stage,tag,split,device):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8);torch.cuda.set_device(device)
    w,x,y,cs=c.read_data();converter=c.build_converter()
    source=c.checkpoint(key,stage,tag)/'converter.safetensors'
    converter.load_state_dict(load_file(str(source)));converter.to(device).eval()
    dest=image_folder(key,stage,tag,split);dest.parent.mkdir(parents=True,exist_ok=True)
    c.write_images(converter,x,w.index[w.split.eq(split)].to_numpy(),w,dest,device,autocast=True)
    c.write(dest/'render_protocol.json',{'key':key,'stage':stage,'tag':tag,'split':split,'converter_sha256':c.sha(source),
       'width':224,'height':224,'image_format':'WebP lossless RGB','blue_channel':128,
       'input':'Original2s/200sample windows; acceleration/gyro triaxial RMS, center196 samples,49 chronological4sample patches',
       'precision':'FP32 converter with BF16 forward, fixed8bit rounding matching STE training forward',
       'windows_sha256':c.sha(ROOT/'dataset/splits/windows.csv'),'splits_sha256':c.sha(ROOT/'dataset/splits/dog_splits.json')})


def result_folder(key,stage,tag,split):
    if split=='test':return c.out(key)
    return c.out(key)/('lr_search/lr_'+c.lr_tag(float(tag))+'/validation' if stage=='pilot' else 'validation/epoch_'+str(int(tag)))


def evaluate(key,stage,tag,split,device,final=False):
    os.environ['CUDA_VISIBLE_DEVICES']=device.split(':')[1]
    import torch
    import pandas as pd
    import importlib.metadata as md
    from transformers import AutoProcessor
    from vllm import LLM
    import qwen_classify as q
    from eval_imu2img_alignment import evaluate as metrics
    torch.set_num_threads(8)
    target=c.model_dir(key) if final else c.model_dir(key)/'evaluation_merged'
    dest=result_folder(key,stage,tag,split);dest.mkdir(parents=True,exist_ok=True)
    w=pd.read_csv(ROOT/'dataset/splits/windows.csv');w=w[w.split.eq(split)]
    folder=image_folder(key,stage,tag,split);im=pd.read_csv(folder/'images.csv')
    assert im.window_id.tolist()==w.window_id.tolist()
    for row in im.itertuples():assert c.sha(ROOT/row.image_file)==row.image_sha256
    base_manifest=json.loads((ROOT/'models/model_manifest.json').read_text())[c.NAMES[key]]
    source=c.checkpoint(key,stage,tag);prompt=(c.out(key)/'prompt.txt').read_text()
    protocol={'model':c.model_dir(key).name,'key':key,'stage':stage,'tag':tag,'split':split,'classes':c.classes(),
      'base_revision':base_manifest['revision'],'expected_windows':len(w),'inference_backend':'vllm','vllm_version':md.version('vllm'),
      'qwen_base_weights_frozen':True,'lora_r':4,'lora_alpha':8,'image_dir':str(folder.relative_to(ROOT)),
      'image_manifest_sha256':c.sha(folder/'images.csv'),'render_protocol_sha256':c.sha(folder/'render_protocol.json'),
      'adapter_sha256':c.sha(source/'adapter_model.safetensors'),'converter_sha256':c.sha(source/'converter.safetensors'),
      'training_protocol_sha256':c.sha(c.out(key)/('lr_search/lr_'+c.lr_tag(float(tag)) if stage=='pilot' else '')/'training_protocol.json'),
      'prompt_sha256':c.sha(c.out(key)/'prompt.txt'),'windows_sha256':c.sha(ROOT/'dataset/splits/windows.csv'),
      'splits_sha256':c.sha(ROOT/'dataset/splits/dog_splits.json'),'image_size':[224,224],'visual_tokens':49,
      'generation':{'temperature':0,'enable_thinking':False,'max_new_tokens':32,'seed':c.SEED,'guided_decoding':False},
      'engine':{'dtype':'bfloat16','tensor_parallel_size':1,'max_model_len':4096,'max_num_seqs':32,'max_num_batched_tokens':8192,
                'gpu_memory_utilization':.85,'enable_prefix_caching':False,'min_pixels':50176,'max_pixels':1474560,'flashinfer_sampler':False}}
    dest.mkdir(parents=True,exist_ok=True);path=dest/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen evaluation protocol differs.'
    else:c.write(path,protocol)
    (dest/'prompt.txt').write_text(prompt)
    raw=dest/'responses.jsonl';done={}
    if raw.exists():
        for line in raw.read_text().splitlines():
            row=json.loads(line);assert row['window_id'] not in done;done[row['window_id']]=row
    assert set(done).issubset(set(w.window_id))
    pending=[r for r in im.to_dict('records') if r['window_id'] not in done]
    if pending:
        start=time.perf_counter();processor=AutoProcessor.from_pretrained(target,local_files_only=True)
        model=LLM(model=str(target),dtype='bfloat16',tensor_parallel_size=1,trust_remote_code=False,seed=c.SEED,
          gpu_memory_utilization=.85,max_model_len=4096,max_num_seqs=32,max_num_batched_tokens=8192,
          limit_mm_per_prompt={'image':1,'video':0},mm_processor_kwargs={'min_pixels':50176,'max_pixels':1474560},
          generation_config='vllm',enable_prefix_caching=False,disable_log_stats=True)
        load_seconds=time.perf_counter()-start;timings=[]
        with raw.open('a') as f:
            for offset in range(0,len(pending),32):
                group=pending[offset:offset+32]
                texts,counts,lengths,truncated,pre,gen,wall=q.generate(model,processor,[ROOT/r['image_file'] for r in group],prompt,device,(224,224))
                timings.append({'n':len(group),'preprocess_seconds':pre,'generation_seconds':gen,'request_wall_seconds':wall})
                for i,item in enumerate(group):
                    prediction,status=q.parse(texts[i],c.classes(),truncated[i])
                    row={name:item[name] for name in ['window_id','image_file','image_sha256']}
                    row.update(response=texts[i],prediction=prediction,status=status,input_tokens=counts[i],generated_tokens=lengths[i],
                               batch_size=len(group),amortized_request_seconds=wall/len(group),inference_backend='vllm')
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');done[row['window_id']]=row
                f.flush()
                state={'completed':len(done),'expected':len(w),'remaining':len(w)-len(done),
                       'status_counts':dict(Counter(r['status'] for r in done.values())),'wall_seconds':time.perf_counter()-start}
                c.write(dest/'progress.json',state)
                if len(done)%320==0 or len(done)==len(w):print('GENERATE '+json.dumps(state),flush=True)
        file=dest/'runtime.json';runtime=json.loads(file.read_text()) if file.exists() else {'sessions':[]}
        runtime['sessions'].append({'load_seconds':load_seconds,'wall_seconds':time.perf_counter()-start,'batch_timings':timings,
            'device':device,'code_sha256':c.sha(Path(__file__)),'common_code_sha256':c.sha(Path(c.__file__)),
            'generation_code_sha256':c.sha(ROOT/'scripts/qwen_classify.py')})
        c.write(file,runtime)
    metrics(dest)


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
    cfg=json.loads((c.SUITE/'protocol.json').read_text());script='scripts/lora_27b.py'
    environment=os.environ.copy();environment.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
    if not (c.SUITE/'batch_selection.json').exists():
        profiles=[json.loads((c.out(k)/'batch_profile.json').read_text()) for k in c.KEYS]
        micro=min(p['safe_max_microbatch'] for p in profiles)
        c.write(c.SUITE/'batch_selection.json',{'microbatch_per_gpu':micro,'world_size':2,'gradient_accumulation':1,
            'effective_batch_size':2*micro,'rule':'Common minimum of largest safe profiled batches, with2GiB reserved headroom',
            'profile_sha256':{k:c.sha(c.out(k)/'batch_profile.json') for k in c.KEYS}})
    # Both capacities must pass DDP before any LR candidate is trained.
    for key in c.KEYS:
        if not (c.out(key)/'ddp_check.json').exists():
            run_process(['-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','2',script,'ddp-check','--key',key],
                        c.out(key)/'ddp_check.log',environment)
    if preflight_only:return
    archived=c.SUITE/'source';archived.mkdir(exist_ok=True)
    for name,digest in sources().items():
        target=archived/Path(name).name
        if target.exists():assert c.sha(target)==digest
        else:shutil.copyfile(ROOT/name,target)
    c.write(c.SUITE/'source_sha256.json',sources())
    for key in c.KEYS:
        if not (c.out(key)/'pilot_training_complete.json').exists():
            run_process(['-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','2',script,'pilot-train','--key',key],
                        c.out(key)/'pilot_train.log',environment)
        if not (c.out(key)/'lr_selection.json').exists():
            for lr in c.LRS:
                dest=result_folder(key,'pilot',lr,'validation')
                if (dest/'metrics.json').exists():continue
                tag=c.lr_tag(lr)
                run_process([script,'export','--key',key,'--stage','pilot','--tag',tag],c.out(key)/('export_pilot_'+tag+'.log'))
                if not (image_folder(key,'pilot',lr,'validation')/'render_protocol.json').exists():
                    run_process([script,'render','--key',key,'--stage','pilot','--tag',tag,'--device','cuda:1'],c.out(key)/('render_pilot_'+tag+'.log'))
                run_process([script,'evaluate','--key',key,'--stage','pilot','--tag',tag,'--device','cuda:0'],c.out(key)/('validation_pilot_'+tag+'.log'))
            select_lr(key)
        if not (c.out(key)/'formal_training_verified.json').exists():
            run_process(['-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','2',script,'train','--key',key],
                        c.out(key)/'train.log',environment)
        if not (c.out(key)/'generation_complete.json').exists():
            scores=[]
            for epoch in [1,2,3]:
                dest=result_folder(key,'formal',epoch,'validation')
                if not (dest/'metrics.json').exists():
                    run_process([script,'export','--key',key,'--tag',str(epoch)],c.out(key)/f'export_epoch{epoch}.log')
                    if not (image_folder(key,'formal',epoch,'validation')/'render_protocol.json').exists():
                        run_process([script,'render','--key',key,'--tag',str(epoch),'--device','cuda:1'],c.out(key)/f'render_epoch{epoch}.log')
                    run_process([script,'evaluate','--key',key,'--tag',str(epoch),'--device','cuda:0'],c.out(key)/f'validation_epoch{epoch}.log')
                m=json.loads((dest/'metrics.json').read_text());scores.append({'epoch':epoch,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
            best=max(scores,key=lambda r:r['macro_f1']);epoch=best['epoch']
            c.write(c.out(key)/'checkpoint_selection.json',{'best_epoch':epoch,'best_macro_f1':best['macro_f1'],'scores':scores,
                    'metric':'Full validation vLLM macro-F1','test_used_for_selection':False,'ties':'earliest epoch'})
            run_process([script,'export','--key',key,'--tag',str(epoch),'--final'],c.out(key)/'export_final.log')
            if not (image_folder(key,'formal',epoch,'test')/'render_protocol.json').exists():
                run_process([script,'render','--key',key,'--tag',str(epoch),'--split','test','--device','cuda:1'],c.out(key)/'render_test.log')
            run_process([script,'evaluate','--key',key,'--tag',str(epoch),'--split','test','--device','cuda:0','--final'],c.out(key)/'test.log')
            temp=c.model_dir(key)/'evaluation_merged'
            if temp.exists():shutil.rmtree(temp)
            c.write(c.out(key)/'generation_complete.json',{'state':'complete','selected_epoch':epoch,'reports_pending':True})
    c.write(c.SUITE/'generation_complete.json',{'state':'complete','keys':c.KEYS,'reports_pending':True})
    print('BOTH27B EXPERIMENTS GENERATED',flush=True)
    run_process(['scripts/validate_lora_27b.py'],c.SUITE/'validation.log')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','profile','ddp-check','pilot-train','train','export','render','evaluate','select-lr','pipeline','preflight'])
    parser.add_argument('--key',choices=c.KEYS);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--mode',choices=['joint','frozen'],default='joint')
    parser.add_argument('--stage',choices=['pilot','formal'],default='formal');parser.add_argument('--tag',default='1')
    parser.add_argument('--split',choices=['validation','test'],default='validation');parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.mode)
    elif args.command in ['pipeline','preflight']:pipeline(args.command=='preflight')
    else:
        assert args.key
        if args.command=='profile':profile(args.key,args.device)
        elif args.command=='ddp-check':train(args.key,'pilot',check_only=True)
        elif args.command=='pilot-train':train(args.key,'pilot')
        elif args.command=='train':train(args.key,'formal')
        elif args.command=='select-lr':select_lr(args.key)
        elif args.command=='export':export(args.key,args.stage,args.tag,args.final)
        elif args.command=='render':render(args.key,args.stage,args.tag,args.split,args.device)
        else:evaluate(args.key,args.stage,args.tag,args.split,args.device,args.final)
