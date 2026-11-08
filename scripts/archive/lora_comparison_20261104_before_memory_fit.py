"""Three matched fresh-LoRA experiments: six plots, fixed IMU2IMG, joint IMU2IMG."""
import argparse
from collections import Counter
from contextlib import nullcontext
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.venv/lora-packages'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/pet-lora-triton')
import train_qwen35_lora as old
from align_imu2img_qwen import quantize_rgb,pack_qwen_pixels,write_images,parameter_fingerprint
from train_imu2img import build_converter,read_data

BASE=ROOT/'models/Qwen3.5-4B'
WARM=ROOT/'models/IMU2IMG-Qwen3.5-4B-aligned/converter.safetensors'
SUITE=ROOT/'results/lora_comparison'
KEYS=['sixplot','imu2img_frozen','imu2img_joint']
SEED=20261101
MICRO=4


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def out(key):return ROOT/'results'/('qwen3.5_4B_lora_'+key)
def model_dir(key):return ROOT/'models'/('Qwen3.5-4B-lora-'+key.replace('_','-'))
def data_dir(key):return ROOT/'dataset'/('vlm' if key=='sixplot' else 'vlm_imu2img_qwen')
def classes():return json.loads((ROOT/'results/protocol.json').read_text())['classes']


def prepare():
    SUITE.mkdir(exist_ok=True)
    folders=[out(k) for k in KEYS]
    if not (SUITE/'prior_files_sha256.json').exists():
        paths=[p for p in (ROOT/'results').rglob('*') if p.is_file() and p.parent!=ROOT/'results'
               and not p.is_relative_to(SUITE) and not any(p.is_relative_to(f) for f in folders)]
        paths += [p for p in (ROOT/'dataset/splits').glob('*') if p.is_file()]
        for d in ['vlm','vlm_acl','vlm_imu2img','vlm_imu2img_qwen']:
            paths += [p for p in (ROOT/'dataset'/d).glob('*') if p.is_file()]
        paths += [WARM,ROOT/'models/Qwen3.5-4B-lora/adapter/adapter_model.safetensors']
        write(SUITE/'prior_files_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    for key in KEYS:
        out(key).mkdir(exist_ok=True);model_dir(key).mkdir(exist_ok=True)
        prompt=data_dir(key)/'prompt.txt';dest=out(key)/'prompt.txt'
        if dest.exists():assert dest.read_bytes()==prompt.read_bytes()
        else:dest.write_bytes(prompt.read_bytes())
    protocol={'state':'authorized_and_fixed_before_training','experiments':KEYS,'seed':SEED,'epochs':3,
       'lora':{'r':4,'alpha':8,'dropout':.05,'target_scopes':['ViT','visual_projection','language_attention'],
               'trainable_parameters':5226496,'initialization':'Fresh PEFT LoRA from the original Qwen3.5-4B, identical seed; no prior ACL adapter'},
       'base_revision':json.loads((ROOT/'models/model_manifest.json').read_text())['Qwen3.5-4B']['revision'],
       'warm_converter_sha256':sha(WARM),'warm_converter':'Original frozen-Qwen-aligned converter epoch3',
       'optimizer':'AdamW','lora_learning_rate':.0002,'joint_converter_learning_rate':.00001,
       'weight_decay':.01,'betas':[.9,.999],'epsilon':1e-8,'warmup_ratio':.05,'scheduler':'linear','max_grad_norm':1.,
       'microbatch_per_gpu':MICRO,'world_size':2,'gradient_accumulation':32//(2*MICRO),'effective_batch_size':32,
       'dtype':'BF16 Qwen base; FP32 LoRA/converter/optimizer; BF16 converter forward and 8-bit straight-through bridge',
       'gradient_checkpointing':False,'train_windows':16461,'validation_windows':4727,'test_windows':3708,
       'selection':'Full validation greedy vLLM macro-F1 over all3 checkpoints; earliest epoch on ties; test excluded',
       'loss':'Unweighted answer JSON plus EOS token cross entropy, all input/padding masked; no resampling or augmentation',
       'prompts':{k:{'path':str((out(k)/'prompt.txt').relative_to(ROOT)),'sha256':sha(out(k)/'prompt.txt')} for k in KEYS},
       'windows_sha256':sha(ROOT/'dataset/splits/windows.csv'),'splits_sha256':sha(ROOT/'dataset/splits/dog_splits.json'),
       'comparisons':'1vs2 compares image representation plus paired prompt; 2vs3 changes only converter optimization (same warm converter, fresh LoRA, prompt and optimizer-step budget).',
       'label_budget_note':'All use the same16461 labeled training windows. IMU2IMG starts from a converter that previously received additional supervised training; warm-start compute is reported separately.',
       'test_note':'This fixed test set has been viewed in earlier experiments; this run is fixed before its new test outputs.'}
    dest=SUITE/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==protocol
    else:write(dest,protocol)
    print('Three experiment protocols and prompts fixed.',flush=True)


class Answers:
    def __init__(self,processor,key):
        import torch
        from PIL import Image
        self.processor=processor;self.key=key
        if key!='sixplot':processor.image_processor.size={'shortest_edge':224**2,'longest_edge':1280*1152}
        image=Image.new('RGB',(1280,1152) if key=='sixplot' else (224,224),(128,128,128))
        prompt=(out(key)/'prompt.txt').read_text()
        conversation=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
        encoded=processor.apply_chat_template(conversation,tokenize=True,return_dict=True,return_tensors='pt',
                                              add_generation_prompt=True,enable_thinking=False)
        self.prefix=encoded['input_ids'][0].tolist();self.types=encoded['mm_token_type_ids'][0].tolist()
        self.grid=encoded['image_grid_thw'][0].tolist()
        expected=[1,72,80] if key=='sixplot' else [1,14,14]
        assert self.grid==expected,(self.grid,expected)
        self.targets={c:processor.tokenizer.encode(json.dumps({'label':c})+processor.tokenizer.eos_token,
                    add_special_tokens=False) for c in classes()}
        self.pad=processor.tokenizer.pad_token_id
        self.check={'prefix_tokens':len(self.prefix),'grid':self.grid,'visual_tokens':math.prod(self.grid)//4,
                    'image_size':[image.width,image.height], 'prompt_sha256':sha(out(key)/'prompt.txt')}

    def __call__(self,rows):
        import torch
        from PIL import Image
        sequences=[self.prefix+self.targets[r['label']] for r in rows]
        length=max(map(len,sequences));ids=torch.full((len(rows),length),self.pad,dtype=torch.long)
        labels=torch.full_like(ids,-100);attention=torch.zeros_like(ids);types=torch.zeros_like(ids)
        types[:,:len(self.prefix)]=torch.tensor(self.types)
        for i,seq in enumerate(sequences):
            ids[i,:len(seq)]=torch.tensor(seq);attention[i,:len(seq)]=1
            labels[i,len(self.prefix):len(seq)]=torch.tensor(self.targets[rows[i]['label']])
        batch={'input_ids':ids,'attention_mask':attention,'mm_token_type_ids':types,'labels':labels,
               'image_grid_thw':torch.tensor([self.grid]*len(rows),dtype=torch.long)}
        if self.key=='sixplot':
            images=[]
            for r in rows:
                with Image.open(ROOT/r['image_file']) as im:
                    assert im.size==(1280,1152);images.append(im.convert('RGB').copy())
            vision=self.processor.image_processor(images=images,return_tensors='pt')
            assert all(g.tolist()==self.grid for g in vision['image_grid_thw'])
            batch['pixel_values']=vision['pixel_values']
        else:
            batch['signals']=torch.stack([SIGNALS[int(r['cache_index'])] for r in rows])
        return batch


def rows(key,partial=False):
    global SIGNALS
    import pandas as pd
    w,SIGNALS,y,cs=read_data()
    selected=w[w.split.isin(['train','validation'])].copy();selected['cache_index']=selected.index
    if key=='sixplot':
        im=pd.read_csv(data_dir(key)/'supervised/images.csv')
        selected=selected.merge(im[['window_id','image_file','image_sha256']],on='window_id',validate='one_to_one',how='inner')
        if not partial:assert selected.split.value_counts().to_dict()=={'train':16461,'validation':4727}
    return selected


def build_system(key):
    import torch
    from torch import nn
    from safetensors.torch import load_file
    qwen,processor,info=old.make_model()
    class System(nn.Module):
        def __init__(self):
            super().__init__();self.qwen=qwen;self.answers=Answers(processor,key)
            self.converter=None
            if key!='sixplot':
                self.converter=build_converter();self.converter.load_state_dict(load_file(str(WARM)))
                self.converter.requires_grad_(key=='imu2img_joint')
            self.debug=False

        def train(self,mode=True):
            super().train(mode)
            if self.converter is not None and key=='imu2img_frozen':self.converter.eval()
            return self

        def forward(self,batch):
            import torch.nn.functional as F
            batch=dict(batch);labels=batch.pop('labels');signals=batch.pop('signals',None)
            if signals is not None:
                with torch.autocast('cuda',dtype=torch.bfloat16):image=self.converter(signals.float())
                unit=quantize_rgb(image,straight_through=key=='imu2img_joint')
                batch['pixel_values']=pack_qwen_pixels(unit).to(torch.bfloat16)
                if self.debug and key=='imu2img_joint':
                    batch['pixel_values'].retain_grad();self.pixels=batch['pixel_values']
            positions=torch.arange(len(self.answers.prefix)-1,batch['input_ids'].shape[1]-1,device=labels.device)
            target=labels[:,len(self.answers.prefix):]
            output=self.qwen(**batch,use_cache=False,logits_to_keep=positions)
            assert output.logits.shape[:2]==target.shape
            total=F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='sum')
            n=target.ne(-100).sum()
            return {'loss':total/n,'nll_sum':total.detach(),'tokens':n.detach()}
    system=System()
    assert sum(p.numel() for p in system.qwen.parameters() if p.requires_grad)==5226496
    assert all('lora_' in n for n,p in system.qwen.named_parameters() if p.requires_grad)
    return system,processor,info


def to_device(batch,device):
    import torch
    return {k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() and k!='signals' else v.dtype) for k,v in batch.items()}


def frozen_fingerprint(system):
    import torch
    digest=hashlib.sha256()
    for name,p in system.qwen.named_parameters():
        if p.requires_grad:continue
        digest.update(name.encode());digest.update(str(p.dtype).encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def smoke(key):
    import torch
    torch.set_num_threads(8);torch.cuda.set_device(0)
    frame=rows(key,partial=True).query("split=='train'").head(MICRO).to_dict('records')
    assert len(frame)==MICRO
    system,processor,info=build_system(key);system.to('cuda:0').train();system.debug=True
    frozen=[p for p in system.qwen.parameters() if not p.requires_grad];versions=[p._version for p in frozen]
    batch=system.answers(frame)
    if key!='sixplot':
        # Exact official preprocessor equivalence on training samples only.
        from PIL import Image
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            unit=quantize_rgb(system.converter(batch['signals'].cuda()),False)
        rgb=torch.round(unit*255).byte().permute(0,2,3,1).cpu().numpy()
        reference=processor.image_processor(images=[Image.fromarray(a) for a in rgb],return_tensors='pt')['pixel_values']
        packed=pack_qwen_pixels(unit).cpu()
        assert torch.equal(packed.to(torch.bfloat16),reference.to(torch.bfloat16))
    began=time.perf_counter();result=system(to_device(batch,'cuda:0'));result['loss'].backward();torch.cuda.synchronize()
    grads={scope:sum(float(p.grad.float().square().sum()) for n,p in system.qwen.named_parameters()
                     if p.requires_grad and old.scope(n)==scope and p.grad is not None)**.5 for scope in info['modules_per_scope']}
    assert all(math.isfinite(v) and v>0 for v in grads.values()),grads
    assert all(p.grad is None for p in frozen) and versions==[p._version for p in frozen]
    conv_grad=None
    if system.converter is not None:
        conv_grad={scope:sum(float(p.grad.float().square().sum()) for n,p in system.converter.named_parameters()
                            if n.startswith(scope) and p.grad is not None)**.5 for scope in ['encoders','decoders']}
        if key=='imu2img_frozen':assert all(v==0 for v in conv_grad.values())
        else:assert all(v>0 for v in conv_grad.values()) and system.pixels.grad.abs().sum()>0
    record={'purpose':'Training-only backward check, no optimizer update','key':key,'microbatch':MICRO,
            'loss':float(result['loss'].detach()),'qwen_base_frozen_no_grad_and_versions_unchanged':True,
            'lora_grad_norms':grads,'converter_grad_norms':conv_grad,'official_processor_equal':key!='sixplot',
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'wall_seconds':time.perf_counter()-began,
            'processor':system.answers.check,'code_sha256':sha(Path(__file__))}
    write(out(key)/'backward_check.json',record);print(json.dumps(record,indent=2),flush=True)


def train(key):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader,DistributedSampler
    from transformers import get_linear_schedule_with_warmup
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    assert world==2;torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(8)
    dist.init_process_group('nccl');config=json.loads((SUITE/'protocol.json').read_text())
    assert sha(WARM)==config['warm_converter_sha256'] and sha(out(key)/'prompt.txt')==config['prompts'][key]['sha256']
    frame=rows(key);system,processor,info=build_system(key);system.to(device)
    frozen=[p for p in system.qwen.parameters() if not p.requires_grad];versions=[p._version for p in frozen]
    initial_hash=frozen_fingerprint(system) if rank==0 else None
    initial_adapter=get_peft_model_state_dict(system.qwen)
    initial_lora_hash=hashlib.sha256(b''.join(n.encode()+p.detach().cpu().contiguous().numpy().tobytes() for n,p in sorted(initial_adapter.items()))).hexdigest()
    del initial_adapter
    torch.manual_seed(SEED+rank);random.seed(SEED+rank);np.random.seed(SEED+rank)
    training=frame[frame.split.eq('train')].to_dict('records')
    sampler=DistributedSampler(training,num_replicas=world,rank=rank,shuffle=True,seed=SEED,drop_last=False)
    loader=DataLoader(training,batch_size=MICRO,sampler=sampler,collate_fn=system.answers,num_workers=0)
    validation=DataLoader(frame[frame.split.eq('validation')].to_dict('records')[rank::world],batch_size=MICRO,
                          shuffle=False,collate_fn=system.answers,num_workers=0)
    groups=[{'params':[p for p in system.qwen.parameters() if p.requires_grad],'lr':config['lora_learning_rate']}]
    if key=='imu2img_joint':groups.append({'params':list(system.converter.parameters()),'lr':config['joint_converter_learning_rate']})
    parameters=[p for group in groups for p in group['params']]
    assert len({id(p) for p in parameters})==len(parameters)
    assert {id(p) for p in parameters}=={id(p) for p in system.parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    accum=config['gradient_accumulation'];per_epoch=math.ceil(len(loader)/accum);steps=3*per_epoch
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(.05*steps),steps)
    if rank==0:
        cfg={**config,'key':key,'model_name':model_dir(key).name,'qwen_base_frozen_hash_before':initial_hash,
             'initial_lora_sha256':initial_lora_hash,'converter_trainable_parameters':835584 if key=='imu2img_joint' else 0,
             'processor':system.answers.check,'trainable_parameters':sum(p.numel() for p in parameters),
             'versions':{p:md.version(p) for p in ['torch','transformers','peft','vllm']},'code_sha256':sha(Path(__file__))}
        path=out(key)/'training_protocol.json'
        if path.exists():assert json.loads(path.read_text())==cfg
        else:write(path,cfg)
        write(out(key)/'target_modules.json',info)
    resume=model_dir(key)/'last_state.pt';first=0;step=0
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        set_peft_model_state_dict(system.qwen,state['adapter'])
        if system.converter is not None:system.converter.load_state_dict(state['converter'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);first=state['epoch'];step=state['step']
    wrapped=DDP(system,device_ids=[local],broadcast_buffers=False)
    began=time.perf_counter();initial_step=step
    for epoch in range(first,3):
        sampler.set_epoch(epoch);wrapped.train();optimizer.zero_grad(set_to_none=True)
        loss_sum=torch.zeros((),device=device,dtype=torch.float64);tokens=torch.zeros_like(loss_sum);epoch_start=time.perf_counter()
        for index,batch in enumerate(loader):
            group_start=index//accum*accum;group_size=min(accum,len(loader)-group_start)
            update=(index+1)%accum==0 or index+1==len(loader)
            with (wrapped.no_sync() if not update else nullcontext()):
                result=wrapped(to_device(batch,device));loss=result['loss'];assert torch.isfinite(loss)
                (loss/group_size).backward()
            loss_sum+=result['nll_sum'].double();tokens+=result['tokens'].double()
            if update:
                assert all(p.grad is None for p in frozen)
                grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                if step%10==0 or index+1==len(loader):
                    totals=torch.stack([loss_sum,tokens]);dist.all_reduce(totals)
                    if rank==0:
                        row={'key':key,'epoch':epoch+1,'optimizer_step':step,'total_steps':steps,
                             'train_answer_nll':float(totals[0]/totals[1]),'learning_rates':scheduler.get_last_lr(),
                             'wall_seconds':time.perf_counter()-began,'peak_allocated_GiB_rank0':torch.cuda.max_memory_allocated()/2**30}
                        write(out(key)/'training_progress.json',row)
                        with (out(key)/'training_history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        print('TRAIN '+json.dumps(row),flush=True)
        system.eval();val_sum=torch.zeros((),device=device,dtype=torch.float64);val_tokens=torch.zeros_like(val_sum)
        with torch.inference_mode():
            for batch in validation:
                result=system(to_device(batch,device));val_sum+=result['nll_sum'].double();val_tokens+=result['tokens'].double()
        totals=torch.stack([val_sum,val_tokens]);dist.all_reduce(totals)
        assert versions==[p._version for p in frozen]
        if rank==0:
            from safetensors.torch import save_file
            checkpoint=model_dir(key)/'checkpoints'/f'epoch_{epoch+1}'
            system.qwen.save_pretrained(checkpoint,safe_serialization=True)
            if system.converter is not None:save_file({k:v.detach().cpu().contiguous() for k,v in system.converter.state_dict().items()},str(checkpoint/'converter.safetensors'))
            torch.save({'epoch':epoch+1,'step':step,'adapter':{k:v.cpu() for k,v in get_peft_model_state_dict(system.qwen).items()},
                        'converter':system.converter.state_dict() if system.converter is not None else None,
                        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},resume)
            record={'epoch':epoch+1,'optimizer_step':step,'validation_answer_nll':float(totals[0]/totals[1]),
                    'validation_tokens':int(totals[1]),'epoch_wall_seconds':time.perf_counter()-epoch_start,
                    'adapter_sha256':sha(checkpoint/'adapter_model.safetensors')}
            with (out(key)/'validation_nll_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('VALIDATION_NLL '+json.dumps(record),flush=True)
        dist.barrier()
    if rank==0:
        final_hash=frozen_fingerprint(system);assert initial_hash==final_hash
        if key=='imu2img_frozen':
            assert all(sha(model_dir(key)/'checkpoints'/f'epoch_{e}'/'converter.safetensors')==sha(WARM) for e in [1,2,3])
        write(out(key)/'training_complete.json',{'state':'complete','epochs':3,'optimizer_steps':step,'initial_lora_sha256':initial_lora_hash,
             'qwen_base_frozen_hash_before':initial_hash,'qwen_base_frozen_hash_after':final_hash,'qwen_base_versions_unchanged':True,
             'trainable_parameters':sum(p.numel() for p in parameters),'session_wall_seconds':time.perf_counter()-began,
             'session_optimizer_steps':step-initial_step,'code_sha256':sha(Path(__file__)),
             'completed_utc':datetime.now(timezone.utc).isoformat(),'selection_pending':'Full validation vLLM macro-F1'})
    dist.destroy_process_group()


def export(key,epoch,final=False):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor
    from peft import PeftModel
    torch.set_num_threads(8)
    checkpoint=model_dir(key)/'checkpoints'/f'epoch_{epoch}'
    target=model_dir(key) if final else model_dir(key)/'validation_merged'
    base=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(base,checkpoint)
    probes={}
    for name,layer in model.named_modules():
        scope=old.scope(name)
        if scope and scope not in probes and hasattr(layer,'lora_A'):
            delta=layer.get_delta_weight('default').detach().float();assert delta.abs().sum()>0
            probes[scope]={'module':name,'delta_l1':float(delta.abs().sum())}
    assert len(probes)==3
    model=model.merge_and_unload(safe_merge=True)
    assert not any('lora_' in n for n,_ in model.named_parameters())
    model.save_pretrained(target,safe_serialization=True,max_shard_size='5GB')
    AutoProcessor.from_pretrained(BASE,local_files_only=True).save_pretrained(target)
    if final:
        adapter=target/'adapter';adapter.mkdir(exist_ok=True)
        for name in ['adapter_config.json','adapter_model.safetensors']:shutil.copyfile(checkpoint/name,adapter/name)
        if key!='sixplot':shutil.copyfile(checkpoint/'converter.safetensors',target/'converter.safetensors')
        write(out(key)/'export_validation.json',{'selected_epoch':epoch,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
              'merged_for_vllm':True,'nonzero_scopes':probes,'merged_files_sha256':{p.name:sha(p) for p in target.glob('*.safetensors')},
              'source_checkpoint':str(checkpoint.relative_to(ROOT))})
    print('EXPORTED '+key+' '+str(epoch)+' final='+str(final),flush=True)


def image_folder(key,epoch,split):
    if key=='sixplot':return data_dir(key)/'supervised' if split=='validation' else data_dir(key)
    if key=='imu2img_frozen':return data_dir(key)/'validation/epoch_3' if split=='validation' else data_dir(key)
    return ROOT/'dataset/vlm_imu2img_qwen_lora_joint'/('validation/epoch_'+str(epoch) if split=='validation' else 'test')


def render(key,epoch,split,device):
    assert key=='imu2img_joint'
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8);torch.cuda.set_device(device)
    w,x,y,cs=read_data();converter=build_converter()
    checkpoint=model_dir(key)/'checkpoints'/f'epoch_{epoch}'/'converter.safetensors'
    converter.load_state_dict(load_file(str(checkpoint)));converter.to(device).eval()
    folder=image_folder(key,epoch,split)
    folder.parent.mkdir(parents=True,exist_ok=True)
    write_images(converter,x,w.index[w.split.eq(split)].to_numpy(),w,folder,device,autocast=True)
    write(folder/'render_protocol.json',{'key':key,'epoch':epoch,'split':split,'converter_sha256':sha(checkpoint),
          'width':224,'height':224,'format':'lossless WebP','B':128,'source_window_samples':200,'encoder_rms_samples':196,
          'patches':49,'precision':'BF16 converter; round to8bit; official Qwen preprocessing at224² minimum',
          'source_protocol_sha256':sha(data_dir(key)/'render_protocol.json')})


def evaluate(key,epoch,split,device,final=False):
    os.environ['CUDA_VISIBLE_DEVICES']=device.split(':')[1]
    import torch
    import pandas as pd
    from transformers import AutoProcessor
    from vllm import LLM
    import qwen_classify as q
    from eval_imu2img_alignment import evaluate as metrics
    torch.set_num_threads(8)
    target=model_dir(key) if final else model_dir(key)/'validation_merged'
    dest=out(key) if split=='test' else out(key)/'validation'/f'epoch_{epoch}'
    dest.mkdir(parents=True,exist_ok=True)
    w=pd.read_csv(ROOT/'dataset/splits/windows.csv');w=w[w.split.eq(split)]
    folder=image_folder(key,epoch,split);im=pd.read_csv(folder/'images.csv');im=im[im.window_id.isin(w.window_id)]
    assert im.window_id.tolist()==w.window_id.tolist()
    for r in im.itertuples():assert sha(ROOT/r.image_file)==r.image_sha256
    prompt=(out(key)/'prompt.txt').read_text();size=(1280,1152) if key=='sixplot' else (224,224)
    p={'model':model_dir(key).name,'base_revision':json.loads((SUITE/'protocol.json').read_text())['base_revision'],
       'key':key,'split':split,'classes':classes(),'expected_windows':len(w),'epoch':epoch,'inference_backend':'vllm',
       'vllm_version':md.version('vllm'),'qwen_base_weights_frozen':True,'lora_r':4,'lora_alpha':8,
       'converter_updated':key=='imu2img_joint','prompt_sha256':sha(out(key)/'prompt.txt'),
       'image_manifest_sha256':sha(folder/'images.csv'),'image_dir':str(folder.relative_to(ROOT)),
       'image_size':list(size),'visual_tokens':1440 if key=='sixplot' else 49,
       'adapter_sha256':sha(model_dir(key)/'checkpoints'/f'epoch_{epoch}'/'adapter_model.safetensors'),
       'training_protocol_sha256':sha(out(key)/'training_protocol.json'),
       'windows_sha256':sha(ROOT/'dataset/splits/windows.csv'),'split_sha256':sha(ROOT/'dataset/splits/dog_splits.json'),
       'generation':{'temperature':0,'enable_thinking':False,'max_new_tokens':32,'seed':SEED,'guided_decoding':False},
       'engine':{'dtype':'bfloat16','tensor_parallel_size':1,'max_model_len':4096,'max_num_seqs':32,
                 'max_num_batched_tokens':8192,'gpu_memory_utilization':.85,'enable_prefix_caching':False,
                 'min_pixels':65536 if key=='sixplot' else 224**2,'max_pixels':1280*1152,'flashinfer_sampler':False}}
    protocol=dest/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==p
    else:write(protocol,p)
    (dest/'prompt.txt').write_text(prompt)
    raw=dest/'responses.jsonl';done={}
    if raw.exists():
        for s in raw.read_text().splitlines():
            row=json.loads(s);assert row['window_id'] not in done;done[row['window_id']]=row
    assert set(done).issubset(set(w.window_id))
    pending=[r for r in im.to_dict('records') if r['window_id'] not in done]
    if pending:
        began=time.perf_counter();processor=AutoProcessor.from_pretrained(target,local_files_only=True)
        model=LLM(model=str(target),dtype='bfloat16',tensor_parallel_size=1,trust_remote_code=False,seed=SEED,
                  gpu_memory_utilization=.85,max_model_len=4096,max_num_seqs=32,max_num_batched_tokens=8192,
                  limit_mm_per_prompt={'image':1,'video':0},mm_processor_kwargs={'min_pixels':p['engine']['min_pixels'],'max_pixels':1280*1152},
                  generation_config='vllm',enable_prefix_caching=False,disable_log_stats=True)
        load_seconds=time.perf_counter()-began;timings=[]
        with raw.open('a') as stream:
            for start in range(0,len(pending),32):
                group=pending[start:start+32]
                texts,counts,lengths,truncated,pre,gen,wall=q.generate(model,processor,[ROOT/r['image_file'] for r in group],prompt,device,size)
                timings.append({'n':len(group),'preprocess_seconds':pre,'generation_seconds':gen,'request_wall_seconds':wall})
                for i,item in enumerate(group):
                    prediction,status=q.parse(texts[i],classes(),truncated[i])
                    row={k:item[k] for k in ['window_id','image_file','image_sha256']}
                    row.update(prediction=prediction,status=status,response=texts[i],input_tokens=counts[i],generated_tokens=lengths[i],
                               batch_size=len(group),amortized_request_seconds=wall/len(group),inference_backend='vllm')
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');done[row['window_id']]=row
                stream.flush()
                progress={'completed':len(done),'expected':len(w),'remaining':len(w)-len(done),
                          'status_counts':dict(Counter(r['status'] for r in done.values())),'wall_seconds':time.perf_counter()-began}
                write(dest/'progress.json',progress)
                if len(done)%320==0 or len(done)==len(w):print(key+' '+split+' '+json.dumps(progress),flush=True)
        runtime_path=dest/'runtime.json';runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {'sessions':[]}
        runtime['sessions'].append({'load_seconds':load_seconds,'wall_seconds':time.perf_counter()-began,'batch_timings':timings,
             'code_sha256':sha(Path(__file__)),'shared_generation_code_sha256':sha(ROOT/'scripts/qwen_classify.py'),'device':device})
        write(runtime_path,runtime)
    metrics(dest)


def run_process(arguments,log,env=None):
    print('RUN '+log.name,flush=True)
    with log.open('a') as f:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-B',*arguments],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                       env=env,check=True)


def pipeline():
    prepare();environment=os.environ.copy();environment.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
    # CPU rendering runs while the two IMU2IMG experiments use the GPUs.
    plot_log=(SUITE/'prepare_sixplot.log').open('a')
    plot_proc=None
    if not (ROOT/'dataset/vlm/supervised/validation.json').exists():
        plot_proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B','scripts/prepare_sixplot_lora.py'],cwd=ROOT,stdout=plot_log,stderr=subprocess.STDOUT)
    script='scripts/lora_comparison.py'
    for key in ['imu2img_frozen','imu2img_joint','sixplot']:
        if key=='sixplot' and plot_proc is not None:
            assert plot_proc.wait()==0,'Sixplot rendering failed.'
        if not (out(key)/'training_complete.json').exists():
            if not (out(key)/'backward_check.json').exists():run_process([script,'smoke','--key',key],out(key)/'backward.log')
            run_process(['-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','2',script,'train','--key',key],out(key)/'train.log',environment)
        if not (out(key)/'complete.json').exists():
            scores=[]
            for epoch in [1,2,3]:
                dest=out(key)/'validation'/f'epoch_{epoch}'
                if not (dest/'metrics.json').exists():
                    run_process([script,'export','--key',key,'--epoch',str(epoch)],out(key)/f'export_epoch{epoch}.log')
                    if key=='imu2img_joint':run_process([script,'render','--key',key,'--epoch',str(epoch),'--split','validation','--device','cuda:1'],out(key)/f'render_epoch{epoch}.log')
                    run_process([script,'evaluate','--key',key,'--epoch',str(epoch),'--split','validation','--device','cuda:0'],out(key)/f'validation_epoch{epoch}.log')
                m=json.loads((dest/'metrics.json').read_text());scores.append({'epoch':epoch,**{k:m[k] for k in ['macro_f1','balanced_accuracy','accuracy']}})
            best=max(scores,key=lambda x:x['macro_f1']);epoch=best['epoch']
            write(out(key)/'checkpoint_selection.json',{'metric':'Full validation vLLM macro-F1','scores':scores,
                  'best_epoch':epoch,'best_macro_f1':best['macro_f1'],'test_used_for_selection':False,'tie_break':'earliest epoch'})
            run_process([script,'export','--key',key,'--epoch',str(epoch),'--final'],out(key)/'export_final.log')
            if key=='imu2img_joint':run_process([script,'render','--key',key,'--epoch',str(epoch),'--split','test','--device','cuda:1'],out(key)/'render_test.log')
            run_process([script,'evaluate','--key',key,'--epoch',str(epoch),'--split','test','--device','cuda:0','--final'],out(key)/'test.log')
            temp=model_dir(key)/'validation_merged'
            if temp.exists():shutil.rmtree(temp)
            write(out(key)/'complete.json',{'state':'predictions_complete','best_epoch':epoch,'reports_pending':True})
    if plot_proc is not None:assert plot_proc.wait()==0
    plot_log.close()
    write(SUITE/'generation_complete.json',{'state':'complete','experiments':KEYS,'reports_pending':True})
    print('ALL THREE EXPERIMENTS GENERATED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','smoke','train','export','render','evaluate','pipeline'])
    parser.add_argument('--key',choices=KEYS);parser.add_argument('--epoch',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--split',choices=['validation','test'],default='validation');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--final',action='store_true');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='pipeline':pipeline()
    else:
        assert args.key
        if args.command=='smoke':smoke(args.key)
        elif args.command=='train':train(args.key)
        elif args.command=='export':export(args.key,args.epoch,args.final)
        elif args.command=='render':render(args.key,args.epoch,args.split,args.device)
        else:evaluate(args.key,args.epoch,args.split,args.device,args.final)
