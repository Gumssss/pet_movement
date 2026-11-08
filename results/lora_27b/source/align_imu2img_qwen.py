"""Supervise only an IMU-to-image converter through a completely frozen Qwen3.5-4B."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/lora-packages'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/pet-lora-triton')
from train_imu2img import build_converter, read_data

SIMPLE = ROOT / 'dataset/vlm_imu2img_simple'
DATA = ROOT / 'dataset/vlm_imu2img_qwen'
OLD = ROOT / 'dataset/vlm_imu2img'
BASE = ROOT / 'models/Qwen3.5-4B'
WARM = ROOT / 'models/IMU2IMG-pet/converter.safetensors'
MODEL = ROOT / 'models/IMU2IMG-Qwen3.5-4B-aligned'
OUT = ROOT / 'results/imu2img_qwen_alignment'
SEED = 20261101


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def quantize_rgb(value, straight_through=False):
    import torch
    unit = ((value.float() + 1) * .5).clamp(0, 1)
    rounded = torch.round(unit * 255) / 255
    return unit + (rounded - unit).detach() if straight_through else rounded


def pack_qwen_pixels(unit):
    """Differentiable exact 224px Qwen2VL image preprocessing, without PIL or detached tensors."""
    # Official Torchvision processor fuses rescale into mean/std: normalize uint8 with127.5/127.5.
    byte_value = unit * 255
    byte_value = byte_value + (byte_value.round() - byte_value).detach()
    normalized = (byte_value - 127.5) / 127.5
    b, c, h, w = normalized.shape
    assert (c, h, w) == (3, 224, 224)
    frames = normalized.unsqueeze(2).expand(b, 3, 2, 224, 224)
    patches = frames.reshape(b, 3, 2, 7, 2, 16, 7, 2, 16)
    return patches.permute(0, 3, 6, 4, 7, 1, 2, 5, 8).reshape(b * 196, 1536)


class AnswerBatch:
    def __init__(self):
        from transformers import AutoProcessor
        from PIL import Image
        self.processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
        self.processor.image_processor.size = {'shortest_edge':224**2, 'longest_edge':1280*1152}
        self.prompt = (DATA / 'prompt.txt').read_text()
        conversation = [{'role':'user','content':[{'type':'image','image':Image.new('RGB',(224,224),(128,128,128))},
                                                  {'type':'text','text':self.prompt}]}]
        encoded = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True,
            return_tensors='pt', add_generation_prompt=True, enable_thinking=False)
        self.prefix = encoded['input_ids'][0].tolist()
        self.types = encoded['mm_token_type_ids'][0].tolist()
        assert encoded['image_grid_thw'].tolist() == [[1,14,14]]
        image_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        assert self.prefix.count(image_id) == 49
        self.classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
        self.targets = [self.processor.tokenizer.encode(json.dumps({'label':s}) + self.processor.tokenizer.eos_token,
                        add_special_tokens=False) for s in self.classes]
        self.pad = self.processor.tokenizer.pad_token_id

    def __call__(self, class_ids, device):
        import torch
        class_ids = [int(i) for i in class_ids]
        sequences = [self.prefix + self.targets[i] for i in class_ids]
        size = max(map(len,sequences)); b = len(sequences)
        ids = torch.full((b,size),self.pad,dtype=torch.long)
        mask = torch.zeros_like(ids); types = torch.zeros_like(ids); labels = torch.full_like(ids,-100)
        types[:,:len(self.prefix)] = torch.tensor(self.types)
        for row,seq in enumerate(sequences):
            ids[row,:len(seq)] = torch.tensor(seq); mask[row,:len(seq)] = 1
            labels[row,len(self.prefix):len(seq)] = torch.tensor(self.targets[class_ids[row]])
        assert labels[:,:len(self.prefix)].eq(-100).all()
        return {k:v.to(device) for k,v in {'input_ids':ids,'attention_mask':mask,'mm_token_type_ids':types,
                'labels':labels,'image_grid_thw':torch.tensor([[1,14,14]]*b,dtype=torch.long)}.items()}


def build_system(device, debug=False):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import Qwen3_5ForConditionalGeneration
    from safetensors.torch import load_file
    torch.manual_seed(SEED)
    class System(nn.Module):
        def __init__(self):
            super().__init__()
            self.converter = build_converter()
            self.converter.load_state_dict(load_file(str(WARM)))
            self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa')
            self.qwen.requires_grad_(False); self.qwen.eval(); self.qwen.config.use_cache=False
            self.answers = AnswerBatch(); self.debug=debug

        def train(self, mode=True):
            super().train(mode); self.qwen.eval(); return self

        def forward(self, signals, class_ids):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value = self.converter(signals.float())
            unit = quantize_rgb(value, straight_through=True)
            pixels = pack_qwen_pixels(unit).to(torch.bfloat16)
            if self.debug:
                pixels.retain_grad(); self.debug_pixels=pixels
            batch = self.answers(class_ids, signals.device)
            labels = batch.pop('labels')
            positions = torch.arange(len(self.answers.prefix)-1,batch['input_ids'].shape[1]-1,device=signals.device)
            target = labels[:,len(self.answers.prefix):]
            output = self.qwen(**batch,pixel_values=pixels,use_cache=False,logits_to_keep=positions)
            assert output.logits.shape[:2] == target.shape
            total = F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),target.reshape(-1),
                                     ignore_index=-100,reduction='sum')
            n=target.ne(-100).sum()
            return {'loss':total/n,'nll_sum':total.detach(),'n_tokens':n.detach()}
    model=System().to(device)
    assert sum(p.numel() for p in model.converter.parameters()) == 835584
    assert all(p.requires_grad for p in model.converter.parameters())
    assert all(not p.requires_grad for p in model.qwen.parameters())
    return model


def parameter_fingerprint(model):
    import torch
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode());digest.update(str(p.dtype).encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_images(converter, x, indices, windows, folder, device, autocast=False):
    import torch
    import numpy as np
    from PIL import Image
    from contextlib import nullcontext
    folder.mkdir(parents=True,exist_ok=True);(folder/'images').mkdir(exist_ok=True)
    fields=['window_id','image_file','image_sha256','render_seconds','b_channel_unique']
    with (folder/'images.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for start in range(0,len(indices),128):
            current=indices[start:start+128];begin=time.perf_counter()
            context=torch.autocast('cuda',dtype=torch.bfloat16) if autocast else nullcontext()
            with torch.inference_mode(),context:
                unit=quantize_rgb(converter(x[current].to(device)))
                rgb=torch.round(unit*255).byte().permute(0,2,3,1).cpu().numpy()
            generated=time.perf_counter()-begin
            for index,pixels in zip(current,rgb):
                wid=windows.window_id.iloc[int(index)]
                path=folder/'images'/('sample_'+hashlib.sha256(wid.encode()).hexdigest()[:20]+'.webp')
                begin=time.perf_counter();Image.fromarray(pixels).save(path,format='WEBP',lossless=True,method=4)
                assert np.array_equal(np.unique(pixels[:,:,2]),[128])
                writer.writerow({'window_id':wid,'image_file':str(path.relative_to(ROOT)),
                    'image_sha256':sha(path),'render_seconds':generated/len(current)+time.perf_counter()-begin,'b_channel_unique':128})
            f.flush()
            print(f'Images {folder.relative_to(ROOT)}: {min(start+128,len(indices))}/{len(indices)}',flush=True)


def prepare(args):
    import torch
    import pandas as pd
    from safetensors.torch import load_file
    from PIL import Image
    torch.set_num_threads(8);torch.cuda.set_device(args.device)
    for d in [SIMPLE,DATA,MODEL,OUT]:d.mkdir(parents=True,exist_ok=True)
    assert (SIMPLE/'prompt.txt').read_bytes()==(DATA/'prompt.txt').read_bytes()
    if not (SIMPLE/'images').exists():(SIMPLE/'images').symlink_to('../vlm_imu2img/images',target_is_directory=True)
    images=pd.read_csv(OLD/'images.csv')
    images['image_file']=images.image_file.map(lambda p:str((SIMPLE/'images'/Path(p).name).relative_to(ROOT)))
    images.to_csv(SIMPLE/'images.csv',index=False)
    (SIMPLE/'preview.webp').write_bytes((OLD/'preview.webp').read_bytes())
    protocol=json.loads((OLD/'render_protocol.json').read_text())
    protocol.update(version='imu2img_original_pixels_simplified_prompt',prompt_change='Removed waveform feature priors; metadata + task + exact labels only')
    write(SIMPLE/'render_protocol.json',protocol)
    windows,x,y,classes=read_data()
    converter=build_converter();converter.load_state_dict(load_file(str(WARM)));converter.to(args.device).eval()
    indices=windows.index[windows.split.eq('validation')].to_numpy()
    write_images(converter,x,indices,windows,SIMPLE/'validation',args.device,autocast=True)
    # Actual old pixels are reused for test; validation is exported using the old BF16 rendering path.
    info={'validation_windows':4727,'test_windows':3708,'warm_converter_sha256':sha(WARM),
          'simple_prompt_sha256':sha(SIMPLE/'prompt.txt'),'test_pixels_reused_unchanged':True,
          'validation_manifest_sha256':sha(SIMPLE/'validation/images.csv')}
    write(SIMPLE/'preparation.json',info)


def training_protocol(system,world):
    import importlib.metadata as md
    return {'experiment':'frozen_qwen_answer_supervised_imu2img_alignment','base_qwen':'Qwen/Qwen3.5-4B',
        'base_revision':json.loads((ROOT/'models/model_manifest.json').read_text())['Qwen3.5-4B']['revision'],
        'qwen_trainable_parameters':0,'converter_trainable_parameters':835584,'qwen_mode':'eval; all weights frozen; autograd through activations enabled',
        'warm_start_converter_sha256':sha(WARM),'warm_start_source':'Original ViT-supervised IMU2IMG best epoch7',
        'prompt_sha256':sha(DATA/'prompt.txt'),'epochs':3,'seed':SEED,'learning_rate':.0001,
        'optimizer':'AdamW','betas':[.9,.999],'epsilon':1e-8,'weight_decay':.01,
        'scheduler':'linear','warmup_ratio':.05,'max_gradient_norm':1.,
        'world_size':world,'microbatch_per_gpu':16,'gradient_accumulation':1 if world==2 else 2,'effective_batch_size':32,
        'ddp_sampler_padding_per_epoch':1 if world==2 else 0,
        'precision':'FP32 converter weights/optimizer with BF16 converter forward, matching original rendering; frozen BF16 Qwen; FP32 quantization/normalization bridge',
        'train_windows':16461,'validation_windows':4727,'test_windows':3708,
        'loss':'Only answer JSON + EOS token cross entropy; all image/user/nonthinking-prefix/padding labels masked',
        'checkpoint_selection':'Full validation greedy vLLM macro-F1 over all3 epoch converters; earliest on ties; test excluded',
        'validation_answer_nll':'Monitored at each epoch; not the final checkpoint selection metric',
        'image_path':'RMS and crop unchanged; FP32 converter -> 8-bit round straight-through -> RGB normalization -> differentiable Qwen patchification',
        'export':'Same 8-bit rounding + lossless WebP; 224x224, B128; Qwen min_pixels224²; no PIL/detach in training path',
        'processor':{'patch_size':16,'temporal_patch_size':2,'merge_size':2,'grid':[1,14,14],
                     'visual_tokens':49,'prefix_tokens':len(system.answers.prefix),'mean':[.5,.5,.5],'std':[.5,.5,.5]},
        'data_source':'Same original2s/200source-sample windows; encoder RMS196points,49x4patches',
        'windows_sha256':sha(ROOT/'dataset/splits/windows.csv'),'split_sha256':sha(ROOT/'dataset/splits/dog_splits.json'),
        'versions':{p:md.version(p) for p in ['torch','transformers','flash-linear-attention','fla-core']}}


def smoke(args):
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(8);torch.cuda.set_device(args.device)
    OUT.mkdir(exist_ok=True);MODEL.mkdir(exist_ok=True)
    windows,x,y,classes=read_data();indices=windows.index[windows.split.eq('train')][:16].to_numpy()
    model=build_system(args.device,debug=True);model.train()
    before=parameter_fingerprint(model.qwen)
    # Forward values of the STE bridge must equal actual exported uint8 RGB plus the official processor.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        rgb=quantize_rgb(model.converter(x[indices[:1]].to(args.device)))
        pixels=torch.round(rgb*255).byte().permute(0,2,3,1).cpu().numpy()[0]
        reference=model.answers.processor.image_processor(images=[Image.fromarray(pixels)],return_tensors='pt')['pixel_values']
        packed=pack_qwen_pixels(rgb).cpu()
        error=float((packed-reference).abs().max());assert error<1e-6,error
        bf16_equal=torch.equal(packed.to(torch.bfloat16),reference.to(torch.bfloat16));assert bf16_equal
    began=time.perf_counter();output=model(x[indices].to(args.device),y[indices].tolist());output['loss'].backward();torch.cuda.synchronize()
    grads={s:sum(float(p.grad.float().square().sum()) for n,p in model.converter.named_parameters()
                 if n.startswith(s) and p.grad is not None)**.5 for s in ['encoders','decoders']}
    assert all(np.isfinite(v) and v>0 for v in grads.values())
    assert model.debug_pixels.grad is not None and float(model.debug_pixels.grad.float().abs().sum())>0
    assert all(p.grad is None and not p.requires_grad for p in model.qwen.parameters())
    after=parameter_fingerprint(model.qwen);assert before==after
    report={'purpose':'16 training-window backward check only, no optimizer update',
            'loss':float(output['loss'].detach()),'gradient_norms':grads,'pixel_gradient_l1':float(model.debug_pixels.grad.float().abs().sum()),
            'qwen_all_frozen_and_no_gradients':True,'qwen_in_memory_sha256_before':before,'qwen_in_memory_sha256_after':after,
            'official_processor_max_abs_error':error,'official_processor_bf16_patches_bitwise_equal':bf16_equal,
            'seconds':time.perf_counter()-began,'peak_allocated_GiB':torch.cuda.max_memory_allocated(args.device)/2**30,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write(OUT/'backward_check.json',report);write(OUT/'processor_check.json',training_protocol(model,2)['processor'])
    print(json.dumps(report,indent=2),flush=True)


def train(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader,DistributedSampler
    from transformers import get_linear_schedule_with_warmup
    from safetensors.torch import save_file,load_file
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    assert world in [1,2];torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(8)
    if world>1:dist.init_process_group('nccl')
    for p in [MODEL,OUT]:p.mkdir(exist_ok=True)
    windows,x,y,classes=read_data();model=build_system(device);config=training_protocol(model,world)
    if rank==0:
        path=OUT/'training_protocol.json'
        if path.exists():assert json.loads(path.read_text())==config,'Frozen training protocol changed.'
        else:write(path,config)
    indices=windows.index[windows.split.eq('train')].tolist();sampler=DistributedSampler(indices,num_replicas=world,rank=rank,shuffle=True,seed=SEED,drop_last=False)
    loader=DataLoader(indices,batch_size=16,sampler=sampler,num_workers=0)
    val_indices=windows.index[windows.split.eq('validation')].to_numpy()[rank::world]
    optimizer=torch.optim.AdamW(model.converter.parameters(),lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    assert {id(p) for group in optimizer.param_groups for p in group['params']}=={id(p) for p in model.converter.parameters()}
    accum=config['gradient_accumulation'];epoch_steps=math.ceil(len(loader)/accum);total_steps=epoch_steps*3
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(.05*total_steps),total_steps)
    start_epoch=0;step=0;state_path=MODEL/'last_state.pt'
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False);model.converter.load_state_dict(state['converter'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start_epoch=state['epoch'];step=state['step']
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    initial_hash=parameter_fingerprint(model.qwen) if rank==0 else None
    versions={n:p._version for n,p in model.qwen.named_parameters()}
    x=x.to(device);begin=time.perf_counter();initial_step=step
    for epoch in range(start_epoch,3):
        sampler.set_epoch(epoch);wrapped.train();optimizer.zero_grad(set_to_none=True);nll=torch.zeros((),device=device,dtype=torch.float64);tokens=torch.zeros_like(nll)
        epoch_begin=time.perf_counter()
        for index,idx in enumerate(loader):
            group_size=min(accum,len(loader)-(index//accum*accum));update=(index+1)%accum==0 or index+1==len(loader)
            from contextlib import nullcontext
            context=wrapped.no_sync() if world>1 and not update else nullcontext()
            with context:
                output=wrapped(x[idx.to(device)],y[idx].tolist());loss=output['loss'];assert torch.isfinite(loss)
                (loss/group_size).backward()
            nll+=output['nll_sum'].double();tokens+=output['n_tokens'].double()
            assert all(p.grad is None and not p.requires_grad for p in model.qwen.parameters())
            if update:
                norm=torch.nn.utils.clip_grad_norm_(model.converter.parameters(),1.);assert torch.isfinite(norm)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                if step%10==0 or index+1==len(loader):
                    stat=torch.stack([nll,tokens])
                    if world>1:dist.all_reduce(stat)
                    if rank==0:
                        row={'epoch':epoch+1,'optimizer_step':step,'total_optimizer_steps':total_steps,
                             'train_answer_nll_so_far':float(stat[0]/stat[1]),'learning_rate':scheduler.get_last_lr()[0],
                             'wall_seconds':time.perf_counter()-begin,'cuda_peak_allocated_GiB_rank0':torch.cuda.max_memory_allocated(device)/2**30}
                        write(OUT/'training_progress.json',row)
                        with (OUT/'training_history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        print('TRAIN '+json.dumps(row),flush=True)
        model.eval();vnll=torch.zeros((),device=device,dtype=torch.float64);vtokens=torch.zeros_like(vnll)
        with torch.inference_mode():
            for offset in range(0,len(val_indices),16):
                idx=torch.as_tensor(val_indices[offset:offset+16]);output=model(x[idx.to(device)],y[idx].tolist())
                vnll+=output['nll_sum'].double();vtokens+=output['n_tokens'].double()
        stat=torch.stack([vnll,vtokens])
        if world>1:dist.all_reduce(stat)
        assert all(p._version==versions[n] for n,p in model.qwen.named_parameters()),'Frozen Qwen parameter version changed.'
        if rank==0:
            directory=MODEL/'checkpoints'/f'epoch_{epoch+1}';directory.mkdir(parents=True,exist_ok=True)
            save_file({k:v.detach().cpu().contiguous() for k,v in model.converter.state_dict().items()},str(directory/'converter.safetensors'))
            state={'converter':{k:v.detach().cpu() for k,v in model.converter.state_dict().items()},
                   'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'step':step}
            temp=MODEL/'last_state.tmp';torch.save(state,temp);temp.replace(state_path)
            record={'epoch':epoch+1,'optimizer_step':step,'validation_answer_nll':float(stat[0]/stat[1]),
                    'validation_tokens':int(stat[1]),'epoch_wall_seconds':time.perf_counter()-epoch_begin,
                    'checkpoint_sha256':sha(directory/'converter.safetensors')}
            with (OUT/'validation_nll_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('VALIDATION_NLL '+json.dumps(record),flush=True)
        if world>1:dist.barrier()
    if rank==0:
        final_hash=parameter_fingerprint(model.qwen);assert final_hash==initial_hash,'Frozen Qwen weights changed.'
        write(OUT/'training_complete.json',{'epochs':3,'optimizer_steps':step,'trainable_parameters':835584,
              'qwen_trainable_parameters':0,'qwen_in_memory_sha256_before':initial_hash,'qwen_in_memory_sha256_after':final_hash,
              'qwen_parameter_versions_unchanged':True,'session_wall_seconds':time.perf_counter()-begin,
              'session_optimizer_steps':step-initial_step,'code_sha256':sha(Path(__file__)),
              'checkpoint_selection_pending':'Full validation vLLM macro-F1 over3 checkpoints; no test used'})
    if world>1:dist.destroy_process_group()


def render(args):
    import torch
    from safetensors.torch import load_file
    from PIL import Image
    torch.set_num_threads(8);torch.cuda.set_device(args.device)
    windows,x,y,classes=read_data()
    checkpoint=MODEL/'checkpoints'/f'epoch_{args.epoch}'/'converter.safetensors'
    converter=build_converter();converter.load_state_dict(load_file(str(checkpoint)));converter.to(args.device).eval()
    if args.split=='validation':folder=DATA/'validation'/f'epoch_{args.epoch}'
    else:folder=DATA
    indices=windows.index[windows.split.eq(args.split)].to_numpy()
    write_images(converter,x,indices,windows,folder,args.device,autocast=True)
    protocol={'version':'imu2img_frozen_qwen_aligned_rgb_v1','width':224,'height':224,'image_format':'WebP lossless RGB',
        'source_window_samples':200,'encoder_rms_samples':196,'patches':49,'tile_size':32,'blue_channel':128,
        'converter_epoch':args.epoch,'converter_weights_sha256':sha(checkpoint),
        'training_protocol_sha256':sha(OUT/'training_protocol.json'),'prompt_sha256':sha(DATA/'prompt.txt'),
        'window_manifest_sha256':sha(ROOT/'dataset/splits/windows.csv'),'split_sha256':sha(ROOT/'dataset/splits/dog_splits.json'),
        'precision':'BF16 converter forward with FP32 parameters; identical 8-bit quantization to training STE forward and original rendering',
        'qwen_finetuned':False,'supervision':'Converter trained on answer labels through frozen Qwen; not a zero-label system'}
    write(folder/'render_protocol.json',protocol)
    if args.split=='test':
        first=int(windows.index[windows.split.eq('train')][0])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):rgb=torch.round(quantize_rgb(converter(x[first:first+1].to(args.device)))*255).byte()[0].permute(1,2,0).cpu().numpy()
        Image.fromarray(rgb).save(DATA/'preview.webp',format='WEBP',lossless=True,method=4)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','smoke','train','render'])
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args();{'prepare':prepare,'smoke':smoke,'train':train,'render':render}[args.command](args)
