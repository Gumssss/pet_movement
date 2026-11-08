"""Native six-channel images and fresh 27B LoRA, isolated to one physical GPU."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.venv/wandb-packages'))
sys.path.insert(0,str(ROOT/'.venv/lora-packages'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/pet-lora-triton')
from train_qwen35_lora import scope
from lora_comparison import to_device,frozen_fingerprint
SEED=20261101
GPU_UUID='GPU-0ded27d3-e266-f20b-2476-f2980be2980e'
KEYS=['qwen38_27b'];NAMES={'qwen38_27b':'Qwen3.8-27B'}
SUITE=ROOT/'results/qwen3.8_27B_lora_sixplot_single_gpu'
PROMPT=ROOT/'dataset/vlm/prompt.txt';DATA=ROOT/'dataset/vlm'
LRS=[1e-4,3e-5,1e-5]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def out(key='qwen38_27b'):assert key in KEYS;return SUITE
def base_dir(key='qwen38_27b'):assert key in KEYS;return ROOT/'models/Qwen3.8-27B'
def model_dir(key='qwen38_27b'):assert key in KEYS;return ROOT/'models/Qwen3.8-27B-lora-sixplot-single-gpu'
def classes():return json.loads((ROOT/'results/protocol.json').read_text())['classes']
def lr_tag(lr):return format(lr,'.0e').replace('e-0','e-')
def checkpoint(key,stage,tag):
    return model_dir(key)/('lr_search/lr_'+lr_tag(float(tag)) if stage=='pilot' else 'checkpoints/epoch_'+str(int(tag)))
def manifest(split):return DATA/('images.csv' if split=='test' else 'supervised/images.csv')


def load_rows(split=None):
    import pandas as pd
    w=pd.read_csv(ROOT/'dataset/splits/windows.csv')
    requested=['train','validation'] if split is None else [split]
    if requested==['test']:im=pd.read_csv(manifest('test'))
    else:im=pd.read_csv(manifest('validation'))
    frame=w[w.split.isin(requested)].merge(im[['window_id','image_file','image_sha256']],on='window_id',validate='one_to_one',how='left',sort=False)
    assert frame.image_file.notna().all()
    expected={'train':16461,'validation':4727,'test':3708}
    assert frame.split.value_counts().to_dict()=={s:expected[s] for s in requested}
    assert frame.window_id.tolist()==w[w.split.isin(requested)].window_id.tolist()
    return frame


class Answers:
    def __init__(self,processor):
        from PIL import Image
        self.processor=processor
        processor.image_processor.size={'shortest_edge':65536,'longest_edge':1280*1152}
        image=Image.new('RGB',(1280,1152),(128,128,128))
        prompt=(SUITE/'prompt.txt').read_text();assert prompt==PROMPT.read_text()
        e=processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}],
              tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
        self.prefix=e['input_ids'][0].tolist();self.types=e['mm_token_type_ids'][0].tolist();self.grid=e['image_grid_thw'][0].tolist()
        assert self.grid==[1,72,80] and self.prefix.count(processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'))==1440
        self.targets={label:processor.tokenizer.encode(json.dumps({'label':label})+processor.tokenizer.eos_token,add_special_tokens=False) for label in classes()}
        self.pad=processor.tokenizer.pad_token_id
        self.check={'prefix_tokens':len(self.prefix),'image_grid_thw':self.grid,'visual_tokens':1440,'image_size':[1280,1152],
                    'min_pixels':65536,'max_pixels':1280*1152,'prompt_sha256':sha(SUITE/'prompt.txt'),
                    'target_token_counts':{k:len(v) for k,v in self.targets.items()},
                    'supervision':'Only assistant JSON answer + EOS; input, image and padding labels -100'}

    def __call__(self,rows):
        import torch
        from PIL import Image
        sequences=[self.prefix+self.targets[r['label']] for r in rows];size=max(map(len,sequences));n=len(rows)
        ids=torch.full((n,size),self.pad,dtype=torch.long);labels=torch.full_like(ids,-100)
        mask=torch.zeros_like(ids);types=torch.zeros_like(ids);types[:,:len(self.prefix)]=torch.tensor(self.types)
        images=[]
        for i,(r,s) in enumerate(zip(rows,sequences)):
            ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1;labels[i,len(self.prefix):len(s)]=torch.tensor(self.targets[r['label']])
            with Image.open(ROOT/r['image_file']) as im:
                assert im.size==(1280,1152);copy=im.convert('RGB').copy();copy.info.clear();images.append(copy)
        vision=self.processor.image_processor(images=images,return_tensors='pt')
        assert all(g.tolist()==self.grid for g in vision['image_grid_thw'])
        assert labels[:,:len(self.prefix)].eq(-100).all()
        return {**vision,'input_ids':ids,'labels':labels,'attention_mask':mask,'mm_token_type_ids':types}


def make_system(key='qwen38_27b',mode=None):
    import torch
    from torch import nn
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    processor=AutoProcessor.from_pretrained(base_dir(key),local_files_only=True)
    qwen=Qwen3_5ForConditionalGeneration.from_pretrained(base_dir(key),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    torch.manual_seed(SEED)
    targets=[name for name,module in qwen.named_modules() if isinstance(module,nn.Linear) and scope(name)]
    counts=Counter(scope(n) for n in targets);assert counts=={'ViT':108,'visual_projection':2,'language_attention':304}
    qwen=get_peft_model(qwen,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM',init_lora_weights=True))
    qwen.config.use_cache=False;qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert all('lora_' in name and scope(name) for name,p in qwen.named_parameters() if p.requires_grad)
    info={'r':4,'alpha':8,'dropout':.05,'scaling':2,'target_modules':targets,'modules_per_scope':dict(counts),
          'lora_trainable_parameters':sum(p.numel() for p in qwen.parameters() if p.requires_grad),
          'gradient_checkpointing':True,'use_reentrant':False,
          'trainable_parameters_by_scope':{s:sum(p.numel() for name,p in qwen.named_parameters() if p.requires_grad and scope(name)==s) for s in counts}}
    info['trainable_parameters']=info['lora_trainable_parameters'];assert info['trainable_parameters']==13881216
    class System(nn.Module):
        def __init__(self):super().__init__();self.qwen=qwen;self.answers=Answers(processor)
        def forward(self,batch):
            import torch.nn.functional as F
            batch=dict(batch);labels=batch.pop('labels')
            positions=torch.arange(len(self.answers.prefix)-1,batch['input_ids'].shape[1]-1,device=labels.device)
            target=labels[:,len(self.answers.prefix):]
            output=self.qwen(**batch,use_cache=False,logits_to_keep=positions);assert output.logits.shape[:2]==target.shape
            total=F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='sum')
            n=target.ne(-100).sum()
            return {'loss':total/n,'nll_sum':total.detach(),'tokens':n.detach()}
    return System(),processor,info
