"""Shared 27B LoRA model, label masks and differentiable IMU image path."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.venv/lora-packages'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/pet-lora-triton')
from train_qwen35_lora import scope
from train_imu2img import read_data,build_converter
from align_imu2img_qwen import quantize_rgb,pack_qwen_pixels,write_images
from lora_comparison import to_device,frozen_fingerprint

SEED=20261101
KEYS=['qwen35_27b','qwen38_27b']
NAMES={'qwen35_27b':'Qwen3.5-27B','qwen38_27b':'Qwen3.8-27B'}
SUITE=ROOT/'results/lora_27b'
WARM=ROOT/'models/IMU2IMG-Qwen3.5-4B-aligned/converter.safetensors'
PROMPT=ROOT/'dataset/vlm_imu2img_qwen/prompt.txt'
LRS=[1e-4,3e-5,1e-5]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def base_dir(key):return ROOT/'models'/NAMES[key]
def out(key):return ROOT/'results'/(NAMES[key].lower().replace('-27b','_27B')+'_lora_imu2img_joint')
def model_dir(key):return ROOT/'models'/(NAMES[key]+'-lora-imu2img-joint')
def data_dir(key):return ROOT/'dataset'/('vlm_imu2img_'+key+'_lora_joint')
def lr_tag(lr):return format(lr,'.0e').replace('e-0','e-')
def classes():return json.loads((ROOT/'results/protocol.json').read_text())['classes']
def checkpoint(key,stage,tag):
    if stage=='pilot':return model_dir(key)/'lr_search'/('lr_'+lr_tag(float(tag)))
    return model_dir(key)/'checkpoints'/('epoch_'+str(int(tag)))


def load_rows():
    global SIGNALS
    w,SIGNALS,y,cs=read_data()
    selected=w[w.split.isin(['train','validation'])].copy();selected['cache_index']=selected.index
    return selected


class Answers:
    def __init__(self,processor,prompt):
        import torch
        from PIL import Image
        self.processor=processor
        processor.image_processor.size={'shortest_edge':224**2,'longest_edge':1280*1152}
        messages=[{'role':'user','content':[{'type':'image','image':Image.new('RGB',(224,224),(128,128,128))},
                                          {'type':'text','text':prompt}]}]
        e=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',
                                       add_generation_prompt=True,enable_thinking=False)
        self.prefix=e['input_ids'][0].tolist();self.types=e['mm_token_type_ids'][0].tolist()
        self.grid=e['image_grid_thw'][0].tolist();assert self.grid==[1,14,14]
        assert self.prefix.count(processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'))==49
        self.targets={c:processor.tokenizer.encode(json.dumps({'label':c})+processor.tokenizer.eos_token,
                      add_special_tokens=False) for c in classes()}
        self.pad=processor.tokenizer.pad_token_id
        self.check={'prefix_tokens':len(self.prefix),'image_grid_thw':self.grid,'visual_tokens':49,
                    'image_size':[224,224],'target_token_counts':{k:len(v) for k,v in self.targets.items()},
                    'supervision':'Answer JSON + EOS only; image/user/padding masked with -100'}

    def __call__(self,rows):
        import torch
        sequences=[self.prefix+self.targets[r['label']] for r in rows];size=max(map(len,sequences))
        ids=torch.full((len(rows),size),self.pad,dtype=torch.long)
        labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids);types=torch.zeros_like(ids)
        types[:,:len(self.prefix)]=torch.tensor(self.types)
        for i,s in enumerate(sequences):
            ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1
            labels[i,len(self.prefix):len(s)]=torch.tensor(self.targets[rows[i]['label']])
        return {'input_ids':ids,'labels':labels,'attention_mask':mask,'mm_token_type_ids':types,
                'image_grid_thw':torch.tensor([self.grid]*len(rows),dtype=torch.long),
                'signals':torch.stack([SIGNALS[int(r['cache_index'])] for r in rows])}


def make_system(key,mode='joint'):
    import torch
    from torch import nn
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    from safetensors.torch import load_file
    processor=AutoProcessor.from_pretrained(base_dir(key),local_files_only=True)
    qwen=Qwen3_5ForConditionalGeneration.from_pretrained(base_dir(key),local_files_only=True,
             dtype=torch.bfloat16,attn_implementation='sdpa')
    torch.manual_seed(SEED)
    targets=[n for n,m in qwen.named_modules() if isinstance(m,nn.Linear) and scope(n)]
    counts=Counter(scope(n) for n in targets)
    cfg=qwen.config
    expected={'ViT':4*cfg.vision_config.depth,'visual_projection':2,
              'language_attention':sum(4 if t=='full_attention' else 5 for t in cfg.text_config.layer_types)}
    assert counts==expected,counts
    qwen=get_peft_model(qwen,LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,target_modules=targets,
                          bias='none',task_type='CAUSAL_LM',init_lora_weights=True))
    qwen.config.use_cache=False
    qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert all('lora_' in n and scope(n) for n,p in qwen.named_parameters() if p.requires_grad)
    info={'r':4,'alpha':8,'dropout':.05,'scaling':2,'target_modules':targets,'modules_per_scope':dict(counts),
          'lora_trainable_parameters':sum(p.numel() for p in qwen.parameters() if p.requires_grad),
          'gradient_checkpointing':True,'use_reentrant':False,
          'trainable_parameters_by_scope':{s:sum(p.numel() for n,p in qwen.named_parameters() if p.requires_grad and scope(n)==s) for s in counts}}
    class System(nn.Module):
        def __init__(self):
            super().__init__();self.qwen=qwen;self.converter=build_converter()
            self.converter.load_state_dict(load_file(str(WARM)))
            self.converter.requires_grad_(mode=='joint')
            self.answers=Answers(processor,(out(key)/'prompt.txt').read_text());self.debug=False

        def train(self,training=True):
            super().train(training)
            if mode!='joint':self.converter.eval()
            return self

        def forward(self,batch):
            import torch.nn.functional as F
            batch=dict(batch);labels=batch.pop('labels');signals=batch.pop('signals')
            with torch.autocast('cuda',dtype=torch.bfloat16):image=self.converter(signals.float())
            batch['pixel_values']=pack_qwen_pixels(quantize_rgb(image,straight_through=mode=='joint')).to(torch.bfloat16)
            if self.debug and mode=='joint':batch['pixel_values'].retain_grad();self.debug_pixels=batch['pixel_values']
            positions=torch.arange(len(self.answers.prefix)-1,batch['input_ids'].shape[1]-1,device=labels.device)
            target=labels[:,len(self.answers.prefix):]
            result=self.qwen(**batch,use_cache=False,logits_to_keep=positions)
            assert result.logits.shape[:2]==target.shape
            loss=F.cross_entropy(result.logits.float().reshape(-1,result.logits.shape[-1]),target.reshape(-1),
                                 ignore_index=-100,reduction='sum')
            tokens=target.ne(-100).sum()
            return {'loss':loss/tokens,'nll_sum':loss.detach(),'tokens':tokens.detach()}
    model=System();info['converter_trainable_parameters']=sum(p.numel() for p in model.converter.parameters() if p.requires_grad)
    info['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model,processor,info
