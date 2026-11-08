"""Train r=4/alpha=8 LoRA on the vision tower, merger and hybrid attention."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/lora-packages'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/pet-lora-triton')
DATA = ROOT / 'dataset/vlm_acl'
BASE = ROOT / 'models/Qwen3.5-4B'
MODEL = ROOT / 'models/Qwen3.5-4B-lora'
OUT = ROOT / 'results/qwen3.5_4B_lora_acl'
SEED = 20261101


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def scope(name):
    if '.visual.blocks.' in name:
        return 'ViT'
    if '.visual.merger.' in name:
        return 'visual_projection'
    if '.language_model.layers.' in name and ('.self_attn.' in name or '.linear_attn.' in name):
        return 'language_attention'
    return None


def metadata():
    import pandas as pd
    w = pd.read_csv(ROOT / 'dataset/splits/windows.csv')
    im = pd.read_csv(DATA / 'supervised/images.csv')
    assert len(im) == im.window_id.nunique() == 21188
    x = w[w.split.isin(['train', 'validation'])].merge(im[['window_id', 'image_file', 'image_sha256']], on='window_id', validate='one_to_one')
    assert x.split.value_counts().to_dict() == {'train': 16461, 'validation': 4727}
    assert not set(x.dog_id) & set(w[w.split.eq('test')].dog_id)
    return x


def make_model():
    import torch
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, local_files_only=True,
                 dtype=torch.bfloat16, attn_implementation='sdpa')
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and scope(n)]
    counts = Counter(scope(n) for n in targets)
    assert counts == {'ViT': 96, 'visual_projection': 2, 'language_attention': 152}, counts
    config = LoraConfig(r=4, lora_alpha=8, target_modules=targets, lora_dropout=.05,
                        bias='none', task_type='CAUSAL_LM', init_lora_weights=True)
    model = get_peft_model(model, config)
    model.config.use_cache = False
    active = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    assert active and all('lora_' in n and scope(n) for n, _ in active)
    info = {'r': 4, 'alpha': 8, 'scaling': 2, 'target_modules': targets,
            'modules_per_scope': dict(counts), 'trainable_parameters': sum(n for _, n in active),
            'total_parameters_including_adapters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters_by_scope': {s: sum(k for n, k in active if scope(n) == s) for s in counts},
            'trainable_parameter_names': [n for n, _ in active],
            'frozen_note': 'Only LoRA A/B matrices are trainable. Patch convolution, norms, embeddings, LM feed-forward layers and base projection weights stay frozen.'}
    return model, processor, info


class ImageDataset:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class Collator:
    def __init__(self, processor):
        import torch
        from PIL import Image
        self.processor = processor
        self.prompt = (DATA / 'prompt.txt').read_text()
        sample = Image.open(DATA / 'preview.webp').convert('RGB')
        conversation = [{'role': 'user', 'content': [{'type': 'image', 'image': sample}, {'type': 'text', 'text': self.prompt}]}]
        encoded = processor.apply_chat_template(conversation, tokenize=True, return_dict=True,
                    return_tensors='pt', add_generation_prompt=True, enable_thinking=False)
        self.prefix = encoded['input_ids'][0].tolist()
        self.prefix_types = encoded['mm_token_type_ids'][0].tolist()
        assert len(self.prefix_types) == len(self.prefix)
        self.grid = encoded['image_grid_thw'][0].tolist()
        assert self.grid == [1, 26, 32]
        image_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        assert self.prefix.count(image_id) == 208
        self.targets = {}
        classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
        for cls in classes:
            text = json.dumps({'label': cls}) + processor.tokenizer.eos_token
            self.targets[cls] = processor.tokenizer.encode(text, add_special_tokens=False)
        self.pad = processor.tokenizer.pad_token_id
        self.check = {'input_tokens': len(self.prefix), 'image_grid_thw': self.grid, 'visual_tokens': 208,
                      'export_size': [512, 409], 'processed_size': [512, 416],
                      'supervision': 'Only JSON answer tokens plus assistant end-of-message EOS. Entire image/user/nonthinking prefix and padding masked with -100.',
                      'target_token_counts': {k: len(v) for k, v in self.targets.items()}}

    def __call__(self, rows):
        import torch
        from PIL import Image
        images = []
        for r in rows:
            with Image.open(ROOT / r['image_file']) as im:
                assert im.size == (512, 409)
                images.append(im.convert('RGB').copy())
        vision = self.processor.image_processor(images=images, return_tensors='pt')
        assert all(g.tolist() == self.grid for g in vision['image_grid_thw'])
        seqs = [self.prefix + self.targets[r['label']] for r in rows]
        length = max(map(len, seqs))
        ids = torch.full((len(rows), length), self.pad, dtype=torch.long)
        labels = torch.full_like(ids, -100); mask = torch.zeros_like(ids)
        mm_types = torch.zeros_like(ids)
        mm_types[:, :len(self.prefix)] = torch.tensor(self.prefix_types)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
            labels[i, len(self.prefix):len(seq)] = torch.tensor(self.targets[rows[i]['label']])
        assert (labels[:, :len(self.prefix)] == -100).all()
        return {**vision, 'input_ids': ids, 'attention_mask': mask, 'mm_token_type_ids': mm_types, 'labels': labels}


def answer_loss(model, batch, prefix_length, device):
    import torch
    import torch.nn.functional as F
    labels = batch.pop('labels').to(device)
    inputs = {k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in batch.items()}
    start = prefix_length - 1
    positions = torch.arange(start, inputs['input_ids'].shape[1] - 1, device=device)
    target = labels[:, start + 1:]
    output = model(**inputs, use_cache=False, logits_to_keep=positions)
    assert output.logits.shape[:2] == target.shape
    total = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                            target.reshape(-1), ignore_index=-100, reduction='sum')
    n = target.ne(-100).sum()
    return total, n


def protocol(info, collator, world_size):
    import importlib.metadata as md
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    return {'model': 'Qwen3.5-4B-lora', 'base_model': 'Qwen/Qwen3.5-4B',
            'base_revision': json.loads((ROOT / 'models/model_manifest.json').read_text())['Qwen3.5-4B']['revision'],
            'lora': {k: info[k] for k in ['r', 'alpha', 'scaling', 'modules_per_scope', 'trainable_parameters', 'trainable_parameters_by_scope']},
            'classes': classes, 'seed': SEED, 'epochs': 3, 'learning_rate': .0002,
            'optimizer': 'AdamW', 'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': .01,
            'scheduler': 'linear', 'warmup_ratio': .05, 'max_grad_norm': 1., 'lora_dropout': .05,
            'dtype': 'bfloat16 base, float32 LoRA/optimizer', 'gradient_checkpointing': False,
            'microbatch_per_gpu': 8, 'world_size': world_size, 'gradient_accumulation': 2 if world_size == 2 else 4,
            'effective_batch_size': 32, 'checkpoint_selection': 'Lowest full validation answer-token negative log likelihood across 3 epochs; test never used for selection.',
            'train_windows': 16461, 'validation_windows': 4727, 'test_windows': 3708,
            'training_labels': 'All original training windows; unweighted assistant-only token cross entropy; no class resampling/augmentation.',
            'ddp_sampler_padding_per_epoch': 1 if world_size == 2 else 0,
            'data_dir': 'dataset/vlm_acl', 'supervised_manifest_sha256': sha(DATA / 'supervised/images.csv'),
            'prompt_sha256': sha(DATA / 'prompt.txt'), 'windows_sha256': sha(ROOT / 'dataset/splits/windows.csv'),
            'split_sha256': sha(ROOT / 'dataset/splits/dog_splits.json'),
            'render_protocol_sha256': sha(DATA / 'render_protocol.json'),
            'normalization_sha256': sha(DATA / 'normalization.json'),
            'processor': collator.check, 'trainer': 'Transformers + PEFT; vLLM for all generated validation/test predictions',
            'versions': {p: md.version(p) for p in ['torch', 'transformers', 'peft', 'flash-linear-attention', 'fla-core']}}


def smoke():
    import torch
    import numpy as np
    torch.set_num_threads(8)
    OUT.mkdir(exist_ok=True); MODEL.mkdir(exist_ok=True)
    rows = metadata().query("split == 'train'").head(8).to_dict('records')
    model, processor, info = make_model(); collator = Collator(processor)
    write(OUT / 'target_modules.json', info); write(OUT / 'processor_check.json', collator.check)
    model.to('cuda:0').train()
    start = time.perf_counter()
    total, n = answer_loss(model, collator(rows), len(collator.prefix), 'cuda:0')
    (total / n).backward(); torch.cuda.synchronize()
    grads = {s: sum(float(p.grad.float().square().sum()) for name, p in model.named_parameters()
                   if p.requires_grad and scope(name) == s and p.grad is not None) ** .5
             for s in info['modules_per_scope']}
    assert all(np.isfinite(v) and v > 0 for v in grads.values()), grads
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    result = {'purpose': 'Training-only forward/backward technical check; no optimizer update or validation/test use.',
              'loss': float(total.detach() / n), 'answer_tokens': int(n), 'gradient_norms': grads,
              'seconds': time.perf_counter() - start, 'cuda_allocated_peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
              'cuda_reserved_peak_GiB': torch.cuda.max_memory_reserved() / 2**30, 'batch_size': 8}
    write(OUT / 'backward_check.json', result); print(json.dumps(result, indent=2), flush=True)


def train():
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import get_linear_schedule_with_warmup
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1)); local = int(os.environ.get('LOCAL_RANK', 0))
    assert world in [1, 2]
    torch.cuda.set_device(local); device = torch.device('cuda', local); torch.set_num_threads(8)
    if world > 1: dist.init_process_group('nccl')
    OUT.mkdir(exist_ok=True); MODEL.mkdir(exist_ok=True)
    data = metadata()
    model, processor, info = make_model(); collator = Collator(processor)
    print(f'rank {rank}: model and supervised inputs ready', flush=True)
    config = protocol(info, collator, world)
    if rank == 0:
        path = OUT / 'training_protocol.json'
        if path.exists(): assert json.loads(path.read_text()) == config, 'Frozen training configuration differs.'
        else: write(path, config)
        write(OUT / 'target_modules.json', info); write(OUT / 'processor_check.json', collator.check)
        if not (OUT / 'prior_files_sha256.json').exists():
            previous = [p for p in (ROOT / 'results').rglob('*') if p.is_file() and p.parent != ROOT / 'results' and not p.is_relative_to(OUT)]
            previous += [ROOT / 'results/protocol.json', ROOT / 'dataset/splits/windows.csv', ROOT / 'dataset/splits/dog_splits.json',
                         DATA / 'prompt.txt', DATA / 'images.csv', DATA / 'normalization.json', DATA / 'render_protocol.json']
            write(OUT / 'prior_files_sha256.json', {str(p.relative_to(ROOT)): sha(p) for p in previous})
    model.to(device)
    print(f'rank {rank}: model on {device}', flush=True)
    torch.manual_seed(SEED + rank); random.seed(SEED + rank); np.random.seed(SEED + rank)
    train_data = ImageDataset(data[data.split.eq('train')].to_dict('records'))
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=SEED, drop_last=False)
    loader = DataLoader(train_data, batch_size=8, sampler=sampler, collate_fn=collator, num_workers=0)
    validation_rows = data[data.split.eq('validation')].to_dict('records')[rank::world]
    validation = DataLoader(ImageDataset(validation_rows), batch_size=8, shuffle=False, collate_fn=collator, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0002,
                                  betas=(.9, .999), eps=1e-8, weight_decay=.01)
    accum = config['gradient_accumulation']; per_epoch = math.ceil(len(loader) / accum); total_steps = per_epoch * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(.05 * total_steps), total_steps)
    resume = MODEL / 'last_state.pt'
    first_epoch = 0; step = 0; best = float('inf'); best_epoch = None
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        first_epoch = state['epoch']; step = state['step']; best = state['best_validation_nll']; best_epoch = state['best_epoch']
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    print(f'rank {rank}: DDP ready; starting optimization', flush=True)
    run_start = time.perf_counter(); initial_step = step
    for epoch in range(first_epoch, 3):
        sampler.set_epoch(epoch); wrapped.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64); token_sum = torch.zeros_like(loss_sum)
        epoch_start = time.perf_counter()
        for index, batch in enumerate(loader):
            group_start = index // accum * accum
            group_size = min(accum, len(loader) - group_start)
            update = (index + 1) % accum == 0 or index + 1 == len(loader)
            from contextlib import nullcontext
            context = wrapped.no_sync() if world > 1 and not update else nullcontext()
            with context:
                total, n = answer_loss(wrapped, batch, len(collator.prefix), device)
                loss = total / n
                assert torch.isfinite(loss), 'Non-finite training loss.'
                (loss / group_size).backward()
            loss_sum += total.detach().double(); token_sum += n.detach().double()
            if update:
                grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                assert torch.isfinite(grad)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0 or index + 1 == len(loader):
                    totals = torch.stack([loss_sum, token_sum])
                    if world > 1: dist.all_reduce(totals)
                    if rank == 0:
                        row = {'epoch': epoch + 1, 'optimizer_step': step, 'total_optimizer_steps': total_steps,
                               'train_answer_nll_so_far': float(totals[0] / totals[1]), 'learning_rate': scheduler.get_last_lr()[0],
                               'session_wall_seconds': time.perf_counter() - run_start, 'cuda_peak_allocated_GiB_rank0': torch.cuda.max_memory_allocated() / 2**30}
                        write(OUT / 'training_progress.json', row)
                        with (OUT / 'training_history.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
                        print(f"epoch {epoch+1}/3 step {step}/{total_steps} train_nll={row['train_answer_nll_so_far']:.5f} wall={row['session_wall_seconds']:.1f}s", flush=True)
        model.eval(); val_sum = torch.zeros((), device=device, dtype=torch.float64); val_tokens = torch.zeros_like(val_sum)
        with torch.inference_mode():
            for batch in validation:
                total, n = answer_loss(model, batch, len(collator.prefix), device)
                val_sum += total.double(); val_tokens += n.double()
        val = torch.stack([val_sum, val_tokens])
        if world > 1: dist.all_reduce(val)
        val_nll = float(val[0] / val[1]); assert math.isfinite(val_nll)
        improved = val_nll < best
        if improved: best = val_nll; best_epoch = epoch + 1
        if rank == 0:
            checkpoint = MODEL / 'checkpoints' / f'epoch_{epoch+1}'
            model.save_pretrained(checkpoint, safe_serialization=True)
            if improved: model.save_pretrained(MODEL / 'adapter', safe_serialization=True)
            from peft import get_peft_model_state_dict
            torch.save({'epoch': epoch + 1, 'step': step, 'best_validation_nll': best, 'best_epoch': best_epoch,
                        'adapter': {k: v.cpu() for k, v in get_peft_model_state_dict(model).items()},
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, resume)
            record = {'epoch': epoch + 1, 'optimizer_step': step, 'validation_answer_nll': val_nll,
                      'validation_answer_tokens': int(val[1]), 'best_epoch': best_epoch, 'best_validation_nll': best,
                      'epoch_wall_seconds': time.perf_counter() - epoch_start}
            with (OUT / 'validation_history.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
            write(OUT / 'checkpoint_selection.json', record)
            print('VALIDATION ' + json.dumps(record), flush=True)
        if world > 1: dist.barrier()
    if rank == 0:
        write(OUT / 'training_complete.json', {'state': 'complete', 'epochs': 3, 'optimizer_steps': step,
                    'best_epoch': best_epoch, 'best_validation_nll': best, 'training_windows': 16461,
                    'sample_presentations': 3 * (16462 if world == 2 else 16461),
                    'world_size': world, 'session_wall_seconds': time.perf_counter() - run_start,
                    'session_optimizer_steps': step - initial_step, 'code_sha256': sha(Path(__file__)),
                    'completed_utc': datetime.now(timezone.utc).isoformat()})
    if world > 1: dist.destroy_process_group()


def export():
    import torch
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    assert (OUT / 'training_complete.json').exists()
    torch.set_num_threads(8)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(base, MODEL / 'adapter')
    probes = {}
    for name, layer in model.named_modules():
        s = scope(name)
        if s and s not in probes and hasattr(layer, 'lora_A'):
            delta = layer.get_delta_weight('default').detach().float()
            assert float(delta.abs().sum()) > 0
            probes[s] = {'module': name, 'delta_l1': float(delta.abs().sum()), 'delta_max': float(delta.abs().max())}
    assert set(probes) == {'ViT', 'visual_projection', 'language_attention'}
    merged = model.merge_and_unload(safe_merge=True)
    assert not any('lora_' in n for n, _ in merged.named_parameters())
    merged.save_pretrained(MODEL, safe_serialization=True, max_shard_size='5GB')
    AutoProcessor.from_pretrained(BASE, local_files_only=True).save_pretrained(MODEL)
    files = {p.name: sha(p) for p in MODEL.glob('*.safetensors')}
    write(OUT / 'export_validation.json', {'merged_for_vllm': True, 'selected_adapter_sha256': sha(MODEL / 'adapter/adapter_model.safetensors'),
                'nonzero_merged_scopes': probes, 'merged_weight_files_sha256': files,
                'best_checkpoint': json.loads((OUT / 'checkpoint_selection.json').read_text())['best_epoch']})
    print('Selected LoRA adapter merged and exported for vLLM.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'train', 'export'])
    args = parser.parse_args()
    {'smoke': smoke, 'train': train, 'export': export}[args.command]()
