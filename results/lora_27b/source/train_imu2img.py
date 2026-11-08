"""Paper-inspired supervised IMU2IMG converter; Qwen remains entirely frozen."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/vlm_imu2img'
MODEL = ROOT / 'models/IMU2IMG-pet'
VIT = ROOT / 'models/vit-base-patch32-224-in21k'
RESULT = ROOT / 'results/imu2img_converter'
SEED = 20261101


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def read_data():
    import numpy as np
    import pandas as pd
    import torch
    w = pd.read_csv(ROOT / 'dataset/splits/windows.csv')
    z = np.load(ROOT / 'dataset/cache/pet_windows.npz')
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    assert np.array_equal(z['dogs'], w.dog_id) and np.array_equal(z['tests'], w.session)
    assert np.array_equal(np.array(classes)[z['y']], w.label) and np.allclose(z['starts'], w.start_sec)
    assert w.split.value_counts().to_dict() == {'train': 16461, 'validation': 4727, 'test': 3708}
    assert not set(w[w.split.eq('train')].dog_id) & set(w[w.split.eq('validation')].dog_id)
    assert not set(w[w.split.ne('test')].dog_id) & set(w[w.split.eq('test')].dog_id)
    raw = z['X'].astype('float64').reshape(-1, 200, 2, 3)
    rms = np.sqrt(np.mean(raw ** 2, axis=-1)).astype('float32')
    # Keep the original 2s/200-sample manifest. This encoder adaptation center-crops 2 samples per end.
    x = torch.from_numpy(np.ascontiguousarray(rms[:, 2:198, :].transpose(0, 2, 1)))
    assert x.shape == (24896, 2, 196) and torch.isfinite(x).all()
    return w, x, torch.from_numpy(z['y'].astype('int64')), classes


def build_converter():
    import torch
    from torch import nn
    from vendor.mantis_tgu import TokenGeneratorUnit
    class Converter(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList([TokenGeneratorUnit(hidden_dim=256, num_patches=49, patch_window_size=4,
                scalar_scales=None, hidden_dim_scalar_enc=32, epsilon_scalar_enc=1.1) for _ in range(2)])
            self.decoders = nn.ModuleList([nn.Linear(256, 32 * 32) for _ in range(2)])

        def forward(self, signals):
            planes = []
            for i in range(2):
                tokens = self.encoders[i](signals[:, i:i + 1, :])
                assert tokens.shape[1:] == (49, 256)
                pixels = self.decoders[i](tokens).reshape(-1, 7, 7, 32, 32)
                plane = pixels.permute(0, 1, 3, 2, 4).reshape(-1, 224, 224).tanh()
                planes.append(plane)
            # No fabricated magnetometer: zero in [-1,1] maps to middle gray when written as RGB.
            return torch.stack([planes[0], planes[1], torch.zeros_like(planes[0])], dim=1)
    return Converter()


def build_model():
    from torch import nn
    from transformers import ViTModel
    class IMU2IMG(nn.Module):
        def __init__(self):
            super().__init__()
            self.converter = build_converter()
            self.vit = ViTModel.from_pretrained(VIT, local_files_only=True, attn_implementation='sdpa')
            self.classifier = nn.Linear(self.vit.config.hidden_size, 7)

        def forward(self, signals):
            image = self.converter(signals)
            return self.classifier(self.vit(pixel_values=image).pooler_output)
    return IMU2IMG()


def configuration(model):
    import importlib.metadata as md
    ref = json.loads((DATA / 'source/source_reference.json').read_text())
    vit = json.loads((VIT / 'source_reference.json').read_text())
    return {'experiment': 'imu2img_pet_2s_two_modality_supervised_converter',
        'paper_doi': '10.1145/3714394.3756208', 'paper_sha256': ref['paper_sha256'],
        'mantis_revision': ref['mantis_revision'], 'mantis_vendor_files_sha256': ref['mantis_vendor_files_sha256'],
        'vit_repo': vit['repo_id'], 'vit_revision': vit['revision'],
        'seed': SEED, 'epochs': 10, 'batch_size': 512, 'learning_rate': .0001,
        'optimizer': 'AdamW', 'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': .01,
        'scheduler': None, 'loss': 'Unweighted seven-class cross entropy',
        'precision': 'Float32 trainable weights/optimizer with BF16 autocast for forward/backward',
        'checkpoint_selection': 'Highest full validation macro-F1 across 10 epochs, earliest epoch on exact ties',
        'train_windows': 16461, 'validation_windows': 4727, 'test_windows': 3708,
        'qwen_training_examples': 0, 'qwen_finetuned': False,
        'trainable_components': ['two modality-specific Mantis-v1 TGU encoders', 'two 256-to-1024 linear patch decoders',
                                 'pretrained ViT-base-patch32-224-in21k', 'new seven-class linear classifier'],
        'parameter_counts': {'converter': sum(p.numel() for p in model.converter.parameters()),
                             'whole_training_system': sum(p.numel() for p in model.parameters())},
        'original_window': {'seconds': 2, 'samples': 200, 'hz': 100, 'channels': 6},
        'source_to_encoder': 'Per-time-step RMS sqrt(mean(x^2,y^2,z^2)) separately for acceleration and gyroscope; center crop original indices [2:198] to 196 samples; 49 non-overlapping 4-sample patches.',
        'paper_adaptations': ['Two modalities instead of three; no magnetometer; B is zero normalized intensity (middle gray in exported RGB)',
                             'Original 2s/200-source-point windows instead of 5s/500; center crop 196 instead of 490 and patch length4 instead of10',
                             'Dog seven-class task and original dog-disjoint train/validation/test splits',
                             'Do not perform original paper final validation-set fine-tuning; validation used only for checkpoint selection',
                             'Generated images subsequently classified by frozen Qwen3.5-4B; this transfer is additional to the paper ViT classifier'],
        'image': {'width': 224, 'height': 224, 'token_grid': [7, 7], 'tile_size': [32, 32],
                  'layout': 'row-major chronological token order', 'channels': {'R': 'learned acceleration RMS', 'G': 'learned gyroscope RMS', 'B': 'constant missing modality'},
                  'pixel_mapping': 'tanh output [-1,1] -> round(127.5*(z+1)) -> uint8; fixed mapping, no per-image rescaling'},
        'windows_sha256': sha(ROOT / 'dataset/splits/windows.csv'),
        'split_sha256': sha(ROOT / 'dataset/splits/dog_splits.json'),
        'versions': {p: md.version(p) for p in ['torch', 'transformers', 'numpy', 'scikit-learn']}}


def init(args):
    import numpy as np
    import torch
    torch.set_num_threads(8); torch.manual_seed(SEED); np.random.seed(SEED)
    torch.cuda.set_device(args.device)
    for path in [DATA, MODEL, RESULT]: path.mkdir(parents=True, exist_ok=True)
    w, x, y, classes = read_data()
    model = build_model().to(args.device)
    config = configuration(model)
    path = RESULT / 'training_protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == config, 'Frozen converter configuration differs.'
    else: write(path, config)
    return w, x, y, classes, model, config


def smoke(args):
    import torch
    import torch.nn.functional as F
    w, x, y, _, model, config = init(args)
    idx = w.index[w.split.eq('train')][:512]
    model.train(); begin = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x[idx].to(args.device))
        loss = F.cross_entropy(logits.float(), y[idx].to(args.device))
    loss.backward(); torch.cuda.synchronize()
    grad = {}
    for name in ['converter.encoders', 'converter.decoders', 'vit', 'classifier']:
        grad[name] = sum(float(p.grad.detach().float().square().sum()) for n, p in model.named_parameters()
                         if n.startswith(name) and p.grad is not None) ** .5
    assert torch.isfinite(loss) and all(v > 0 for v in grad.values())
    record = {'purpose': '512 training-window forward/backward check only; no optimizer update', 'loss': float(loss.detach()),
              'gradient_norms': grad, 'seconds': time.perf_counter() - begin,
              'peak_allocated_GiB': torch.cuda.max_memory_allocated(args.device) / 2**30}
    write(RESULT / 'backward_check.json', record); print(json.dumps(record, indent=2), flush=True)


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
    w, x, y, classes, model, config = init(args)
    train_idx = w.index[w.split.eq('train')].to_numpy(); val_idx = w.index[w.split.eq('validation')].to_numpy()
    x = x.to(args.device); y = y.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    rng = np.random.default_rng(SEED); first_epoch = 0; best = -1.; best_epoch = None
    last = MODEL / 'last_state.pt'
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        first_epoch = state['epoch']; best = state['best_macro_f1']; best_epoch = state['best_epoch']; rng.bit_generator.state = state['rng_state']
    begin = time.perf_counter()
    for epoch in range(first_epoch, 10):
        model.train(); start = time.perf_counter(); order = rng.permutation(train_idx); train_loss = 0.; n = 0
        for step, offset in enumerate(range(0, len(order), 512)):
            idx = torch.as_tensor(order[offset:offset + 512], device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x[idx]); loss = F.cross_entropy(logits.float(), y[idx])
            assert torch.isfinite(loss), 'Non-finite converter training loss.'
            loss.backward()
            grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach())
                        for p in model.parameters() if p.grad is not None]))
            assert torch.isfinite(grad_norm), 'Non-finite converter gradient norm.'
            optimizer.step(); train_loss += float(loss.detach()) * len(idx); n += len(idx)
            if (step + 1) % 5 == 0 or n == len(order):
                progress = {'epoch': epoch + 1, 'epochs': 10, 'train_windows_seen_this_epoch': n,
                            'train_windows_per_epoch': len(order), 'train_loss': train_loss / n,
                            'wall_seconds': time.perf_counter() - begin,
                            'cuda_peak_allocated_GiB': torch.cuda.max_memory_allocated(args.device) / 2**30}
                write(RESULT / 'progress.json', progress); print('TRAIN ' + json.dumps(progress), flush=True)
        model.eval(); preds = []; val_loss = 0.
        with torch.inference_mode():
            for offset in range(0, len(val_idx), 512):
                idx = torch.as_tensor(val_idx[offset:offset + 512], device=args.device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x[idx]); loss = F.cross_entropy(logits.float(), y[idx])
                preds.extend(logits.argmax(-1).cpu().tolist()); val_loss += float(loss) * len(idx)
        truth = y[torch.as_tensor(val_idx, device=args.device)].cpu().numpy()
        f1 = f1_score(truth, preds, labels=list(range(7)), average='macro', zero_division=0)
        if f1 > best:
            best = f1; best_epoch = epoch + 1
            torch.save({'model': {k: v.cpu() for k, v in model.state_dict().items()}, 'epoch': best_epoch,
                        'validation_macro_f1': best, 'protocol_sha256': sha(RESULT / 'training_protocol.json')}, MODEL / 'best.pt')
        record = {'epoch': epoch + 1, 'train_loss': train_loss / n, 'validation_loss': val_loss / len(val_idx),
                  'validation_macro_f1': float(f1), 'validation_balanced_accuracy': float(balanced_accuracy_score(truth, preds)),
                  'validation_accuracy': float(accuracy_score(truth, preds)), 'best_epoch': best_epoch, 'best_macro_f1': float(best),
                  'epoch_wall_seconds': time.perf_counter() - start}
        with (RESULT / 'history.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
        write(RESULT / 'checkpoint_selection.json', record)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                    'best_macro_f1': best, 'best_epoch': best_epoch, 'rng_state': rng.bit_generator.state}, last)
        print('VALIDATION ' + json.dumps(record), flush=True)
    write(RESULT / 'training_complete.json', {'epochs': 10, 'best_epoch': best_epoch, 'best_validation_macro_f1': float(best),
          'train_windows': 16461, 'validation_windows': 4727, 'qwen_finetuned': False,
          'session_wall_seconds': time.perf_counter() - begin, 'code_sha256': sha(Path(__file__)),
          'best_checkpoint_sha256': sha(MODEL / 'best.pt')})
    print('IMU2IMG converter training complete; Qwen was never loaded or updated.', flush=True)


def render(args):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    torch.set_num_threads(8); torch.cuda.set_device(args.device)
    w, x, y, classes = read_data()
    state = torch.load(MODEL / 'best.pt', map_location='cpu', weights_only=False)
    converter = build_converter()
    converter.load_state_dict({k.removeprefix('converter.'): v for k, v in state['model'].items() if k.startswith('converter.')})
    save_file({k: v.contiguous() for k, v in converter.state_dict().items()}, str(MODEL / 'converter.safetensors'))
    converter.to(args.device).eval()
    config = json.loads((RESULT / 'training_protocol.json').read_text())
    protocol = {'version': 'imu2img_learned_rgb_pet_v1', 'paper_doi': config['paper_doi'],
        'window_manifest_sha256': sha(ROOT / 'dataset/splits/windows.csv'), 'split_sha256': sha(ROOT / 'dataset/splits/dog_splits.json'),
        'converter_training_protocol_sha256': sha(RESULT / 'training_protocol.json'),
        'converter_weights_sha256': sha(MODEL / 'converter.safetensors'), 'selected_epoch': state['epoch'],
        'width': 224, 'height': 224, 'input_signal': config['source_to_encoder'], 'image': config['image'],
        'image_format': 'WebP lossless RGB; no private metadata', 'test_windows': 3708,
        'out_of_range': 'Fixed tanh [-1,1] to uint8 mapping; no adaptive contrast/stretching; numerical range and pixel hashes logged',
        'supervision_note': 'Image converter and ViT were supervised on original training labels; Qwen receives no examples and no training updates.'}
    write(DATA / 'render_protocol.json', protocol)
    (DATA / 'images').mkdir(exist_ok=True)
    def export_batch(indices):
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            values = converter(x[indices].to(args.device)).float()
        assert torch.isfinite(values).all() and values.min() >= -1 and values.max() <= 1
        pixels = torch.round((values + 1) * 127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        seconds = time.perf_counter() - start
        return pixels, seconds
    first = int(w.index[w.split.eq('train')][0]); preview, _ = export_batch([first])
    Image.fromarray(preview[0]).save(DATA / 'preview.webp', format='WEBP', lossless=True, method=4)
    fields = ['window_id', 'image_file', 'image_sha256', 'render_seconds', 'pixel_min', 'pixel_max', 'b_channel_unique']
    indices = w.index[w.split.eq('test')].to_numpy()
    with (DATA / 'images.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for start in range(0, len(indices), 128):
            current = indices[start:start + 128]; pixels, gen_seconds = export_batch(current)
            for index, rgb in zip(current, pixels):
                wid = w.window_id.iloc[index]; path = DATA / 'images' / ('sample_' + hashlib.sha256(wid.encode()).hexdigest()[:20] + '.webp')
                saved = time.perf_counter(); Image.fromarray(rgb).save(path, format='WEBP', lossless=True, method=4)
                assert np.array_equal(np.unique(rgb[:, :, 2]), [128])
                writer.writerow({'window_id': wid, 'image_file': str(path.relative_to(ROOT)), 'image_sha256': sha(path),
                    'render_seconds': gen_seconds / len(current) + time.perf_counter() - saved,
                    'pixel_min': int(rgb.min()), 'pixel_max': int(rgb.max()), 'b_channel_unique': 128})
            f.flush(); print(f'IMU2IMG images {min(start + 128,len(indices))}/3708', flush=True)
    write(DATA / 'input_validation.json', {'test_windows': 3708, 'fixed_window_order': True, 'blue_channel_fixed_gray': 128,
          'selected_training_only_converter': True, 'qwen_finetuned': False,
          'manifest_sha256': sha(DATA / 'images.csv'), 'converter_weights_sha256': sha(MODEL / 'converter.safetensors')})
    print('All 3708 learned RGB test images exported.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'train', 'render'])
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    {'smoke': smoke, 'train': train, 'render': render}[args.command](args)
