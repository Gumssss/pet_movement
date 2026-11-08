"""Frozen Qwen3.5-4B/vLLM evaluation of ordinary plots or learned IMU2IMG images."""
import argparse
from collections import Counter
import importlib.metadata as md
import json
import os
from pathlib import Path
import time
import qwen_classify as q

ROOT = q.ROOT
METHODS = {
    'ordinary': ('dataset/vlm', 'results/qwen3.5_4B_sixplot_en'),
    'imu2img': ('dataset/vlm_imu2img', 'results/qwen3.5_4B_imu2img_en'),
}


def load_model(args):
    if args.method == 'ordinary':
        return q.load_model(args)
    # Preserve the paper's 224x224 raster: its 7x7 32px tiles also align with Qwen's merged visual-token grid.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[1]
    import torch
    from transformers import AutoProcessor
    from vllm import LLM
    torch.set_num_threads(8); torch.manual_seed(20261101)
    path = ROOT / 'models/Qwen3.5-4B'
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    model = LLM(model=str(path), dtype='bfloat16', tensor_parallel_size=1, trust_remote_code=False,
                seed=20261101, gpu_memory_utilization=.85, max_model_len=4096, max_num_seqs=32,
                max_num_batched_tokens=8192, limit_mm_per_prompt={'image': 1, 'video': 0},
                mm_processor_kwargs={'min_pixels': 224*224, 'max_pixels': 1280*1152},
                generation_config='vllm', enable_prefix_caching=False, disable_log_stats=True)
    return model, processor, time.perf_counter() - started


def setup(args):
    data, folder = METHODS[args.method]
    data = ROOT / data
    args.model = '4B'; args.output_dir = ROOT / folder
    dest = args.output_dir; dest.mkdir(exist_ok=True)
    q.DATA = data
    render = json.loads((data / 'render_protocol.json').read_text())
    assert render['window_manifest_sha256'] == q.sha(ROOT / 'dataset/splits/windows.csv')
    assert render['split_sha256'] == q.sha(ROOT / 'dataset/splits/dog_splits.json')
    expected_size = (render['width'], render['height'])
    prompt = (data / 'prompt.txt').read_text(); assert prompt.isascii()
    records = q.inputs(); images = q.read_rendered()
    assert len(records) == len(images) == 3708
    for r in records:
        assert q.sha(ROOT / r['image_file']) == images[r['window_id']]['image_sha256']
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    model = json.loads((ROOT / 'models/model_manifest.json').read_text())['Qwen3.5-4B']
    frozen = {'model': 'Qwen3.5-4B', 'model_revision': model['revision'], 'model_repo_id': model['repo_id'],
        'experiment': 'qwen35_4b_' + args.method + '_vllm_en', 'image_method': args.method,
        'classes': classes, 'inference_backend': 'vllm', 'backend_version': md.version('vllm'),
        'prompt_language': 'en', 'evaluation_split': 'test', 'expected_windows': 3708,
        'qwen_training_examples': 0, 'qwen_finetuned': False, 'prompt_examples': 0,
        'data_dir': str(data.relative_to(ROOT)), 'input_image_size': list(expected_size),
        'prompt_sha256': q.sha(data / 'prompt.txt'), 'render_protocol_sha256': q.sha(data / 'render_protocol.json'),
        'images_manifest_sha256': q.sha(data / 'images.csv'),
        'windows_sha256': q.sha(ROOT / 'dataset/splits/windows.csv'),
        'split_sha256': q.sha(ROOT / 'dataset/splits/dog_splits.json'),
        'generation': {'temperature': 0, 'enable_thinking': False, 'max_new_tokens': 32, 'seed': 20261101,
                       'dtype': 'bfloat16', 'guided_decoding': False, 'speculative_decoding': False},
        'engine': {'tensor_parallel_size': 1, 'max_model_len': 4096, 'max_num_seqs': 32,
                   'max_num_batched_tokens': 8192, 'gpu_memory_utilization': .85,
                   'enable_prefix_caching': False, 'flashinfer_sampler': False},
        'parsing': 'Same strict label-only JSON parser; parse failures/refusals/truncation counted incorrect; no semantic retries.',
        'timing_scope': 'Observed vLLM request wall time includes internal processing, scheduling, compute and decoding.'}
    if args.method == 'imu2img':
        frozen.update(converter_training_windows=16461, converter_training_protocol_sha256=render['converter_training_protocol_sha256'],
                      converter_weights_sha256=render['converter_weights_sha256'],
                      method_sha256=q.sha(data / 'method.md'),
                      paper_source_reference_sha256=q.sha(data / 'source/source_reference.json'),
                      image_processor_limits={'min_pixels': 224*224, 'max_pixels': 1280*1152},
                      image_processor_note='Lower minimum pixel budget keeps original 224x224 raster and 7x7 merged visual-token grid; ordinary large images unaffected by their existing 256x256 minimum.',
                      pipeline_note='Qwen is frozen/zero-shot; the image converter was supervised using training labels. This is not a zero-label pipeline or a full reproduction of the original SHL system.')
    else:
        frozen.update(converter_training_windows=0, pipeline_note='Original raw-value six-subplot pixels reused unchanged; zero-shot Qwen with the existing English prompt.')
    path = dest / 'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == frozen, 'Frozen experiment changed.'
    else: q.write_json(path, frozen)
    (dest / 'prompt.txt').write_text(prompt)
    return data, dest, records, images, prompt, classes, expected_size


def run(args):
    data, dest, records, images, prompt, classes, size = setup(args)
    file = dest / 'responses.jsonl'
    rows = [json.loads(s) for s in file.read_text().splitlines()] if file.exists() else []
    done = {r['window_id']: r for r in rows}; assert len(rows) == len(done)
    pending = [r for r in records if r['window_id'] not in done]
    if not pending:
        print('Already complete; no requests repeated.', flush=True); return
    began = time.perf_counter()
    model, processor, load_seconds = load_model(args)
    memory = q.process_gpu_memory(args.device); peak = memory['observed_process_memory_GiB']
    if not (dest / 'preflight.json').exists():
        texts, counts, lengths, trunc, pre, gen, wall = q.generate(model, processor, [data / 'preview.webp'], prompt, args.device, size)
        label, status = q.parse(texts[0], classes, trunc[0])
        q.write_json(dest / 'preflight.json', {'split': 'train', 'purpose': 'Local technical check only; excluded from test scores',
                     'response': texts[0], 'prediction': label, 'status': status,
                     'input_tokens': counts[0], 'generated_tokens': lengths[0], 'seconds': wall})
        if args.method == 'imu2img':
            from PIL import Image
            with Image.open(data / 'preview.webp') as im:
                content = [{'type': 'image', 'image': im.convert('RGB')}, {'type': 'text', 'text': prompt}]
                template = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                              add_generation_prompt=True, enable_thinking=False)
            plain_tokens = len(processor.tokenizer.encode(template, add_special_tokens=False))
            visual_tokens = counts[0] - plain_tokens + 1
            assert visual_tokens == 49, f'224x224 image should produce 49 visual tokens, got {visual_tokens}'
            q.write_json(dest / 'processor_check.json', {'export_size': [224,224], 'actual_vllm_visual_tokens': visual_tokens,
                    'total_input_tokens': counts[0], 'raw_template_tokens_with_one_image_placeholder': plain_tokens,
                    'image_grid_thw': [1,14,14], 'merged_visual_grid': [7,7],
                    'note': 'Actual engine prompt token count equals template tokens minus one placeholder plus 49 image tokens; no resizing away from paper grid.'})
    timings = []; initial = len(done)
    with file.open('a') as stream:
        for start in range(0, len(pending), args.batch_size):
            current = pending[start:start + args.batch_size]
            texts, counts, lengths, trunc, pre, gen, wall = q.generate(model, processor,
                [ROOT / r['image_file'] for r in current], prompt, args.device, size)
            timings.append({'n': len(current), 'preprocess_seconds': pre, 'generation_seconds': gen, 'image_to_response_seconds': wall})
            for i, item in enumerate(current):
                prediction, status = q.parse(texts[i], classes, trunc[i])
                row = {**item, 'image_sha256': images[item['window_id']]['image_sha256'],
                       'response': texts[i], 'prediction': prediction, 'status': status,
                       'input_tokens': counts[i], 'generated_tokens': lengths[i], 'batch_size': len(current),
                       'inference_backend': 'vllm', 'amortized_image_to_response_seconds': wall / len(current)}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); done[item['window_id']] = row
            stream.flush()
            peak = max(peak, q.process_gpu_memory(args.device)['observed_process_memory_GiB'])
            progress = {'completed': len(done), 'expected': 3708, 'remaining': 3708 - len(done),
                        'status_counts': dict(Counter(r['status'] for r in done.values())),
                        'wall_seconds': time.perf_counter() - began}
            q.write_json(dest / 'progress.json', progress)
            if len(done) % 160 == 0 or len(done) == 3708: print(json.dumps(progress), flush=True)
    session = {'inference_backend': 'vllm', 'versions': {'vllm': md.version('vllm')}, 'load_seconds': load_seconds,
        'session_model_run_wall_seconds': time.perf_counter() - began, 'batch_timings': timings,
        'peak_cuda_allocated_GiB': None, 'observed_gpu_process_memory_GiB_peak': peak,
        'memory_measurement': 'NVML process-tree sampling after load and each batch; includes cache reservation, not an exact allocator peak.',
        'gpu': memory['gpu'], 'device': args.device, 'session_completed': len(done) - initial,
        'code_sha256': q.sha(Path(__file__)), 'shared_generation_code_sha256': q.sha(ROOT / 'scripts/qwen_classify.py')}
    path = dest / 'runtime.json'; runtime = json.loads(path.read_text()) if path.exists() else {'sessions': []}
    runtime['sessions'].append(session); q.write_json(path, runtime)
    print('COMPLETE ' + args.method, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'evaluate'])
    parser.add_argument('--method', choices=list(METHODS), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1: parser.error('batch-size must be positive')
    if args.command == 'run': run(args)
    else:
        setup(args)
        q.evaluate(args)
