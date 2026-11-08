"""Matched vLLM evaluations of Qwen3.5-4B and its merged vision/attention LoRA."""
import argparse
from collections import Counter
import importlib.metadata as md
import json
from pathlib import Path
import time
import qwen_classify as q

ROOT = q.ROOT
DATA = ROOT / 'dataset/vlm_acl'


def setup(args):
    q.DATA = DATA
    q.MODEL_SPECS['4B-lora'] = {'name': 'Qwen3.5-4B-lora'}
    args.model = '4B' if args.variant == 'base' else '4B-lora'
    name = 'Qwen3.5-4B' if args.variant == 'base' else 'Qwen3.5-4B-lora'
    folder = 'qwen3.5_4B_acl' if args.variant == 'base' else 'qwen3.5_4B_lora_acl'
    args.output_dir = ROOT / 'results' / folder
    dest = args.output_dir; dest.mkdir(exist_ok=True)
    records = q.inputs(); rendered = q.read_rendered()
    assert len(rendered) == len(records) == 3708
    for r in records:
        assert q.sha(ROOT / r['image_file']) == rendered[r['window_id']]['image_sha256']
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    base_revision = json.loads((ROOT / 'models/model_manifest.json').read_text())['Qwen3.5-4B']['revision']
    config = {'model': name, 'experiment': 'imu_2s_acl_shared_axis_en_' + ('supervised_lora' if args.variant == 'lora' else 'zero_shot_base4b'),
              'classes': classes, 'base_model_revision': base_revision, 'inference_backend': 'vllm',
              'backend_version': md.version('vllm'), 'prompt_language': 'en', 'evaluation_split': 'test', 'expected_windows': 3708,
              'data_dir': 'dataset/vlm_acl', 'input_image_size': [512, 409],
              'prompt_sha256': q.sha(DATA / 'prompt.txt'), 'images_manifest_sha256': q.sha(DATA / 'images.csv'),
              'render_protocol_sha256': q.sha(DATA / 'render_protocol.json'),
              'windows_sha256': q.sha(ROOT / 'dataset/splits/windows.csv'),
              'split_sha256': q.sha(ROOT / 'dataset/splits/dog_splits.json'),
              'generation': {'temperature': 0, 'enable_thinking': False, 'max_new_tokens': 32, 'seed': 20261101,
                             'dtype': 'bfloat16', 'guided_decoding': False, 'speculative_decoding': False},
              'engine': {'tensor_parallel_size': 1, 'max_model_len': 4096, 'max_num_seqs': 32, 'max_num_batched_tokens': 8192,
                         'gpu_memory_utilization': .85, 'enable_prefix_caching': False, 'flashinfer_sampler': False},
              'training_examples': 16461 if args.variant == 'lora' else 0,
              'comparison_note': 'Same Qwen3.5-4B base revision, ACL image pixels, English prompt, tokenizer/processor, vLLM settings and 3708 test windows; only the selected merged LoRA differs.'}
    if args.variant == 'lora':
        training = json.loads((dest / 'training_complete.json').read_text())
        config.update(selected_adapter_sha256=q.sha(ROOT / 'models/Qwen3.5-4B-lora/adapter/adapter_model.safetensors'),
                      selected_epoch=training['best_epoch'], checkpoint_selection='minimum validation answer-token NLL',
                      training_protocol_sha256=q.sha(dest / 'training_protocol.json'), export_validation_sha256=q.sha(dest / 'export_validation.json'))
    path = dest / 'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == config, 'Frozen evaluation changed.'
    else: q.write_json(path, config)
    (dest / 'prompt.txt').write_bytes((DATA / 'prompt.txt').read_bytes())
    return dest, records, rendered, classes


def run(args):
    dest, records, rendered, classes = setup(args)
    raw = dest / 'responses.jsonl'
    rows = [json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
    assert len(rows) == len({r['window_id'] for r in rows})
    done = {r['window_id']: r for r in rows}
    pending = [r for r in records if r['window_id'] not in done]
    if not pending:
        print('All 3708 responses already exist; no model generation repeated.', flush=True)
        return
    began = time.perf_counter()
    model, processor, load_seconds = q.load_model(args)
    memory = q.process_gpu_memory(args.device); peak = memory['observed_process_memory_GiB']
    prompt = (DATA / 'prompt.txt').read_text()
    if not (dest / 'preflight.json').exists():
        text, counts, lengths, truncated, pre, gen, wall = q.generate(model, processor, [DATA / 'preview.webp'], prompt, args.device, (512, 409))
        prediction, status = q.parse(text[0], classes, truncated[0])
        q.write_json(dest / 'preflight.json', {'split': 'train', 'purpose': 'technical check only; excluded from test metrics',
                   'response': text[0], 'prediction': prediction, 'status': status, 'input_tokens': counts[0],
                   'generated_tokens': lengths[0], 'seconds': wall})
    timings = []; first_done = len(done)
    with raw.open('a') as stream:
        for start in range(0, len(pending), args.batch_size):
            group = pending[start:start + args.batch_size]
            texts, counts, lengths, truncated, pre, gen, wall = q.generate(model, processor,
                          [ROOT / r['image_file'] for r in group], prompt, args.device, (512, 409))
            timings.append({'n': len(group), 'preprocess_seconds': pre, 'generation_seconds': gen, 'image_to_response_seconds': wall})
            for i, item in enumerate(group):
                prediction, status = q.parse(texts[i], classes, truncated[i])
                row = {**item, 'image_sha256': rendered[item['window_id']]['image_sha256'],
                       'response': texts[i], 'prediction': prediction, 'status': status,
                       'input_tokens': counts[i], 'generated_tokens': lengths[i], 'batch_size': len(group),
                       'inference_backend': 'vllm', 'batch_preprocess_seconds': pre, 'batch_generation_seconds': gen,
                       'batch_image_to_response_seconds': wall, 'amortized_image_to_response_seconds': wall / len(group)}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); done[item['window_id']] = row
            stream.flush()
            peak = max(peak, q.process_gpu_memory(args.device)['observed_process_memory_GiB'])
            progress = {'completed': len(done), 'expected': 3708, 'remaining': 3708 - len(done),
                        'status_counts': dict(Counter(r['status'] for r in done.values())), 'wall_seconds': time.perf_counter() - began}
            q.write_json(dest / 'progress.json', progress)
            if len(done) % 160 == 0 or len(done) == 3708:
                print(args.variant + ': ' + json.dumps(progress), flush=True)
    session = {'inference_backend': 'vllm', 'versions': {'vllm': md.version('vllm')}, 'load_seconds': load_seconds,
               'session_model_run_wall_seconds': time.perf_counter() - began, 'batch_timings': timings,
               'peak_cuda_allocated_GiB': None, 'observed_gpu_process_memory_GiB_peak': peak,
               'memory_measurement': 'NVML process-tree samples after load and each batch; includes cache reservation; not an exact allocator peak.',
               'gpu': memory['gpu'], 'device': args.device, 'session_completed': len(done) - first_done,
               'code_sha256': q.sha(Path(__file__)), 'shared_generation_code_sha256': q.sha(ROOT / 'scripts/qwen_classify.py')}
    path = dest / 'runtime.json'
    runtime = json.loads(path.read_text()) if path.exists() else {'sessions': []}
    runtime['sessions'].append(session); q.write_json(path, runtime)
    print('Complete: ' + args.variant, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'evaluate'])
    parser.add_argument('--variant', choices=['base', 'lora'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1: parser.error('batch-size must be positive')
    if args.command == 'run': run(args)
    else:
        setup(args)
        q.evaluate(args)
