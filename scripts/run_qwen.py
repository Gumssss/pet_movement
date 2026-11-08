"""Offline vLLM text/image inference with the downloaded official Qwen models."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['4B', '27B', '3.8-27B', '4B-lora'], default='4B')
    parser.add_argument('--prompt', default='请用中文简短介绍你自己。')
    parser.add_argument('--image', type=Path, help='Optional local image')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if args.image and not args.image.is_file():
        parser.error('Image file does not exist')

    if not args.device.startswith('cuda:') or not args.device.split(':')[1].isdigit():
        parser.error('Use a CUDA device such as cuda:0')
    os.environ['CUDA_VISIBLE_DEVICES']=args.device.split(':')[1]
    from qwen_classify import load_model, process_gpu_memory, MODEL_SPECS
    MODEL_SPECS['4B-lora'] = {'name': 'Qwen3.5-4B-lora'}
    from vllm import SamplingParams
    names = {'4B':'Qwen3.5-4B', '27B':'Qwen3.5-27B', '3.8-27B':'Qwen3.8-27B', '4B-lora':'Qwen3.5-4B-lora'}
    model_path = ROOT / 'models' / names[args.model]
    if not (model_path / 'model.safetensors.index.json').is_file():
        parser.error(f'Model download is incomplete: {model_path}')
    started = time.monotonic()
    model, processor, load_seconds = load_model(args)
    loaded = time.monotonic()
    content = []
    if args.image:
        from PIL import Image
        with Image.open(args.image) as image:
            content.append({'type': 'image', 'image': image.convert('RGB')})
    content.append({'type': 'text', 'text': args.prompt})
    # Render a template string; vLLM expands visual placeholders and runs the model.
    text = processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                                        add_generation_prompt=True,enable_thinking=args.thinking)
    request={'prompt':text}
    if args.image:
        request['multi_modal_data']={'image':content[0]['image']}
    outputs=model.generate([request],sampling_params=SamplingParams(
        temperature=0.,max_tokens=args.max_new_tokens,seed=20261101),use_tqdm=False)
    result=outputs[0]
    input_tokens=len(result.prompt_token_ids)
    response=result.outputs[0].text.strip()
    if not response:
        raise RuntimeError('Model generated an empty response')
    print(response, flush=True)
    if args.report:
        report = {
            'model': str(model_path), 'device': args.device,'inference_backend':'vllm',
            'dtype': 'bfloat16', 'prompt': args.prompt,
            'image': str(args.image.resolve()) if args.image else None,
            'response': response, 'input_tokens': input_tokens,
            'generated_tokens': len(result.outputs[0].token_ids),
            'load_seconds': round(loaded - started, 3),
            'generation_seconds': round(time.monotonic() - loaded, 3),
            'observed_gpu_process_memory_GiB':process_gpu_memory(args.device)['observed_process_memory_GiB'],
            'timing_scope':'vLLM generation includes internal preprocessing, scheduling, GPU execution and detokenization',
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    main()
