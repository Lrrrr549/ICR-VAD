import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import math

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("This benchmark requires PyTorch") from e

from PIL import Image
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:  # pragma: no cover
    # transformers==4.37.x fallback
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText, AutoProcessor

# Local imports
sys.path.append(os.path.dirname(__file__))

from model import CLIPVAD  # noqa: E402
from utils.benchmark import time_callable  # noqa: E402


def _label_map_from_ckpt(ckpt_path: str) -> Dict[str, str]:
    if "ucf" in (ckpt_path or "").lower():
        return {
            "Normal": "Normal",
            "Abuse": "Abuse",
            "Arrest": "Arrest",
            "Arson": "Arson",
            "Assault": "Assault",
            "Burglary": "Burglary",
            "Explosion": "Explosion",
            "Fighting": "Fighting",
            "RoadAccidents": "RoadAccidents",
            "Robbery": "Robbery",
            "Shooting": "Shooting",
            "Shoplifting": "Shoplifting",
            "Stealing": "Stealing",
            "Vandalism": "Vandalism",
        }

    return {
        "A": "normal",
        "B1": "fighting",
        "B2": "shooting",
        "B4": "riot",
        "B5": "abuse",
        "B6": "car accident",
        "G": "explosion",
    }


def _random_rgb_images(n: int, h: int, w: int, seed: Optional[int] = None) -> List[Image.Image]:
    rng = np.random.default_rng(seed)
    imgs: List[Image.Image] = []
    for _ in range(int(n)):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs


def bench_vadclip_fps(
    *,
    vadclip_ckpt: str,
    device: str,
    visual_length: int,
    visual_width: int,
    prompt_prefix: int,
    prompt_postfix: int,
    visual_head: int,
    visual_layers: int,
    attn_window: int,
    embed_dim: int,
    classes_num: int,
    batch_size: int,
    warmup: int,
    repeat: int,
) -> Dict[str, Any]:
    torch_device = torch.device(device)

    label_map = _label_map_from_ckpt(vadclip_ckpt)
    prompt_text = list(label_map.values())

    if len(label_map) == 7:
        # XD defaults from pipeline
        visual_layers = 1
        attn_window = 64
        classes_num = 7

    model = CLIPVAD(
        classes_num,
        embed_dim,
        visual_length,
        visual_width,
        visual_head,
        visual_layers,
        attn_window,
        prompt_prefix,
        prompt_postfix,
        torch_device,
    ).to(torch_device)

    state = torch.load(vadclip_ckpt, map_location=torch_device)
    model.load_state_dict(state)
    model.eval()

    bs = max(1, int(batch_size))
    # Random features: [B, T, D]
    visual = torch.randn((bs, visual_length, visual_width), device=torch_device, dtype=torch.float16)
    lengths = torch.full((bs,), int(visual_length), device=torch_device, dtype=torch.int64)

    def _run_once():
        with torch.no_grad():
            _ = model(visual, None, prompt_text, lengths)

    timing = time_callable(_run_once, devices=[torch_device], warmup=warmup, repeat=repeat)

    frames_per_call = bs * int(visual_length)
    mean_s = float(timing.mean_s)
    fps = float(frames_per_call / mean_s) if mean_s > 0 else 0.0

    return {
        "frames_per_call": int(frames_per_call),
        "fps": fps,
        "latency": timing.to_json(),
        "device": str(torch_device),
        "batch_size": int(bs),
        "visual_length": int(visual_length),
        "visual_width": int(visual_width),
        "classes_num": int(classes_num),
    }


def bench_qwen3vl_fps(
    *,
    qwen_path: str,
    device: str,
    num_images: int,
    image_size: int,
    max_new_tokens: int,
    warmup: int,
    repeat: int,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    torch_device = torch.device(device)

    llm = AutoModelForImageTextToText.from_pretrained(
        qwen_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().to(torch_device)
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)

    frames = _random_rgb_images(num_images, image_size, image_size, seed=seed)
    anomal_labels = [
        "abuse",
        "arson",
        "assault",
        "burglary",
        "explosion",
        "fighting",
        "robbery",
        "shooting",
        "shoplifting",
        "stealing",
        "vandalism",
        "car accident",
        "riot",
    ]
    prompt = f"Is there any {anomal_labels} in video frames? Answer with anomal labels or 'No'"

    messages = [{"role": "user", "content": ([{"type": "image", "image": img} for img in frames] + [{"type": "text", "text": prompt}])}]

    # Preferred path: chat template
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=frames, videos=None, do_resize=False, return_tensors="pt")
    except Exception:
        # Fallback: plain text
        inputs = processor(text=prompt, images=frames, return_tensors="pt")

    inputs = inputs.to(torch_device)

    def _run_once():
        with torch.no_grad():
            _ = llm.generate(**inputs, max_new_tokens=int(max_new_tokens))

    timing = time_callable(_run_once, devices=[torch_device], warmup=warmup, repeat=repeat)

    frames_per_call = int(num_images)
    mean_s = float(timing.mean_s)
    fps = float(frames_per_call / mean_s) if mean_s > 0 else 0.0

    return {
        "frames_per_call": frames_per_call,
        "fps": fps,
        "latency": timing.to_json(),
        "device": str(torch_device),
        "image_size": int(image_size),
        "max_new_tokens": int(max_new_tokens),
    }


def bench_end2end_overall_fps(
    *,
    device: str,
    # simulated video timeline
    video_frames: int,
    frame_stride: int,
    select_frames: int,
    window_size: int,
    # vadclip
    vadclip_ckpt: str,
    visual_length: int,
    visual_width: int,
    vad_batch: int,
    # qwen
    qwen_path: str,
    qwen_image_size: int,
    qwen_max_new_tokens: int,
    # timing
    warmup: int,
    repeat: int,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Simulate full pipeline speed: VadCLIP runs over whole video, Qwen runs `select_frames` times.

    Assumptions (matches pipeline logic at a high level):
    - Feature length ~= ceil(video_frames / frame_stride)
    - VadCLIP runs in chunks of `visual_length` (like process_split/pad)
    - Qwen is invoked `select_frames` times
    - Each Qwen call sees ~ (2*(window_size//2)+1) images (like window_indices with half_window)

    Returns a JSON-serializable dict with overall FPS + breakdown.
    """

    torch_device = torch.device(device)
    vf = max(1, int(video_frames))
    fs = max(1, int(frame_stride))
    feat_len = int(math.ceil(vf / fs))
    chunks = int(math.ceil(feat_len / max(1, int(visual_length))))

    # --- Build VadCLIP ---
    label_map = _label_map_from_ckpt(vadclip_ckpt)
    prompt_text = list(label_map.values())

    prompt_prefix = 10
    prompt_postfix = 10
    visual_head = 1
    visual_layers = 2
    attn_window = 8
    embed_dim = 512
    classes_num = 14
    if len(label_map) == 7:
        visual_layers = 1
        attn_window = 64
        classes_num = 7

    vad_model = CLIPVAD(
        classes_num,
        embed_dim,
        int(visual_length),
        int(visual_width),
        visual_head,
        visual_layers,
        attn_window,
        prompt_prefix,
        prompt_postfix,
        torch_device,
    ).to(torch_device)
    state = torch.load(vadclip_ckpt, map_location=torch_device)
    vad_model.load_state_dict(state)
    vad_model.eval()

    bs = max(1, int(vad_batch))
    # One chunk input (reused across chunks). This measures compute, not data loading.
    visual = torch.randn((bs, int(visual_length), int(visual_width)), device=torch_device, dtype=torch.float16)
    lengths = torch.full((bs,), int(visual_length), device=torch_device, dtype=torch.int64)

    # --- Build Qwen ---
    llm = AutoModelForImageTextToText.from_pretrained(
        qwen_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().to(torch_device)
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)

    half_window = max(0, int(window_size) // 2)
    qwen_images_per_call = int(2 * half_window + 1) if int(window_size) > 0 else 0
    qwen_images_per_call = max(1, qwen_images_per_call)
    frames = _random_rgb_images(qwen_images_per_call, int(qwen_image_size), int(qwen_image_size), seed=seed)
    anomal_labels = [
        "abuse",
        "arson",
        "assault",
        "burglary",
        "explosion",
        "fighting",
        "robbery",
        "shooting",
        "shoplifting",
        "stealing",
        "vandalism",
        "car accident",
        "riot",
    ]
    prompt = f"Is there any {anomal_labels} in video frames? Answer with anomal labels or 'No'"
    messages = [{"role": "user", "content": ([{"type": "image", "image": img} for img in frames] + [{"type": "text", "text": prompt}])}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        qwen_inputs = processor(text=text, images=frames, videos=None, do_resize=False, return_tensors="pt")
    except Exception:
        qwen_inputs = processor(text=prompt, images=frames, return_tensors="pt")
    qwen_inputs = qwen_inputs.to(torch_device)

    q_calls = max(0, int(select_frames))

    def _run_end2end_once():
        with torch.no_grad():
            # VadCLIP over whole video
            for _ in range(chunks):
                _ = vad_model(visual, None, prompt_text, lengths)

            # Qwen over sampled windows
            for _ in range(q_calls):
                _ = llm.generate(**qwen_inputs, max_new_tokens=int(qwen_max_new_tokens))

    timing = time_callable(_run_end2end_once, devices=[torch_device], warmup=warmup, repeat=repeat)
    mean_s = float(timing.mean_s)
    overall_fps = float(vf / mean_s) if mean_s > 0 else 0.0

    return {
        "video_frames": int(vf),
        "frame_stride": int(fs),
        "feature_length": int(feat_len),
        "vadclip_chunks": int(chunks),
        "select_frames": int(q_calls),
        "qwen_images_per_call": int(qwen_images_per_call),
        "overall_fps": overall_fps,
        "latency": timing.to_json(),
        "device": str(torch_device),
        "notes": "overall_fps = video_frames / (mean end-to-end latency per simulated video)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VadCLIP + Qwen3-VL-8B inference speed (FPS) using random inputs")

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--vadclip-ckpt", type=str, default="/root/VadCLIP/model_ucf.pth")
    parser.add_argument("--skip-vadclip", action="store_true")
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--visual-width", type=int, default=512)
    parser.add_argument("--vad-batch", type=int, default=1)

    parser.add_argument("--qwen-path", type=str, default="/root/autodl-tmp/Qwen3-VL-8B-Instruct")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--qwen-num-images", type=int, default=16, help="Number of images per Qwen call (treated as frames).")
    parser.add_argument("--qwen-image-size", type=int, default=448)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=64)

    # End-to-end simulation options
    parser.add_argument("--end2end", action="store_true", help="Simulate full video: VadCLIP full pass + Qwen sampled calls, report overall FPS.")
    parser.add_argument("--video-frames", type=int, default=3000, help="Simulated video length in frames for end2end.")
    parser.add_argument("--frame-stride", type=int, default=16, help="Feature stride (frames per feature) for end2end.")
    parser.add_argument("--select-frames", type=int, default=36, help="Number of Qwen calls (sampled frames) for end2end.")
    parser.add_argument("--window-size", type=int, default=12, help="Window size (controls images per Qwen call) for end2end.")

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    result: Dict[str, Any] = {
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "device": str(args.device),
    }

    if not args.skip_vadclip:
        result["vadclip"] = bench_vadclip_fps(
            vadclip_ckpt=args.vadclip_ckpt,
            device=args.device,
            visual_length=args.visual_length,
            visual_width=args.visual_width,
            prompt_prefix=10,
            prompt_postfix=10,
            visual_head=1,
            visual_layers=2,
            attn_window=8,
            embed_dim=512,
            classes_num=14,
            batch_size=args.vad_batch,
            warmup=args.warmup,
            repeat=args.repeat,
        )

    if not args.skip_qwen:
        result["qwen3vl"] = bench_qwen3vl_fps(
            qwen_path=args.qwen_path,
            device=args.device,
            num_images=args.qwen_num_images,
            image_size=args.qwen_image_size,
            max_new_tokens=args.qwen_max_new_tokens,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
        )

    if args.end2end:
        # End-to-end requires both models.
        result["end2end"] = bench_end2end_overall_fps(
            device=args.device,
            video_frames=args.video_frames,
            frame_stride=args.frame_stride,
            select_frames=args.select_frames,
            window_size=args.window_size,
            vadclip_ckpt=args.vadclip_ckpt,
            visual_length=args.visual_length,
            visual_width=args.visual_width,
            vad_batch=args.vad_batch,
            qwen_path=args.qwen_path,
            qwen_image_size=args.qwen_image_size,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
