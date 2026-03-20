import os
import csv
import argparse
import json
import sys
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Ensure local imports work
try:
    from pipeline_ats_qwen3vl import build_pipeline
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from pipeline_ats_qwen3vl import build_pipeline

# Ensure local imports work for utils/* as well.
sys.path.append(os.path.dirname(__file__))
from utils.benchmark import time_callable

def _stem_from_path(p: str) -> str:
    base = os.path.basename(p)
    base = os.path.splitext(base)[0]
    # UCF feature list uses: <Stem>__<idx>.npy
    if '__' in base:
        base = base.split('__')[0]
    return base


def parse_ucf_temporal_annotations(annotation_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parse VadCLIP/list/Temporal_Anomaly_Annotation.txt.

    Expected line format (whitespace-separated):
      <video_stem>  <label>  s1  e1  s2  e2
    with -1 meaning absent.
    """
    ann: Dict[str, List[Tuple[int, int]]] = {}
    with open(annotation_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            stem = parts[0]
            try:
                s1, e1, s2, e2 = (int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]))
            except Exception:
                continue
            segs: List[Tuple[int, int]] = []
            if s1 >= 0 and e1 >= 0 and e1 > s1:
                segs.append((s1, e1))
            if s2 >= 0 and e2 >= 0 and e2 > s2:
                segs.append((s2, e2))
            ann[stem] = segs
    return ann


def build_gt_vector(num_frames: int, segments: Optional[List[Tuple[int, int]]]) -> np.ndarray:
    y = np.zeros((max(0, int(num_frames)),), dtype=np.int64)
    if not segments:
        return y
    for s, e in segments:
        s = max(0, min(num_frames, int(s)))
        e = max(0, min(num_frames, int(e)))
        if e > s:
            y[s:e] = 1
    return y


def load_ucf_test_list(csv_path: str) -> List[Tuple[str, str]]:
    """Return list of (path,label).

    UCF list file is typically a CSV with header: path,label
    where path is often a feature .npy path.
    """
    items: List[Tuple[str, str]] = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == 'path':
                continue
            p = row[0].strip()
            label = row[1].strip() if len(row) > 1 else ''
            if p:
                items.append((p, label))
    return items


def infer_ucf_video_path(
    sample_path: str,
    label: str,
    video_root: str,
) -> str:
    if sample_path.lower().endswith(('.mp4', '.avi', '.mkv')):
        return sample_path
    stem = _stem_from_path(sample_path)
    # Handle Normal videos housed under a different folder.
    if label and label.lower() == 'normal':
        normal_dir = 'Testing_Normal_Videos_Anomaly'
        video_path = os.path.join(video_root, normal_dir, f'{stem}.mp4')
    else:
        category = label or os.path.basename(os.path.dirname(sample_path))
        video_path = os.path.join(video_root, category, f'{stem}.mp4')
    return video_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate ATS (Holmes-VAU) + Qwen3-VL AUC')
    parser.add_argument('--test-list', type=str, default='/root/VadCLIP/list/ucf_CLIP_rgbtest.csv')
    parser.add_argument('--annotation-path', type=str, default='/root/VadCLIP/list/Temporal_Anomaly_Annotation.txt')
    parser.add_argument('--ucf-video-root', type=str, default='/root/autodl-tmp/ucf_crime/Anomaly-Videos')
    parser.add_argument('--ats-ckpt', type=str, default='/root/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth')
    parser.add_argument('--encoder-path', type=str, default='/root/autodl-tmp/InternVL2-2B')
    parser.add_argument('--qwen-path', type=str, default='/root/autodl-tmp/Qwen3-VL-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--qwen-device', type=str, default=None)
    parser.add_argument('--clip-device', type=str, default='cpu')
    parser.add_argument('--select-frames', type=int, default=12)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--save-json', type=str, default='/root/results/ucf_crime_results_ats_qwen3vl.json')
    parser.add_argument('--benchmark', action='store_true', help='Measure inference latency/FPS during evaluation.')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup runs per video for benchmark (excluded from stats).')
    parser.add_argument('--repeat', type=int, default=1, help='Timed runs per video for benchmark (mean used).')
    args = parser.parse_args()

    samples = load_ucf_test_list(args.test_list)
    if args.limit is not None:
        samples = samples[:max(1, args.limit)]

    annotations = parse_ucf_temporal_annotations(args.annotation_path)
    pipeline = build_pipeline(
        ats_ckpt=args.ats_ckpt,
        qwen_path=args.qwen_path,
        encoder_path=args.encoder_path,
        device=args.device,
        qwen_device=args.qwen_device,
        clip_device=args.clip_device,
    )

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    per_video_results = {}
    per_video_benchmark = {}
    all_video_latencies_s: List[float] = []
    all_video_fps: List[float] = []
    total_benchmark_frames = 0
    total_benchmark_latency_s = 0.0
    eval_t0 = time.perf_counter()

    for _, (src_path, label) in tqdm(enumerate(samples), total=len(samples)):
        video_path = infer_ucf_video_path(src_path, label, args.ucf_video_root)
        if not os.path.exists(video_path):
            print(f"[WARN] Missing video: {video_path} (from {src_path})")
            continue

        try:
            if args.benchmark:
                last_res = {}

                def _run_once():
                    nonlocal last_res
                    last_res = pipeline.run(
                        video_path=video_path,
                        select_frames=args.select_frames,
                        weight=args.weight,
                        window_size=args.window_size,
                    )
                    return last_res

                devices = [getattr(pipeline, 'device', None), getattr(pipeline, 'qwen_device', None)]
                timing = time_callable(_run_once, devices=devices, warmup=args.warmup, repeat=args.repeat)
                res = last_res
            else:
                res = pipeline.run(
                    video_path=video_path,
                    select_frames=args.select_frames,
                    weight=args.weight,
                    window_size=args.window_size,
                )
        except Exception as e:
            print(f"[ERROR] Failed {video_path}: {e}")
            continue

        # Use per-frame outputs aligned to the actual video length.
        y_pred = np.asarray(res.get('full_video_scores_frames', res['full_video_scores']), dtype=float)
        num_frames = int(np.asarray(res.get('num_frames', [len(y_pred)])).reshape(-1)[0])
        y_pred = y_pred[:num_frames]

        stem = _stem_from_path(video_path)
        if label and label.lower() == 'normal':
            segs = []
        else:
            segs = annotations.get(stem)
            if segs is None:
                print(f"[WARN] No annotation for {stem}; skipping")
                continue
        y_true = build_gt_vector(num_frames, segs)
        if len(y_true) != len(y_pred):
            L = min(len(y_true), len(y_pred))
            y_true = y_true[:L]
            y_pred = y_pred[:L]

        all_preds.append(y_pred)
        all_labels.append(y_true)

        vid_key = os.path.splitext(os.path.basename(res.get('video_path', video_path)))[0]
        per_video_results[vid_key] = {
            'auc': float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
        }

        if args.benchmark:
            # timing may not exist if args.benchmark is False
            mean_latency_s = float(timing.mean_s)  # type: ignore[name-defined]
            fps = float(num_frames / mean_latency_s) if mean_latency_s > 0 else 0.0
            per_video_benchmark[vid_key] = {
                'latency': timing.to_json(),  # type: ignore[name-defined]
                'num_frames': int(num_frames),
                'fps_effective': fps,
            }
            all_video_latencies_s.append(mean_latency_s)
            all_video_fps.append(fps)
            total_benchmark_frames += int(num_frames)
            total_benchmark_latency_s += float(mean_latency_s)

    if not all_preds:
        print('No results.')
        return

    y_pred_cat = np.concatenate(all_preds)
    y_true_cat = np.concatenate(all_labels)
    auc = roc_auc_score(y_true_cat, y_pred_cat)
    ap = average_precision_score(y_true_cat, y_pred_cat)

    eval_t1 = time.perf_counter()

    summary = {
        'auc': float(auc),
        'ap': float(ap),
        'num_videos': len(all_preds),
        'eval_wall_time_s': float(eval_t1 - eval_t0),
    }
    benchmark_summary = None
    if args.benchmark and all_video_latencies_s:
        # Aggregate per-video (not per-frame) latency/FPS.
        lat = np.asarray(all_video_latencies_s, dtype=float)
        fps = np.asarray(all_video_fps, dtype=float)
        benchmark_summary = {
            'videos_count': int(lat.shape[0]),
            'latency_s_mean': float(lat.mean()),
            'latency_s_p50': float(np.percentile(lat, 50)),
            'latency_s_p90': float(np.percentile(lat, 90)),
            'fps_effective_mean': float(fps.mean()) if fps.size else 0.0,
            'fps_effective_p50': float(np.percentile(fps, 50)) if fps.size else 0.0,
            # Overall throughput computed from totals (more stable than mean of ratios)
            'total_frames': int(total_benchmark_frames),
            'total_latency_s': float(total_benchmark_latency_s),
            'fps_effective_overall': float(total_benchmark_frames / total_benchmark_latency_s)
            if total_benchmark_latency_s > 0
            else 0.0,
        }
    overall_print = {'summary': summary}
    if args.benchmark:
        overall_print['benchmark_summary'] = benchmark_summary
    print(json.dumps(overall_print, indent=2))

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, 'w') as f:
        payload = {'summary': summary, 'per_video': per_video_results}
        if args.benchmark:
            payload['benchmark'] = {
                'summary': benchmark_summary,
                'per_video': per_video_benchmark,
                'warmup': int(args.warmup),
                'repeat': int(args.repeat),
            }
        json.dump(payload, f)


if __name__ == '__main__':
    main()
