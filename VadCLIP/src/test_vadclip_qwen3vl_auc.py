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
    from pipeline_vadclip_qwen3vl import build_pipeline
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from pipeline_vadclip_qwen3vl import build_pipeline

sys.path.append(os.path.dirname(__file__))
from utils.benchmark import time_callable


def load_test_list(csv_path: str) -> List[str]:
    feats: List[str] = []
    with open(csv_path, 'r') as f:
        sample = f.read(4096)
        f.seek(0)
        if ',' in sample:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                p = row[0].strip()
                if p: feats.append(p)
        else:
            for line in f:
                p = line.strip()
                if p: feats.append(p)
    return feats


def load_gt_any(gt_path: str) -> Tuple[str, object]:
    gt_raw = np.load(gt_path, allow_pickle=True)
    if isinstance(gt_raw, np.ndarray) and gt_raw.dtype != object and gt_raw.ndim == 1:
        return 'vector', gt_raw.astype(np.int64)

    def normalize_key(k: str) -> str:
        return os.path.splitext(os.path.basename(str(k)))[0]

    normalized: Dict[str, np.ndarray] = {}
    if isinstance(gt_raw, dict):
        for k, v in gt_raw.items():
            normalized[normalize_key(k)] = np.asarray(v).astype(np.int64)
        return 'map', normalized

    if isinstance(gt_raw, np.ndarray) and gt_raw.shape == ():
        try:
            obj = gt_raw.item()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    normalized[normalize_key(k)] = np.asarray(v).astype(np.int64)
                return 'map', normalized
        except Exception: pass

    raise ValueError(f"Unsupported gt format at {gt_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VadCLIP + Qwen3-VL AUC on UCF-Crime')
    parser.add_argument('--test-list', type=str, default='/root/VadCLIP/list/ucf_CLIP_rgbtest.csv')
    parser.add_argument('--gt-path', type=str, default='/root/VadCLIP/list/gt_ucf.npy')
    parser.add_argument('--vadclip-ckpt', type=str, default='/root/VadCLIP/model_ucf.pth')
    parser.add_argument('--qwen-path', type=str, default='/root/autodl-tmp/Qwen3-VL-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--select-frames', type=int, default=12)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--method', action='store_true')
    parser.add_argument('--save-json', type=str, default='/root/results/ucf_crime_results_qwen3vl.json')
    parser.add_argument('--benchmark', action='store_true', help='Measure inference latency/throughput during evaluation.')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup runs per video for benchmark (excluded from stats).')
    parser.add_argument('--repeat', type=int, default=1, help='Timed runs per video for benchmark (mean used).')
    args = parser.parse_args()

    feats = load_test_list(args.test_list)
    if args.limit is not None:
        feats = feats[:max(1, args.limit)]

    gt_mode, gt_data = load_gt_any(args.gt_path)
    pipeline = build_pipeline(args.vadclip_ckpt, args.qwen_path, args.device)

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    per_video_results = {}
    per_video_benchmark = {}
    all_video_latencies_s: List[float] = []
    all_video_units_per_s: List[float] = []
    total_units = 0
    total_latency_s = 0.0
    gt_offset = 0
    eval_t0 = time.perf_counter()

    for idx, feat_path in tqdm(enumerate(feats)):
        if not os.path.exists(feat_path):
            print(f"[WARN] Missing: {feat_path}")
            continue
        # print(f"[{idx+1}/{len(feats)}] Running: {feat_path}")
        # try:
        #     res = pipeline.run(
        #         feature_path=feat_path,
        #         select_frames=args.select_frames,
        #         weight=args.weight,
        #         window_size=args.window_size,
        #     )
        # except Exception as e:
        #     print(f"[ERROR] Failed {feat_path}: {e}")
        #     continue

        if args.benchmark:
            last_res = {}

            def _run_once():
                nonlocal last_res
                last_res = pipeline.run(
                    feature_path=feat_path,
                    select_frames=args.select_frames,
                    weight=args.weight,
                    window_size=args.window_size,
                )
                return last_res

            devices = [getattr(pipeline, 'device', None)]
            timing = time_callable(_run_once, devices=devices, warmup=args.warmup, repeat=args.repeat)
            res = last_res
        else:
            res = pipeline.run(
                feature_path=feat_path,
                select_frames=args.select_frames,
                weight=args.weight,
                window_size=args.window_size,
            )
        y_pred = np.asarray(res['full_video_scores'], dtype=float)
        
        if gt_mode == 'vector':
            if gt_offset >= len(gt_data): break
            L = min(len(y_pred), len(gt_data) - gt_offset)
            y_true = gt_data[gt_offset:gt_offset + L]
            gt_offset += L
            y_pred = y_pred[:L]
        else:
            # Simplified map matching for brevity
            stem = os.path.splitext(os.path.basename(feat_path))[0].split('__')[0]
            y_true = gt_data.get(stem)
            if y_true is None: continue
            L = min(len(y_pred), len(y_true))
            y_pred, y_true = y_pred[:L], y_true[:L]

        all_preds.append(y_pred)
        all_labels.append(y_true)
        
        vid_key = os.path.splitext(os.path.basename(res.get('video_path', feat_path)))[0]
        per_video_results[vid_key] = {'auc': float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0}

        if args.benchmark:
            units = int(len(y_pred))
            mean_latency_s = float(timing.mean_s)  # type: ignore[name-defined]
            units_per_s = float(units / mean_latency_s) if mean_latency_s > 0 else 0.0
            per_video_benchmark[vid_key] = {
                'latency': timing.to_json(),  # type: ignore[name-defined]
                'pred_len': units,
                'units_per_s': units_per_s,
            }
            all_video_latencies_s.append(mean_latency_s)
            all_video_units_per_s.append(units_per_s)
            total_units += units
            total_latency_s += float(mean_latency_s)

    if not all_preds:
        print("No results.")
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
        lat = np.asarray(all_video_latencies_s, dtype=float)
        ups = np.asarray(all_video_units_per_s, dtype=float)
        benchmark_summary = {
            'videos_count': int(lat.shape[0]),
            'latency_s_mean': float(lat.mean()),
            'latency_s_p50': float(np.percentile(lat, 50)),
            'latency_s_p90': float(np.percentile(lat, 90)),
            'units_per_s_mean': float(ups.mean()) if ups.size else 0.0,
            'units_per_s_p50': float(np.percentile(ups, 50)) if ups.size else 0.0,
            'total_units': int(total_units),
            'total_latency_s': float(total_latency_s),
            'units_per_s_overall': float(total_units / total_latency_s) if total_latency_s > 0 else 0.0,
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
