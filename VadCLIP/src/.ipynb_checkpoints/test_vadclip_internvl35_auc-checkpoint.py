import os
import csv
import argparse
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure local imports work when running from repo root
try:
    from pipeline_vadclip_internvl35 import build_pipeline
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from pipeline_vadclip_internvl35 import build_pipeline


def load_test_list(csv_path: str) -> List[str]:
    """Load feature paths from a CSV list.

    Accepts either one-path-per-line or CSV with feature_path in the first column.
    """
    feats: List[str] = []
    with open(csv_path, 'r') as f:
        # Try CSV reader, fallback to raw lines if needed
        sample = f.read(4096)
        f.seek(0)
        # Heuristic: if there's a comma, assume CSV, else plain paths
        if ',' in sample:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                p = row[0].strip()
                if p:
                    feats.append(p)
        else:
            for line in f:
                p = line.strip()
                if p:
                    feats.append(p)
    return feats


def video_key_from_feature_path(feature_path: str) -> Tuple[str, str]:
    """Derive (category, stem) from feature path like .../Abuse/Abuse028_x264__5.npy."""
    base = os.path.basename(feature_path)
    stem = base.split('__')[0].replace('.npy', '')  # e.g. Abuse028_x264
    category = os.path.basename(os.path.dirname(feature_path))
    return category, stem


def load_gt_any(gt_path: str) -> Tuple[str, object]:
    """Load GT either as a concatenated frame vector (preferred per ucf_test.py)
    or as a mapping from video key to frame labels. Returns (mode, data) where
    mode is 'vector' or 'map'.
    """
    gt_raw = np.load(gt_path, allow_pickle=True)

    # Preferred: plain ndarray of frame-level labels concatenated over test list
    if isinstance(gt_raw, np.ndarray) and gt_raw.dtype != object and gt_raw.ndim == 1:
        return 'vector', gt_raw.astype(np.int64)

    # Map-like formats
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
        except Exception:
            pass

    if isinstance(gt_raw, np.ndarray) and hasattr(gt_raw, 'dtype') and gt_raw.dtype.names:
        names = list(gt_raw.dtype.names)
        key_field = None
        label_field = None
        for nm in names:
            if nm.lower() in ('video', 'video_name', 'name', 'file', 'key', 'vid'):
                key_field = nm
            if nm.lower() in ('labels', 'gt', 'y', 'target'):
                label_field = nm
        if key_field is None and names:
            key_field = names[0]
        if label_field is None and len(names) > 1:
            label_field = names[1]
        if key_field and label_field:
            for row in gt_raw:
                k = row[key_field]
                v = row[label_field]
                normalized[normalize_key(k)] = np.asarray(v).astype(np.int64)
            return 'map', normalized

    if isinstance(gt_raw, np.ndarray):
        for elem in gt_raw:
            if isinstance(elem, dict):
                k = None
                v = None
                for kk in ('key', 'video', 'name', 'file', 'vid'):
                    if kk in elem:
                        k = elem[kk]
                        break
                for vv in ('labels', 'gt', 'y', 'target'):
                    if vv in elem:
                        v = elem[vv]
                        break
                if k is not None and v is not None:
                    normalized[normalize_key(k)] = np.asarray(v).astype(np.int64)
                    continue
            if isinstance(elem, (list, tuple)) and len(elem) >= 2:
                k, v = elem[0], elem[1]
                normalized[normalize_key(k)] = np.asarray(v).astype(np.int64)
                continue
        if normalized:
            return 'map', normalized

    raise ValueError(f"Unsupported gt format at {gt_path}: {type(gt_raw)} (dtype={getattr(gt_raw, 'dtype', None)})")


def match_gt_for_video(gt_map: Dict[str, np.ndarray], feature_path: str, video_path: Optional[str]) -> Optional[np.ndarray]:
    """Try multiple key variants to locate GT for a video."""
    cat, stem = video_key_from_feature_path(feature_path)
    candidates = [
        stem,
        f"{stem}.mp4",
        os.path.splitext(os.path.basename(video_path))[0] if video_path else None,
        f"{cat}/{stem}",
    ]
    for key in candidates:
        if not key:
            continue
        k0 = os.path.splitext(os.path.basename(key))[0]
        if k0 in gt_map:
            return gt_map[k0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate VadCLIP + InternVL3.5 AUC on UCF-Crime')
    parser.add_argument('--test-list', type=str, default='/root/VadCLIP/list/ucf_CLIP_rgbtest.csv')
    parser.add_argument('--gt-path', type=str, default='/root/VadCLIP/list/gt_ucf.npy')
    parser.add_argument('--vadclip-ckpt', type=str, default='/root/VadCLIP/model_ucf.pth')
    parser.add_argument('--internvl-path', type=str, default='/root/autodl-tmp/InternVL3_5-8B')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--select-frames', type=int, default=12)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None, help='Process only first N items for quick tests')
    parser.add_argument('--save-json', type=str, default='/root/results/ucf_crime_results_internvl35.json')
    args = parser.parse_args()

    feats = load_test_list(args.test_list)
    if args.limit is not None:
        feats = feats[:max(1, args.limit)]

    gt_mode, gt_data = load_gt_any(args.gt_path)
    pipeline = build_pipeline(args.vadclip_ckpt, args.internvl_path, args.device)

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    per_video_results = {}
    gt_offset = 0  # for vector mode

    for idx, feat_path in enumerate(feats):
        if not os.path.exists(feat_path):
            print(f"[WARN] Feature file missing, skip: {feat_path}")
            continue
        print(f"[{idx+1}/{len(feats)}] Running: {feat_path}")
        try:
            res = pipeline.run(
                feature_path=feat_path,
                video_path=None,
                select_frames=args.select_frames,
                weight=args.weight,
                window_size=args.window_size,
            )
        except Exception as e:
            print(f"[ERROR] Inference failed for {feat_path}: {e}")
            continue

        y_pred = np.asarray(res['full_video_scores'], dtype=float)
        video_path = res.get('video_path')

        # Choose GT slice depending on mode
        if gt_mode == 'vector':
            # Slice contiguous segment per test list order (as in ucf_test.py)
            if gt_offset >= len(gt_data):
                print(f"[WARN] GT vector exhausted at video #{idx+1}; stopping accumulation.")
                break
            # Prefer predicted length; if remaining GT shorter, clip
            remaining = len(gt_data) - gt_offset
            L = min(len(y_pred), remaining)
            y_true = gt_data[gt_offset:gt_offset + L]
            gt_offset += L
            if len(y_pred) != L:
                print(f"[INFO] Length mismatch (vector): pred={len(y_pred)} gt_slice={L} -> use {L}")
            y_pred = y_pred[:L]
        else:
            # Map mode: match per video
            y_true = match_gt_for_video(gt_data, feat_path, video_path)
            if y_true is None:
                print(f"[WARN] No GT matched for {video_path or feat_path}; skipping from AUC")
                continue
            L = min(len(y_pred), len(y_true))
            if len(y_pred) != len(y_true):
                print(f"[INFO] Length mismatch (map) for {video_path}: pred={len(y_pred)} gt={len(y_true)} -> use {L}")
            y_pred = y_pred[:L]
            y_true = y_true[:L]

        all_preds.append(y_pred)
        all_labels.append(y_true)

        vid_key = os.path.splitext(os.path.basename(video_path if video_path else feat_path))[0]
        per_video_results[vid_key] = {
            'video_path': video_path,
            'feature_path': feat_path,
            'pred_len': int(len(y_pred)),
            'gt_len': int(len(y_true)),
            'pred_mean': float(float(np.mean(y_pred))) if len(y_pred) > 0 else 0.0,
            'anom_ratio': float(float(np.mean(y_true))) if len(y_true) > 0 else 0.0,
        }

    if not all_preds or not all_labels:
        print('[ERROR] No valid predictions/labels collected; AUC cannot be computed.')
        return

    y_pred_cat = np.concatenate(all_preds, axis=0)
    y_true_cat = np.concatenate(all_labels, axis=0)

    try:
        auc = roc_auc_score(y_true_cat, y_pred_cat)
    except Exception as e:
        print(f"[ERROR] AUC computation failed: {e}")
        return

    summary = {
        'auc': float(auc),
        'num_videos': int(len(all_preds)),
        'total_frames': int(len(y_true_cat)),
        'test_list': args.test_list,
        'gt_path': args.gt_path,
        'vadclip_ckpt': args.vadclip_ckpt,
        'internvl_path': args.internvl_path,
        'select_frames': int(args.select_frames),
        'weight': float(args.weight),
        'window_size': int(args.window_size),
    }

    print(json.dumps(summary, indent=2))

    # Save JSON
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, 'w') as f:
        json.dump({'summary': summary, 'per_video': per_video_results}, f)
    print(f"Saved results to {args.save_json}")


if __name__ == '__main__':
    main()
