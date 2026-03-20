"""
Counterfactual variant of the VadCLIP + Qwen3-VL pipeline.

Compared with `pipeline_vadclip_qwen3vl.py`, this file keeps System-1 unchanged
and replaces System-2 with structured counterfactual reasoning:

- observe the sampled window
- propose the strongest normal explanation
- list evidence that contradicts that normal explanation
- choose an anomaly label and estimate anomaly-vs-normal confidence

The counterfactual judgment is converted into a signed System-2 score:
- positive: supports anomaly
- negative: supports normal
"""

import argparse
import json
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from scipy.ndimage import gaussian_filter1d
from qwen_vl_utils import process_vision_info

from pipeline_vadclip_qwen3vl import VadclipQwen3VLPipeline
from utils.tools import get_batch_mask, process_split


class VadclipQwen3VLCounterfactualPipeline(VadclipQwen3VLPipeline):
    def __init__(
        self,
        vadclip_ckpt: str,
        qwen_path: str = '/root/autodl-tmp/Qwen3-VL-8B-Instruct',
        device: str = 'cuda:0',
        counterfactual_max_new_tokens: int = 256,
        contradiction_bonus_weight: float = 0.15,
    ):
        super().__init__(
            vadclip_ckpt=vadclip_ckpt,
            qwen_path=qwen_path,
            device=device,
        )
        self.counterfactual_max_new_tokens = int(counterfactual_max_new_tokens)
        self.contradiction_bonus_weight = float(contradiction_bonus_weight)
        self._normal_aliases = {'normal', 'no', 'none', 'no anomaly', 'no anomalies'}

    def _build_counterfactual_prompt(self) -> str:
        labels = ', '.join(self.prompt_text)
        anomaly_labels = [label for label in self.prompt_text if label.lower() != 'normal']
        anomaly_label_text = ', '.join(anomaly_labels)
        return (
            "You are verifying whether the video window contains an anomaly.\n"
            f"Allowed labels: {labels}.\n"
            f"Abnormal labels are: {anomaly_label_text}.\n"
            "Reason counterfactually:\n"
            "1. Briefly describe the visible event.\n"
            "2. Propose the strongest plausible normal explanation.\n"
            "3. List the visual evidence that contradicts the normal explanation.\n"
            "4. Choose exactly one label from the allowed labels.\n"
            "5. Estimate anomaly_confidence and normal_confidence in [0, 1].\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "observed_event": "short description",\n'
            '  "normal_hypothesis": "short description",\n'
            '  "normal_support": ["evidence 1", "evidence 2"],\n'
            '  "normal_contradictions": ["evidence 1", "evidence 2"],\n'
            f'  "anomaly_label": "one of [{labels}]",\n'
            '  "anomaly_confidence": 0.0,\n'
            '  "normal_confidence": 0.0\n'
            "}\n"
            "If the frames are normal, set anomaly_label to Normal."
        )

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, object]]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r'\{', text):
            start = match.start()
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _coerce_confidence(self, value: object, default: float) -> float:
        if isinstance(value, (int, float)):
            conf = float(value)
        elif isinstance(value, str):
            stripped = value.strip().rstrip('%')
            try:
                conf = float(stripped)
            except ValueError:
                conf = default
            else:
                if value.strip().endswith('%'):
                    conf /= 100.0
        else:
            conf = default
        return float(max(0.0, min(1.0, conf)))

    def _coerce_string_list(self, value: object) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in re.split(r'[;\n,]+', value) if part.strip()]
        return []

    def _normalize_label(self, label: object) -> str:
        text = str(label or '').strip()
        if not text:
            return 'Normal'

        lowered = text.lower()
        if lowered in self._normal_aliases or re.search(r'\b(no|none|normal)\b', lowered):
            return 'Normal'

        canon_map = {re.sub(r'[^a-z0-9]+', '', item.lower()): item for item in self.prompt_text}
        key = re.sub(r'[^a-z0-9]+', '', lowered)
        if key in canon_map:
            return canon_map[key]

        for item in self.prompt_text:
            item_norm = re.sub(r'[^a-z0-9]+', '', item.lower())
            if item_norm and (item_norm in key or key in item_norm):
                return item

        return text

    def _fallback_label_from_text(self, text: str) -> str:
        normalized_text = str(text or '').strip()
        if not normalized_text:
            return 'Normal'

        label = self._normalize_label(normalized_text)
        if label in self.prompt_text or label == 'Normal':
            return label

        lowered = normalized_text.lower()
        for item in self.prompt_text:
            if item.lower() in lowered:
                return item

        return 'Normal' if re.search(r'\b(no|none|normal)\b', lowered) else normalized_text

    def _parse_counterfactual_response(self, output_text: str) -> Dict[str, object]:
        parsed = self._extract_first_json_object(output_text)
        if not parsed:
            fallback_label = self._fallback_label_from_text(output_text)
            return {
                'observed_event': output_text.strip(),
                'normal_hypothesis': '',
                'normal_support': [],
                'normal_contradictions': [],
                'anomaly_label': fallback_label,
                'anomaly_confidence': 0.7 if fallback_label != 'Normal' else 0.0,
                'normal_confidence': 0.7 if fallback_label == 'Normal' else 0.3,
                'raw_response': output_text.strip(),
                'parse_ok': False,
            }

        anomaly_label = self._normalize_label(parsed.get('anomaly_label', 'Normal'))
        return {
            'observed_event': str(parsed.get('observed_event', '')).strip(),
            'normal_hypothesis': str(parsed.get('normal_hypothesis', '')).strip(),
            'normal_support': self._coerce_string_list(parsed.get('normal_support', [])),
            'normal_contradictions': self._coerce_string_list(parsed.get('normal_contradictions', [])),
            'anomaly_label': anomaly_label,
            'anomaly_confidence': self._coerce_confidence(parsed.get('anomaly_confidence', 0.0), 0.0),
            'normal_confidence': self._coerce_confidence(parsed.get('normal_confidence', 0.0), 0.0),
            'raw_response': output_text.strip(),
            'parse_ok': True,
        }

    def _generate_counterfactual_judgment(self, frames) -> Dict[str, object]:
        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": self._build_counterfactual_prompt()})

        messages = [{"role": "user", "content": content}]

        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(text=texts, images=images, videos=videos, do_resize=False, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.llm.generate(**inputs, max_new_tokens=self.counterfactual_max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        judgment = self._parse_counterfactual_response(output_text)
        judgment['raw_response'] = output_text
        return judgment

    def _compute_counterfactual_score(self, judgment: Dict[str, object]) -> float:
        anomaly_label = self._normalize_label(judgment.get('anomaly_label', 'Normal'))
        anomaly_conf = self._coerce_confidence(judgment.get('anomaly_confidence', 0.0), 0.0)
        normal_conf = self._coerce_confidence(judgment.get('normal_confidence', 0.0), 0.0)
        contradictions = self._coerce_string_list(judgment.get('normal_contradictions', []))
        contradiction_bonus = min(len(contradictions), 3) / 3.0 * self.contradiction_bonus_weight

        margin = anomaly_conf - normal_conf
        if anomaly_label == 'Normal' or margin <= 0.0:
            support_normal = max(normal_conf - anomaly_conf, 0.0)
            return float(-min(1.0, support_normal))

        label_score = self._compute_score_from_category(anomaly_label)
        score = label_score * max(0.0, margin)
        score += contradiction_bonus * max(anomaly_conf, 0.0)
        return float(min(1.0, score))

    def _write_signed_window_score(
        self,
        full_system2_scores: np.ndarray,
        start_idx: int,
        end_idx: int,
        score: float,
    ) -> None:
        current = full_system2_scores[start_idx:end_idx]
        if current.size == 0:
            return
        candidate = np.full_like(current, np.float32(score))
        replace_mask = np.abs(candidate) > np.abs(current)
        full_system2_scores[start_idx:end_idx] = np.where(replace_mask, candidate, current)

    def run(
        self,
        feature_path: str,
        video_path: Optional[str] = None,
        select_frames: int = 12,
        weight: float = 0.5,
        window_size: int = 5,
    ) -> Dict[str, np.ndarray]:
        video_path = video_path or self._infer_video_path(feature_path)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        features = np.load(feature_path)
        clip_length = features.shape[0]

        split_feat, clip_length = process_split(features, self.visual_length)
        visual = torch.tensor(split_feat, device=self.device, dtype=torch.float32)
        if visual.dim() == 2:
            visual = visual.unsqueeze(0)
        lengths = self._build_lengths(clip_length)
        padding_mask = get_batch_mask(lengths, self.visual_length).to(self.device)

        with torch.no_grad():
            visual_features = self.vad_model.encode_video(visual, padding_mask, lengths)
            logits1 = self.vad_model.classifier(visual_features + self.vad_model.mlp2(visual_features))

        logits1_flat = logits1.reshape(-1, logits1.shape[2])
        system1_scores = torch.sigmoid(logits1_flat[:clip_length].squeeze(-1)).cpu().numpy()
        system1_scores_repeated = np.repeat(system1_scores, self.frame_stride)

        if weight <= 0.0 or select_frames <= 0:
            final_scores = gaussian_filter1d(system1_scores_repeated, sigma=2.0)
            full_video_scores = np.clip(final_scores, 0, 1)
            return {
                'full_video_scores': full_video_scores,
                'system1_scores': system1_scores_repeated,
                'system2_scores': np.zeros_like(system1_scores_repeated, dtype=np.float32),
                'counterfactual_judgments': [],
                'video_path': video_path,
                'feature_path': feature_path,
            }

        dense_indices = [min(num_frames - 1, i * self.frame_stride) for i in range(clip_length)]
        sampled_relative_idxs = self._densities_sample(system1_scores, select_frames)
        sampled_frame_indices = [dense_indices[i] for i in sampled_relative_idxs]

        full_system2_scores = np.zeros(len(system1_scores_repeated), dtype=np.float32)
        half_window = window_size // 2
        counterfactual_judgments: List[Dict[str, object]] = []

        for rel_idx, frame_idx in zip(sampled_relative_idxs, sampled_frame_indices):
            frame_interval = 5
            window_indices = [frame_idx + i * frame_interval for i in range(-half_window, half_window + 1)]
            window_indices = sorted(list(set([max(0, min(num_frames - 1, i)) for i in window_indices])))

            start_idx = window_indices[0]
            end_idx = window_indices[-1] + 1

            frames = self._get_frames(vr, window_indices)
            judgment = self._generate_counterfactual_judgment(frames)
            score = self._compute_counterfactual_score(judgment)

            judgment['sampled_relative_idx'] = int(rel_idx)
            judgment['sampled_frame_idx'] = int(frame_idx)
            judgment['window_indices'] = window_indices
            judgment['system2_score'] = float(score)
            counterfactual_judgments.append(judgment)

            self._write_signed_window_score(full_system2_scores, start_idx, end_idx, score)

        final_scores = system1_scores_repeated + weight * full_system2_scores
        final_scores = gaussian_filter1d(final_scores, sigma=2.0)
        full_video_scores = np.clip(final_scores, 0, 1)

        return {
            'full_video_scores': full_video_scores,
            'system1_scores': system1_scores_repeated,
            'system2_scores': full_system2_scores,
            'counterfactual_judgments': counterfactual_judgments,
            'video_path': video_path,
            'feature_path': feature_path,
        }


def build_pipeline(
    vadclip_ckpt: str = '/root/VadCLIP/model_ucf.pth',
    qwen_path: str = '/root/autodl-tmp/Qwen3-VL-8B-Instruct',
    device: str = 'cuda:0',
) -> VadclipQwen3VLCounterfactualPipeline:
    return VadclipQwen3VLCounterfactualPipeline(
        vadclip_ckpt=vadclip_ckpt,
        qwen_path=qwen_path,
        device=device,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VadCLIP + Qwen3-VL Counterfactual Pipeline')
    parser.add_argument('--feature-path', type=str, required=True, help='Path to VadCLIP npy feature file.')
    parser.add_argument('--video-path', type=str, default=None, help='Path to raw video.')
    parser.add_argument('--vadclip-ckpt', type=str, default='/root/VadCLIP/model_ucf.pth')
    parser.add_argument('--qwen-path', type=str, default='/root/autodl-tmp/Qwen3-VL-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--select-frames', type=int, default=12)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=5)
    args = parser.parse_args()

    pipeline = build_pipeline(args.vadclip_ckpt, args.qwen_path, args.device)
    _ = pipeline.run(
        feature_path=args.feature_path,
        video_path=args.video_path,
        select_frames=args.select_frames,
        weight=args.weight,
        window_size=args.window_size,
    )
