"""
Pipeline that fuses Holmes-VAU ATS (URDMU) coarse scores with Qwen3-VL-8B-Instruct summaries.

This file is a drop-in variant of `pipeline_vadclip_qwen3vl.py` where:
- System-1 (VadCLIP) is replaced by Holmes-VAU ATS anomaly scorer (URDMU).
- All other logic (density sampling, Qwen summarization, CLIP text similarity scoring, fusion)
  is kept the same as the reference pipeline.

Expected `feature_path`:
- A .npy file containing dense temporal features of shape [T, D] (e.g., CLIP/ViT CLS tokens).
  ATS consumes these features and outputs an anomaly score curve of length T.

Notes:
- This pipeline keeps the same frame-indexing assumptions as the reference: `frame_stride`
  maps feature index i -> video frame i * frame_stride.
"""

import os
import sys
import torch
import numpy as np
import argparse
import re
from typing import Dict, List, Optional
from PIL import Image
from decord import VideoReader, cpu
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from transformers import AutoModel
try:
    # Newer transformers / Qwen-VL stacks
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:  # pragma: no cover
    # transformers==4.37.x fallback
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText, AutoProcessor
from clip import clip as openai_clip

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Minimal fallback to keep this pipeline runnable without the external qwen_vl_utils.
    # For our usage here (a single user message with multiple PIL images), we can pass
    # the extracted PIL images directly into the processor.
    def process_vision_info(messages, image_patch_size=16):
        images = []
        videos = []
        for msg in messages:
            for item in msg.get('content', []) or []:
                if item.get('type') == 'image' and 'image' in item:
                    images.append(item['image'])
                elif item.get('type') == 'video' and 'video' in item:
                    videos.append(item['video'])
        return images, videos

# Add paths for local imports
sys.path.append('/root/HolmesVAU')

from holmesvau.ATS.anomaly_scorer import URDMU  # noqa: E402
from holmesvau.internvl_utils import build_transform, dynamic_preprocess  # noqa: E402


class ATSQwen3VLPipeline:
    def __init__(
        self,
        ats_ckpt: str,
        encoder_path: str = '/root/autodl-tmp/InternVL2-2B',
        qwen_path: str = '/root/autodl-tmp/Qwen3-VL-8B-Instruct',
        device: str = 'cuda:0',
        qwen_device: Optional[str] = None,
        clip_device: str = 'cpu',
        frame_stride: int = 16,
        tau: float = 1e-3,
        clip_name: str = 'ViT-B/32',
        input_size: int = 448,
        max_num: int = 1,
        encoder_batch_size: int = 16,
    ):
        self.device = torch.device(device)
        self.qwen_device = torch.device(qwen_device or device)
        self.clip_device = torch.device(clip_device)
        self.frame_stride = frame_stride
        self.tau = float(tau)
        self.input_size = int(input_size)
        self.max_num = int(max_num)
        self.encoder_batch_size = int(encoder_batch_size)

        self.qwen_path = qwen_path

        # Dataset label maps (kept consistent with the reference pipeline)
        self._ucf_label_map = {
            'Normal': 'Normal',
            'Abuse': 'Abuse',
            'Arrest': 'Arrest',
            'Arson': 'Arson',
            'Assault': 'Assault',
            'Burglary': 'Burglary',
            'Explosion': 'Explosion',
            'Fighting': 'Fighting',
            'RoadAccidents': 'RoadAccidents',
            'Robbery': 'Robbery',
            'Shooting': 'Shooting',
            'Shoplifting': 'Shoplifting',
            'Stealing': 'Stealing',
            'Vandalism': 'Vandalism',
        }
        self._xd_label_map = {
            'A': 'normal',
            'B1': 'fighting',
            'B2': 'shooting',
            'B4': 'riot',
            'B5': 'abuse',
            'B6': 'car accident',
            'G': 'explosion',
        }

        # Initialize label_map; will be auto-adjusted per-run from feature_path.
        self._set_label_map_from_hint(ats_ckpt)

        # Load ATS (URDMU)
        print(f"Loading Holmes-VAU ATS (URDMU) from {ats_ckpt}...")
        self.ats_model = URDMU().to(self.device)
        self.ats_model.load_state_dict(torch.load(ats_ckpt, map_location=self.device))
        self.ats_model.eval()

        # The ATS scorer is trained for a fixed feature dimension (usually 1024 for InternVL CLS tokens).
        # Some feature dumps (e.g., CLIP ViT-B/32) are 512-d. We adapt by zero-padding / truncation.
        try:
            self.ats_input_dim = int(self.ats_model.embedding.conv_1[0].in_channels)
        except Exception:
            self.ats_input_dim = 1024

        # Load CLIP (text encoder)
        print(f"Loading CLIP ({clip_name}) for text similarity...")
        self.clip_model, _ = openai_clip.load(clip_name, device=self.clip_device, jit=False)
        self.clip_model.eval()

        # Load Encoder (must provide `vision_model(pixel_values=...)` like ATS code expects)
        print(f"Loading encoder (InternVL-style) from {encoder_path}...")
        self.encoder = AutoModel.from_pretrained(
            encoder_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(self.device)
        self.transform = build_transform(input_size=self.input_size)

        # Qwen3-VL is large; load lazily to avoid OOM when only System-1 is evaluated.
        self.llm = None
        self.processor = None

    def _ensure_qwen_loaded(self) -> None:
        if self.llm is not None and self.processor is not None:
            return
        print(f"Loading Qwen3-VL from {self.qwen_path} to {self.qwen_device}...")
        self.llm = AutoModelForImageTextToText.from_pretrained(
            self.qwen_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(self.qwen_device)
        self.processor = AutoProcessor.from_pretrained(self.qwen_path, trust_remote_code=True)

    def _set_label_map_from_hint(self, hint: str) -> None:
        hint = (hint or '').lower()
        if 'ucf' in hint:
            self.label_map = self._ucf_label_map
            # print("Using UCF-Crime label map for Qwen3-VL.")
        elif 'xd' in hint or 'violence' in hint:
            self.label_map = self._xd_label_map
            # print("Using XD label map for Qwen3-VL.")
        else:
            # Default to UCF (richer label space) when hint is ambiguous.
            self.label_map = self._ucf_label_map
            # print("Using UCF-Crime label map for Qwen3-VL (default).")

        self.prompt_text = list(self.label_map.values())

    def _densities_sample(self, scores: np.ndarray, select_frames: int) -> List[int]:
        if len(scores) <= select_frames or float(scores.sum()) < 1.0:
            return np.rint(np.linspace(0, len(scores) - 1, select_frames)).astype(int).tolist()

        adjusted = scores + self.tau
        score_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(adjusted)), axis=0)
        max_score_cumsum = np.round(score_cumsum[-1]).astype(int)
        f_upsample = interpolate.interp1d(
            score_cumsum,
            np.arange(len(scores) + 1),
            kind='linear',
            axis=0,
            fill_value='extrapolate',
        )
        scale_x = np.linspace(1, max_score_cumsum, select_frames)
        sampled = f_upsample(scale_x)
        sampled = [min(len(scores) - 1, max(0, int(idx))) for idx in sampled]
        return sampled

    def _get_frames(self, vr: VideoReader, frame_indices: List[int]) -> List[Image.Image]:
        frames = []
        for idx in frame_indices:
            frames.append(Image.fromarray(vr[idx].asnumpy()).convert('RGB'))
        return frames

    def _get_pixel_values(self, vr: VideoReader, frame_indices: List[int]) -> torch.Tensor:
        """Build InternVL-style pixel_values tensor of shape [N, 3, H, W]."""
        pixel_values_list: List[torch.Tensor] = []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            tiles = dynamic_preprocess(
                img,
                image_size=self.input_size,
                use_thumbnail=True,
                max_num=self.max_num,
            )
            pix = [self.transform(tile) for tile in tiles]
            pix = torch.stack(pix, dim=0)  # [num_patches, 3, H, W]
            # ATS scorer path expects one patch per frame (as used in HolmesVAU utils).
            if pix.shape[0] != 1:
                pix = pix[:1]
            pixel_values_list.append(pix[0])
        return torch.stack(pixel_values_list, dim=0)

    @torch.no_grad()
    def _compute_system1_scores_from_video(self, vr: VideoReader) -> Dict[str, object]:
        """Compute System-1 scores using encoder -> CLS tokens -> ATS anomaly scorer.

        Returns:
          - dense_indices: List[int]
          - system1_scores: np.ndarray of shape [N]
        """
        num_frames = len(vr)
        dense_indices = list(range(num_frames))[:: self.frame_stride]
        if not dense_indices:
            dense_indices = [0]

        pixel_values = self._get_pixel_values(vr, dense_indices)

        cls_tokens: List[torch.Tensor] = []
        bs = max(1, self.encoder_batch_size)
        for i in range(0, pixel_values.shape[0], bs):
            batch = pixel_values[i : i + bs].to(torch.bfloat16).to(self.device)
            vit_out = self.encoder.vision_model(
                pixel_values=batch,
                output_hidden_states=False,
                return_dict=True,
            )
            cls_tokens.append(vit_out.last_hidden_state[:, 0, :].to(torch.float32).cpu())

        cls_tokens_t = torch.cat(cls_tokens, dim=0).to(self.device)  # [N, D]

        # Adapt dim if needed (should usually already match ATS input dim)
        if cls_tokens_t.shape[-1] != self.ats_input_dim:
            if cls_tokens_t.shape[-1] < self.ats_input_dim:
                pad = self.ats_input_dim - cls_tokens_t.shape[-1]
                cls_tokens_t = torch.nn.functional.pad(cls_tokens_t, (0, pad), value=0.0)
            else:
                cls_tokens_t = cls_tokens_t[:, : self.ats_input_dim]

        vid_feats = cls_tokens_t.unsqueeze(0)  # [1, N, D]
        system1_scores = self.ats_model(vid_feats)["anomaly_scores"][0].detach().cpu().numpy()
        return {"dense_indices": dense_indices, "system1_scores": system1_scores}

    def _generate_summary(self, frames: List[Image.Image]) -> str:
        self._ensure_qwen_loaded()
        content = []
        for img in frames:
            content.append({"type": "image", "image": img})

        anomal_labels = [label for label in self.label_map.values() if label.lower() != 'normal']
        prompt = f"Is there any {anomal_labels} in video frames? Answer with anomal labels or 'No'"
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(text=texts, images=images, videos=videos, do_resize=False, return_tensors="pt")
        inputs = inputs.to(self.qwen_device)
        with torch.no_grad():
            generated_ids = self.llm.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def _compute_score_from_category(self, output_category: str) -> float:
        """Compute anomaly confidence (0~1) from Qwen's short category-like output."""
        if not output_category:
            return 0.0

        token_output = openai_clip.tokenize([output_category], truncate=True).to(self.clip_device)
        anomaly_labels = [label for label in self.prompt_text if label.lower() != 'normal']
        token_labels = openai_clip.tokenize(anomaly_labels, truncate=True).to(self.clip_device)

        with torch.no_grad():
            # This repo's CLIP implementation expects encode_text(word_embedding, token).
            # Keep a fallback for the vanilla OpenAI CLIP signature encode_text(token).
            if hasattr(self.clip_model, 'encode_token'):
                word_emb_out = self.clip_model.encode_token(token_output)
                feat_output = self.clip_model.encode_text(word_emb_out, token_output)
                word_emb_labels = self.clip_model.encode_token(token_labels)
                feat_labels = self.clip_model.encode_text(word_emb_labels, token_labels)
            else:  # pragma: no cover
                feat_output = self.clip_model.encode_text(token_output)
                feat_labels = self.clip_model.encode_text(token_labels)

            feat_output = feat_output / (feat_output.norm(dim=-1, keepdim=True) + 1e-6)
            feat_labels = feat_labels / (feat_labels.norm(dim=-1, keepdim=True) + 1e-6)

            similarity = feat_output @ feat_labels.t()
            score = similarity.max().item()

        if score < 0.3:
            score = 0.0
        return float(score)

    def run(
        self,
        video_path: Optional[str] = None,
        feature_path: Optional[str] = None,
        select_frames: int = 12,
        weight: float = 0.5,
        window_size: int = 5,
    ) -> Dict[str, np.ndarray]:
        # Allow passing `feature_path` only to infer the raw video path.
        if video_path is None:
            if not feature_path:
                raise ValueError('Either video_path or feature_path must be provided')
            # Align label map with dataset implied by feature_path.
            self._set_label_map_from_hint(feature_path)
            video_path = self._infer_video_path(feature_path)
        else:
            # Align label map with dataset implied by video_path.
            self._set_label_map_from_hint(video_path)

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)

        # System-1: encoder -> CLS tokens -> ATS
        sys1 = self._compute_system1_scores_from_video(vr)
        dense_indices = sys1["dense_indices"]
        system1_scores = sys1["system1_scores"]

        sampled_relative_idxs = self._densities_sample(system1_scores, select_frames)
        sampled_frame_indices = [dense_indices[i] for i in sampled_relative_idxs]

        # Keep the original behavior: repeat coarse scores by stride (16).
        system1_scores_repeated = np.repeat(system1_scores, self.frame_stride)
        full_len = int(len(system1_scores_repeated))

        full_system2_scores = np.zeros(full_len, dtype=np.float32)
        half_window = window_size // 2

        # If System-2 has zero contribution, skip loading / running Qwen entirely.
        if weight <= 0.0 or select_frames <= 0:
            final_scores = system1_scores_repeated
            final_scores = gaussian_filter1d(final_scores, sigma=2.0)
            full_video_scores = np.clip(final_scores, 0, 1)
            return {
                'full_video_scores': full_video_scores,
                'full_video_scores_frames': full_video_scores[:num_frames],
                'system1_scores': system1_scores_repeated,
                'system1_scores_frames': system1_scores_repeated[:num_frames],
                'num_frames': np.asarray([num_frames], dtype=np.int64),
                'video_path': video_path,
                'feature_path': feature_path,
            }

        for rel_idx, frame_idx in zip(sampled_relative_idxs, sampled_frame_indices):
            frame_interval = 5
            window_indices = [frame_idx + i * frame_interval for i in range(-half_window, half_window + 1)]
            window_indices = sorted(list(set([max(0, min(num_frames - 1, i)) for i in window_indices])))

            frames = self._get_frames(vr, window_indices)
            description = self._generate_summary(frames)

            if re.search(r'\b(no|none|normal)\b', description.lower()):
                score = 0.0
            else:
                score = self._compute_score_from_category(description)

            # Write System-2 scores in the same (stride-repeated) timeline.
            # We map real frame indices to this timeline by clamping.
            start_idx = max(0, min(full_len - 1, window_indices[0]))
            end_idx = max(start_idx + 1, min(full_len, window_indices[-1] + 1))
            full_system2_scores[start_idx:end_idx] = np.maximum(full_system2_scores[start_idx:end_idx], score)

        final_scores = system1_scores_repeated + weight * full_system2_scores
        final_scores = gaussian_filter1d(final_scores, sigma=2.0)
        full_video_scores = np.clip(final_scores, 0, 1)

        return {
            'full_video_scores': full_video_scores,
            'full_video_scores_frames': full_video_scores[:num_frames],
            'system1_scores': system1_scores_repeated,
            'system1_scores_frames': system1_scores_repeated[:num_frames],
            'system2_scores': full_system2_scores,
            'system2_scores_frames': full_system2_scores[:num_frames],
            'num_frames': np.asarray([num_frames], dtype=np.int64),
            'video_path': video_path,
            'feature_path': feature_path,
        }

    def _infer_video_path(self, feature_path: str) -> str:
        if 'ucf' in feature_path.lower():
            base = os.path.basename(feature_path)
            if not base.endswith('.npy') or '__' not in base:
                raise ValueError('feature_path format unexpected, cannot infer video path')

            stem = base.split('__')[0]
            category = os.path.basename(os.path.dirname(feature_path))
            root_dir = '/root/autodl-tmp/ucf_crime/Anomaly-Videos'
            if category.lower() == 'normal':
                normal_dir = 'Testing_Normal_Videos_Anomaly'
                video_path = os.path.join(root_dir, normal_dir, f'{stem}.mp4')
            else:
                video_path = os.path.join(root_dir, category, f'{stem}.mp4')
            if not os.path.exists(video_path):
                raise FileNotFoundError(f'Inferred video path not found: {video_path}')
        else:
            base = os.path.basename(feature_path)
            if not base.endswith('.npy'):
                raise ValueError('feature_path format unexpected, cannot infer video path')
            stem = base.split('/')[-1].split('__0')[0]
            video_path = os.path.join('/root/autodl-tmp/xd-violence/data/video/test_videos', f'{stem}.mp4')
            if not os.path.exists(video_path):
                raise FileNotFoundError(f'Inferred video path not found: {video_path}')
        return video_path


def build_pipeline(
    ats_ckpt: str = '/root/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth',
    qwen_path: str = '/root/autodl-tmp/Qwen3-VL-8B-Instruct',
    encoder_path: str = '/root/autodl-tmp/InternVL2-2B',
    device: str = 'cuda:0',
    qwen_device: Optional[str] = None,
    clip_device: str = 'cpu',
) -> ATSQwen3VLPipeline:
    return ATSQwen3VLPipeline(
        ats_ckpt=ats_ckpt,
        encoder_path=encoder_path,
        qwen_path=qwen_path,
        device=device,
        qwen_device=qwen_device,
        clip_device=clip_device,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ATS (Holmes-VAU) + Qwen3-VL Pipeline')
    parser.add_argument('--video-path', type=str, default=None, help='Path to raw video.')
    parser.add_argument('--feature-path', type=str, default=None, help='Optional: only used to infer video path, npy will NOT be loaded.')
    parser.add_argument('--ats-ckpt', type=str, default='/root/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth')
    parser.add_argument('--encoder-path', type=str, default='/root/autodl-tmp/InternVL2-2B')
    parser.add_argument('--qwen-path', type=str, default='/root/autodl-tmp/Qwen3-VL-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--select-frames', type=int, default=12)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--frame-stride', type=int, default=16)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--encoder-batch-size', type=int, default=16)
    args = parser.parse_args()

    pipeline = ATSQwen3VLPipeline(
        ats_ckpt=args.ats_ckpt,
        encoder_path=args.encoder_path,
        qwen_path=args.qwen_path,
        device=args.device,
        frame_stride=args.frame_stride,
        tau=args.tau,
        encoder_batch_size=args.encoder_batch_size,
    )

    _ = pipeline.run(
        video_path=args.video_path,
        feature_path=args.feature_path,
        select_frames=args.select_frames,
        weight=args.weight,
        window_size=args.window_size,
    )
