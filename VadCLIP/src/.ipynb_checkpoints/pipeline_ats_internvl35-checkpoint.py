"""
Pipeline that fuses Holmes-VAU ATS (URDMU) coarse scores with InternVL3.5 summaries.

System-1: ATS anomaly scorer fed with CLS tokens from an InternVL-style encoder.
System-2: InternVL3.5 multi-modal LLM summarizes sampled frames; summary text is scored
           with CLIP text encoder against anomaly labels; scores are fused with System-1.

This mirrors `pipeline_ats_qwen3vl.py` but swaps Qwen3-VL for InternVL3.5 as the LLM.
"""

import os
import sys
import torch
import numpy as np
import argparse
import re
import copy
from typing import Dict, List, Optional
from PIL import Image
from decord import VideoReader, cpu
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from clip import clip as openai_clip

# Add paths for local imports
sys.path.append('/root/HolmesVAU')

from holmesvau.ATS.anomaly_scorer import URDMU  # noqa: E402
from holmesvau.internvl_utils import build_transform, dynamic_preprocess  # noqa: E402


class ATSInternVL35Pipeline:
    def __init__(
        self,
        ats_ckpt: str,
        encoder_path: str = '/root/autodl-tmp/InternVL2-2B',
        llm_path: str = '/root/autodl-tmp/InternVL3_5-2B',
        device: str = 'cuda:0',
        llm_device: Optional[str] = None,
        clip_device: str = 'cpu',
        frame_stride: int = 16,
        tau: float = 1e-3,
        clip_name: str = 'ViT-B/32',
        input_size: int = 448,
        max_num: int = 1,
        encoder_batch_size: int = 16,
    ):
        self.device = torch.device(device)
        self.llm_device = torch.device(llm_device or device)
        self.clip_device = torch.device(clip_device)
        self.frame_stride = frame_stride
        self.tau = float(tau)
        self.input_size = int(input_size)
        self.max_num = int(max_num)
        self.encoder_batch_size = int(encoder_batch_size)

        self.llm_path = llm_path

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

        self._set_label_map_from_hint(ats_ckpt)

        # Load ATS (URDMU)
        print(f"Loading Holmes-VAU ATS (URDMU) from {ats_ckpt}...")
        self.ats_model = URDMU().to(self.device)
        self.ats_model.load_state_dict(torch.load(ats_ckpt, map_location=self.device))
        self.ats_model.eval()

        try:
            self.ats_input_dim = int(self.ats_model.embedding.conv_1[0].in_channels)
        except Exception:
            self.ats_input_dim = 1024

        # CLIP (text encoder)
        print(f"Loading CLIP ({clip_name}) for text similarity...")
        self.clip_model, _ = openai_clip.load(clip_name, device=self.clip_device, jit=False)
        self.clip_model.eval()

        # Encoder for System-1 CLS tokens
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

        # InternVL3.5 LLM (lazy load)
        self.llm = None
        self.tokenizer = None

    def _ensure_llm_loaded(self) -> None:
        if self.llm is not None and self.tokenizer is not None:
            return
        print(f"Loading InternVL3.5 from {self.llm_path} to {self.llm_device}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(self.llm_device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)

    def _set_label_map_from_hint(self, hint: str) -> None:
        hint = (hint or '').lower()
        if 'ucf' in hint:
            self.label_map = self._ucf_label_map
        elif 'xd' in hint or 'violence' in hint:
            self.label_map = self._xd_label_map
        else:
            self.label_map = self._ucf_label_map
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
            pix = torch.stack(pix, dim=0)
            if pix.shape[0] != 1:
                pix = pix[:1]
            pixel_values_list.append(pix[0])
        return torch.stack(pixel_values_list, dim=0)

    @torch.no_grad()
    def _compute_system1_scores_from_video(self, vr: VideoReader) -> Dict[str, object]:
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

        cls_tokens_t = torch.cat(cls_tokens, dim=0).to(self.device)

        if cls_tokens_t.shape[-1] != self.ats_input_dim:
            if cls_tokens_t.shape[-1] < self.ats_input_dim:
                pad = self.ats_input_dim - cls_tokens_t.shape[-1]
                cls_tokens_t = torch.nn.functional.pad(cls_tokens_t, (0, pad), value=0.0)
            else:
                cls_tokens_t = cls_tokens_t[:, : self.ats_input_dim]

        vid_feats = cls_tokens_t.unsqueeze(0)
        system1_scores = self.ats_model(vid_feats)["anomaly_scores"][0].detach().cpu().numpy()
        return {"dense_indices": dense_indices, "system1_scores": system1_scores}

    def _generate_summary(self, pixel_values: torch.Tensor, num_frames: int) -> str:
        """Use InternVL3.5 generate flow with image tokens (mirrors pipeline_vadclip_internvl35)."""
        self._ensure_llm_loaded()
        pixel_values = pixel_values.to(torch.bfloat16).to(self.llm_device)

        anomal_labels = [label for label in self.label_map.values() if label.lower() != 'normal']
        prompt = ''.join([f'Frame{i + 1}: <image>\n' for i in range(num_frames)])
        prompt += f"Is there any {anomal_labels} in video frames? Answer with anomal labels or 'Normal'"
        # prompt += f"Answer with descriptions of the frames."

        try:
            template = copy.deepcopy(self.llm.conv_template)
            template.system_message = getattr(self.llm, 'system_message', '')
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            img_start = '<img>'
            img_end = '</img>'
            img_context = '<IMG_CONTEXT>'
            self.llm.img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context)
            for _ in range(num_frames):
                image_tokens = img_start + img_context * self.llm.num_image_token * self.max_num + img_end
                query = query.replace('<image>', image_tokens, 1)

            model_inputs = self.tokenizer(query, return_tensors='pt')
            input_ids = model_inputs['input_ids'].to(self.llm_device)
            attention_mask = model_inputs['attention_mask'].to(self.llm_device)

            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
            gen_config = {
                'max_new_tokens': 128,
                'do_sample': False,
                'use_cache': True,
                'eos_token_id': eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id or eos_token_id,
            }
            if self.tokenizer.bos_token_id is not None:
                gen_config['bos_token_id'] = self.tokenizer.bos_token_id
            elif hasattr(self.llm.config, 'llm_config'):
                gen_config['bos_token_id'] = getattr(self.llm.config.llm_config, 'bos_token_id', None)

            output = self.llm.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(**gen_config),
            )

            description = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            description = description.split(template.sep.strip())[0].strip()
            return description
        except Exception:
            # Last-resort fallback: text-only prompt
            try:
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.llm_device)
                out = self.llm.generate(**inputs, max_new_tokens=64)
                return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            except Exception:
                return ''

    def _compute_score_from_category(self, output_category: str) -> float:
        if not output_category:
            return 0.0
        token_output = openai_clip.tokenize([output_category], truncate=True).to(self.clip_device)
        anomaly_labels = [label for label in self.prompt_text if label.lower() != 'normal']
        token_labels = openai_clip.tokenize(anomaly_labels, truncate=True).to(self.clip_device)

        with torch.no_grad():
            if hasattr(self.clip_model, 'encode_token'):
                word_emb_out = self.clip_model.encode_token(token_output)
                feat_output = self.clip_model.encode_text(word_emb_out, token_output)
                word_emb_labels = self.clip_model.encode_token(token_labels)
                feat_labels = self.clip_model.encode_text(word_emb_labels, token_labels)
            else:
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
        if video_path is None:
            if not feature_path:
                raise ValueError('Either video_path or feature_path must be provided')
            self._set_label_map_from_hint(feature_path)
            video_path = self._infer_video_path(feature_path)
        else:
            self._set_label_map_from_hint(video_path)

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)

        sys1 = self._compute_system1_scores_from_video(vr)
        dense_indices = sys1["dense_indices"]
        system1_scores = sys1["system1_scores"]

        sampled_relative_idxs = self._densities_sample(system1_scores, select_frames)
        sampled_frame_indices = [dense_indices[i] for i in sampled_relative_idxs]

        system1_scores_repeated = np.repeat(system1_scores, self.frame_stride)
        full_len = int(len(system1_scores_repeated))

        full_system2_scores = np.zeros(full_len, dtype=np.float32)
        half_window = window_size // 2

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

            pixel_values = self._get_pixel_values(vr, window_indices)
            description = self._generate_summary(pixel_values, len(window_indices))
            # print(description)

            if re.search(r'\b(no|none|normal)\b', description.lower()):
                score = 0.0
            else:
                score = self._compute_score_from_category(description)

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
    llm_path: str = '/root/autodl-tmp/InternVL3_5-2B',
    encoder_path: str = '/root/autodl-tmp/InternVL2-2B',
    device: str = 'cuda:0',
    llm_device: Optional[str] = None,
    clip_device: str = 'cpu',
) -> ATSInternVL35Pipeline:
    return ATSInternVL35Pipeline(
        ats_ckpt=ats_ckpt,
        encoder_path=encoder_path,
        llm_path=llm_path,
        device=device,
        llm_device=llm_device,
        clip_device=clip_device,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ATS + InternVL3.5 Pipeline')
    parser.add_argument('--video-path', type=str, default=None, help='Path to raw video.')
    parser.add_argument('--feature-path', type=str, default=None, help='Optional: only to infer video path; npy will NOT be loaded.')
    parser.add_argument('--ats-ckpt', type=str, default='/root/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth')
    parser.add_argument('--encoder-path', type=str, default='/root/autodl-tmp/InternVL2-2B')
    parser.add_argument('--llm-path', type=str, default='/root/autodl-tmp/InternVL3_5-2B')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--llm-device', type=str, default=None)
    parser.add_argument('--select-frames', type=int, default=12)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--frame-stride', type=int, default=16)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--encoder-batch-size', type=int, default=16)
    args = parser.parse_args()

    pipeline = ATSInternVL35Pipeline(
        ats_ckpt=args.ats_ckpt,
        encoder_path=args.encoder_path,
        llm_path=args.llm_path,
        device=args.device,
        llm_device=args.llm_device,
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
