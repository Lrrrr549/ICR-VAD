"""
Pipeline that fuses VadCLIP coarse scores with InternVL3.5-8B summaries.

Steps:
- Use pre-extracted VadCLIP features (npy, stride 16) to get logits1 (System-1).
- Density sample clips according to logits1.
- For sampled clips, call InternVL3.5-8B to summarize nearby frames.
- Encode summaries with the frozen CLIP text encoder and compute similarity (logits2).
- Fuse two score streams as HolmesVAUPlus does.
"""

import copy
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from scipy import interpolate
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from clip import clip as openai_clip

sys.path.append('/root/HolmesVAU')

from holmesvau.internvl_utils import build_transform, dynamic_preprocess  
from model import CLIPVAD  
from utils.tools import get_batch_mask, process_split  


class VadclipInternVL35Pipeline:
    def __init__(
        self,
        vadclip_ckpt: str,
        internvl_path: str = '/root/autodl-tmp/InternVL3_5-8B',
        device: str = 'cuda:0',
        visual_length: int = 256,
        visual_width: int = 512,
        prompt_prefix: int = 10,
        prompt_postfix: int = 10,
        visual_head: int = 1,
        visual_layers: int = 2,
        attn_window: int = 8,
        embed_dim: int = 512,
        classes_num: int = 14,
        frame_stride: int = 16,
        tau: float = 1e-3,
    ):
        self.device = torch.device(device)
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.frame_stride = frame_stride
        self.tau = tau

        if 'ucf' in vadclip_ckpt.lower():
            self.label_map = {
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
            print("Using UCF-Crime label map for Qwen3-VL.")
        else: 
            self.label_map = {
            'A': 'normal',
            'B1': 'fighting',
            'B2': 'shooting',
            'B4': 'riot',
            'B5': 'abuse',
            'B6': 'car accident',
            'G': 'explosion',   
            }
            print("Using XD label map for Qwen3-VL.")
            visual_layers = 1 
            attn_window = 64  
            classes_num = 7
        self.prompt_text = list(self.label_map.values())

        self.vad_model = CLIPVAD(
            classes_num,
            embed_dim,
            visual_length,
            visual_width,
            visual_head,
            visual_layers,
            attn_window,
            prompt_prefix,
            prompt_postfix,
            self.device,
        ).to(self.device)
        self.vad_model.load_state_dict(torch.load(vadclip_ckpt, map_location=self.device))
        self.vad_model.eval()

        print(f"Loading InternVL3.5-8B from {internvl_path}...")
        self.llm = AutoModel.from_pretrained(
            internvl_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

        self.input_size = 448
        self.max_num = 1
        self.transform = build_transform(input_size=self.input_size)
        self.generation_config = dict(max_new_tokens=256, do_sample=False, use_cache=True)

    def _build_lengths(self, clip_length: int) -> torch.Tensor:
        lengths = torch.zeros(int(clip_length / self.visual_length) + 1, dtype=torch.int)
        length_left = clip_length
        for j in range(len(lengths)):
            if j == 0 and length_left < self.visual_length:
                lengths[j] = length_left
            elif j == 0 and length_left > self.visual_length:
                lengths[j] = self.visual_length
                length_left -= self.visual_length
            elif length_left > self.visual_length:
                lengths[j] = self.visual_length
                length_left -= self.visual_length
            else:
                lengths[j] = length_left
        return lengths

    def _densities_sample(self, scores: np.ndarray, select_frames: int) -> List[int]:
        if len(scores) <= select_frames or float(scores.sum()) < 1.0:
            return self._evenly_spaced_indices(len(scores), select_frames)

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
    
    def _evenly_spaced_indices(self, length: int, select_frames: int) -> List[int]:
        if length <= 0 or select_frames <= 0:
            return []
        span = np.linspace(0, max(length - 1, 0), select_frames)
        return np.clip(np.rint(span).astype(int), 0, max(length - 1, 0)).tolist()

    def _uniform_sample(self, scores: np.ndarray, select_frames: int) -> List[int]:
        return self._evenly_spaced_indices(len(scores), select_frames)

    def _top_k_sample(self, scores: np.ndarray, select_frames: int) -> List[int]:
        length = len(scores)
        if length == 0 or select_frames <= 0:
            return []
        select_count = min(select_frames, length)
        if select_count == length:
            return list(range(length))
        topk_idx = np.argpartition(scores, -select_count)[-select_count:]
        return sorted(np.clip(topk_idx, 0, length - 1).tolist())

    def _get_pixel_values(self, vr: VideoReader, frame_indices: List[int]) -> torch.Tensor:
        pixel_values_list = []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img_tiles = dynamic_preprocess(img, image_size=self.input_size, use_thumbnail=True, max_num=self.max_num)
            pixel_values = [self.transform(tile) for tile in img_tiles]
            pixel_values_list.append(torch.stack(pixel_values))
        return torch.cat(pixel_values_list)

    def _generate_summary(self, pixel_values: torch.Tensor, num_frames: int) -> str:
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(num_frames)])
        anomal_labels = [label for label in self.label_map.values() if label.lower() != 'normal']
        # prompt = video_prefix + f"If there is any anomaly (like {anomal_labels}), describe the specific event concisely, else just describe it concisely."
        prompt = video_prefix + f"Is there any {anomal_labels} in video frames? Answer with anomal labels or 'No'"
        # prompt = video_prefix + f"Is there any anomaly in video frames? Answer with brief anomaly description"
        # Analyze the video frames. Determine if there is {anomal_labels}.
        # Return the result strictly in JSON format:
        # {"is_anomaly": true/false, "reason": "concise description"}
        # Do not output anything else.
        template = copy.deepcopy(self.llm.conv_template)
        template.system_message = self.llm.system_message
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
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)

        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        gen_config = self.generation_config.copy()
        gen_config['eos_token_id'] = eos_token_id
        if self.tokenizer.bos_token_id is not None:
            gen_config['bos_token_id'] = self.tokenizer.bos_token_id
        elif hasattr(self.llm.config, 'llm_config'):
            gen_config['bos_token_id'] = self.llm.config.llm_config.bos_token_id
        gen_config['pad_token_id'] = self.tokenizer.pad_token_id or eos_token_id

        output = self.llm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(**gen_config),
        )

        description = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        description = description.split(template.sep.strip())[0].strip()
        # print(description)
        return description

    def _compute_similarity_raw(self, visual_feat: torch.Tensor, description: str) -> float:
        tokens = openai_clip.tokenize([description], truncate=True).to(self.device)
        with torch.no_grad():
            word_embedding = self.vad_model.clipmodel.encode_token(tokens)
            text_features = self.vad_model.clipmodel.encode_text(word_embedding, tokens)

        visual_feat = visual_feat.unsqueeze(0)
        visual_norm = visual_feat / (visual_feat.norm(dim=-1, keepdim=True) + 1e-6)
        text_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        logit_scale = self.vad_model.clipmodel.logit_scale.exp()
        logits = (visual_norm @ text_norm.t())
        # logits = logit_scale * (visual_norm @ text_norm.t())
        return logits.squeeze().item()
    
    def _compute_score_from_category(self, output_category: str) -> float:
        """
        输入: InternVL 输出的类别字符串 (e.g., "Explosion" or "Normal")
        输出: 异常置信度 (0.0 ~ 1.0)
        """
        
        # 语义匹配 (Text-Text Similarity)
        # 如果模型输出了 "Fighting"，我们要看它和预定义的异常列表匹配度有多高
        
        # 1. Tokenize 输出的词
        token_output = openai_clip.tokenize([output_category], truncate=True).to(self.device)
        # 2. Tokenize 所有异常标签 (排除 Normal)
        anomaly_labels = [label for label in self.prompt_text if label != 'normal']
        token_labels = openai_clip.tokenize(anomaly_labels, truncate=True).to(self.device)

        with torch.no_grad():
            # 3. 编码
            word_embedding = self.vad_model.clipmodel.encode_token(token_output)
            feat_output = self.vad_model.clipmodel.encode_text(word_embedding, token_output)
            word_embedding_labels = self.vad_model.clipmodel.encode_token(token_labels)
            feat_labels = self.vad_model.clipmodel.encode_text(word_embedding_labels, token_labels)

            # 4. 归一化
            feat_output /= feat_output.norm(dim=-1, keepdim=True)
            feat_labels /= feat_labels.norm(dim=-1, keepdim=True)

            # 5. 计算相似度 [1, dim] @ [13, dim].T -> [1, 13]
            similarity = (feat_output @ feat_labels.t())
            
            # 6. 取最大值
            # 逻辑：只要输出的词和任何一个异常标签很像，就认为是异常
            score = similarity.max().item()
            
            # (可选) 阈值截断：如果最相似的标签相似度都很低(比如输出乱码)，置0
            if score < 0.3: 
                score = 0.0
                
        return score

    def run(
        self,
        feature_path: str,
        video_path: Optional[str] = None,
        select_frames: int = 12,
        sampling_method: str = 'density',
        weight: float = 0.5,
        window_size: int = 4,
    ) -> Dict[str, np.ndarray]:
        video_path = video_path or self._infer_video_path(feature_path)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        features = np.load(feature_path)
        clip_length = features.shape[0]
        raw_clip_features = torch.tensor(features, device=self.device, dtype=torch.float32)

        split_feat, clip_length = process_split(features, self.visual_length)
        visual = torch.tensor(split_feat, device=self.device, dtype=torch.float32)
        if visual.dim() == 2:  # when clip_length < visual_length, add batch dim
            visual = visual.unsqueeze(0)
        lengths = self._build_lengths(clip_length)
        padding_mask = get_batch_mask(lengths, self.visual_length).to(self.device)

        with torch.no_grad():
            visual_features = self.vad_model.encode_video(visual, padding_mask, lengths)
            logits1 = self.vad_model.classifier(visual_features + self.vad_model.mlp2(visual_features))

        logits1_flat = logits1.reshape(-1, logits1.shape[2])
        system1_scores = torch.sigmoid(logits1_flat[:clip_length].squeeze(-1)).cpu().numpy()
        
        # print(system1_scores)
        visual_features_flat = visual_features.reshape(-1, visual_features.shape[2])[:clip_length]
        dense_indices = [min(num_frames - 1, i * self.frame_stride) for i in range(clip_length)]

        # sampled_relative_idxs = self._densities_sample(system1_scores, select_frames)
        method = sampling_method.lower()
        if method not in {'density', 'uniform', 'topk'}:
            raise ValueError(f'Unsupported sampling method: {sampling_method}')

        if method == 'uniform':
            sampled_relative_idxs = self._uniform_sample(system1_scores, select_frames)
        elif method == 'topk':
            sampled_relative_idxs = self._top_k_sample(system1_scores, select_frames)
        else:
            sampled_relative_idxs = self._densities_sample(system1_scores, select_frames)

        sampled_frame_indices = [dense_indices[i] for i in sampled_relative_idxs]
        full_system2_scores = np.zeros(len(np.repeat(system1_scores, self.frame_stride)), dtype=np.float32)
        half_window = window_size // 2

        for rel_idx, frame_idx in zip(sampled_relative_idxs, sampled_frame_indices):
            frame_interval = 5
            # 以 frame_idx 为中心，前后各采样 half_window 个帧，间隔为 frame_interval
            window_indices = [frame_idx + i * frame_interval for i in range(-half_window, half_window + 1)]
            # 限制在视频范围内并去重排序
            window_indices = sorted(list(set([max(0, min(num_frames - 1, i)) for i in window_indices])))
            
            # 更新分数的覆盖范围为实际采样的物理跨度
            start_idx = window_indices[0]
            end_idx = window_indices[-1] + 1
            
            pixel_values = self._get_pixel_values(vr, window_indices)
            description = self._generate_summary(pixel_values, len(window_indices))
            # 使用正则表达式匹配独立的单词 "no", "none", "normal"，避免匹配到 "anomaly" 中的 "no"
            if re.search(r'\b(no|none|normal)\b', description.lower()):
                score = -0.15
                # print(f"Description indicates no anomaly: '{description}'. Assigned score: {score}")
            else:
                score = self._compute_score_from_category(description)
                # print(f"Computed score for description '{description}': {score}")
            # score = self._compute_similarity_raw(raw_clip_features[rel_idx], description)
            
            # 使用上面修改后的“Yes/No”概率逻辑
            # score = self._get_anomaly_probability(pixel_values, len(window_indices))
            
            # 【核心修改】：不要插值，直接把分数填入对应的窗口
            # 只有在这个窗口内，System 2 的结论才是可信的
            # 使用 np.maximum 确保如果有多个采样覆盖同一帧，保留最高的异常置信度
            full_system2_scores[start_idx:end_idx] = np.maximum(
                full_system2_scores[start_idx:end_idx], 
                score
            )
        # system2_scores = []
        # half_window = window_size // 2
        # for rel_idx, frame_idx in zip(sampled_relative_idxs, sampled_frame_indices):
        #     start_idx = max(0, frame_idx - half_window)
        #     end_idx = min(num_frames, frame_idx + half_window + 1)
        #     window_indices = list(range(start_idx, end_idx))
        #     pixel_values = self._get_pixel_values(vr, window_indices)
        #     description = self._generate_summary(pixel_values, len(window_indices))
        #     score = self._compute_similarity_raw(raw_clip_features[rel_idx], description)
        #     system2_scores.append(score)

        # system2_scores = np.array(system2_scores)
        # system2_scores = np.nan_to_num(system2_scores, nan=0.0)
        # print("system 2 scores: ", system2_scores)
        # f_interp = interpolate.interp1d(sampled_relative_idxs, system2_scores, kind='linear', fill_value='extrapolate')
        # system2_scores_interpolated = f_interp(np.arange(len(system1_scores)))
        # system2_scores_interpolated = np.nan_to_num(system2_scores_interpolated, nan=0.0)
        # print(system2_scores_interpolated)
        # print(len(system2_scores_interpolated))
        gate = (system1_scores > 0.3).astype(float)
        # final_scores = system1_scores + weight * (system2_scores_interpolated * gate)
        system1_scores = np.repeat(system1_scores, self.frame_stride)
        # print("len system1 scores:", len(system1_scores))
        # print("len system2 scores:", len(full_system2_scores))
        final_scores = system1_scores + weight * full_system2_scores
        # final_scores = system1_scores
        from scipy.ndimage import gaussian_filter1d
        final_scores = gaussian_filter1d(final_scores, sigma=2.0)
        # f_final = interpolate.interp1d(dense_indices, final_scores, kind='linear', fill_value='extrapolate')
        # full_video_scores = f_final(np.arange(num_frames))
        # full_video_scores = np.repeat(final_scores, self.frame_stride)
        full_video_scores = np.clip(final_scores, 0, 1)

        return {
            'full_video_scores': full_video_scores,
            'system1_scores': system1_scores,
            # 'system2_scores': system2_scores,
            # 'system2_scores_interpolated': system2_scores_interpolated,
            # 'sampled_indices': sampled_frame_indices,
            # 'dense_indices': dense_indices,
            'video_path': video_path,
            'feature_path': feature_path,
        }

    def _infer_video_path(self, feature_path: str) -> str:
        # Infer raw video path from feature npy path.
        base = os.path.basename(feature_path)
        if not base.endswith('.npy') or '__' not in base:
            raise ValueError('feature_path format unexpected, cannot infer video path')

        stem = base.split('__')[0]  # Abuse028_x264
        category = os.path.basename(os.path.dirname(feature_path))
        video_path = os.path.join('/root/autodl-tmp/ucf_crime/Anomaly-Videos', category, f'{stem}.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Inferred video path not found: {video_path}')
        return video_path


def build_pipeline(
    vadclip_ckpt: str = '/root/VadCLIP/model_ucf.pth',
    internvl_path: str = '/root/autodl-tmp/InternVL3_5-8B',
    device: str = 'cuda:0',
) -> VadclipInternVL35Pipeline:
    return VadclipInternVL35Pipeline(
        vadclip_ckpt=vadclip_ckpt,
        internvl_path=internvl_path,
        device=device,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VadCLIP + InternVL3.5 Pipeline')
    parser.add_argument('--feature-path', type=str, required=True, help='Path to VadCLIP npy feature file (stride 16).')
    parser.add_argument('--video-path', type=str, default=None, help='Path to raw video; if omitted, inferred from feature path.')
    parser.add_argument('--vadclip-ckpt', type=str, default='/root/VadCLIP/model_ucf.pth')
    parser.add_argument('--internvl-path', type=str, default='/root/autodl-tmp/InternVL3_5-8B')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--select-frames', type=int, default=12)
    parser.add_argument('--sampling-method', type=str, choices=['density', 'uniform', 'topk'], default='density')
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=5)
    args = parser.parse_args()

    pipeline = build_pipeline(args.vadclip_ckpt, args.internvl_path, args.device)
    results = pipeline.run(
        feature_path=args.feature_path,
        video_path=args.video_path,
        select_frames=args.select_frames,
        sampling_method=args.sampling_method,
        weight=args.weight,
        window_size=args.window_size,
    )
    print('Full video scores:', results['full_video_scores'])