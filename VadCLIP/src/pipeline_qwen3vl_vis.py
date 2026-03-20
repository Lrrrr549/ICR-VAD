"""
Pipeline that fuses VadCLIP coarse scores with Qwen3-VL-8B-Instruct summaries.

Steps:
- Use pre-extracted VadCLIP features (npy, stride 16) to get logits1 (System-1).
- Density sample clips according to logits1.
- For sampled clips, call Qwen3-VL-8B to summarize nearby frames.
- Encode summaries with the frozen CLIP text encoder and compute similarity (logits2).
- Fuse two score streams as HolmesVAUPlus does.
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
import torchvision.transforms as T
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:  # transformers 4.37 fallback
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText, AutoProcessor
from clip import clip as openai_clip
try:
    from qwen_vl_utils import process_vision_info
except ModuleNotFoundError:  # minimal fallback
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

from model import CLIPVAD  # noqa: E402
from utils.tools import get_batch_mask, process_split  # noqa: E402


class VadclipQwen3VLPipeline:
    def __init__(
        self,
        vadclip_ckpt: str,
        qwen_path: str = '/root/autodl-tmp/Qwen3-VL-8B-Instruct',
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

        # Load VadCLIP
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

        # Load Qwen3-VL
        print(f"Loading Qwen3-VL-8B from {qwen_path}...")
        self.llm = AutoModelForImageTextToText.from_pretrained(
            qwen_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            # attn_implementation="flash_attention_2",
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)

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

    def _generate_summary(self, frames: List[Image.Image]) -> str:
        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
        
        # prompt = "Is there any anomaly in video frames? Answer with brief anomaly description"
        anomal_labels = [label for label in self.label_map.values() if label.lower() != 'normal']
        prompt = f"Is there any {anomal_labels} in video frames? Answer with brief anomal description or 'No'"
        # prompt = f"If there is any anomaly (like {anomal_labels}), describe the specific event concisely, else just describe it concisely."
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, image_patch_size=16)
        # images, videos = process_vision_info(messages)
        # inputs = self.processor(text=[texts], images=images, videos=videos, do_resize=False, return_tensors="pt", padding=True)
        inputs = self.processor(text=texts, images=images, videos=videos, do_resize=False, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            generated_ids = self.llm.generate(**inputs, max_new_tokens=128)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        description = output_text.strip()
        return description

    def _clip_safe_text(self, text: str, max_words: int = 60) -> str:
        """Trim text to be safe for CLIP's 77-token context."""
        if not text:
            return "Normal"
        parts = re.findall(r"\w+|[^\w\s]", text)
        if len(parts) > max_words:
            parts = parts[:max_words]
        return " ".join(parts)

    def _encode_frame_feature(self, frame: Image.Image) -> torch.Tensor:
        """Extract CLIP image feature from a PIL frame using the VadCLIP clipmodel."""
        preprocess = T.Compose(
            [
                T.Resize(self.vad_model.clipmodel.visual.input_resolution, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.vad_model.clipmodel.visual.input_resolution),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        img = preprocess(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.vad_model.clipmodel.encode_image(img)
        return feat.squeeze(0)

    def _compute_similarity_raw(self, frame: Image.Image, description: str) -> float:
        """Compute similarity between CLIP image embedding of frame and text description."""
        desc = self._clip_safe_text(description, max_words=60)
        try:
            tokens = openai_clip.tokenize([desc], truncate=True).to(self.device)
        except TypeError:
            tokens = openai_clip.tokenize([desc]).to(self.device)

        with torch.no_grad():
            word_embedding = self.vad_model.clipmodel.encode_token(tokens)
            text_features = self.vad_model.clipmodel.encode_text(word_embedding, tokens)

        visual_feat = self._encode_frame_feature(frame)
        visual_norm = visual_feat / (visual_feat.norm(dim=-1, keepdim=True) + 1e-6)
        text_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        logits = (visual_norm @ text_norm.t())
        return logits.squeeze().item()

    def _compute_score_from_category(self, output_category: str) -> float:
        """
        输入: Qwen 输出的类别字符串 (e.g., "Explosion" or "Normal")
        输出: 异常置信度 (0.0 ~ 1.0)
        """
        # 1. Tokenize 输出的词
        token_output = openai_clip.tokenize([output_category], truncate=True).to(self.device)
        # 2. Tokenize 所有异常标签 (排除 Normal)
        anomaly_labels = [label for label in self.prompt_text if label.lower() != 'normal']
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
            score = similarity.max().item()
            
            if score < 0.3: 
                score = 0.0
                
        return score

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
        raw_clip_features = torch.tensor(features, device=self.device, dtype=torch.float32)

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
        
        dense_indices = [min(num_frames - 1, i * self.frame_stride) for i in range(clip_length)]

        sampled_relative_idxs = self._densities_sample(system1_scores, select_frames)
        sampled_frame_indices = [dense_indices[i] for i in sampled_relative_idxs]

        full_system2_scores = np.zeros(len(np.repeat(system1_scores, self.frame_stride)), dtype=np.float32)
        half_window = window_size // 2

        window_centers: List[int] = []
        window_center_scores: List[float] = []
        system1_scores_repeated = np.repeat(system1_scores, self.frame_stride)

        for rel_idx, frame_idx in zip(sampled_relative_idxs, sampled_frame_indices):
            frame_interval = 5
            window_indices = [frame_idx + i * frame_interval for i in range(-half_window, half_window + 1)]
            window_indices = sorted(list(set([max(0, min(num_frames - 1, i)) for i in window_indices])))
            
            start_idx = window_indices[0]
            end_idx = window_indices[-1] + 1
            
            frames = self._get_frames(vr, window_indices)
            description = self._generate_summary(frames)

            # center frame for similarity
            center_frame = frames[len(frames) // 2]
            if re.search(r'\b(no|none|normal)\b', description.lower()):
                # 对应区间的 score2 取 -0.7 * score1baseline
                score_slice = system1_scores_repeated[start_idx:end_idx]
                score_block = -0.4 * score_slice
                full_system2_scores[start_idx:end_idx] = score_block
                score = float(score_block.mean()) if len(score_block) else 0.0
                print(f"Detected 'no anomaly'; assigning score2 = -0.1 * score1 in window (mean {score:.3f})")
            else:
                score = self._compute_similarity_raw(center_frame, description)
                full_system2_scores[start_idx:end_idx] = score + 0.4
                print(f"Computed score for description '{description}': {score}")

            window_centers.append(frame_idx)
            window_center_scores.append(score)

        final_scores = system1_scores_repeated + weight * full_system2_scores
        final_scores = gaussian_filter1d(final_scores, sigma=2.0)
        full_video_scores = np.clip(final_scores, 0, 1)

        return {
            'full_video_scores': full_video_scores,
            'system1_scores': system1_scores_repeated,
            'video_path': video_path,
            'feature_path': feature_path,
            "system2_scores": full_system2_scores,
            "window_centers": window_centers,
            "window_center_scores": window_center_scores,
        }

    def _infer_video_path(self, feature_path: str) -> str:
        if 'ucf' in feature_path.lower():
            base = os.path.basename(feature_path)
            if not base.endswith('.npy') or '__' not in base:
                raise ValueError('feature_path format unexpected, cannot infer video path')

            stem = base.split('__')[0]
            category = os.path.basename(os.path.dirname(feature_path))
            video_path = os.path.join('/root/autodl-tmp/ucf_crime/Anomaly-Videos', category, f'{stem}.mp4')
            if not os.path.exists(video_path):
                raise FileNotFoundError(f'Inferred video path not found: {video_path}')
        else:
            # /root/autodl-tmp/XD-Violence/XDTestClipFeatures/Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6__0.npy,G-B2-B6
            base = os.path.basename(feature_path)
            if not base.endswith('.npy'):
                # print(f'feature_path: {base}')
                raise ValueError('feature_path format unexpected, cannot infer video path')
            stem = base.split('/')[-1].split('__0')[0]
            video_path = os.path.join('/root/autodl-tmp/xd-violence/data/video/test_videos', f'{stem}.mp4')
            if not os.path.exists(video_path):    
                raise FileNotFoundError(f'Inferred video path not found: {video_path}')
        return video_path


def build_pipeline(
    vadclip_ckpt: str = '/root/VadCLIP/model_ucf.pth',
    qwen_path: str = '/root/autodl-tmp/Qwen3-VL-8B-Instruct',
    device: str = 'cuda:0',
) -> VadclipQwen3VLPipeline:
    return VadclipQwen3VLPipeline(
        vadclip_ckpt=vadclip_ckpt,
        qwen_path=qwen_path,
        device=device,
    )

import numpy as np
import matplotlib.pyplot as plt

def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    pad_len = target_len - len(arr)
    return np.concatenate([arr, np.full((pad_len,), arr[-1] if len(arr) > 0 else 0.0, dtype=arr.dtype)])


def plot_score_overlay(score1, score2, save_path="scores_overlay.png", video_len: int = None, marker_indices: Optional[List[int]] = None):
    """
    score1: baseline system1 scores (per frame)
    score2: system2 incremental scores (per frame), can be positive or negative
    video_len: optional, pad/truncate curves to cover full video length
    """
    # font size
    plt.rcParams.update({'font.size': 18})
    s1 = np.asarray(score1, dtype=float)
    s2 = np.asarray(score2, dtype=float)
    target_len = max(len(s1), len(s2))
    if video_len is not None:
        target_len = max(target_len, int(video_len))
    s1 = _pad_to_length(s1, target_len)
    s2 = _pad_to_length(s2, target_len)
    s2 = np.clip(s2, -1.0, 1.0) * 0.5
    x = np.arange(target_len)
    plt.figure(figsize=(15, 4))
    s1_plus_s2 = np.clip(s1 + s2, 0, 1.0)
    # Baseline
    plt.plot(x, s1, color="black", linestyle="--", label="Score1", linewidth=2)
    # Score2 alone (shows正负)
    plt.plot(x, s2, color="red",  label="Score2", linewidth=2)
    # Fused
    plt.plot(x, s1_plus_s2, color="blue", label="Final Score", linewidth=2)
    plt.xlabel("Frame Number")
    plt.ylabel("Score Value")
    plt.ylim(bottom=min(0.0, s2.min()-0.1 if len(s2) else 0), top=s1_plus_s2.max()+0.1)
    if marker_indices:
        xs = [int(i)*16 for i in marker_indices]
        ys = [0.0 for _ in xs]
        plt.scatter(xs, ys, color="black", s=12, label="Sampled frames", zorder=5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Visualization saved to {save_path}")

import os
import numpy as np
from PIL import Image


def save_center_keyframes(
    centers: List[int],
    center_scores: List[float],
    video_reader,
    save_dir: str = "keyframes",
):
    """保存每个窗口中心帧；不做间隔过滤、不丢弃 score=0。"""

    pairs = sorted(zip(centers, center_scores), key=lambda x: x[0])

    def save_frames(pairs_in, subfolder):
        os.makedirs(os.path.join(save_dir, subfolder), exist_ok=True)
        dup_counter = {}
        saved_indices = []
        for idx, sc in pairs_in:
            key = (subfolder, idx)
            dup_counter[key] = dup_counter.get(key, 0) + 1
            suffix = '' if dup_counter[key] == 1 else f"_dup{dup_counter[key]-1}"
            frame = Image.fromarray(video_reader[idx].asnumpy()).convert("RGB")
            frame_path = os.path.join(save_dir, subfolder, f"frame_{idx:06d}{suffix}.jpg")
            frame.save(frame_path)
            saved_indices.append(idx)
            print(f"[Saved] {subfolder} keyframe: frame={idx}, score2={sc:.3f}")
        return saved_indices

    pos_pairs = [(i, s) for i, s in pairs if s > 0]
    neg_pairs = [(i, s) for i, s in pairs if s < 0]
    zero_pairs = [(i, s) for i, s in pairs if s == 0]

    pos_saved = save_frames(pos_pairs, "positive")
    neg_saved = save_frames(neg_pairs, "negative")
    zero_saved = save_frames(zero_pairs, "zero")

    print(
        f"\n中心帧保存完成：正向 {len(pos_pairs)} 张，负向 {len(neg_pairs)} 张，零分 {len(zero_pairs)} 张\n"
        f"目录: {save_dir}/positive & {save_dir}/negative & {save_dir}/zero"
    )

    return pos_saved, neg_saved, zero_saved



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VadCLIP + Qwen3-VL Pipeline')
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
    results = pipeline.run(
        feature_path=args.feature_path,
        video_path=args.video_path,
        select_frames=args.select_frames,
        weight=args.weight,
        window_size=args.window_size,
    )

    score1 = results["system1_scores"]
    score2 = results["system2_scores"]
    score1_plot = np.repeat(np.asarray(score1, dtype=float), 16)
    score2_plot = np.repeat(np.asarray(score2, dtype=float), 16)
    video = VideoReader(results["video_path"])

    vis_dir = "/root/VadCLIP/src/vis"
    os.makedirs(vis_dir, exist_ok=True)
    centers = results.get("window_centers", [])

    plot_score_overlay(
        score1_plot,
        score2_plot,
        save_path=os.path.join(vis_dir, "scores_overlay.png"),
        video_len=len(video),
        # marker_indices=centers,
    )

    # 保存每个 window 的中心帧
    center_scores = results.get("window_center_scores", [])
    pos_idx, neg_idx, zero_idx = save_center_keyframes(
        centers,
        center_scores,
        video_reader=video,
        save_dir=os.path.join(vis_dir, "keyframes_example")
    )

    # 保存 score1 / score2 文本
    np.savetxt(os.path.join(vis_dir, "score1.txt"), np.asarray(score1, dtype=float))
    np.savetxt(os.path.join(vis_dir, "score2.txt"), np.asarray(score2, dtype=float))
    # print('Full video scores:', results['full_video_scores'])
