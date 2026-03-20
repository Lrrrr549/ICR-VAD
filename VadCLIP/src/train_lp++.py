import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import argparse
from collections import OrderedDict
from sklearn.metrics import average_precision_score, roc_auc_score

# ==========================================
# 1. 依赖导入 (假设这些文件在你的 utils 目录下)
# ==========================================
try:
    from clip import clip
    from utils.layers import GraphConvolution, DistanceAdj
    from utils.dataset import UCFDataset
    from utils.tools import get_prompt_text, get_batch_label, get_batch_mask
    from utils.ucf_detectionMAP import getDetectionMAP as dmAP
    import ucf_option
except ImportError as e:
    print("错误: 缺少必要的 utils 文件或 ucf_option.py。请确保这些文件在当前目录下。")
    print(f"详细错误: {e}")
    exit(1)

# ==========================================
# 2. 模型定义 (CLIPVAD & Components)
# ==========================================
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CLIPVAD(nn.Module):
    def __init__(self, num_class, embed_dim, visual_length, visual_width, visual_head, visual_layers, attn_window, prompt_prefix, prompt_postfix, device):
        super().__init__()
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width, layers=visual_layers, heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) 
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True) 
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]; adj2 = tmp; adj2 = F.threshold(adj2, 0.7, 0); adj2 = soft(adj2); output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]; adj2 = tmp; adj2 = F.threshold(adj2, 0.7, 0); adj2 = soft(adj2); output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj)); x2_h = self.gelu(self.gc3(x, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj)); x2 = self.gelu(self.gc4(x2_h, disadj))
        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        return x # [B, T, D]

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)
        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]
        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        # 原始 Forward，LP++ 模式下不使用此方法
        pass

# ==========================================
# 3. LP++ 封装类 (核心创新点)
# ==========================================
class CLIPVAD_LP_Plus(nn.Module):
    def __init__(self, backbone: CLIPVAD, alpha=0.5):
        super().__init__()
        self.backbone = backbone
        self.alpha = alpha
        self.device = backbone.device
        
        # 1. 冻结骨干网络 (Freeze Backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. 定义可学习的分类头 (Trainable Head)
        # 输入维度: visual_width, 输出维度: 1 (Anomaly Score)
        self.linear_probe = nn.Linear(self.backbone.visual_width, 1, bias=False)
        
        # 3. 存储文本先验 (Text Priors)
        self.text_prior_embeddings = None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # Learnable temperature for prior

    def init_weights_with_text(self, label_map):
        """利用 CLIP 文本特征初始化分类头，实现 LP++"""
        print(">>> LP++ Initialization: Computing Text Embeddings...")
        
        # 构建 Prompt
        normal_prompts = ["Normal video", "Daily life", "Peaceful street view", "Safe environment"]
        # 提取 UCF-Crime 所有 13 个异常类别
        anomaly_types = [v for k, v in label_map.items() if k != 'Normal']
        anomaly_prompts = [f"A video of {atype}" for atype in anomaly_types]
        
        with torch.no_grad():
            # 编码 Normal
            norm_embeds = self.backbone.encode_textprompt(normal_prompts)
            norm_embeds = norm_embeds / norm_embeds.norm(dim=-1, keepdim=True)
            norm_center = torch.mean(norm_embeds, dim=0) # [D]
            
            # 编码 Abnormal
            abnorm_embeds = self.backbone.encode_textprompt(anomaly_prompts)
            abnorm_embeds = abnorm_embeds / abnorm_embeds.norm(dim=-1, keepdim=True)
            abnorm_center = torch.mean(abnorm_embeds, dim=0) # [D]
            
            # 计算方向向量: Normal -> Abnormal
            # 这样特征在方向上投影越大，越代表异常
            direction = abnorm_center - norm_center
            direction = direction / direction.norm()
            
            # 初始化 Linear Probe
            self.linear_probe.weight.data.copy_(direction.unsqueeze(0))
            
            # 保存 Prior 用于 Zero-shot 分支
            self.text_prior_embeddings = direction.clone().detach() # [D]

        # 开启分类头梯度
        for param in self.linear_probe.parameters():
            param.requires_grad = True
        self.logit_scale.requires_grad = True
        print(">>> LP++ Initialization Done.")

    def forward(self, visual, padding_mask, lengths):
        # 1. 提取特征 (Frozen)
        with torch.no_grad():
            # [B, T, D]
            visual_features = self.backbone.encode_video(visual, padding_mask, lengths)
            visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        
        # 2. 分支 A: Learnable Linear Probe (Few-shot refined)
        # [B, T, 1]
        logits_learn = self.linear_probe(visual_features)
        
        # 3. 分支 B: Zero-shot Text Prior
        # [B, T, D] @ [D, 1] -> [B, T, 1]
        logits_prior = visual_features_norm @ self.text_prior_embeddings.view(-1, 1)
        logits_prior = logits_prior * self.logit_scale.exp()
        
        # 4. LP++ 融合
        # 简单的加权融合 logits
        logits_final = self.alpha * logits_learn + (1 - self.alpha) * logits_prior
        
        return logits_final

# ==========================================
# 4. 弱监督损失函数 (MIL Ranking Loss)
# ==========================================
class MILRankingLoss(nn.Module):
    def __init__(self, margin=100.0, lambda_smooth=8e-5, lambda_sparse=8e-5):
        # margin 设大一点，因为 logits 没有经过 sigmoid
        super().__init__()
        self.margin = margin
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse

    def forward(self, logits, labels, lengths):
        # logits: [B, T, 1]
        # labels: [B, 1] (0 for normal, 1 for anomaly)
        
        # 对 Logits 进行 Sigmoid 归一化到 [0, 1] 区间计算 Loss
        scores = torch.sigmoid(logits).squeeze(-1) # [B, T]
        
        # 区分 Normal 和 Anomaly Video
        # 假设 DataLoader 是配对的 (batch_size // 2 正常, batch_size // 2 异常)
        # 为了通用性，我们用 mask
        labels = labels.squeeze()
        abnormal_mask = (labels == 1).bool()
        normal_mask = (labels == 0).bool()
        
        if abnormal_mask.sum() == 0 or normal_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        scores_abnormal = scores[abnormal_mask] # [N_a, T]
        scores_normal = scores[normal_mask]     # [N_n, T]
        
        # 这里的 lengths 需要处理，因为经过 padding。我们只取有效长度内的 max
        # 简单起见，取 top-k mean 或者 max
        
        # 获取 Instance 最具异常性的分数
        # Top-k mean approach (referencing CLASM logic)
        def get_topk_mean(s, l):
            means = []
            for i in range(s.shape[0]):
                k = max(1, int(l[abnormal_mask][i] / 16 + 1))
                tmp, _ = torch.topk(s[i, :l[abnormal_mask][i]], k=k)
                means.append(torch.mean(tmp))
            return torch.stack(means)
            
        def get_max(s, l):
            maxs = []
            for i in range(s.shape[0]):
                valid_scores = s[i, :l[normal_mask][i]]
                if valid_scores.numel() > 0:
                    maxs.append(torch.max(valid_scores))
                else:
                    maxs.append(torch.tensor(0.0).to(s.device))
            return torch.stack(maxs)

        # 简化版 MIL: Max(Anomaly) > Max(Normal)
        max_abnormal, _ = torch.max(scores_abnormal, dim=1)
        max_normal, _ = torch.max(scores_normal, dim=1)
        
        # Ranking Loss
        loss_rank = F.relu(self.margin - max_abnormal.mean() + max_normal.mean()) # 这种写法是 Batch Mean 比较
        # 也可以做 Pairwise:
        # loss_rank = torch.mean(torch.relu(1.0 - max_abnormal.unsqueeze(1) + max_normal.unsqueeze(0)))

        # Smoothness & Sparsity
        loss_smooth = torch.mean(torch.sum((scores_abnormal[:, 1:] - scores_abnormal[:, :-1]) ** 2, dim=1))
        loss_sparse = torch.mean(torch.sum(scores_abnormal, dim=1))
        
        # 由于我们用了 Sigmoid 后的分数，margin 设为 1.0 比较合理
        # 如果 self.margin 是 100 (针对 logits)，这里应该用 logits。
        # 建议直接用 Sigmoid 后的分数计算 Ranking，margin=1.0 (standard MIL)
        
        loss = loss_rank + self.lambda_smooth * loss_smooth + self.lambda_sparse * loss_sparse
        return loss

# ==========================================
# 5. 训练流程 (LP++ Training)
# ==========================================
def train_lp_plus(model_lp, normal_loader, anomaly_loader, testloader, args, label_map, device):
    optimizer = torch.optim.AdamW([
        {'params': model_lp.linear_probe.parameters()},
        {'params': model_lp.logit_scale}
    ], lr=args.lr, weight_decay=0.01) # 只优化分类头
    
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    criterion = MILRankingLoss(margin=1.0) # Margin for probability space
    
    ap_best = 0
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    prompt_text = get_prompt_text(label_map) # For test function compatibility

    print(">>> Start LP++ Training...")
    
    for e in range(args.max_epoch):
        model_lp.train()
        # 确保 Backbone 处于 eval 模式 (关闭 Dropout, 稳定 BN/LN)
        model_lp.backbone.eval()
        
        loss_epoch = 0
        steps = 0
        
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        
        # 每个 Epoch 迭代次数取决于较短的那个 Loader
        iter_len = min(len(normal_loader), len(anomaly_loader))
        
        for i in range(iter_len):
            try:
                n_feat, n_label, n_len = next(normal_iter)
                a_feat, a_label, a_len = next(anomaly_iter)
            except StopIteration:
                break
                
            # 拼接数据 [B_n + B_a, T, D]
            visual = torch.cat([n_feat, a_feat], dim=0).to(device)
            # 构造弱监督标签: Normal=0, Anomaly=1
            # n_label 是 one-hot string 还是 index? 根据原代码是 string list
            # 这里简单构造 binary labels
            labels = torch.cat([torch.zeros(len(n_label)), torch.ones(len(a_label))]).to(device)
            lengths = torch.cat([n_len, a_len]).to(device)
            
            # Forward
            logits = model_lp(visual, None, lengths) # [B, T, 1]
            
            # Loss
            loss = criterion(logits, labels.unsqueeze(-1), lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()
            steps += 1
            
            if steps % 50 == 0:
                print(f"Epoch {e+1} | Step {steps}/{iter_len} | Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {e+1} Finished. Avg Loss: {loss_epoch/steps:.4f}")
        
        # Test every epoch
        if (e + 1) % 1 == 0:
            auc, ap = test_lp_plus(model_lp, testloader, args.visual_length, gt, gtsegments, gtlabels, device)
            if auc > ap_best:
                ap_best = auc
                torch.save(model_lp.state_dict(), args.model_path.replace('.pth', '_lp_best.pth'))
                print(f"New Best AUC: {ap_best:.4f}, Model Saved.")

# ==========================================
# 6. 测试流程 (LP++ Testing)
# ==========================================
def test_lp_plus(model_lp, testdataloader, maxlen, gt, gtsegments, gtlabels, device):
    model_lp.eval()
    preds = []
    
    # 获取特征维度，用于强制 reshape
    feature_dim = model_lp.backbone.visual_width
    
    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            # 1. 获取输入并移动到 Device
            visual = item[0].to(device)
            length = int(item[2])
            
            # 2. 【关键修复】强制 Flatten 为 [Total_Frames, D]
            # 无论 DataLoader 给的是 [1, T, D] 还是 [1, 1, T, D]，这一步都能修正为 [T, D]
            visual = visual.view(-1, feature_dim)
            
            # 3. 按 maxlen 切片 (Split)
            # chunks 中的每个 chunk 形状将是 [T_chunk, D]
            chunks = torch.split(visual, maxlen, dim=0)
            vid_scores = []
            
            for chunk in chunks:
                # 4. 增加 Batch 维度 -> [1, T_chunk, D]
                # 这样就严格符合模型要求的 3 维输入 [B, T, D]
                chunk = chunk.unsqueeze(0)
                
                # 计算当前切片的长度
                l = torch.tensor([chunk.shape[1]]).to(device)
                
                # Forward
                logits = model_lp(chunk, None, l) # [1, T_chunk, 1]
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                # 处理单帧情况 (防止变成 scalar)
                if prob.ndim == 0: 
                    prob = np.array([prob])
                    
                vid_scores.append(prob)
                
            # 拼接所有切片结果
            vid_scores = np.concatenate(vid_scores)
            
            # 截取有效长度 (去除 padding)
            if len(vid_scores) > length:
                vid_scores = vid_scores[:length]
            
            # 16帧重复 (对应特征提取步长)
            vid_scores_expanded = np.repeat(vid_scores, 16)
            preds.append(vid_scores_expanded)

    # 拼接所有视频的预测
    preds = np.concatenate(preds)
    
    # GT 对齐处理
    if len(preds) != len(gt):
        min_len = min(len(preds), len(gt))
        preds = preds[:min_len]
        gt_eval = gt[:min_len]
    else:
        gt_eval = gt

    auc = roc_auc_score(gt_eval, preds)
    ap = average_precision_score(gt_eval, preds)
    
    print(f"TEST RESULT -> AUC: {auc:.4f} | AP: {ap:.4f}")
    return auc, ap

# ==========================================
# 7. 主程序
# ==========================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    # 1. 设置参数
    parser = ucf_option.parser
    # 如果需要在命令行覆盖，可以在这里 add_argument
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    
    print(f"Using Device: {device}")

    # 2. 准备数据集
    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})
    
    # 假设 dataset 返回: feature, label_string, length
    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 3. 初始化骨干模型 (Backbone)
    # 加载预训练权重 (User's checkpoint or initialized)
    base_model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    
    # 加载已有的预训练权重 (如果有)
    # 这一步非常重要，因为 LP++ 依赖于骨干网络已经有较好的特征提取能力
    # 如果没有预训练权重，Transformers 的随机初始化特征可能无法与 CLIP Text 对齐
    try:
        if args.use_checkpoint:
            checkpoint = torch.load(args.checkpoint_path)
            base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded pretrained checkpoint for Backbone.")
        else:
            print("Warning: No checkpoint loaded. Backbone initialized randomly (LP++ might fail without aligned features).")
    except Exception as e:
        print(f"Checkpoint loading failed: {e}")

    # 4. 构建 LP++ 模型
    model_lp = CLIPVAD_LP_Plus(base_model, alpha=0.5).to(device)
    
    # 5. LP++ 初始化 (Text Priors)
    model_lp.init_weights_with_text(label_map)
    
    # 6. 开始训练
    train_lp_plus(model_lp, normal_loader, anomaly_loader, test_loader, args, label_map, device)