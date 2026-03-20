import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from ucf_test import dmAP  # 假设这是原有的评价函数引用
from model import CLIPVAD
from utils.tools import get_batch_mask,get_prompt_text
import ucf_option
from utils.dataset import UCFDataset
from torch.utils.data import DataLoader

def test_with_lp(model, testdataloader, args, prompt_text, gt, gtsegments, gtlabels, device):
    """
    Args:
        model: 原有的 VadCLIP 模型 (Backbone)
        args: 参数配置
        prompt_text: 原有的 prompt list (用于原有逻辑对比)
        gt, gtsegments, gtlabels: 用于计算 AUC 和 dmAP 的真值
    """
    model.eval()
    
    # -----------------------------------------------------------
    # 1. 加载训练好的 LP++ Adapter 权重
    # -----------------------------------------------------------
    print("Loading LP++ Adapter weights...")
    lp_ckpt = torch.load("/root/VadCLIP/lp_plus_adapter.pth", map_location=device)
    
    # 提取参数并转为 Parameter 或 Tensor
    W_lp = lp_ckpt['classifier_weight'].to(device)   # [2, D]
    b_lp = lp_ckpt['classifier_bias'].to(device)     # [2]
    alpha = lp_ckpt['alpha'].to(device)              # Scalar
    text_protos = lp_ckpt['text_prototypes'].to(device) # [2, D]
    
    maxlen = args.visual_length # 通常是 256 或类似

    ap1_list = [] # Original Visual (Linear)
    ap2_list = [] # Original Text (MIL)
    ap_lp_list = [] # New LP++ Score
    
    element_logits_lp_stack = [] # 用于计算 LP++ 的 dmAP

    print(f"Starting Inference on {len(testdataloader)} videos...")
    
    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            # --- 数据预处理 (保持原逻辑不变) ---
            visual = item[0].squeeze(0)
            length = item[2]
            length = int(length)
            len_cur = length
            
            # 处理 Batch 维度
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            
            visual = visual.to(device)

            # 处理长视频切片逻辑 (Batching strategy)
            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int).to(device)
            
            # 获取 batch mask
            # 注意：原代码的 get_batch_mask 可能需要 import，这里假设你有
            from utils.tools import get_batch_mask 
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            # -----------------------------------------------------------
            # 2. 获取特征 (关键修改)
            # -----------------------------------------------------------
            # 我们不调用 model(...)，而是调用 model.encode_video 直接拿特征
            # 这样既可以用于 LP++，原有的 logits 也可以通过特征简单计算得到(如果需要严格对比，建议跑两次)
            
            # A. 跑原模型 Forward (为了获取 logits1, logits2 做对比基准)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            
            # B. 再次获取纯特征用于 LP++ (或者修改 model 代码让其返回 feature)
            # 为了不修改 model 代码，这里显式调用 encode_video
            # 注意：encode_video 返回 [B, T, D]
            features = model.encode_video(visual, padding_mask, lengths)
            
            # -----------------------------------------------------------
            # 3. 处理输出形状
            # -----------------------------------------------------------
            # Flatten: [B, T, D] -> [B*T, D]
            B, T, D = features.shape
            features = features.reshape(B * T, D)
            
            # 原有 Logits Reshape
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

            # -----------------------------------------------------------
            # 4. LP++ 推理逻辑
            # -----------------------------------------------------------
            # Step 1: 归一化 (LP++ 必须)
            features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Step 2: Visual Branch (Linear) -> [B*T, 2]
            # F.linear(input, weight, bias) = input @ weight.T + bias
            vis_logits_lp = F.linear(features_norm, W_lp, b_lp)
            
            # Step 3: Text Branch (Zero-shot) -> [B*T, 2]
            text_logits_lp = features_norm @ text_protos.t()
            
            # Step 4: Fusion -> [B*T, 2]
            final_logits_lp = vis_logits_lp + alpha * text_logits_lp
            
            # Step 5: Probability (Softmax) -> 取 Class 1 (Anomaly)
            # [B*T, 2] -> [B*T]
            prob_lp = F.softmax(final_logits_lp, dim=-1)[:, 1]

            # -----------------------------------------------------------
            # 5. 截断与收集 (Truncate padding)
            # -----------------------------------------------------------
            # 原有概率
            prob2_ori = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1_ori = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            
            # LP++ 概率
            prob_lp_final = prob_lp[0:len_cur]

            if i == 0:
                ap1 = prob1_ori
                ap2 = prob2_ori
                ap3 = prob_lp_final # LP++
            else:
                ap1 = torch.cat([ap1, prob1_ori], dim=0)
                ap2 = torch.cat([ap2, prob2_ori], dim=0)
                ap3 = torch.cat([ap3, prob_lp_final], dim=0)

            # 收集用于 dmAP 计算的 logits (需要转 numpy)
            # LP++ 输出的是 [T]，dmAP 需要 [T, 2] 的 softmax 结果或者类似形式
            # 原代码 element_logits2 是 logits2 的 softmax
            # 我们构造 LP++ 的 element logits: [T, 2]
            lp_softmax = F.softmax(final_logits_lp[0:len_cur], dim=-1).detach().cpu().numpy()
            lp_softmax = np.repeat(lp_softmax, 16, 0) # 帧级扩展
            element_logits_lp_stack.append(lp_softmax)

    # -----------------------------------------------------------
    # 6. 评估计算
    # -----------------------------------------------------------
    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap3 = ap3.cpu().numpy() # LP++

    print("Total Score length:", len(ap1))
    
    # 扩展 GT 到帧级 (UCF-Crime 的做法通常是 repeat 16)
    # 假设 input 的 gt 是 video-level 或 segment-level，这里保持原逻辑 repeat 16
    gt_expanded = gt # 如果 gt 已经是帧级
    # 注意：roc_auc_score 需要 y_true 和 y_score 长度一致
    # 原代码逻辑：ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    # 这意味着 gt 是帧级的，而 ap1 是 16帧级(segment级)的，需要扩展
    
    pred1 = np.repeat(ap1, 16)
    pred2 = np.repeat(ap2, 16)
    pred_lp = np.repeat(ap3, 16)

    ROC1 = roc_auc_score(gt, pred1)
    AP1 = average_precision_score(gt, pred1)
    
    ROC2 = roc_auc_score(gt, pred2)
    AP2 = average_precision_score(gt, pred2)
    
    ROC_LP = roc_auc_score(gt, pred_lp)
    AP_LP = average_precision_score(gt, pred_lp)

    print("-" * 30)
    print(f"Original Visual (AUC): {ROC1:.4f} | AP: {AP1:.4f}")
    print(f"Original Text   (AUC): {ROC2:.4f} | AP: {AP2:.4f}")
    print(f"★ LP++ Enhanced (AUC): {ROC_LP:.4f} | AP: {AP_LP:.4f}")
    print("-" * 30)

    # 计算 dmAP (如果需要)
    print("Calculating dmAP for LP++ result...")
    # dmAP 函数通常需要 element_logits_stack (list of numpy arrays)
    # 注意：dmAP 的输入格式取决于具体实现，通常是 [T_frames, num_classes]
    # 这里我们只关注 Abnormal 类，或者保持原格式
    dmap, iou = dmAP(element_logits_lp_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('LP++ Average MAP: {:.2f}'.format(averageMAP))
    
    return ROC_LP, AP_LP

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    print("GT length:", len(gt))
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path, map_location=device)
    model.load_state_dict(model_param)
    model.to(device)

    test_with_lp(model, testdataloader, args, get_prompt_text(label_map), gt, gtsegments, gtlabels, device)