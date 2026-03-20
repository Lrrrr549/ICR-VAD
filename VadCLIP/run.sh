# python src/test_vadclip_internvl35_auc.py --test-list /root/VadCLIP/list/xd_CLIP_rgbtest.csv --gt-path /root/VadCLIP/list/gt.npy --internvl-path /root/autodl-tmp/InternVL3_5-2B --weight 0.5 --window-size 8 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_xd.pth

# python src/test_vadclip_internvl35_auc.py --test-list /root/VadCLIP/list/xd_CLIP_rgbtest.csv --gt-path /root/VadCLIP/list/gt.npy --internvl-path /root/autodl-tmp/InternVL3_5-8B --weight 0.5 --window-size 8 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_xd.pth

# python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 4 --select-frames 36 --vadclip-ckpt /root/VadCLIP/model_ucf.pth

# python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 8 --select-frames 36 --vadclip-ckpt /root/VadCLIP/model_ucf.pth

# python src/test.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_ucf.pth --sampling-method uniform

# python src/test.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_ucf.pth --sampling-method topk

# python src/test.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_ucf.pth 

# python src/test.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 12 --vadclip-ckpt /root/VadCLIP/model_ucf.pth --sampling-method uniform

# python src/test.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 12 --vadclip-ckpt /root/VadCLIP/model_ucf.pth --sampling-method topk

# python src/test.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 12 --vadclip-ckpt /root/VadCLIP/model_ucf.pth 

# python /root/VadCLIP/src/test_ats_internvl35_auc.py --window-size 12 --select-frames 36 --llm-path /root/autodl-tmp/InternVL3_5-2B

# python /root/VadCLIP/src/test_ats_internvl35_auc.py --window-size 12 --select-frames 36 --llm-path /root/autodl-tmp/InternVL3_5-8B

# python /root/VadCLIP/src/test_ats_internvl35_auc.py --window-size 12 --select-frames 36 --llm-path /root/HolmesVAU/ckpt/HolmesVAU-2B

# python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 4 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_ucf.pth
# python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 8 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_ucf.pth
# python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_ucf.pth
python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 16 --select-frames 24 --vadclip-ckpt /root/VadCLIP/model_ucf.pth
python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 4 --select-frames 12 --vadclip-ckpt /root/VadCLIP/model_ucf.pth
python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 8 --select-frames 12 --vadclip-ckpt /root/VadCLIP/model_ucf.pth
python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 12 --select-frames 12 --vadclip-ckpt /root/VadCLIP/model_ucf.pth
python src/test_vadclip_qwen3vl_auc.py   --test-list /root/VadCLIP/list/ucf_CLIP_rgbtest.csv   --gt-path /root/VadCLIP/list/gt_ucf.npy   --qwen-path /root/autodl-tmp/Qwen3-VL-8B-Instruct  --weight 0.5 --window-size 16 --select-frames 12 --vadclip-ckpt /root/VadCLIP/model_ucf.pth