import os
import sys
import torch
import joblib
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

from core.config import (
    REPO_DIR, WEIGHTS, CLF_PATH, SCALER_PATH,
    PATCH_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, COLORMAPS
)

# ── 模型单例 ──────────────────────────────────────
_model  = None
_clf    = None
_scaler = None
_device = torch.device("cpu")


def load_model(device: str = "cuda"):
    """启动时调用，同时加载 DINOv3 + 分割头"""
    global _model, _clf, _scaler, _device

    print("正在加载 DINOv3 模型...")
    if device == "cuda" and torch.cuda.is_available():
        _device = torch.device("cuda")
    elif device == "cuda":
        print("CUDA 不可用，自动回退到 CPU")
        _device = torch.device("cpu")
    else:
        _device = torch.device(device)

    sys.path.insert(0, os.path.dirname(REPO_DIR))  # dinov3_test/
    sys.path.insert(0, REPO_DIR)                    # dinov3_test/dinov3/

    _model = torch.hub.load(
        REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHTS
    )
    _model = _model.to(_device)
    _model.eval()
    print(f"DINOv3 加载成功，当前 device: {_device}")

    print("正在加载前景分割头...")
    _clf    = joblib.load(CLF_PATH)
    _scaler = joblib.load(SCALER_PATH)
    print("分割头加载成功")


def is_model_ready():
    return _model is not None


def is_seg_ready():
    return _clf is not None and _scaler is not None


# ── 特征提取 ──────────────────────────────────────
def load_and_extract(image_path, scale):
    img_rgb = np.array(Image.open(image_path).convert('RGB'))
    orig_h, orig_w = img_rgb.shape[:2]

    base_scale = IMAGE_SIZE / min(orig_h, orig_w)
    base_h = int(round(orig_h * base_scale / PATCH_SIZE)) * PATCH_SIZE
    base_w = int(round(orig_w * base_scale / PATCH_SIZE)) * PATCH_SIZE
    new_h  = (base_h * scale // PATCH_SIZE) * PATCH_SIZE
    new_w  = (base_w * scale // PATCH_SIZE) * PATCH_SIZE

    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    h_patches   = new_h // PATCH_SIZE
    w_patches   = new_w // PATCH_SIZE

    tensor = TF.normalize(
        TF.to_tensor(Image.fromarray(img_resized)),
        IMAGENET_MEAN, IMAGENET_STD
    ).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _model.get_intermediate_layers(tensor, n=1, return_class_token=True)

    patch_tokens  = features[0][0].squeeze(0).detach().cpu().numpy()
    norms         = np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8
    tokens_normed = patch_tokens / norms

    return img_resized, tokens_normed, patch_tokens, h_patches, w_patches, orig_w, orig_h


# ── 前景分割 ──────────────────────────────────────
def predict_fg_mask(patch_tokens_raw, h_patches, w_patches):
    """
    用分割头预测每个 patch 的前景概率。
    返回:
        prob_map   : float32 [h_patches, w_patches]  0~1 概率
        mask_binary: bool    [h_patches, w_patches]  True=前景
    """
    scaled     = _scaler.transform(patch_tokens_raw)
    probs      = _clf.predict_proba(scaled)[:, 1]
    prob_map   = probs.reshape(h_patches, w_patches).astype(np.float32)
    mask_binary = prob_map > 0.5
    return prob_map, mask_binary


# ── 热力图 / PCA ──────────────────────────────────
def sim_to_heatmap(sim_map, colormap_name):
    sim_u8 = ((sim_map + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(sim_u8, COLORMAPS[colormap_name])


def compute_pca_rgb(patch_tokens, h_patches, w_patches, fg_mask=None):
    """
    PCA 降维到 3 维 → RGB 图。
    fg_mask: bool [h_patches, w_patches]，不为 None 时只对前景 patch 做 PCA，
             背景 patch 显示为黑色。
    """
    if fg_mask is not None:
        flat_mask = fg_mask.flatten()          # [N]
        fg_tokens = patch_tokens[flat_mask]    # 只取前景行

        if fg_tokens.shape[0] < 3:
            # 前景 patch 太少，退回全图 PCA
            fg_mask = None
        else:
            pca    = PCA(n_components=3)
            coords = pca.fit_transform(fg_tokens)   # [N_fg, 3]
            for i in range(3):
                c = coords[:, i]
                coords[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-8) * 255

            # 把前景结果填回完整网格，背景保持黑色
            rgb_flat = np.zeros((patch_tokens.shape[0], 3), dtype=np.uint8)
            rgb_flat[flat_mask] = coords.astype(np.uint8)
            return rgb_flat.reshape(h_patches, w_patches, 3)

    # 全图 PCA（fg_mask 为 None 或前景太少时）
    pca    = PCA(n_components=3)
    coords = pca.fit_transform(patch_tokens)
    for i in range(3):
        c = coords[:, i]
        coords[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-8) * 255
    return coords.reshape(h_patches, w_patches, 3).astype(np.uint8)
