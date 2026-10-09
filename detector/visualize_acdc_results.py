# -*- coding: utf-8 -*-
# find_best_detections_by_gain.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ================== 所有自定义模块定义 ==================
class LightweightMultiScaleAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads, self.head_dim, self.scale = num_heads, channels // num_heads, (channels // num_heads) ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.proj, self.norm, self.drop = nn.Conv2d(channels, channels, 1), nn.GroupNorm(32, channels), nn.Dropout(dropout)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x); q, k, v = qkv.chunk(3, dim=1)
        q = q + self.dw3(q) + self.dw5(q); k = k + self.dw3(k) + self.dw5(k); v = v + self.dw3(v) + self.dw5(v)
        q = q.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        attn = self.drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        out = (attn @ v).transpose(-2, -1).contiguous().reshape(B, C, H, W)
        return self.norm(self.proj(out) + x)

class ScaleAdaptiveAttention(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.small = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.medium = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.large = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//4, 1), nn.BatchNorm2d(channels//4), nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels//8, 1), nn.BatchNorm2d(channels//8), nn.ReLU(inplace=True), nn.Conv2d(channels//8, num_scales, 1))
        self.temperature, self.norm, self.dropout = temperature, nn.GroupNorm(32, channels), nn.Dropout(dropout)
    def forward(self, x, epoch=None):
        s, m, l = self.small(x), self.medium(x), self.large(x)
        w = self.weight_net(x)
        t = self.temperature * (1.0 - 0.5 * epoch/300) if epoch is not None else self.temperature
        w = (w / t).softmax(1)
        return self.dropout(self.norm(w[:,0:1]*s + w[:,1:2]*m + w[:,2:3]*l + x))

class DWTBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.channels = int(channels)
        d = float(dropout) if float(dropout) <= 1.0 else float(dropout) / 10.0
        d = max(0.0, min(1.0, d))
        self.conv_ll = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_lh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hl = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv2d(channels * 4, channels, 1), nn.ReLU(inplace=True))
        self.norm, self.dropout = nn.GroupNorm(32, channels), nn.Dropout(d)
        self._register_dwt_kernels()
    def _register_dwt_kernels(self):
        self.register_buffer('kernel_low', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0)
        self.register_buffer('kernel_high', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 4.0)
    def forward(self, x):
        B, C, H, W = x.shape
        orig, coeffs = x, self._dwt_decompose(x)
        ll, lh, hl, hh = self.conv_ll(coeffs[0]), self.conv_lh(coeffs[1]), self.conv_hl(coeffs[2]), self.conv_hh(coeffs[3])
        del coeffs
        ll = F.interpolate(ll, size=(H, W), mode='nearest')
        lh = F.interpolate(lh, size=(H, W), mode='nearest')
        hl = F.interpolate(hl, size=(H, W), mode='nearest')
        hh = F.interpolate(hh, size=(H, W), mode='nearest')
        out = self.fusion(torch.cat([ll, lh, hl, hh], dim=1))
        return self.dropout(self.norm(out + orig))
    def _dwt_decompose(self, x):
        kl, kh = self.kernel_low.to(x.device), self.kernel_high.to(x.device)
        return [self._apply_filter(x, kl, kl), self._apply_filter(x, kl, kh),
                self._apply_filter(x, kh, kl), self._apply_filter(x, kh, kh)]
    def _apply_filter(self, x, kernel_h, kernel_v):
        B, C, H, W = x.shape
        xd, xf = x.dtype, x.float()
        xr = xf.reshape(B*C, 1, H, W)
        kh, kv = kernel_h.float().unsqueeze(0).unsqueeze(0), kernel_v.float().unsqueeze(0).unsqueeze(0)
        xv = F.conv2d(F.conv2d(xr, kh, padding=1), kv, padding=1)[:, :, ::2, ::2]
        _, _, h, w = xv.shape
        return xv.reshape(B, C, h, w).to(xd)

class ScaleAdaptiveDWTBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.ReLU(inplace=True))
        self.norm, self.dropout = nn.GroupNorm(32, channels), nn.Dropout(dropout)
    def forward(self, x, epoch=None):
        out1, out2 = self.scale_adaptive(x, epoch), self.dwt(x)
        fused = self.fusion(torch.cat([out1, out2], dim=1))
        return self.dropout(self.norm(fused + x))

class GatedFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.gate_conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                       nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.norm, self.dropout = nn.GroupNorm(32, channels), nn.Dropout(dropout)
    def forward(self, x, epoch=None):
        out_attn, out_dwt = self.scale_adaptive(x, epoch), self.dwt(x)
        gate = self.gate_conv(out_attn)
        fused = gate * out_dwt + (1 - gate) * out_attn
        return self.dropout(self.norm(fused + x))

class MultiScaleGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch3 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.Conv2d(channels, channels, 1))
        self.branch5 = nn.Sequential(nn.Conv2d(channels, channels, 5, padding=2, groups=channels), nn.Conv2d(channels, channels, 1))
        self.weight, self.sigmoid = nn.Parameter(torch.zeros(2)), nn.Sigmoid()
    def forward(self, x): w = self.weight.softmax(dim=0); return self.sigmoid(w[0] * self.branch3(x) + w[1] * self.branch5(x))

class MultiScaleGatedFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.gate_conv, self.norm, self.dropout = MultiScaleGate(channels), nn.GroupNorm(32, channels), nn.Dropout(dropout)
    def forward(self, x, epoch=None):
        out_attn, out_dwt = self.scale_adaptive(x, epoch), self.dwt(x)
        gate = self.gate_conv(out_attn)
        return self.dropout(self.norm(gate * out_dwt + (1 - gate) * out_attn + x))

# 注册所有模块
for cls in [ScaleAdaptiveAttention, LightweightMultiScaleAttention, DWTBlock, ScaleAdaptiveDWTBlock,
            GatedFusionBlock, MultiScaleGate, MultiScaleGatedFusionBlock]:
    sys.modules['__main__'].__dict__[cls.__name__] = cls

# ================== 场景最优模型映射 ==================
SCENE_OPTIMAL = {
    "Fog":   r"C:\Users\KUST\xlw\results-2\09_scale_adaptive_dwt_lightweight\weights\best.pt",   # exp09
    "Night": r"C:\Users\KUST\xlw\results-2\12_multiscale_gated_fusion\weights\best.pt",          # exp12
    "Rain":  r"C:\Users\KUST\xlw\results-2\04_dwt_consistent\weights\best.pt",                    # exp04
    "Snow":  r"C:\Users\KUST\xlw\results-2\11_gated_fusion\weights\best.pt",                      # exp11
}

BASELINE = r"C:\Users\KUST\xlw\results-2\01_baseline_consistent\weights\best.pt"

ACDC_ROOT = r"C:\Users\KUST\xlw\acdc\rgb_anon"
SCENE_DIRS = {
    "Fog":   os.path.join(ACDC_ROOT, "fog", "val"),
    "Night": os.path.join(ACDC_ROOT, "night", "val"),
    "Rain":  os.path.join(ACDC_ROOT, "rain", "val"),
    "Snow":  os.path.join(ACDC_ROOT, "snow", "val"),
}

TOP_K = 5
SAVE_FIGS = True

def collect_all_pngs_recursive(scene_dir):
    png_files = []
    for root, _, files in os.walk(scene_dir):
        for f in files:
            if f.endswith('.png'):
                png_files.append(os.path.join(root, f))
    return png_files

def avg_confidence(results):
    boxes = results.boxes
    if boxes is None or len(boxes.conf) == 0:
        return 0.0
    return boxes.conf.mean().item()

if __name__ == '__main__':
    for scene_name, scene_dir in SCENE_DIRS.items():
        optimal_path = SCENE_OPTIMAL[scene_name]
        print(f"\n=== {scene_name} (optimal model: {Path(optimal_path).parent.name}) ===")
        
        optimal_model = YOLO(optimal_path)
        baseline_model = YOLO(BASELINE)
        all_imgs = collect_all_pngs_recursive(scene_dir)
        scored = []
        for img_path in all_imgs:
            res_opt = optimal_model.predict(img_path, imgsz=416, save=False, verbose=False)
            res_base = baseline_model.predict(img_path, imgsz=416, save=False, verbose=False)
            conf_opt = avg_confidence(res_opt[0])
            conf_base = avg_confidence(res_base[0])
            gain = conf_opt - conf_base   # 提升幅度
            scored.append((img_path, conf_opt, conf_base, gain))
        
        scored.sort(key=lambda x: x[3], reverse=True)
        top = scored[:TOP_K]
        print(f"Top {TOP_K} by confidence gain (Ours - Baseline):")
        for rank, (path, conf_opt, conf_base, gain) in enumerate(top, 1):
            print(f"  {rank}. {os.path.basename(path)}  (Baseline={conf_base:.4f}, Ours={conf_opt:.4f}, Gain={gain:.4f})")
        
        if SAVE_FIGS:
            save_dir = Path(f"best_imgs_by_gain/{scene_name}")
            save_dir.mkdir(parents=True, exist_ok=True)
            for rank, (img_path, _, _, _) in enumerate(top, 1):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                b_res = baseline_model.predict(img_path, imgsz=416, save=False, verbose=False)
                o_res = optimal_model.predict(img_path, imgsz=416, save=False, verbose=False)
                axes[0].imshow(b_res[0].plot()[:, :, ::-1])
                axes[0].set_title("Baseline"); axes[0].axis('off')
                axes[1].imshow(o_res[0].plot()[:, :, ::-1])
                axes[1].set_title(f"Ours ({Path(optimal_path).parent.name})"); axes[1].axis('off')
                plt.suptitle(f"{scene_name} - Gain {scored[rank-1][3]:.4f} (Top {rank})")
                plt.tight_layout()
                save_path = save_dir / f"top{rank}_{Path(img_path).name}"
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"  saved {save_path}")

    print("\nDone.")