# -*- coding: utf-8 -*-
# measure_latency_fixed.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from pathlib import Path
from ultralytics import YOLO

# ================== 所有自定义模块定义（与前相同，已压缩） ==================
class LightweightMultiScaleAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(32, channels)
        self.drop = nn.Dropout(dropout)
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
        d = float(dropout) if dropout <= 1.0 else dropout / 10.0
        d = max(0.0, min(1.0, d))
        self.conv_ll = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_lh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hl = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv2d(channels * 4, channels, 1), nn.ReLU(inplace=True))
        self.norm, self.dropout = nn.GroupNorm(32, channels), nn.Dropout(d)
        self._register_dwt_kernels(); self._init_weights()
    def _register_dwt_kernels(self):
        self.register_buffer('kernel_low', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0)
        self.register_buffer('kernel_high', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 4.0)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)): nn.init.constant_(m.weight, 1)
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
        kh = kernel_h.float().unsqueeze(0).unsqueeze(0); kv = kernel_v.float().unsqueeze(0).unsqueeze(0)
        xv = F.conv2d(F.conv2d(xr, kh, padding=1), kv, padding=1)[:, :, ::2, ::2]
        _, _, h, w = xv.shape
        return xv.reshape(B, C, h, w).to(xd)

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma)); k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class ECASABlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.eca = ECABlock(channels)
        self.attn = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, epoch=None):
        x = self.eca(x)
        return self.dropout(self.norm(self.attn(x, epoch) + x))

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
        return self.dropout(self.norm(gate * out_dwt + (1 - gate) * out_attn + x))

class MultiScaleGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch3 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.Conv2d(channels, channels, 1))
        self.branch5 = nn.Sequential(nn.Conv2d(channels, channels, 5, padding=2, groups=channels), nn.Conv2d(channels, channels, 1))
        self.weight = nn.Parameter(torch.zeros(2))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): w = self.weight.softmax(dim=0); return self.sigmoid(w[0] * self.branch3(x) + w[1] * self.branch5(x))

class MultiScaleGatedFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.gate_conv = MultiScaleGate(channels)
        self.norm, self.dropout = nn.GroupNorm(32, channels), nn.Dropout(dropout)
    def forward(self, x, epoch=None):
        out_attn, out_dwt = self.scale_adaptive(x, epoch), self.dwt(x)
        gate = self.gate_conv(out_attn)
        return self.dropout(self.norm(gate * out_dwt + (1 - gate) * out_attn + x))

class SequentialFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.norm, self.dropout = nn.GroupNorm(32, channels), nn.Dropout(dropout)
    def forward(self, x, epoch=None):
        out_attn = self.scale_adaptive(x, epoch)
        mixed = x + self.norm(out_attn)
        return self.dropout(self.dwt(mixed) + x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//reduction, 1),
                                nn.ReLU(inplace=True), nn.Conv2d(channels//reduction, channels, 1), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, channels//reduction, 1, bias=False),
                                nn.ReLU(inplace=True), nn.Conv2d(channels//reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid(avg_out + max_out)
        avg_out = torch.mean(x, dim=1, keepdim=True); max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg_out, max_out], dim=1))

class CABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)); self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels//reduction, 1); self.bn = nn.BatchNorm2d(channels//reduction)
        self.conv_h = nn.Conv2d(channels//reduction, channels, 1); self.conv_w = nn.Conv2d(channels//reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).transpose(-2, -1)
        y = torch.cat([x_h, x_w], dim=2); y = self.bn(self.conv1(y)); y = F.relu(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        a_h, a_w = self.sigmoid(self.conv_h(x_h)), self.sigmoid(self.conv_w(x_w.transpose(-2, -1)))
        return x * a_h * a_w

# 注册
for cls in [ScaleAdaptiveAttention, LightweightMultiScaleAttention, DWTBlock, ECABlock, ECASABlock,
            ScaleAdaptiveDWTBlock, GatedFusionBlock, MultiScaleGate, MultiScaleGatedFusionBlock,
            SequentialFusionBlock, SEBlock, CBAMBlock, CABlock]:
    sys.modules['__main__'].__dict__[cls.__name__] = cls

# ================== 模型路径 ==================
MODELS = {
    "exp01 (Baseline)": r"C:\Users\KUST\xlw\results-2\01_baseline_consistent\weights\best.pt",
    "exp04 (DWT only)": r"C:\Users\KUST\xlw\results-2\04_dwt_consistent\weights\best.pt",
    "exp08 (AMAR)":      r"C:\Users\KUST\xlw\results-2\08_scale_adaptive_lightweight\weights\best.pt",
    "exp09 (Concat Fusion)": r"C:\Users\KUST\xlw\results-2\09_scale_adaptive_dwt_lightweight\weights\best.pt",
    "exp10 (Concat Fusion, l6)": r"C:\Users\KUST\xlw\results-2\10_scale_adaptive_dwt_layer6\weights\best.pt",
    "exp11 (Gated Fusion)": r"C:\Users\KUST\xlw\results-2\11_gated_fusion\weights\best.pt",
    "exp12 (Multi-Gated Fusion)": r"C:\Users\KUST\xlw\results-2\12_multiscale_gated_fusion\weights\best.pt",
    "exp13 (Sequential Fusion)": r"C:\Users\KUST\xlw\results-2\13_sequential_fusion\weights\best.pt",
    "exp15 (ECASA)": r"C:\Users\KUST\xlw\results-2\15_ecasa\weights\best.pt",
    "comp01 (SE)": r"C:\Users\KUST\xlw\results-2\comp01_se\weights\best.pt",
    "comp02 (CBAM)": r"C:\Users\KUST\xlw\results-2\comp02_cbam\weights\best.pt",
    "comp03 (ECA)": r"C:\Users\KUST\xlw\results-2\comp03_eca\weights\best.pt",
    "comp04 (CA)": r"C:\Users\KUST\xlw\results-2\comp04_ca\weights\best.pt",
}

DEVICE = 'cuda:0'
IMG_SIZE = 416
WARMUP = 50
TEST_ITERS = 100

def measure_inference_latency(model_path):
    if not os.path.exists(model_path):
        return None
    m = YOLO(model_path)
    model = m.model
    model.eval().to(DEVICE)
    # 确保使用 float32（与训练一致）
    model.float()
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE).float()

    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(dummy)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(TEST_ITERS):
            _ = model(dummy)
        torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / TEST_ITERS * 1000

def measure_end2end_latency(model_path):
    if not os.path.exists(model_path):
        return None
    m = YOLO(model_path)
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE).float()

    with torch.inference_mode():
        for _ in range(WARMUP):
            # 显式关闭 half，使用 FP32 推理，与训练配置对齐
            m.predict(dummy, imgsz=IMG_SIZE, half=False, verbose=False)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(TEST_ITERS):
            m.predict(dummy, imgsz=IMG_SIZE, half=False, verbose=False)
        torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / TEST_ITERS * 1000

if __name__ == '__main__':
    print("=" * 95)
    print("Latency Measurement (batch=1, 416x416, RTX 3090, FP32)")
    print("=" * 95)
    print(f"{'Model':35s} {'Inference (ms)':>14s}  {'End-to-End (ms)':>15s}  {'FPS (End-to-End)':>16s}")
    print("-" * 85)
    for name, path in MODELS.items():
        infer_ms = measure_inference_latency(path)
        end2end_ms = measure_end2end_latency(path)

        infer_str = f"{infer_ms:.2f}" if infer_ms is not None else "N/A"
        end2end_str = f"{end2end_ms:.2f}" if end2end_ms is not None else "N/A"
        fps_str = f"{1000/end2end_ms:.1f}" if (end2end_ms is not None and end2end_ms > 0) else "N/A"
        print(f"{name:35s}  {infer_str:>14s}  {end2end_str:>15s}  {fps_str:>16s}")
    print("=" * 95)
    print("Done.")