# -*- coding: utf-8 -*-
# measure_all_latency.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from ultralytics import YOLO

# ================== 所有自定义模块定义（与训练/评估脚本一致） ==================
class LightweightMultiScaleAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(32, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q + self.dw3(q) + self.dw5(q)
        k = k + self.dw3(k) + self.dw5(k)
        v = v + self.dw3(v) + self.dw5(v)
        q = q.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(-2, -1).contiguous().reshape(B, C, H, W)
        out = self.proj(out)
        return self.norm(out + x)

class ScaleAdaptiveAttention(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.small = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.medium = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.large = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4), nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels//8, 1),
            nn.BatchNorm2d(channels//8), nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, num_scales, 1)
        )
        self.temperature = temperature
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

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
        dropout = float(dropout) if dropout <= 1.0 else dropout / 10.0
        dropout = max(0.0, min(1.0, dropout))
        self.conv_ll = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_lh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hl = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv2d(channels * 4, channels, 1), nn.ReLU(inplace=True))
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)
        self._register_dwt_kernels()
        self._init_weights()

    def _register_dwt_kernels(self):
        kernel_low = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0
        kernel_high = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 4.0
        self.register_buffer('kernel_low', kernel_low)
        self.register_buffer('kernel_high', kernel_high)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_orig = x
        coeffs = self._dwt_decompose(x)
        ll = self.conv_ll(coeffs[0]); lh = self.conv_lh(coeffs[1])
        hl = self.conv_hl(coeffs[2]); hh = self.conv_hh(coeffs[3])
        del coeffs
        ll = F.interpolate(ll, size=(H, W), mode='nearest')
        lh = F.interpolate(lh, size=(H, W), mode='nearest')
        hl = F.interpolate(hl, size=(H, W), mode='nearest')
        hh = F.interpolate(hh, size=(H, W), mode='nearest')
        fused = torch.cat([ll, lh, hl, hh], dim=1)
        out = self.fusion(fused)
        del ll, lh, hl, hh, fused
        out = out + x_orig
        out = self.norm(out)
        out = self.dropout(out)
        return out

    def _dwt_decompose(self, x):
        B, C, H, W = x.shape
        kl = self.kernel_low.to(x.device); kh = self.kernel_high.to(x.device)
        ll = self._apply_filter(x, kl, kl); lh = self._apply_filter(x, kl, kh)
        hl = self._apply_filter(x, kh, kl); hh = self._apply_filter(x, kh, kh)
        return [ll, lh, hl, hh]

    def _apply_filter(self, x, kernel_h, kernel_v):
        B, C, H, W = x.shape
        xd = x.dtype; xf = x.float(); xr = xf.reshape(B*C, 1, H, W)
        kh = kernel_h.float().unsqueeze(0).unsqueeze(0); kv = kernel_v.float().unsqueeze(0).unsqueeze(0)
        xh = F.conv2d(xr, kh, padding=1); xv = F.conv2d(xh, kv, padding=1)
        xv = xv[:, :, ::2, ::2]
        _, _, h, w = xv.shape
        return xv.reshape(B, C, h, w).to(xd)

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
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
        out = self.attn(x, epoch)
        return self.dropout(self.norm(out + x))

class ScaleAdaptiveDWTBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.ReLU(inplace=True))
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch=None):
        out1 = self.scale_adaptive(x, epoch)
        out2 = self.dwt(x)
        fused = torch.cat([out1, out2], dim=1)
        fused = self.fusion(fused)
        fused = fused + x
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused

class GatedFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch=None):
        out_attn = self.scale_adaptive(x, epoch)
        out_dwt = self.dwt(x)
        gate = self.gate_conv(out_attn)
        fused = gate * out_dwt + (1 - gate) * out_attn
        fused = fused + x
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused

class MultiScaleGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, 1)
        )
        self.weight = nn.Parameter(torch.zeros(2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out3 = self.branch3(x)
        out5 = self.branch5(x)
        w = self.weight.softmax(dim=0)
        fused = w[0] * out3 + w[1] * out5
        return self.sigmoid(fused)

class MultiScaleGatedFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.gate_conv = MultiScaleGate(channels)
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch=None):
        out_attn = self.scale_adaptive(x, epoch)
        out_dwt = self.dwt(x)
        gate = self.gate_conv(out_attn)
        fused = gate * out_dwt + (1 - gate) * out_attn
        fused = fused + x
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused

class SequentialFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch=None):
        out_attn = self.scale_adaptive(x, epoch)
        mixed = x + self.norm(out_attn)
        out_dwt = self.dwt(mixed)
        return self.dropout(out_dwt + x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid(avg_out + max_out)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x

class CABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.bn = nn.BatchNorm2d(channels // reduction)
        self.conv_h = nn.Conv2d(channels // reduction, channels, 1)
        self.conv_w = nn.Conv2d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, H, W = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(-2, -1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.bn(self.conv1(y))
        y = nn.functional.relu(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w.transpose(-2, -1)))
        return x * a_h * a_w

# ================== 注册所有模块到 __main__ ==================
for cls in [ScaleAdaptiveAttention, LightweightMultiScaleAttention, DWTBlock, ECABlock, ECASABlock,
            ScaleAdaptiveDWTBlock, GatedFusionBlock, MultiScaleGate, MultiScaleGatedFusionBlock,
            SequentialFusionBlock, SEBlock, CBAMBlock, CABlock]:
    sys.modules['__main__'].__dict__[cls.__name__] = cls

# ================== 模型路径列表（所有13个） ==================
MODEL_PATHS = {
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
WARMUP_ITERS = 50
TEST_ITERS = 100

def measure_latency(model_path):
    if not os.path.exists(model_path):
        print(f"  [SKIP] Missing: {model_path}")
        return None

    model = YOLO(model_path)
    model.model.eval().to(DEVICE)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # 预热
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model.model(dummy_input)
        torch.cuda.synchronize()

    # 正式测量
    start = time.time()
    with torch.no_grad():
        for _ in range(TEST_ITERS):
            _ = model.model(dummy_input)
        torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) / TEST_ITERS * 1000
    return avg_ms

if __name__ == '__main__':
    print("=" * 60)
    print("Measuring Inference Latency for ALL 13 Models")
    print("=" * 60)

    results = {}
    for name, path in MODEL_PATHS.items():
        print(f"\nProcessing {name}...")
        ms = measure_latency(path)
        if ms is not None:
            results[name] = round(ms, 2)
            print(f"  Inference Time: {ms:.2f} ms")
        else:
            results[name] = None

    # 打印汇总表
    print("\n" + "=" * 80)
    print("SUMMARY: Inference Time (ms) - batch=1, 416x416, RTX 3090")
    print("=" * 80)
    print(f"{'Model':40s} {'Latency (ms)':15s}")
    print("-" * 55)
    for name, ms in results.items():
        if ms is not None:
            print(f"{name:40s} {ms:<15.2f}")
    print("=" * 80)