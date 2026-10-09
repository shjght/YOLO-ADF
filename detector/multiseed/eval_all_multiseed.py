# -*- coding: utf-8 -*-
"""
eval_all_multiseed.py
对指定 seed 的所有模型 best.pt 一次性跑完 BDD + ACDC 评测，
输出论文所需的全部数值（域内 mAP50/mAP50-95 + 跨域四场景 mAP50 + SSDI）。

用法：
  $env:XLW_SEED=0; python multiseed/eval_all_multiseed.py
  $env:XLW_SEED=1; python multiseed/eval_all_multiseed.py
  $env:XLW_SEED=2; python multiseed/eval_all_multiseed.py

输出：results-2/_eval_all_s{seed}.csv（一行一个模型，包含所有列）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics import YOLO

# ================== SEED 配置 ==================
SEED = int(os.environ.get('XLW_SEED', '0'))
SUF = '' if SEED == 0 else f'_s{SEED}'

RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
BDD_YAML    = r'C:\Users\KUST\xlw\bdd100k.yaml'
ACDC_BASE   = r'C:\Users\KUST\xlw\ACDC'
EVAL_CLASSES = [0, 1, 2, 3, 4, 5, 8]  # BDD100K ∩ ACDC 共有 7 类

# ================== 自定义模块定义 ==================
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
        qkv = self.qkv(x); q, k, v = qkv.chunk(3, dim=1)
        q = q + self.dw3(q) + self.dw5(q)
        k = k + self.dw3(k) + self.dw5(k)
        v = v + self.dw3(v) + self.dw5(v)
        q = q.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1); attn = self.drop(attn)
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
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4), nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels//8, 1),
            nn.BatchNorm2d(channels//8), nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, num_scales, 1))
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
        d = float(dropout) if float(dropout) <= 1.0 else float(dropout) / 10.0
        d = max(0.0, min(1.0, d))
        self.conv_ll = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_lh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hl = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.conv_hh = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv2d(channels * 4, channels, 1), nn.ReLU(inplace=True))
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(d)
        self.register_buffer('kernel_low', torch.tensor([[1,1],[1,1]], dtype=torch.float32) / 4.0)
        self.register_buffer('kernel_high', torch.tensor([[1,-1],[-1,1]], dtype=torch.float32) / 4.0)
    def forward(self, x):
        B, C, H, W = x.shape; orig = x; coeffs = self._dwt(x)
        ll = self.conv_ll(coeffs[0]); lh = self.conv_lh(coeffs[1])
        hl = self.conv_hl(coeffs[2]); hh = self.conv_hh(coeffs[3]); del coeffs
        ll = F.interpolate(ll, size=(H,W), mode='nearest')
        lh = F.interpolate(lh, size=(H,W), mode='nearest')
        hl = F.interpolate(hl, size=(H,W), mode='nearest')
        hh = F.interpolate(hh, size=(H,W), mode='nearest')
        return self.dropout(self.norm(self.fusion(torch.cat([ll,lh,hl,hh], dim=1)) + orig))
    def _dwt(self, x):
        kl, kh = self.kernel_low.to(x.device), self.kernel_high.to(x.device)
        return [self._f(x,kl,kl), self._f(x,kl,kh), self._f(x,kh,kl), self._f(x,kh,kh)]
    def _f(self, x, kh, kv):
        B,C,H,W = x.shape; xr = x.float().reshape(B*C,1,H,W)
        xv = F.conv2d(F.conv2d(xr, kh.float().unsqueeze(0).unsqueeze(0), padding=1),
                      kv.float().unsqueeze(0).unsqueeze(0), padding=1)[:, :, ::2, ::2]
        return xv.reshape(B, C, xv.shape[2], xv.shape[3]).to(x.dtype)

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
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        return x * self.sigmoid(y)

class ECASABlock(nn.Module):
    def __init__(self, channels, **kw):
        super().__init__()
        self.eca = ECABlock(channels)
        self.attn = ScaleAdaptiveAttention(channels, **kw)
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(kw.get('dropout', 0.0))
    def forward(self, x, epoch=None):
        return self.dropout(self.norm(self.attn(self.eca(x), epoch) + x))

class ScaleAdaptiveDWTBlock(nn.Module):
    def __init__(self, channels, **kw):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, **kw)
        self.dwt = DWTBlock(channels, kw.get('dropout', 0.0))
        self.fusion = nn.Sequential(nn.Conv2d(channels*2, channels, 1), nn.ReLU(inplace=True))
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(kw.get('dropout', 0.0))
    def forward(self, x, epoch=None):
        return self.dropout(self.norm(self.fusion(
            torch.cat([self.scale_adaptive(x, epoch), self.dwt(x)], dim=1)) + x)

class GatedFusionBlock(nn.Module):
    def __init__(self, channels, **kw):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, **kw)
        self.dwt = DWTBlock(channels, kw.get('dropout', 0.0))
        self.gate_conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                       nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(kw.get('dropout', 0.0))
    def forward(self, x, epoch=None):
        a, d = self.scale_adaptive(x, epoch), self.dwt(x)
        g = self.gate_conv(a)
        return self.dropout(self.norm(g*d + (1-g)*a + x))

class MultiScaleGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b3 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                 nn.Conv2d(channels, channels, 1))
        self.b5 = nn.Sequential(nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
                                 nn.Conv2d(channels, channels, 1))
        self.weight = nn.Parameter(torch.zeros(2))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.weight.softmax(dim=0)
        return self.sigmoid(w[0]*self.b3(x) + w[1]*self.b5(x))

class MultiScaleGatedFusionBlock(nn.Module):
    def __init__(self, channels, **kw):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, **kw)
        self.dwt = DWTBlock(channels, kw.get('dropout', 0.0))
        self.gate_conv = MultiScaleGate(channels)
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(kw.get('dropout', 0.0))
    def forward(self, x, epoch=None):
        a, d = self.scale_adaptive(x, epoch), self.dwt(x)
        g = self.gate_conv(a)
        return self.dropout(self.norm(g*d + (1-g)*a + x))

class SequentialFusionBlock(nn.Module):
    def __init__(self, channels, **kw):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, **kw)
        self.dwt = DWTBlock(channels, kw.get('dropout', 0.0))
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(kw.get('dropout', 0.0))
    def forward(self, x, epoch=None):
        return self.dropout(self.dwt(x + self.norm(self.scale_adaptive(x, epoch))) + x)

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
        x = x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        a = torch.mean(x, dim=1, keepdim=True); m, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([a, m], dim=1))

class CABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)); self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels//reduction, 1); self.bn = nn.BatchNorm2d(channels//reduction)
        self.conv_h = nn.Conv2d(channels//reduction, channels, 1); self.conv_w = nn.Conv2d(channels//reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, H, W = x.shape
        y = torch.cat([self.pool_h(x), self.pool_w(x).transpose(-2,-1)], dim=2)
        y = F.relu(self.bn(self.conv1(y)))
        xh, xw = torch.split(y, [H, W], dim=2)
        return x * self.sigmoid(self.conv_h(xh)) * self.sigmoid(self.conv_w(xw.transpose(-2,-1)))

# 注册所有模块
for cls in [ScaleAdaptiveAttention, LightweightMultiScaleAttention, DWTBlock, ECABlock, ECASABlock,
            ScaleAdaptiveDWTBlock, GatedFusionBlock, MultiScaleGate, MultiScaleGatedFusionBlock,
            SequentialFusionBlock, SEBlock, CBAMBlock, CABlock]:
    sys.modules['__main__'].__dict__[cls.__name__] = cls

# ================== 模型列表 ==================
MODELS = {
    'Baseline':  r'01_baseline_consistent',
    'DWT':       r'04_dwt_consistent',
    'AMAR':      r'08_scale_adaptive_lightweight',
    'ADF-F':     r'09_scale_adaptive_dwt_lightweight',
    'ADF-S':     r'11_gated_fusion',
    'ADF-N':     r'12_multiscale_gated_fusion',
    'ADF-R':     r'13_sequential_fusion',
    'SE':        r'comp01_se',
    'CBAM':      r'comp02_cbam',
    'ECA':       r'comp03_eca',
    'CA':        r'comp04_ca',
}

ACDC_SCENARIOS = {
    'Fog':   os.path.join(ACDC_BASE, 'acdc_fog.yaml'),
    'Night': os.path.join(ACDC_BASE, 'acdc_night.yaml'),
    'Rain':  os.path.join(ACDC_BASE, 'acdc_rain.yaml'),
    'Snow':  os.path.join(ACDC_BASE, 'acdc_snow.yaml'),
}

# ================== 评测主流程 ==================
out_csv = os.path.join(RESULTS_DIR, f'_eval_all_s{SEED}.csv')
results = []

print('='*70)
print(f'Full Evaluation (BDD + ACDC) — Seed={SEED}')
print('='*70)

for name, expdir in MODELS.items():
    model_path = os.path.join(RESULTS_DIR, expdir + SUF, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f'\nSKIP: {name} (best.pt not found)')
        results.append({'Model': name, 'BDD_mAP50': '', 'BDD_mAP5095': '',
                        'Fog': '', 'Night': '', 'Rain': '', 'Snow': '', 'SSDI': ''})
        continue

    print(f'\n{"="*60}\nEvaluating: {name}\n{"="*60}')
    row = {'Model': name}

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f'  ERROR loading: {e}')
        results.append({**row, 'BDD_mAP50':'', 'BDD_mAP5095':'', 'Fog':'', 'Night':'', 'Rain':'', 'Snow':'', 'SSDI':''})
        continue

    # --- BDD 域内 ---
    try:
        m = model.val(data=BDD_YAML, split='val', workers=0, verbose=False)
        row['BDD_mAP50'] = round(m.box.map50, 4)
        row['BDD_mAP5095'] = round(m.box.map, 4)
        print(f'  BDD:  mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}')
    except Exception as e:
        print(f'  BDD ERROR: {e}')
        row['BDD_mAP50'] = ''; row['BDD_mAP5095'] = ''

    # --- ACDC 跨域 ---
    ssdi_vals = []
    for sc, yaml_path in ACDC_SCENARIOS.items():
        try:
            m = model.val(data=yaml_path, split='val', workers=0, classes=EVAL_CLASSES, verbose=False)
            v = round(m.box.map50, 4)
            row[sc] = v
            ssdi_vals.append(v)
            print(f'  {sc:5s}: mAP50={v:.4f}')
        except Exception as e:
            print(f'  {sc} ERROR: {e}')
            row[sc] = ''

    # --- SSDI ---
    if len(ssdi_vals) == 4:
        row['SSDI'] = round(sum(ssdi_vals) / 4, 4)
        print(f'  SSDI: {row["SSDI"]:.4f}')
    else:
        row['SSDI'] = ''

    results.append(row)

# ================== 保存 CSV ==================
fields = ['Model', 'BDD_mAP50', 'BDD_mAP5095', 'Fog', 'Night', 'Rain', 'Snow', 'SSDI']
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f'\n{"="*70}')
print(f'Done! Results saved to: {out_csv}')
print(f'{"="*70}')
