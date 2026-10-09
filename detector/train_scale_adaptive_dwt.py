# train_exp09_scale_adaptive_dwt_final.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 根据空闲卡调整

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# ================== 轻量模块定义 ==================
class LightweightMultiScaleAttention(nn.Module):
    """轻量版多尺度注意力（与 exp08 完全一致）"""
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
        # 数值稳定性 L2 归一化
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(-2, -1).contiguous().reshape(B, C, H, W)
        out = self.proj(out)
        return self.norm(out + x)

class ScaleAdaptiveAttention(nn.Module):
    """多尺度自适应注意力（与 exp08 完全一致）"""
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
        s = self.small(x); m = self.medium(x); l = self.large(x)
        w = self.weight_net(x)
        if epoch is not None:
            t = self.temperature * (1.0 - 0.5 * epoch/300)
        else:
            t = self.temperature
        w = (w / t).softmax(1)
        fused = w[:,0:1]*s + w[:,1:2]*m + w[:,2:3]*l
        return self.dropout(self.norm(fused + x))

# ================== DWT 模块定义（与 exp04 完全一致） ==================
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

# ================== 融合模块定义 ==================
class ScaleAdaptiveDWTBlock(nn.Module):
    """Scale-Adaptive Attention + DWT 双分支融合（轻量版）"""
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.scale_adaptive = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.dwt = DWTBlock(channels, dropout)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch=None):
        out1 = self.scale_adaptive(x, epoch)
        out2 = self.dwt(x)
        fused = torch.cat([out1, out2], dim=1)
        fused = self.fusion(fused)
        fused = fused + x  # 残差连接
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused

# ================== 注册自定义模块 ==================
import ultralytics.nn.tasks as tasks
tasks.ScaleAdaptiveDWTBlock = ScaleAdaptiveDWTBlock
tasks.ScaleAdaptiveAttention = ScaleAdaptiveAttention
tasks.DWTBlock = DWTBlock
tasks.LightweightMultiScaleAttention = LightweightMultiScaleAttention
sys.modules['__main__'].ScaleAdaptiveDWTBlock = ScaleAdaptiveDWTBlock
sys.modules['__main__'].ScaleAdaptiveAttention = ScaleAdaptiveAttention
sys.modules['__main__'].DWTBlock = DWTBlock
sys.modules['__main__'].LightweightMultiScaleAttention = LightweightMultiScaleAttention

if __name__ == '__main__':
    DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'
    RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
    EXP_NAME = '09_scale_adaptive_dwt_lightweight'

    # 1. 加载基线模型，获取缩放后的配置
    base_model = YOLO('yolo11m.yaml', verbose=False)
    cfg = base_model.model.yaml
    nc = 10

    # 2. 计算第8层实际输入通道数 (512)
    with torch.no_grad():
        dummy = torch.randn(1, 3, 416, 416)
        feat = base_model.model.model[:8](dummy)
        actual_channels = feat.shape[1]
    print(f"Actual input channels for layer 8: {actual_channels}")

    # 3. 修改 backbone 配置：替换第8层为融合模块
    backbone = cfg['backbone']
    backbone[8] = [-1, 1, 'ScaleAdaptiveDWTBlock', [actual_channels, 8, 3, 4.0, 0.0]]

    # 4. 防止二次缩放
    cfg['width_multiple'] = 1.0
    cfg['depth_multiple'] = 1.0

    # 5. 重建模型
    model = DetectionModel(cfg, ch=3, nc=nc, verbose=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.info()

    # 6. 包装为 YOLO 对象并训练
    final_model = YOLO('yolo11m.yaml', verbose=False)
    final_model.model = model
    final_model.model.nc = nc
    final_model.model.yaml = cfg

    final_model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=416,
        batch=32,
        device=2,
        workers=2,               # 避免 Windows 多进程死锁
        amp=True,
        optimizer='AdamW',
        lr0=0.001,               # 学习率与 exp01/exp04/exp08 完全一致
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-3,
        warmup_epochs=50,        # 延长预热，保证注意力分支稳定（与 exp08 相同）
        warmup_momentum=0.8,
        warmup_bias_lr=0.00005,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.2,
        copy_paste=0.0, erasing=0.0,
        close_mosaic=15,
        nbs=64,
        save=True, save_period=10,
        project=RESULTS_DIR, name=EXP_NAME,
        exist_ok=True,
        patience=50,
        plots=False, val=True,
        cache=False, verbose=True,
        deterministic=True, seed=0,
        cos_lr=True,
        fraction=1.0,
    )