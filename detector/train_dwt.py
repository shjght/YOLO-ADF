# train_dwt_fixed.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 根据空闲卡调整

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# ================== DWTBlock definition (same as in exp09) ==================
class DWTBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.channels = int(channels)
        dropout = float(dropout) if dropout <= 1.0 else dropout / 10.0
        dropout = max(0.0, min(1.0, dropout))

        self.conv_ll = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.ReLU(inplace=True),
        )
        self.conv_lh = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.ReLU(inplace=True),
        )
        self.conv_hl = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.ReLU(inplace=True),
        )
        self.conv_hh = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels, 1),
            nn.ReLU(inplace=True),
        )

        self.norm = nn.GroupNorm(32, self.channels)
        self.dropout = nn.Dropout(dropout)

        self._register_dwt_kernels()
        self._init_weights()

    def _register_dwt_kernels(self):
        kernel_low = torch.tensor([[1, 1],
                                   [1, 1]], dtype=torch.float32) / 4.0
        kernel_high = torch.tensor([[1, -1],
                                    [-1, 1]], dtype=torch.float32) / 4.0
        self.register_buffer('kernel_low', kernel_low)
        self.register_buffer('kernel_high', kernel_high)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_orig = x

        coeffs = self._dwt_decompose(x)

        ll = self.conv_ll(coeffs[0])
        lh = self.conv_lh(coeffs[1])
        hl = self.conv_hl(coeffs[2])
        hh = self.conv_hh(coeffs[3])
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
        kernel_low = self.kernel_low.to(x.device)
        kernel_high = self.kernel_high.to(x.device)

        ll = self._apply_filter(x, kernel_low, kernel_low)
        lh = self._apply_filter(x, kernel_low, kernel_high)
        hl = self._apply_filter(x, kernel_high, kernel_low)
        hh = self._apply_filter(x, kernel_high, kernel_high)
        return [ll, lh, hl, hh]

    def _apply_filter(self, x, kernel_h, kernel_v):
        B, C, H, W = x.shape
        x_dtype = x.dtype
        x_float = x.float()
        x_flat = x_float.reshape(B * C, 1, H, W)

        kernel_h = kernel_h.float().unsqueeze(0).unsqueeze(0)
        kernel_v = kernel_v.float().unsqueeze(0).unsqueeze(0)

        x_h = F.conv2d(x_flat, kernel_h, padding=1)
        x_v = F.conv2d(x_h, kernel_v, padding=1)
        x_v = x_v[:, :, ::2, ::2]

        _, _, h, w = x_v.shape
        x_v = x_v.reshape(B, C, h, w)
        return x_v.to(x_dtype)

# Register
import ultralytics.nn.tasks as tasks
tasks.DWTBlock = DWTBlock
sys.modules['__main__'].DWTBlock = DWTBlock

if __name__ == '__main__':
    DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'
    RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
    EXP_NAME = '04_dwt_consistent'

    # Load baseline and get actual channels
    base_model = YOLO('yolo11m.yaml', verbose=False)
    cfg = base_model.model.yaml
    nc = 10

    with torch.no_grad():
        dummy = torch.randn(1, 3, 416, 416)
        feat = base_model.model.model[:8](dummy)
        actual_channels = feat.shape[1]
    print(f"Actual input channels for layer 8: {actual_channels}")

    # Replace layer 8
    backbone = cfg['backbone']
    backbone[8] = [-1, 1, 'DWTBlock', [actual_channels, 0.0]]

    cfg['width_multiple'] = 1.0
    cfg['depth_multiple'] = 1.0

    model = DetectionModel(cfg, ch=3, nc=nc, verbose=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.info()

    final_model = YOLO('yolo11m.yaml', verbose=False)
    final_model.model = model
    final_model.model.nc = nc
    final_model.model.yaml = cfg

    final_model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=416,
        batch=32,
        device=0,
        workers=2,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-3,
        warmup_epochs=20,
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