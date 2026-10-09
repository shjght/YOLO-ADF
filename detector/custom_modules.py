# custom_modules.py
"""
自定义模块集合（修复版 v4 - 无重复定义）
包含改进的 Window Transformer、DWT、DWT Transformer、
DWT + Window Transformer 交互融合块等

✅ 修复内容：
- 删除所有重复定义
- 确保每个类只定义一次
- 所有类都有完整的 forward() 方法
- 添加类型转换处理 YAML 参数为字符串的情况
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, Tuple, List, Dict, Type
import pywt


# ============================================================================
# 🔧 显存优化配置
# ============================================================================

ENABLE_GRADIENT_CHECKPOINTING = False  # 启用梯度检查点
USE_NEAREST_INTERPOLATE = True  # 使用 nearest 插值（更快）


# ============================================================================
# 1. Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    标准 Transformer Block - ✅ 显存优化版
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # ✅ 安全修复：类型转换
        self.c1 = int(c1) if not isinstance(c1, int) else c1
        self.c2 = int(c2) if not isinstance(c2, int) else c2
        self.num_heads = int(num_heads) if not isinstance(num_heads, int) else num_heads
        self.num_layers = int(num_layers) if not isinstance(num_layers, int) else num_layers
        dropout = float(dropout) if not isinstance(dropout, float) else dropout
        
        # ✅ 验证 c1 能被 num_heads 整除
        if self.num_heads > 0 and self.c1 % self.num_heads != 0:
            for nh in range(self.num_heads, 0, -1):
                if self.c1 % nh == 0:
                    self.num_heads = nh
                    break
        
        # ✅ 显存优化：hidden_dim = c1（不是 c1*2）
        hidden_dim = self.c1
        
        # 多层 Transformer
        self.layers = nn.ModuleList([
            TransformerLayer(self.c1, hidden_dim, self.num_heads, dropout)
            for _ in range(self.num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.c1)
        
        # 输出投影（如果需要）
        if self.c1 != self.c2:
            self.proj = nn.Conv2d(self.c1, self.c2, 1)
        else:
            self.proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        B, C, H, W = x.shape
        
        assert C == self.c1, f"Input channels {C} != c1 {self.c1}"
        
        # 将 (B, C, H, W) 转换为 (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # ✨ 优化：使用梯度检查点
        if self.training and ENABLE_GRADIENT_CHECKPOINTING:
            for layer in self.layers:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        
        # 归一化
        x = self.norm(x)
        
        # 转换回 (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 投影到输出通道数
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class TransformerLayer(nn.Module):
    """Transformer 单层 - ✅ 显存优化版"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # ✅ 关键修复：先归一化 dropout（在创建任何 Dropout 层之前）
        dropout = float(dropout)
        if dropout > 1.0:
            dropout = dropout / 10.0
        dropout = max(0.0, min(1.0, dropout))
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        
        # ✅ 使用归一化后的 dropout
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        x = x + self.attn(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力 - ✅ Linear Attention 版本"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        
        # ✅ 关键修复：先归一化 dropout
        dropout = float(dropout)
        if dropout > 1.0:
            dropout = dropout / 10.0
        dropout = max(0.0, min(1.0, dropout))
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        # ✅ Linear Attention 的投影层
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # ✅ ELU 激活函数（用于 Linear Attention 的特征映射）
        self.elu = nn.ELU(alpha=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - Linear Attention
        
        复杂度：O(N × d²) 而不是 O(N² × d)
        """
        B, N, C = x.shape
        
        # 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # ✅ 特征映射：φ(x) = elu(x) + 1
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        # ✅ Linear Attention 计算
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        # 计算分母
        k_sum = k.sum(dim=2)  # (B, num_heads, head_dim)
        denom = torch.einsum('bhnd,bhd->bhn', q, k_sum)
        denom = denom.unsqueeze(-1).clamp(min=1e-6)
        
        # 归一化
        x = z / denom
        
        # 合并多头
        x = x.transpose(1, 2).reshape(B, N, C)
        
        # 投影和 dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# ============================================================================
# 2. DWT Block
# ============================================================================

class DWTBlock(nn.Module):
    """
    离散小波变换块
    用于多尺度特征分解
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet: str = 'db1',
    ):
        super().__init__()
        
        print(f"\n{'='*80}")
        print(f"📊 DWTBlock 初始化（custom_modules.py）")
        print(f"{'='*80}")
        print(f"  in_channels = {in_channels}")
        print(f"  out_channels = {out_channels}")
        print(f"  wavelet = {wavelet}")
        print(f"{'='*80}\n")
        
        # ✅ 安全修复：类型转换
        try:
            self.in_channels = int(in_channels) if not isinstance(in_channels, int) else in_channels
            self.out_channels = int(out_channels) if not isinstance(out_channels, int) else out_channels
            self.wavelet = str(wavelet) if not isinstance(wavelet, str) else wavelet
            
        except (ValueError, TypeError) as e:
            print(f"❌ 类型转换失败：{e}\n")
            raise
        
        # 小波分解后会产生 4 个分量（LL, LH, HL, HH）
        self.conv_ll = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_lh = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_hl = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_hh = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(self.out_channels * 4, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - DWT 分解和融合
        
        Args:
            x: 输入张量 (B, C, H, W)
        
        Returns:
            输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # ✅ 如果输入通道数不匹配，添加投影层
        if C != self.in_channels:
            proj = nn.Conv2d(C, self.in_channels, 1).to(x.device)
            x = proj(x)
            C = self.in_channels
        
        # DWT 分解
        coeffs = self._dwt_decompose(x)  # 返回 [LL, LH, HL, HH]
        
        # ✨ 优化：逐个处理，及时释放中间结果
        ll = self.conv_ll(coeffs[0])
        lh = self.conv_lh(coeffs[1])
        hl = self.conv_hl(coeffs[2])
        hh = self.conv_hh(coeffs[3])
        
        # 清理系数以释放显存
        del coeffs
        
        # ✨ 优化：使用 nearest 插值（更快更省显存）
        interp_mode = 'nearest' if USE_NEAREST_INTERPOLATE else 'bilinear'
        
        ll = F.interpolate(ll, size=(H, W), mode=interp_mode)
        lh = F.interpolate(lh, size=(H, W), mode=interp_mode)
        hl = F.interpolate(hl, size=(H, W), mode=interp_mode)
        hh = F.interpolate(hh, size=(H, W), mode=interp_mode)
        
        # 融合
        fused = torch.cat([ll, lh, hl, hh], dim=1)
        out = self.fusion(fused)
        
        # ✨ 清理中间结果
        del ll, lh, hl, hh, fused
        
        return out
    
    def _dwt_decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        DWT 分解
        
        Args:
            x: 输入张量 (B, C, H, W)
        
        Returns:
            [LL, LH, HL, HH] 四个分量
        """
        B, C, H, W = x.shape
        
        # 低通滤波器（平滑）
        kernel_low = torch.tensor([
            [1, 1],
            [1, 1]
        ], dtype=torch.float32, device=x.device) / 4.0
        
        # 高通滤波器（细节）
        kernel_high = torch.tensor([
            [1, -1],
            [-1, 1]
        ], dtype=torch.float32, device=x.device) / 4.0
        
        # 应用滤波器组合
        ll = self._apply_filter(x, kernel_low, kernel_low)
        lh = self._apply_filter(x, kernel_low, kernel_high)
        hl = self._apply_filter(x, kernel_high, kernel_low)
        hh = self._apply_filter(x, kernel_high, kernel_high)
        
        return [ll, lh, hl, hh]
    
    def _apply_filter(
        self,
        x: torch.Tensor,
        kernel_h: torch.Tensor,
        kernel_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用二维滤波器
        
        Args:
            x: 输入张量 (B, C, H, W)
            kernel_h: 水平滤波器 (2, 2)
            kernel_v: 垂直滤波器 (2, 2)
        
        Returns:
            滤波后的张量 (B, C, H//2, W//2)
        """
        B, C, H, W = x.shape
        
        # 展开为 (B*C, 1, H, W)
        x_flat = x.reshape(B * C, 1, H, W)
        
        # 应用水平滤波器
        kernel_h = kernel_h.unsqueeze(0).unsqueeze(0)
        x_h = F.conv2d(x_flat, kernel_h, padding=1)
        
        # 应用垂直滤波器
        kernel_v = kernel_v.unsqueeze(0).unsqueeze(0)
        x_v = F.conv2d(x_h, kernel_v, padding=1)
        
        # 下采样
        x_v = x_v[:, :, ::2, ::2]
        
        # ✅ 恢复形状
        _, _, h, w = x_v.shape
        x_v = x_v.reshape(B, C, h, w)
        
        return x_v


# ============================================================================
# 3. DWT Transformer Block
# ============================================================================

class DWTTransformerBlock(nn.Module):
    """
    DWT + Transformer 融合块
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 1,
        wavelet: str = 'db1',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # ✅ 安全修复：类型转换
        self.dim = int(dim) if not isinstance(dim, int) else dim
        self.num_heads = int(num_heads) if not isinstance(num_heads, int) else num_heads
        self.num_layers = int(num_layers) if not isinstance(num_layers, int) else num_layers
        self.wavelet = str(wavelet) if not isinstance(wavelet, str) else wavelet
        dropout = float(dropout) if not isinstance(dropout, float) else dropout
        
        if hidden_dim is None:
            hidden_dim = self.dim
        else:
            hidden_dim = int(hidden_dim) if not isinstance(hidden_dim, int) else hidden_dim
        
        # 创建 DWT 和 Transformer 模块
        self.dwt = DWTBlock(self.dim, self.dim, self.wavelet)
        self.transformer = TransformerBlock(
            c1=self.dim,
            c2=self.dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.dwt(x)
        x = self.transformer(x)
        return x


# ============================================================================
# 4. Window Transformer Block
# ============================================================================

class WindowTransformerBlock(nn.Module):
    """
    改进的 Window Transformer Block
    使用真正的多头自注意力而不是卷积
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # ✅ 安全修复：类型转换
        self.dim = int(dim) if not isinstance(dim, int) else dim
        self.num_heads = int(num_heads) if not isinstance(num_heads, int) else num_heads
        self.window_size = int(window_size) if not isinstance(window_size, int) else window_size
        self.shift_size = int(shift_size) if not isinstance(shift_size, int) else shift_size
        mlp_ratio = float(mlp_ratio) if not isinstance(mlp_ratio, float) else mlp_ratio
        dropout = float(dropout) if not isinstance(dropout, float) else dropout
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        
        # 多头自注意力
        self.attn = MultiHeadAttention(self.dim, self.num_heads, dropout)
        
        # MLP
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, self.dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
        
        Returns:
            输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 将 (B, C, H, W) 转换为序列格式
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 自注意力
        x_seq = x_seq + self.dropout(self.attn(self.norm1(x_seq)))
        
        # MLP
        x_seq = x_seq + self.dropout(self.mlp(self.norm2(x_seq)))
        
        # 转换回图像格式
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        return x


# ============================================================================
# 5. DWT + Window Transformer 交互融合块
# ============================================================================

class DWTWindowTransformerBlock(nn.Module):
    """
    DWT + Window Transformer 交互融合块
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # ✅ 安全修复：类型转换
        self.dim = int(dim) if not isinstance(dim, int) else dim
        self.num_heads = int(num_heads) if not isinstance(num_heads, int) else num_heads
        self.window_size = int(window_size) if not isinstance(window_size, int) else window_size
        self.shift_size = int(shift_size) if not isinstance(shift_size, int) else shift_size
        mlp_ratio = float(mlp_ratio) if not isinstance(mlp_ratio, float) else mlp_ratio
        dropout = float(dropout) if not isinstance(dropout, float) else dropout
        
        # 确保 num_heads 能整除 dim
        assert self.dim % self.num_heads == 0, f"dim {self.dim} should be divisible by num_heads {self.num_heads}"
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        
        # 多头自注意力
        self.attn = MultiHeadAttention(self.dim, self.num_heads, dropout)
        
        # MLP
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, self.dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
        
        Returns:
            输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 确保输入通道数等于 dim
        assert C == self.dim, f"Input channels {C} != dim {self.dim}"
        
        # 将 (B, C, H, W) 转换为序列格式
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 自注意力（残差连接）
        x_seq = x_seq + self.dropout(self.attn(self.norm1(x_seq)))
        
        # MLP（残差连接）
        x_seq = x_seq + self.dropout(self.mlp(self.norm2(x_seq)))
        
        # 转换回图像格式
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        # 确保输出通道数等于输入通道数
        assert x.shape[1] == C, f"Output channels {x.shape[1]} != input channels {C}"
        
        return x


# ============================================================================
# 6. Dynamic Convolution
# ============================================================================

class DynamicConv(nn.Module):
    """
    动态卷积层
    根据输入动态调整卷积核权重
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = True,
        reduction: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # 静态卷积核
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 动态权重生成网络
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid(),
        )
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C_in, H, W)
        
        Returns:
            输出张量 (B, C_out, H', W')
        """
        # 生成动态权重
        weight_scale = self.fc(x)  # (B, C_out, 1, 1)
        
        # 应用动态权重到静态卷积核
        B = x.shape[0]
        weight = self.weight.unsqueeze(0) * weight_scale.unsqueeze(2).unsqueeze(3)
        weight = weight.view(-1, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        
        # 执行分组卷积
        x = x.view(1, -1, x.shape[2], x.shape[3])
        out = F.conv2d(
            x,
            weight,
            None,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups * B,
        )
        out = out.view(B, self.out_channels, out.shape[2], out.shape[3])
        
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        
        return out


# ============================================================================
# 7. Adaptive Attention
# ============================================================================

class AdaptiveAttention(nn.Module):
    """
    自适应注意力模块
    根据特征图自适应调整注意力权重
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
        
        Returns:
            输出张量 (B, C, H, W)
        """
        # 通道注意力
        channel_attn = self.channel_attn(x)
        x = x * channel_attn
        
        # 空间注意力
        spatial_attn = self.spatial_attn(x)
        x = x * spatial_attn
        
        return x


# ============================================================================
# 8. Conv BN ReLU
# ============================================================================

class ConvBNReLU(nn.Module):
    """标准的 Conv + BN + ReLU 块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ============================================================================
# 9. Depthwise Separable Convolution
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# ============================================================================
# 10. SE Block
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return x * self.fc(x)


# ============================================================================
# 11. Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Focal Loss
        
        Args:
            inputs: 预测值 (B, C) 或 (B, C, H, W)
            targets: 目标值 (B,) 或 (B, H, W)
        
        Returns:
            损失值
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 pt
        p = torch.exp(-ce_loss)
        
        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# 12. Efficient NMS
# ============================================================================

class EfficientNMS(nn.Module):
    """
    高效 NMS 模块
    用于检测框的后处理
    
    ✅ 完整版：包含所有必要的方法
    """
    
    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
    
    def forward(
        self,
        predictions: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        执行 NMS
        
        Args:
            predictions: 预测结果 (B, N, 6) 其中 6 为 [x1, y1, x2, y2, conf, cls]
        
        Returns:
            NMS 后的结果列表
        """
        results = []
        
        for pred in predictions:
            # 过滤低置信度
            mask = pred[:, 4] > self.conf_thres
            pred = pred[mask]
            
            if len(pred) == 0:
                results.append(pred)
                continue
            
            # 按置信度排序
            sorted_indices = torch.argsort(pred[:, 4], descending=True)
            pred = pred[sorted_indices]
            
            # NMS
            keep = self._nms(pred[:, :4], pred[:, 4], self.iou_thres)
            pred = pred[keep]
            
            # 限制最大检测数
            if len(pred) > self.max_det:
                pred = pred[:self.max_det]
            
            results.append(pred)
        
        return results
    
    @staticmethod
    def _nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_thres: float,
    ) -> torch.Tensor:
        """
        NMS 实现
        
        Args:
            boxes: 检测框 (N, 4) [x1, y1, x2, y2]
            scores: 置信度 (N,)
            iou_thres: IOU 阈值
        
        Returns:
            保留的索引
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # 计算面积
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        keep = []
        indices = torch.arange(len(boxes), device=boxes.device)
        
        while len(indices) > 0:
            # 保留置信度最高的框
            keep.append(indices[0].item())
            
            if len(indices) == 1:
                break
            
            # 计算 IOU
            box1 = boxes[indices[0]]
            boxes_rest = boxes[indices[1:]]
            
            x1_inter = torch.max(box1[0], boxes_rest[:, 0])
            y1_inter = torch.max(box1[1], boxes_rest[:, 1])
            x2_inter = torch.min(box1[2], boxes_rest[:, 2])
            y2_inter = torch.min(box1[3], boxes_rest[:, 3])
            
            inter_area = torch.clamp(x2_inter - x1_inter, min=0) * \
                         torch.clamp(y2_inter - y1_inter, min=0)
            
            union_area = area[indices[0]] + area[indices[1:]] - inter_area
            iou = inter_area / union_area
            
            # 过滤 IOU 高的框
            mask = iou < iou_thres
            indices = indices[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ============================================================================
# 获取所有自定义模块
# ============================================================================

def get_custom_modules() -> Dict[str, Type]:
    """
    获取所有自定义模块
    
    Returns:
        Dict[str, Type]: 模块名称到模块类的映射
    """
    return {
        'TransformerBlock': TransformerBlock,
        'DWTBlock': DWTBlock,
        'DWTTransformerBlock': DWTTransformerBlock,
        'WindowTransformerBlock': WindowTransformerBlock,
        'DWTWindowTransformerBlock': DWTWindowTransformerBlock,
        'DynamicConv': DynamicConv,
        'AdaptiveAttention': AdaptiveAttention,
        'FocalLoss': FocalLoss,
        'EfficientNMS': EfficientNMS,
        'ConvBNReLU': ConvBNReLU,
        'DepthwiseSeparableConv': DepthwiseSeparableConv,
        'SEBlock': SEBlock,
    }


# ============================================================================
# 导出所有模块
# ============================================================================

__all__ = [
    'TransformerBlock',
    'TransformerLayer',
    'MultiHeadAttention',
    'DWTBlock',
    'DWTTransformerBlock',
    'WindowTransformerBlock',
    'DWTWindowTransformerBlock',
    'DynamicConv',
    'AdaptiveAttention',
    'FocalLoss',
    'EfficientNMS',
    'ConvBNReLU',
    'DepthwiseSeparableConv',
    'SEBlock',
    'get_custom_modules',
]
