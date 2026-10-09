# yolo_compatible_modules.py
"""
YOLO 兼容的模块包装器
将自定义模块适配到 YOLO 的 c1, c2 格式
"""

import torch
import torch.nn as nn
from custom_modules import (
    TransformerBlock as OriginalTransformerBlock,
    DWTBlock as OriginalDWTBlock,
 #   DWTTransformerBlock as OriginalDWTTransformerBlock,
    WindowTransformerBlock as OriginalWindowTransformerBlock,
    DWTWindowTransformerBlock as OriginalDWTWindowTransformerBlock,
    DynamicConv as OriginalDynamicConv,
    AdaptiveAttention as OriginalAdaptiveAttention,
    ConvBNReLU as OriginalConvBNReLU,
    DepthwiseSeparableConv as OriginalDepthwiseSeparableConv,
    SEBlock as OriginalSEBlock,
)


# ============================================================================
# TransformerBlock 包装器
# ============================================================================

class TransformerBlock(nn.Module):
    """YOLO 兼容的 TransformerBlock"""
    
    def __init__(self, c1, *args, **kwargs):
        super().__init__()
        
        # ✅ 参数提取
        if kwargs:
            num_heads = int(kwargs.get('num_heads', 8))
            num_layers = int(kwargs.get('num_layers', 1))
            dropout = float(kwargs.get('dropout', 0.0))
        elif len(args) >= 2:
            num_heads = int(args[0])
            num_layers = int(args[1])
            dropout = float(args[2]) if len(args) > 2 else 0.0
        elif len(args) == 1:
            num_heads = int(args[0])
            num_layers = 1
            dropout = 0.0
        else:
            num_heads = 8
            num_layers = 1
            dropout = 0.0
        
        # ✅ 参数验证
        c1 = int(c1)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        dropout = float(dropout)
        
        assert c1 > 0
        assert num_heads > 0
        assert num_layers > 0
        
        # ✅ 调整 num_heads
        if num_heads > c1:
            num_heads = c1
        if c1 % num_heads != 0:
            for nh in range(num_heads, 0, -1):
                if c1 % nh == 0:
                    num_heads = nh
                    break
        
        self.c1 = c1
        self.c2 = c1
        
        # ✅ 创建模块
        self.transformer = OriginalTransformerBlock(
            c1=c1,
            c2=c1,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        if c1 != self.c2:
            self.proj = nn.Conv2d(c1, self.c2, 1)
        else:
            self.proj = None
    
    def forward(self, x):
        x = self.transformer(x)
        if self.proj is not None:
            x = self.proj(x)
        return x







# ============================================================================
# DWTBlock 包装器
# ============================================================================

class DWTBlock(nn.Module):
    """
    YOLO 兼容的 DWTBlock
    
    YAML 格式：
    - [-1, 1, DWTBlock, [wavelet, level]]
    
    示例：
    - [-1, 1, DWTBlock, [db1, 1]]
    """
    
    def __init__(self, c1, c2, wavelet='db1', level=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # 使用原始模块
        self.dwt = OriginalDWTBlock(
            in_channels=c1,
            out_channels=c2,
            wavelet=wavelet
        )
    
    def forward(self, x):
        return self.dwt(x)


# ============================================================================
# DWTTransformerBlock 包装器
# ============================================================================
class DWTTransformerBlock(nn.Module):
    """YOLO 兼容的 DWTTransformerBlock
    
    ✅ 完全重写：正确处理 YOLO 的参数传递方式
    
    YAML 格式：
    - [-1, 1, DWTTransformerBlock, [num_heads, num_layers, hidden_dim, dropout]]
    
    示例：
    - [-1, 1, DWTTransformerBlock, [8, 4, 1024, 0.0]]
    
    参数说明：
    - c1: 输入通道数（由 YOLO 自动传入，从前一层获取）✅ 这是实际通道数！
    - c2: 第1个参数（num_heads）
    - args[0]: 第2个参数（num_layers）
    - args[1]: 第3个参数（hidden_dim）
    - args[2]: 第4个参数（dropout）
    """
    
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()
        
        # ✅ 关键修复：先保存实际的输入通道数！
        actual_c1 = int(c1) if not isinstance(c1, int) else c1
        
        # ✅ 初始化所有变量（防止 UnboundLocalError）
        num_heads = 8
        num_layers = 1
        hidden_dim = actual_c1
        dropout = 0.0
        
        # ✅ 正确解析参数
        # YAML 格式：[num_heads, num_layers, hidden_dim, dropout]
        # c2 是 num_heads，args 是 (num_layers, hidden_dim, dropout)
        try:
            if len(args) >= 3:
                # 完整格式：[8, 4, 1024, 0.0]
                num_heads = int(c2) if not isinstance(c2, int) else c2
                num_layers = int(args[0]) if not isinstance(args[0], int) else args[0]
                hidden_dim = int(args[1]) if not isinstance(args[1], int) else args[1]
                dropout = float(args[2]) if not isinstance(args[2], (float, int)) else args[2]
                
            elif len(args) == 2:
                # 简化格式：[8, 4, 1024]
                num_heads = int(c2) if not isinstance(c2, int) else c2
                num_layers = int(args[0]) if not isinstance(args[0], int) else args[0]
                hidden_dim = int(args[1]) if not isinstance(args[1], int) else args[1]
                dropout = 0.0
                
            elif len(args) == 1:
                # 最小格式：[8, 4]
                num_heads = int(c2) if not isinstance(c2, int) else c2
                num_layers = int(args[0]) if not isinstance(args[0], int) else args[0]
                hidden_dim = actual_c1  # 默认使用 actual_c1
                dropout = 0.0
                
            else:
                # 无参数格式：使用默认值
                num_heads = 8
                num_layers = 1
                hidden_dim = actual_c1
                dropout = 0.0
                
        except (ValueError, TypeError) as e:
            # 如果解析失败，使用默认值
            num_heads = 8
            num_layers = 1
            hidden_dim = actual_c1
            dropout = 0.0
        
        # ✅ 关键修复：保存实际的输入输出通道数
        self.c1 = actual_c1  # ✅ 实际输入通道数
        self.c2 = actual_c1  # ✅ 输出通道数 = 输入通道数
        
        # ✅ 参数验证
        assert self.c1 > 0, f"❌ c1={self.c1} 必须 > 0"
        assert num_heads > 0, f"❌ num_heads={num_heads} 必须 > 0"
        assert num_layers > 0, f"❌ num_layers={num_layers} 必须 > 0"
        
        # ✅ 调整 num_heads 以确保能整除 c1
        if self.c1 % num_heads != 0:
            for nh in range(num_heads, 0, -1):
                if self.c1 % nh == 0:
                    num_heads = nh
                    break
        
        assert self.c1 % num_heads == 0, f"❌ c1={self.c1} 必须能被 num_heads={num_heads} 整除"
        
        # ✅ 直接创建 DWT 和 Transformer 模块
        self.dwt = OriginalDWTBlock(
            in_channels=self.c1,
            out_channels=self.c1,
            wavelet='db1'
        )
        
        self.transformer = OriginalTransformerBlock(
            c1=self.c1,
            c2=self.c1,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.dwt(x)
        x = self.transformer(x)
        return x













# ============================================================================
# WindowTransformerBlock 包装器
# ============================================================================

class WindowTransformerBlock(nn.Module):
    """YOLO 兼容的 WindowTransformerBlock"""
    
    def __init__(self, c1, *args, **kwargs):
        super().__init__()
        
        # ✅ 参数提取
        if kwargs:
            num_heads = int(kwargs.get('num_heads', 8))
            num_layers = int(kwargs.get('num_layers', 1))
            window_size = int(kwargs.get('window_size', 0))
            mlp_ratio = float(kwargs.get('mlp_ratio', 4.0))
            dropout = float(kwargs.get('dropout', 0.0))
        elif len(args) >= 3:
            num_heads = int(args[0])
            num_layers = int(args[1])
            window_size = int(args[2])
            mlp_ratio = float(args[3]) if len(args) > 3 else 4.0
            dropout = float(args[4]) if len(args) > 4 else 0.0
        else:
            num_heads = 8
            num_layers = 1
            window_size = 0
            mlp_ratio = 4.0
            dropout = 0.0
        
        # ✅ 参数验证
        c1 = int(c1)
        num_heads = int(num_heads)
        
        assert c1 > 0
        assert num_heads > 0
        assert num_layers > 0
        
        # ✅ 调整 num_heads
        if num_heads > c1:
            num_heads = c1
        if c1 % num_heads != 0:
            for nh in range(num_heads, 0, -1):
                if c1 % nh == 0:
                    num_heads = nh
                    break
        
        self.c1 = c1
        self.c2 = c1
        
        # ✅ 创建模块
        self.window_transformer = OriginalWindowTransformerBlock(
            dim=c1,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        if c1 != self.c2:
            self.proj = nn.Conv2d(c1, self.c2, 1)
        else:
            self.proj = None
    
    def forward(self, x):
        x = self.window_transformer(x)
        if self.proj is not None:
            x = self.proj(x)
        return x







# ============================================================================
# DWTWindowTransformerBlock 包装器（✅ 完全修复版）
# ============================================================================

class DWTWindowTransformerBlock(nn.Module):
    """YOLO 兼容的 DWTWindowTransformerBlock"""
    
    def __init__(self, c1, *args, **kwargs):
   

        super().__init__()
        
        # ✅ 参数提取
        if kwargs:
            num_heads = int(kwargs.get('num_heads', 8))
            num_layers = int(kwargs.get('num_layers', 1))
            window_size = int(kwargs.get('window_size', 0))
            mlp_ratio = float(kwargs.get('mlp_ratio', 4.0))
            dropout = float(kwargs.get('dropout', 0.0))
        elif len(args) >= 3:
            num_heads = int(args[0])
            num_layers = int(args[1])
            window_size = int(args[2])
            mlp_ratio = float(args[3]) if len(args) > 3 else 4.0
            dropout = float(args[4]) if len(args) > 4 else 0.0
        else:
            num_heads = 8
            num_layers = 1
            window_size = 0
            mlp_ratio = 4.0
            dropout = 0.0
        
        # ✅ 参数验证
        c1 = int(c1)
        num_heads = int(num_heads)
        
        assert c1 > 0
        assert num_heads > 0
        assert num_layers > 0
        
        # ✅ 调整 num_heads
        if num_heads > c1:
            num_heads = c1
        if c1 % num_heads != 0:
            for nh in range(num_heads, 0, -1):
                if c1 % nh == 0:
                    num_heads = nh
                    break
        
        self.c1 = c1
        self.c2 = c1
        
        # ✅ 创建模块
        self.dwt_window_transformer = OriginalDWTWindowTransformerBlock(
            dim=c1,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        if c1 != self.c2:
            self.proj = nn.Conv2d(c1, self.c2, 1)
        else:
            self.proj = None
    
    def forward(self, x):
        x = self.dwt_window_transformer(x)
        if self.proj is not None:
            x = self.proj(x)
        return x






# ============================================================================
# DynamicConv 包装器
# ============================================================================

class DynamicConv(nn.Module):
    """
    YOLO 兼容的 DynamicConv
    
    YAML 格式：
    - [-1, 1, DynamicConv, [k, s, p, g]]
    
    示例：
    - [-1, 1, DynamicConv, [3, 1, None, 1]]
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        if p is None:
            p = k // 2
        
        # 使用原始模块
        self.conv = OriginalDynamicConv(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=True,
            reduction=16
        )
    
    def forward(self, x):
        return self.conv(x)


# ============================================================================
# AdaptiveAttention 包装器
# ============================================================================

class AdaptiveAttention(nn.Module):
    """
    YOLO 兼容的 AdaptiveAttention
    
    YAML 格式：
    - [-1, 1, AdaptiveAttention, [reduction, num_heads]]
    
    示例：
    - [-1, 1, AdaptiveAttention, [16, 8]]
    """
    
    def __init__(self, c1, c2, reduction=16, num_heads=8):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # 使用原始模块
        self.attention = OriginalAdaptiveAttention(
            channels=c1,
            reduction=reduction,
            num_heads=num_heads
        )
        
        # 如果输出通道不同，添加投影层
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1)
        else:
            self.proj = None
    
    def forward(self, x):
        x = self.attention(x)
        if self.proj is not None:
            x = self.proj(x)
        return x


# ============================================================================
# ConvBNReLU 包装器
# ============================================================================

class ConvBNReLU(nn.Module):
    """
    YOLO 兼容的 ConvBNReLU
    
    YAML 格式：
    - [-1, 1, ConvBNReLU, [k, s, p, g]]
    
    示例：
    - [-1, 1, ConvBNReLU, [3, 1, None, 1]]
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        if p is None:
            p = k // 2
        
        # 使用原始模块
        self.conv = OriginalConvBNReLU(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=True
        )
    
    def forward(self, x):
        return self.conv(x)


# ============================================================================
# DepthwiseSeparableConv 包装器
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    YOLO 兼容的 DepthwiseSeparableConv
    
    YAML 格式：
    - [-1, 1, DepthwiseSeparableConv, [k, s, p]]
    
    示例：
    - [-1, 1, DepthwiseSeparableConv, [3, 1, None]]
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        if p is None:
            p = k // 2
        
        # 使用原始模块
        self.conv = OriginalDepthwiseSeparableConv(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p
        )
    
    def forward(self, x):
        return self.conv(x)


# ============================================================================
# SEBlock 包装器
# ============================================================================

class SEBlock(nn.Module):
    """
    YOLO 兼容的 SEBlock
    
    YAML 格式：
    - [-1, 1, SEBlock, [reduction]]
    
    示例：
    - [-1, 1, SEBlock, [16]]
    """
    
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # 使用原始模块
        self.se = OriginalSEBlock(
            channels=c1,
            reduction=reduction
        )
        
        # 如果输出通道不同，添加投影层
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1)
        else:
            self.proj = None
    
    def forward(self, x):
        x = self.se(x)
        if self.proj is not None:
            x = self.proj(x)
        return x


# ============================================================================
# FocalLoss - 特殊处理（不是 Layer）
# ============================================================================

class FocalLoss(nn.Module):
    """
    YOLO 兼容的 FocalLoss
    
    用于损失函数，不是网络层
    """
    
    def __init__(self, c1=None, c2=None, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        计算 Focal Loss
        
        Args:
            inputs: 预测值 (N, C)
            targets: 目标值 (N,)
        
        Returns:
            loss: 损失值
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# EfficientNMS - 特殊处理（不是 Layer）
# ============================================================================

class EfficientNMS(nn.Module):
    """
    YOLO 兼容的 EfficientNMS
    
    用于后处理，不是网络层
    """
    
    def __init__(self, c1=None, c2=None, conf_thres=0.25, iou_thres=0.45, max_det=300):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
    
    def forward(self, x):
        """
        执行 NMS
        
        Args:
            x: 检测结果
        
        Returns:
            NMS 后的结果
        """
        return x


# ============================================================================
# 导出所有模块
# ============================================================================

def get_yolo_compatible_modules():
    """
    获取所有 YOLO 兼容的模块
    
    Returns:
        dict: 模块名称到模块类的映射
    """
    return {
        'TransformerBlock': TransformerBlock,
        'DWTBlock': DWTBlock,
        'DWTTransformerBlock': DWTTransformerBlock,
        'WindowTransformerBlock': WindowTransformerBlock,
        'DWTWindowTransformerBlock': DWTWindowTransformerBlock,
        'DynamicConv': DynamicConv,
        'AdaptiveAttention': AdaptiveAttention,
        'ConvBNReLU': ConvBNReLU,
        'DepthwiseSeparableConv': DepthwiseSeparableConv,
        'SEBlock': SEBlock,
        'FocalLoss': FocalLoss,
        'EfficientNMS': EfficientNMS,
    }
