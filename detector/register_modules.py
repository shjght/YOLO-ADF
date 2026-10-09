# register_modules.py
"""
自定义模块注册器
将所有自定义模块注册到 ultralytics.nn.modules

✅ 修复：添加模块级别的 CustomUpsample 类
保留原有的 create_upsample_module() 函数（向后兼容）
✅ 新增：全局标志防止重复注册（但保留所有功能）
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch.nn as nn
import torch.nn.functional as F

# ✅ 全局标志：防止重复注册（但保留所有功能）
_MODULES_REGISTERED = False
_REGISTRATION_DETAILS = {}


# ============================================================================
# ✅ CustomUpsample - 模块级别（新增，用于 pickle 序列化）
# ============================================================================

class CustomUpsample(nn.Module):
    """
    Upsample 模块 - YOLO 兼容
    
    ✅ 新增：模块级别定义，可以被 pickle 正确序列化
    这样训练时保存权重不会出错
    """
    
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor if scale_factor is not None else 2
        self.mode = mode
    
    def forward(self, x):
        kwargs = {'mode': self.mode}
        if self.size is not None:
            kwargs['size'] = self.size
        else:
            kwargs['scale_factor'] = self.scale_factor
        
        if self.mode != 'nearest':
            kwargs['align_corners'] = False
        
        return F.interpolate(x, **kwargs)


# ============================================================================
# ✅ 原有的 create_upsample_module() 函数（保留，向后兼容）
# ============================================================================

def create_upsample_module():
    """
    创建 Upsample 模块
    
    ✅ 保留原有的嵌套类实现（向后兼容）
    但在注册时使用模块级别的 CustomUpsample
    """
    import torch.nn.functional as F
    import torch.nn as nn
    
    class Upsample(nn.Module):
        """Upsample 模块 - YOLO 兼容"""
        def __init__(self, size=None, scale_factor=None, mode='nearest'):
            super().__init__()
            self.size = size
            self.scale_factor = scale_factor if scale_factor is not None else 2
            self.mode = mode
        
        def forward(self, x):
            kwargs = {'mode': self.mode}
            if self.size is not None:
                kwargs['size'] = self.size
            else:
                kwargs['scale_factor'] = self.scale_factor
            
            if self.mode != 'nearest':
                kwargs['align_corners'] = False
            
            return F.interpolate(x, **kwargs)
    
    return Upsample


def register_custom_modules():
    """
    注册所有自定义模块到 ultralytics.nn.modules
    
    ✅ 关键特性：
    - 使用全局标志防止重复注册（但保留所有功能）
    - 导入修复版本的模块（yolo_compatible_modules_fixed.py）
    - 同时注册到 nn_modules 和 tasks_module
    - 创建并注册 Upsample 模块（✅ 使用 CustomUpsample）
    - 保留原有的 create_upsample_module() 函数（向后兼容）
    - 返回注册详情（保留所有功能）
    """
    global _MODULES_REGISTERED
    global _REGISTRATION_DETAILS
    
    # ✅ 如果已经注册过，直接返回注册详情（防止重复注册但保留功能）
    if _MODULES_REGISTERED:
        print("✅ 模块已注册，跳过重复注册\n")
        return _REGISTRATION_DETAILS
    
    _MODULES_REGISTERED = True
    
    print(f"\n{'='*100}")
    print("🔧 注册自定义模块")
    print(f"{'='*100}\n")
    
    try:
        # ✅ Step 1: 导入 YOLO 兼容的模块（从修复版本）
        print("📦 导入 YOLO 兼容的模块（从 yolo_compatible_modules_fixed.py）...")
        from yolo_compatible_modules_fixed import (  # ✅ 改成 _fixed
            TransformerBlock,
            DWTBlock,
            DWTTransformerBlock,
            WindowTransformerBlock,
            DWTWindowTransformerBlock,
            DynamicConv,
            AdaptiveAttention,
            ConvBNReLU,
            DepthwiseSeparableConv,
            SEBlock,
            FocalLoss,
            EfficientNMS,
        )
        print("✅ 导入成功\n")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}\n")
        import traceback
        traceback.print_exc()
        raise
    
    # ✅ Step 2: 获取 ultralytics.nn.modules 模块
    print("📝 获取 ultralytics.nn.modules...")
    try:
        import ultralytics.nn.modules as nn_modules
        print("✅ 获取成功\n")
    except ImportError as e:
        print(f"❌ 导入 ultralytics.nn.modules 失败: {e}\n")
        raise
    
    # ✅ Step 3: 创建模块字典
    modules_to_register = {
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
    
    # ✅ Step 4: 注册每个模块到 nn_modules
    print("📝 注册模块到 ultralytics.nn.modules:\n")
    registered_count = 0
    for name, cls in modules_to_register.items():
        try:
            setattr(nn_modules, name, cls)
            print(f"   ✅ {name}")
            registered_count += 1
            _REGISTRATION_DETAILS[f'nn_modules_{name}'] = True
        except Exception as e:
            print(f"   ❌ {name} 注册失败: {e}")
            _REGISTRATION_DETAILS[f'nn_modules_{name}'] = False
    
    print(f"\n✅ 成功注册 {registered_count}/{len(modules_to_register)} 个模块到 nn_modules\n")
    
    # ✅ Step 5: 同时注册到 ultralytics.nn.tasks（备选）
    print("📝 同时注册到 ultralytics.nn.tasks（备选）:\n")
    try:
        import ultralytics.nn.tasks as tasks_module
        
        for name, cls in modules_to_register.items():
            try:
                setattr(tasks_module, name, cls)
                print(f"   ✅ {name}")
                _REGISTRATION_DETAILS[f'tasks_{name}'] = True
            except Exception as e:
                print(f"   ⚠️  {name} 注册失败: {e}")
                _REGISTRATION_DETAILS[f'tasks_{name}'] = False
        
        print()
    except ImportError as e:
        print(f"   ⚠️  无法导入 tasks_module: {e}\n")
    
    # ✅ Step 6: 创建并注册 Upsample（关键！✅ 使用 CustomUpsample）
    print("📝 创建并注册 Upsample 模块（关键 - ✅ 使用 CustomUpsample）:\n")
    try:
        # ✅ 使用模块级别的 CustomUpsample（可以被 pickle 序列化）
        Upsample = CustomUpsample  # ✅ 改成使用 CustomUpsample
        
        # 注册到 nn_modules
        setattr(nn_modules, 'Upsample', Upsample)
        print(f"   ✅ Upsample (registered to nn_modules)")
        _REGISTRATION_DETAILS['nn_modules_Upsample'] = True
        
        # 注册到 tasks_module
        try:
            import ultralytics.nn.tasks as tasks_module
            setattr(tasks_module, 'Upsample', Upsample)
            print(f"   ✅ Upsample (registered to nn_tasks)\n")
            _REGISTRATION_DETAILS['tasks_Upsample'] = True
        except:
            print()
            _REGISTRATION_DETAILS['tasks_Upsample'] = False
        
    except Exception as e:
        print(f"   ❌ 创建 Upsample 失败: {e}\n")
        import traceback
        traceback.print_exc()
    
    return modules_to_register


def verify_registration():
    """验证所有模块是否已正确注册"""
    
    print(f"\n{'='*100}")
    print("✅ 验证模块注册")
    print(f"{'='*100}\n")
    
    try:
        import ultralytics.nn.modules as nn_modules
        
        required_modules = [
            'TransformerBlock',
            'DWTBlock',
            'DWTTransformerBlock',
            'WindowTransformerBlock',
            'DWTWindowTransformerBlock',
            'DynamicConv',
            'AdaptiveAttention',
            'ConvBNReLU',
            'DepthwiseSeparableConv',
            'SEBlock',
            'FocalLoss',
            'EfficientNMS',
            'Upsample',
        ]
        
        all_registered = True
        verified_count = 0
        
        print("检查 ultralytics.nn.modules:\n")
        for module_name in required_modules:
            if hasattr(nn_modules, module_name):
                print(f"   ✅ {module_name}")
                verified_count += 1
            else:
                print(f"   ❌ {module_name} 未注册")
                all_registered = False
        
        print(f"\n✅ 验证完成：{verified_count}/{len(required_modules)} 个模块已注册\n")
        
        return all_registered
            
    except Exception as e:
        print(f"❌ 验证失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def get_registered_modules():
    """获取所有已注册的模块"""
    
    import ultralytics.nn.modules as nn_modules
    
    required_modules = [
        'TransformerBlock',
        'DWTBlock',
        'DWTTransformerBlock',
        'WindowTransformerBlock',
        'DWTWindowTransformerBlock',
        'DynamicConv',
        'AdaptiveAttention',
        'ConvBNReLU',
        'DepthwiseSeparableConv',
        'SEBlock',
        'FocalLoss',
        'EfficientNMS',
        'Upsample',
    ]
    
    registered = {}
    for module_name in required_modules:
        if hasattr(nn_modules, module_name):
            registered[module_name] = getattr(nn_modules, module_name)
    
    return registered


def get_registration_details():
    """获取注册详情"""
    global _REGISTRATION_DETAILS
    return _REGISTRATION_DETAILS


def print_registration_summary():
    """打印注册摘要"""
    
    print(f"\n{'='*100}")
    print("📊 注册摘要")
    print(f"{'='*100}\n")
    
    details = get_registration_details()
    
    if not details:
        print("❌ 尚未执行注册\n")
        return
    
    nn_modules_count = sum(1 for k, v in details.items() if k.startswith('nn_modules_') and v)
    tasks_count = sum(1 for k, v in details.items() if k.startswith('tasks_') and v)
    
    print(f"✅ nn_modules 注册数：{nn_modules_count}")
    print(f"✅ tasks 注册数：{tasks_count}")
    print(f"✅ 总注册数：{nn_modules_count + tasks_count}\n")


if __name__ == '__main__':
    # 注册模块
    print("\n" + "="*100)
    print("🚀 开始注册自定义模块")
    print("="*100)
    
    register_custom_modules()
    
    # 验证注册
    if verify_registration():
        print("✅ 所有模块注册成功！\n")
    else:
        print("❌ 部分模块注册失败！\n")
    
    # 打印摘要
    print_registration_summary()
