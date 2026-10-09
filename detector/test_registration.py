"""
测试模块注册 - 直接运行这个文件
"""

import sys
from pathlib import Path

print("=" * 80)
print("🧪 测试模块注册")
print("=" * 80)

# 第一步：注册模块
print("\n[1/3] 注册自定义模块...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from register_modules import register_custom_modules
    
    custom_modules = register_custom_modules()
    print(f"✅ 注册成功，共 {len(custom_modules)} 个模块")
except Exception as e:
    print(f"❌ 注册失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 第二步：验证注册
print("\n[2/3] 验证模块注册...")
try:
    import ultralytics
    import ultralytics.nn.tasks as tasks_module
    
    modules_to_check = [
        'WindowTransformerBlock',
        'DynamicConv',
        'AdaptiveAttention',
        'FocalLoss',
        'EfficientNMS',
    ]
    
    for module_name in modules_to_check:
        if hasattr(tasks_module, module_name):
            print(f"   ✅ {module_name} 已注册")
        else:
            print(f"   ❌ {module_name} 未注册")
    
except Exception as e:
    print(f"❌ 验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 第三步：加载模型
print("\n[3/3] 加载模型...")
try:
    import yaml
    from ultralytics.nn.tasks import DetectionModel
    
    # 补丁 parse_model
    from ultralytics.nn.tasks import parse_model as original_parse_model
    import ultralytics.nn.tasks as tasks_module
    
    def patched_parse_model(d, ch=3, verbose=True):
        for name, module_class in custom_modules.items():
            setattr(tasks_module, name, module_class)
        return original_parse_model(d, ch=ch, verbose=verbose)
    
    tasks_module.parse_model = patched_parse_model
    ultralytics.nn.tasks.parse_model = patched_parse_model
    
    # 加载 YAML
    with open('yolo11m_window_transformer_layer8.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"   ✅ YAML 加载成功 (nc={cfg.get('nc')})")
    
    # 加载模型
    model = DetectionModel(cfg, ch=3, nc=cfg.get('nc', 10))
    print("   ✅ 模型加载成功！")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - 总参数量: {total_params / 1e6:.2f}M")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！")
print("=" * 80)
