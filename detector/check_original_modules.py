# check_original_modules.py
"""
检查原始模块的参数签名
"""

import inspect
from custom_modules import DWTWindowTransformerBlock

print("\n" + "="*100)
print("🔍 检查 DWTWindowTransformerBlock 的参数签名")
print("="*100 + "\n")

# 获取 __init__ 方法的签名
sig = inspect.signature(DWTWindowTransformerBlock.__init__)

print("📋 参数列表：\n")
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    
    default = param.default
    if default == inspect.Parameter.empty:
        print(f"   - {param_name}: (必需)")
    else:
        print(f"   - {param_name}: {default}")

print("\n✅ 完整签名：")
print(f"   {DWTWindowTransformerBlock.__init__}{sig}\n")
