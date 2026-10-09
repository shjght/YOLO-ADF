# train_exp01_baseline_consistent.py


"""


exp01 基线训练脚本 - YOLOv11m 原始模型


模型：YOLOv11m（无任何修改）


参数：与 exp04/exp08/exp09 完全一致


梯度累积：batch=32, nbs=64 → 有效 batch=64


用途：消融实验基准、模型对比基础


"""





import os


SEED = int(os.environ.get('XLW_SEED', '0'))  # 多seed重跑：环境变量 XLW_SEED=0/1/2

import sys


import gc


import warnings


from pathlib import Path


from datetime import datetime


from typing import Optional, List, Dict, Type, Tuple





# ============================================================================


# 🔧 环境配置


# ============================================================================


os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('XLW_GPU', '2')          # 根据空闲卡调整


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


os.environ['MPLBACKEND'] = 'Agg'


os.environ['CUDA_LAUNCH_BLOCKING'] = '2'





warnings.filterwarnings('ignore')





import torch


import torch.nn as nn





# ============================================================================


# 🔬 exp01 基线配置 - 与 exp04/exp08/exp09 完全一致


# ============================================================================


DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'


MODEL_YAML = 'yolo11m.yaml'


RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'


EXP_NAME = '01_baseline_consistent' + f'_s{SEED}'





EPOCHS = 300


BATCH_SIZE = 32              # 物理 batch（与所有实验一致）


LEARNING_RATE = 0.001        # 与所有实验一致


WEIGHT_DECAY = 5e-3          # 与所有实验一致


WARMUP_EPOCHS = 20           # 与所有实验一致


GPU_ID = 2





NUM_WORKERS = 2              # 与所有实验一致


PIN_MEMORY = True


CACHE = False





# ============================================================================


# 📦 模块注册（baseline 不需要自定义模块）


# ============================================================================


class ModuleRegistry:


    _registry: Dict[str, Type] = {}


    @classmethod


    def register(cls, name, module_class):


        cls._registry[name] = module_class


    @classmethod


    def get_all(cls):


        return cls._registry.copy()





print("\n  exp01 baseline 不需要自定义模块\n")





def install_parse_model_hook():


    print(" exp01 baseline 不需要安装 Hook\n")





def load_model(config_path):


    from ultralytics import YOLO


    model = YOLO(config_path)


    return model





def train():


    print(f" 基线训练（batch=32, nbs=64 → 有效 batch=64）")


    install_parse_model_hook()


    torch.cuda.set_device(0)





    model = load_model(MODEL_YAML)


    gc.collect()


    torch.cuda.empty_cache()





    model.train(


        data=DATA_YAML,


        epochs=EPOCHS,


        imgsz=416,


        batch=BATCH_SIZE,


        device=0,


        workers=NUM_WORKERS,





        optimizer='AdamW',


        lr0=LEARNING_RATE,


        lrf=0.01,


        momentum=0.937,


        weight_decay=WEIGHT_DECAY,





        warmup_epochs=WARMUP_EPOCHS,


        warmup_momentum=0.8,


        warmup_bias_lr=0.00005,





        box=7.5, cls=0.5, dfl=1.5,





        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,


        degrees=0.0, translate=0.1, scale=0.5,


        flipud=0.0, fliplr=0.5,


        mosaic=1.0, mixup=0.2,


        copy_paste=0.0, erasing=0.0,


        close_mosaic=15,





        amp=True, half=False, nbs=64,


        save=True, save_period=10,


        project=RESULTS_DIR, name=EXP_NAME,


        exist_ok=True,


        patience=50,


        plots=False, val=True,


        cache=CACHE, verbose=True,


        deterministic=True, seed=SEED,


        cos_lr=True,


        single_cls=False, rect=False,


        fraction=1.0,


    )


    print("✅ 训练完成")





if __name__ == '__main__':


    train()