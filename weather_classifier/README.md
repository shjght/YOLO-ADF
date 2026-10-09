# 天气分类器基准（回应审稿意见 R1-b，兼顾 R3）

> 位置：本地 `F:\xlw\weather_clf\` ⇄ 训练机 `C:\Users\KUST\xl\weather_clf\`
> （把整个 `weather_clf` 文件夹拷到训练机 `C:\Users\KUST\xl\` 下即可，数据路径自动探测两台机器）

## 目的

YOLO-ADF 的 WADI 指标假设"知道当前天气后选对 ADF 分支"。R1-b 建议补一个
**轻量天气分类器的精度/延迟基准**，把这个假设变成实测证据：
分类器开销 X ms + 精度 Y% → WADI 的现实意义成立。

## 设计（与论文跨域协议完全对称，不污染任何实验）

| 环节 | 数据 | 说明 |
|---|---|---|
| 训练 | BDD100K train（~6.3 万帧） | 与检测器同域；标签直接来自官方 `attributes.weather` + `timeofday`，**零人工标注** |
| 域内评估 | BDD100K val（10k） | 5 类：clear / rain / snow / fog / night |
| 跨域评估 | **ACDC val**（fog/rain/snow/night 各 100，合计 400） | 只评估、不训练 → 跨域协议不变；补上 BDD100K 雾天稀缺的问题。运行时打印每类张数，**每类应为 100**，不符说明数据目录有出入 |

**关键事实**：BDD100K 里 foggy 只有 130/69,863 帧（val 仅 13 帧）——这是数据集本身
的性质，不是我们的缺陷。论文里如实报告，并指出 fog 分支的路由靠 ACDC 跨域评估补证，
正好和现有 Table 3 叙事呼应。

**类别映射**：night 优先于天气（timeofday==night → night，与 ADF-N 设计一致）；
clear/partly cloudy/cloudy/overcast → clear；dawn/dusk 与 undefined 剔除。

## 三步运行（在训练机上）

```bat
cd C:\Users\KUST\xl\weather_clf
:: 1. 生成标签 csv（打印类别分布）
python build_labels.py
:: 2. 训练（单卡即可，3090 上 <1 小时；不用占满 4 卡）
set CUDA_VISIBLE_DEVICES=0 && python train.py --model mobilenet_v3_small
:: 3. 基准：域内+跨域精度、GPU FP32/FP16 延迟
python benchmark.py --model mobilenet_v3_small --cpu-latency
```

换 backbone 一行切换（用于对比表）：
```bat
python train.py --model resnet18 && python benchmark.py --model resnet18
python train.py --model shufflenet_v2_x1_0 && python benchmark.py --model shufflenet_v2_x1_0
```
可选 backbone：`mobilenet_v3_small`（部署候选，~2.5M）、`mobilenet_v3_large`、
`shufflenet_v2_x1_0`、`resnet18`、`efficientnet_b0`。

## 产物位置

| 文件 | 内容 |
|---|---|
| `weights/weatherclf_<model>_best.pt` | 验证集最优权重 |
| `weights/trainlog_<model>.json` | 每 epoch 损失/精度 |
| `results/benchmark_<model>.json` | 全部指标（精度+每类 P/R/F1+延迟） |
| `results/table_<model>.md` | **可直接贴进论文的 markdown 表** |

## 延迟测量协议（审稿可辩护）

batch=1、224×224、100 次预热 + 500 次计时、CUDA event 计时（含 synchronize）；
报告 FP32 与 FP16 的 mean±std ms 及换算 FPS；`--cpu-latency` 附测 CPU 延迟
（用于回应 R3-4"桌面 GPU vs 车载硬件"：CPU 数字就是保守上界）。
FLOPs 用 thop（没装则跳过，不影响其它指标）。

## 数据集下载（仅新机器需要，两台现有机器都不用）

```bat
python download_bdd100k.py --labels        # 只补标签 JSON（~300MB，ETH 公开镜像免登录）
python download_bdd100k.py --with-images   # 连图像 5.3GB 一起下
```
镜像不通时脚本会打印官方注册站（bdd-data.berkeley.edu / scal.ai）与 Kaggle 镜像命令。
注意训练机 `C:\Users\KUST\xl\bdd100k_final` 与 `ACDC` 已齐全，无需下载。

## 依赖

torch + torchvision（训练环境已有，torchvision ≥ 0.13 即可用预训练权重 API）；
Pillow；可选 thop（`pip install thop`，仅 FLOPs 用）。

## 论文里怎么写（草稿）

> A lightweight MobileNetV3-Small classifier (2.5 M parameters) trained on
> BDD100K weather attributes achieves XX% in-domain accuracy (BDD100K val) and
> XX% on the ACDC cross-domain validation set, at an overhead of X.X ms per
> frame (FP16, RTX 3090) — below Y% of the detector's inference budget. Fog
> routing is additionally validated on ACDC, as BDD100K contains only 130 foggy
> training frames.

数字出来后由 Claude 填进中英两版 JICV 修订版，并同步改 WADI 段落 + Limitations
（对应 R1-b + R3-1）。
