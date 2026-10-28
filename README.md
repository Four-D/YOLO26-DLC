# YOLO26-DLC# YOLOv26-DLC 分类训练/推理脚本（standalone）

对应论文：
**YOLOv26-DLC: Defect-Preserving, Local Pooling and Class-Specific Loss for Insulator Flashover Classification**

本脚本在**不依赖 ultralytics 库**的前提下，从零实现 YOLO26n 图像分类模型（结构与官方 `yolo26n-cls` 逐层一致），并在其基础上实现论文提出的三个改进模块，可完整复现论文消融实验（Group0–Group3）。

---

## 1. 三大改进（对应论文 DLC）

| 缩写 | 全称 | 作用 | 代码位置 |
| --- | --- | --- | --- |
| **DPE** | Defect-Preserving Erasure | 保缺陷数据增强：取消随机擦除、颜色扰动、随机缩放，仅保留上下/水平翻转与 90°/180° 旋转，避免增强过程破坏缺陷像素 | `ClassTransforms`（`aug_mode="dpe"`） |
| **LMP** | Local Multi-scale Pooling | 局部多尺度池化分类头：保留 GAP 全局路径，并联 2×2、3×3 网格局部池化分支，增强对焦斑、电弧等局部小尺度特征的感知 | `ClassifyLMP`（`--lmp`） |
| **CSFL** | Class-Specific Focal Loss | 类特异焦点损失：闪络类使用更高调制系数 γ，其余类使用较小 γ，抑制易分样本对难样本的梯度淹没 | `ClassSpecificFocalLoss`（`--loss csfl`） |

---

## 2. 环境依赖

- Python 3.13（开发环境：`C:\Users\eccom\AppData\Local\Programs\Python\Python313\python.exe`）
- PyTorch（含 CUDA 支持）
- opencv-python
- numpy
- pyyaml

**不需要安装 ultralytics**。脚本仅使用 `torch`、`cv2`、`numpy`、`argparse`、`pathlib` 等标准依赖。

---

## 3. 数据准备

数据集为 ImageFolder 风格目录结构：

```
<data_root>/
├── mini_train/            # 训练子集（可换名，用 --train-split 指定）
│   ├── background/        # 类别目录：背景
│   │   ├── xxx.jpg
│   │   └── ...
│   ├── jyzsl/             # 类别目录：闪络（flashover）
│   └── zcjyz/             # 类别目录：正常绝缘子（normal）
├── mini_test/             # 测试子集（每类 128 张）
├── train/                 # 原始全量训练集（可选）
└── test/                  # 原始全量测试集（可选）
```

论文实验使用的类别均衡子集：`mini_train` 共 5120 张（background 1024 / jyzsl 2048 / zcjyz 2048），`mini_test` 共 384 张（每类 128）。

类别名自动扫描子目录确定；CSFL 的"闪络类"通过 `--flashover-class` 指定（默认 `jyzsl`，与数据集目录名一致）。

---

## 4. 快速开始

### 4.1 训练（论文 Full Model：DPE + LMP + CSFL）

```powershell
python yolo26DLC.py train `
  --data D:\datasets\jyz\jyz_sl_v7 `
  --train-split mini_train `
  --val-split mini_test `
  --epochs 10 `
  --batch 128 `
  --imgsz 224 `
  --aug dpe `
  --lmp `
  --loss csfl `
  --gamma-flashover 1.5 `
  --gamma-other 1.0 `
  --pretrained yolo26n-cls.pt `
  --out ab_G3_best
```

### 4.2 推理单图/目录

```powershell
python yolo26DLC.py infer test_qingtoujun.jpg --weights ab_G3_best_best.pt --topk 3
python yolo26DLC.py infer D:\datasets\jyz\jyz_sl_v7\mini_test --weights ab_G3_best_best.pt
```

推理目录时自动统计 top-1 精度（按父目录类别名判断）。

---

## 5. 消融实验复现（Group0–Group3）

四个实验组共享同一套训练超参（SGD momentum 0.937、weight decay 5e-4、余弦退火 + 3 轮 warmup、EMA、固定随机种子），仅改动增强/损失/头结构，保证组间可比：

| 组 | 配置 | 命令要点 |
| --- | --- | --- |
| **G0** Baseline | YOLOv26 + 原生随机擦除增强 + 原生 Focal Loss | `--aug full --loss focal --gamma 2.0`（不加 `--lmp`） |
| **G1** +DPE | G0 + 仅保缺陷增强 | `--aug dpe --loss focal --gamma 2.0` |
| **G2** +LMP | G1 + LMP 多尺度池化头 | `--aug dpe --loss focal --gamma 2.0 --lmp` |
| **G3** Full | G2 + CSFL 类特异损失（γ_flashover=1.5） | `--aug dpe --loss csfl --gamma-flashover 1.5 --gamma-other 1.0 --lmp` |

示例（G0）：

```powershell
python yolo26DLC.py train `
  --data D:\datasets\jyz\jyz_sl_v7 `
  --train-split mini_train --val-split mini_test `
  --epochs 10 --batch 128 --imgsz 224 `
  --aug full --loss focal --gamma 2.0 `
  --pretrained yolo26n-cls.pt `
  --out ab_G0_best
```

其余各组仅替换命令要点中的参数并修改 `--out` 前缀。论文 mini_test 结果（每类 128 张）：

| 组 | top-1 | background | flashover | normal |
| --- | --- | --- | --- | --- |
| G0 | 0.7891 | 112 | 107 | 84 |
| G1 | 0.7969 | 116 | 107 | 83 |
| G2 | 0.8307 | 116 | 116 | 87 |
| G3 (γ=1.5) | **0.8542** | 116 | 113 | 99 |

CSFL γ 敏感性（G3 结构下）：γ_flashover=2.5 → 0.8281；2.0 → 0.8307；**1.5 → 0.8542**（最优）。

---

## 6. 参数说明

### train 子命令

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data` | `D:\datasets\jyz\jyz_sl_v6` | 数据集根目录（含 train/val 子目录） |
| `--train-split` | `train` | 训练子目录名 |
| `--val-split` | `test` | 验证子目录名 |
| `--epochs` | 10 | 训练轮数 |
| `--batch` | 256 | 批大小（论文实验用 128） |
| `--imgsz` | 224 | 输入分辨率 |
| `--lr` / `--lrf` | 0.01 / 0.01 | 初始学习率 / 余弦退火末段比例 |
| `--workers` | 4 | DataLoader 进程数 |
| `--seed` | 0 | 随机种子（固定保证可复现） |
| `--max-samples` | None | 调试用：按类均匀限制训练样本数 |
| `--aug` | `full` | `full`=完整增强(crop+翻转+HSV+擦除)；`flip_only`=仅水平翻转；`dpe`=DPE 保缺陷几何增强 |
| `--pretrained` | None | 官方预训练权重路径（如 `yolo26n-cls.pt`）；不传则随机初始化 |
| `--lmp` | False | 启用 LMP 多尺度池化分类头 |
| `--loss` | `ce` | `ce`=CrossEntropy；`focal`=原生 Focal Loss；`csfl`=类特异 Focal Loss |
| `--gamma` | 2.0 | 原生 Focal Loss 的 γ |
| `--gamma-flashover` | 2.5 | CSFL：闪络类 γ |
| `--gamma-other` | 1.0 | CSFL：其余类 γ |
| `--flashover-class` | `jyzsl` | CSFL：闪络类在数据目录中的名称 |
| `--out` | `cls_trained` | 输出权重前缀（生成 `<out>_best.pt` 与 `<out>_last.pt`） |

### infer 子命令

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `path` | （必填） | 单张图片或目录路径 |
| `--weights` | `cls_trained_best.pt` | checkpoint 路径 |
| `--topk` | 5 | 输出 top-k 类别 |
| `--imgsz` | 224 | 输入分辨率 |

---

## 7. 代码结构

| 模块 | 说明 |
| --- | --- |
| `Conv` / `Bottleneck` / `C3k` / `C2f` / `C3k2` / `Attention` / `PSABlock` / `C2PSA` | 基础组件，与官方 `nn/modules/conv.py`、`block.py` 逐层一致 |
| `Classify` | 官方分类头（Conv 1×1 → GAP → Dropout → Linear） |
| `ClassifyLMP` | **LMP 创新头**：GAP 全局路径 + 多尺度局部池化分支（2×2、3×3 → 1×1 降维）拼接 |
| `ClassificationModel` | YOLO26n 分类模型（backbone 9 层 + 头），scale n（depth 0.50 / width 0.25 / max_channels 1024） |
| `ClassTransforms` | 预处理与增强；`aug_mode="dpe"` 为 DPE 保缺陷几何增强 |
| `ClassificationDataset` | ImageFolder 风格数据集（兼容中文路径） |
| `FocalLoss` | 原生 Focal Loss（G0–G2） |
| `ClassSpecificFocalLoss` | **CSFL 创新损失**：按样本所属类取独立 γ |
| `ClassificationTrainer` | 训练器：SGD + warmup + cosine + EMA + top1/top5 验证 |
| `ModelEMA` | 指数滑动平均（decay 0.9999，前 2000 步线性 ramp） |
| `Classifier` / `run_infer` | 推理：加载 checkpoint（自动识别 LMP 结构）输出 top-k |

---

## 8. 与官方 yolo26n-cls 的关键对齐点

- **网络结构**：backbone 9 层（Conv×2 + C3k2×2 + Conv + C3k2 + Conv + C3k2 + Conv + C3k2 + C2PSA）+ 分类头，逐层通道数与官方 `yolo26-cls.yaml`（scale n）一致。
- **初始化**：BatchNorm eps=1e-3、momentum=0.03；Linear 权重 ~ Normal(0, 0.01)、bias=0（官方 `initialize_weights` 行为）。
- **输入预处理**：像素值 /255（mean=0, std=1）；验证/推理用 Resize(短边=imgsz) + CenterCrop。
- **输出语义**：train 模式返回 logits；eval 模式返回 (softmax 概率, logits)（与官方 Classify 头一致）。
- **训练流程**：SGD(momentum=0.937, wd=5e-4)、3 轮 warmup、余弦退火、EMA、预训练权重初始化。

---

## 9. 输出说明

训练结束生成两个权重文件（`<out>_best.pt` / `<out>_last.pt`），均为 dict，字段：

| 字段 | 说明 |
| --- | --- |
| `state_dict` | 模型权重（EMA 权重；best 为验证 top1 最优时的 EMA 状态） |
| `names` | 类别名列表（按类别目录排序） |
| `top1` | 对应 top-1 准确率 |
| `epoch` | 训练轮数 |

推理时自动根据 `state_dict` 中是否存在 `model.10.lmp.*` 键判断是否为 LMP 结构，无需手动指定。

---

## 10. 论文结果复现环境备注

论文实验固定：224×224、batch 128、10 epochs、SGD(momentum 0.937, wd 5e-4)、余弦退火 + 3 轮 warmup、EMA、`yolo26n-cls.pt` 官方预训练初始化、随机种子固定（`--seed 0`），四个消融组共享同一训练脚本与超参。
