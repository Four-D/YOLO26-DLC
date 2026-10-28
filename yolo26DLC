# -*- coding: utf-8 -*-
"""
YOLO26n 图像分类 standalone —— 不依赖 ultralytics，仅 torch + opencv-python + numpy + pyyaml
结构与官方 ultralytics yolo26n-cls（yolo26-cls.yaml, scale=n）逐层一致：
  backbone: Conv×2 + C3k2×2 + Conv + C3k2 + Conv + C3k2 + Conv + C3k2 + C2PSA（共 9 层）
  head:     Classify（Conv(256→1280) + AdaptiveAvgPool2d + Linear(1280→nc)）
关键对齐点：
  - scale n = [depth 0.50, width 0.25, max_channels 1024]：通道 = min(c2, 1024) * 0.25 取 8 的倍数
  - C3k2 层 6/8 使用 c3k=True 变体（C3k 结构）；层 2/4 为 e=0.25 的 Bottleneck 变体
  - 所有 BatchNorm2d 统一 eps=1e-3, momentum=0.03（官方 initialize_weights 行为）
  - Linear 权重 ~ normal(0, 0.01)、bias = 0（官方 initialize_weights）
  - eval 模式返回 (softmax 概率, logits)，train 模式返回 logits（与官方 Classify 头一致）
  - 输入归一化：像素值 /255（官方 mean=0, std=1）；验证/推理 Resize(短边=imgsz)+CenterCrop

用法：
  python yolo26n_cls_standalone.py train --data D:\\datasets\\jyz\\jyz_sl_v6 --epochs 10 --batch 256 --train-split train --val-split test
  python yolo26n_cls_standalone.py infer <img_or_dir> --weights cls_trained_best.pt --topk 3
"""
  
import argparse
import math
import os

import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 基础组件（与官方 nn/modules/conv.py、block.py 一致）
# ---------------------------------------------------------------------------


def autopad(k, p=None, d=1):
    """自动计算 padding，与官方 autopad 一致"""
    if d > 1:
        k = k if isinstance(k, int) else [int(x) for x in k]
        if isinstance(k, int):
            k = [k]
        k = [k[0] * (d - 1) + 1, k[1] * (d - 1) + 1] if len(k) == 2 else [k[0] * (d - 1) + 1]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def imread_unicode(path):
    """cv2 读取图片，兼容中文/特殊字符路径（Windows: np.fromfile + imdecode）；文件缺失/损坏返回 None"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class Conv(nn.Module):
    """标准卷积: Conv2d + BN + SiLU"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准瓶颈: Conv(1x1) + Conv(kxk) + 可选残差"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    """C3k 模块（CSP Bottleneck with 3 convolutions）"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions（C3k2 基类）"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2(C2f):
    """C3k2（官方同名：c3k=True 时用 C3k 块；attn=True 时用 Bottleneck+PSABlock）"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class Attention(nn.Module):
    """PSA 注意力（qkv 拼接 + depthwise 位置编码，与官方 block.py 一致）"""

    format = None

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """PSA 块: attention + ffn 双残差"""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2PSA（官方 block.py: cv1 拆分 + PSABlock 序列 + cv2）"""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Classify(nn.Module):
    """YOLO 分类头（官方 head.py Classify: Conv(1x1) + GAP + Dropout + Linear）"""

    export = False

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        c_ = 1280  # efficientnet_b0 中间宽度（官方固定）
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # 官方 eval 模式返回 (softmax 概率, logits)
        return y if self.export else (y, x)


class ClassifyLMP(nn.Module):
    """LMP 分类头：保留 GAP（全局路径），在 backbone 末端并联多尺度局部池化分支
    （AdaptiveAvgPool2d(scales) → 1x1 降维 → flatten），与 GAP 拼接后进 Linear。
    对应消融 Group2/3 的 LMP（Local Multi-scale Pooling）结构创新。"""

    export = False

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, scales=(2, 3), proj=128):
        super().__init__()
        c_ = 1280  # 与官方 Classify 中间宽度一致
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局路径（保留原 GAP）
        self.drop = nn.Dropout(p=0.0, inplace=True)
        # LMP：多尺度局部池化分支，每尺度 1x1 降维到 proj，避免 Linear 输入爆炸
        self.lmp = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(sc), Conv(c_, proj, 1, 1)) for sc in scales
        )
        lmp_dim = sum(proj * sc * sc for sc in scales)
        self.linear = nn.Linear(c_ + lmp_dim, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.conv(x)
        g = self.pool(x).flatten(1)
        l = [m(x).flatten(1) for m in self.lmp]
        x = self.linear(self.drop(torch.cat([g] + l, 1)))
        if self.training:
            return x
        y = x.softmax(1)
        return y if self.export else (y, x)


# ---------------------------------------------------------------------------
# 模型（官方 yolo26-cls.yaml, scale=n）
# ---------------------------------------------------------------------------
class ClassificationModel(nn.Module):
    """YOLO26n 分类模型：backbone 9 层 + Classify 头，与官方 yolo26n-cls 逐层一致"""

    def __init__(self, nc=3, lmp=False):
        super().__init__()
        self.nc = nc
        # scale n: depth=0.50, width=0.25, max_channels=1024 → 通道=min(c2,1024)*0.25（8 的倍数）
        self.model = nn.Sequential(
            Conv(3, 16, 3, 2),  # 0-P1/2
            Conv(16, 32, 3, 2),  # 1-P2/4
            C3k2(32, 64, 1, False, 0.25),  # 2
            Conv(64, 64, 3, 2),  # 3-P3/8
            C3k2(64, 128, 1, False, 0.25),  # 4
            Conv(128, 128, 3, 2),  # 5-P4/16
            C3k2(128, 128, 1, True),  # 6（c3k=True）
            Conv(128, 256, 3, 2),  # 7-P5/32
            C3k2(256, 256, 1, True),  # 8（c3k=True）
            C2PSA(256, 256, 1),  # 9
            ClassifyLMP(256, nc) if lmp else Classify(256, nc),  # 10
        )
        self._apply_official_init()

    def _apply_official_init(self):
        """官方 torch_utils.initialize_weights：
        - BatchNorm2d: eps=1e-3, momentum=0.03
        - Linear: weight ~ normal(0, 0.01), bias = 0（与官方逐位一致的关键）"""
        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t is nn.Linear:
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True):
        """加载权重（忽略形状不匹配的键，返回 (成功加载键, 未加载键)）"""
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        missing = [k for k in model_dict if k not in pretrained_dict]
        super().load_state_dict(model_dict, strict=False)
        return pretrained_dict, missing


# ---------------------------------------------------------------------------
# 数据：分类数据集（ImageFolder 风格）+ 增强
# ---------------------------------------------------------------------------
class ClassTransforms:
    """分类预处理：
    train(full)      = RandomResizedCrop+翻转+HSV 扰动+擦除
    train(flip_only) = Resize+CenterCrop+水平翻转
    train(dpe)       = DPE 保缺陷增强：Resize+CenterCrop + 几何变换（上下/水平翻转、90°/180°旋转），
                       无随机擦除、无颜色扰动、无随机缩放（缺陷区域不被破坏）
    val/infer        = Resize+CenterCrop"""

    def __init__(self, img_size=224, train=False, aug_mode="full"):
        self.size = img_size
        self.train = train
        self.aug_mode = aug_mode

    @staticmethod
    def _random_resized_crop(img, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        """cv2 版 RandomResizedCrop：随机面积/宽高比裁剪后 resize 到 (size, size)"""
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect = math.exp(random.uniform(*log_ratio))
            tw = int(round(math.sqrt(target_area * aspect)))
            th = int(round(math.sqrt(target_area / aspect)))
            if 0 < tw <= w and 0 < th <= h:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                return cv2.resize(img[y1:y1 + th, x1:x1 + tw], (size, size), interpolation=cv2.INTER_LINEAR)
        # 回退：中心裁剪
        tw, th = min(w, size), min(h, size)
        x1, y1 = (w - tw) // 2, (h - th) // 2
        return cv2.resize(img[y1:y1 + th, x1:x1 + tw], (size, size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _resize_center_crop(img, size):
        """官方 classify_transforms：先等比缩放短边=size，再中心裁剪 size×size"""
        h, w = img.shape[:2]
        if h <= w:
            nh, nw = size, max(int(w * size / h), 1)
        else:
            nh, nw = max(int(h * size / w), 1), size
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        y1 = (nh - size) // 2
        x1 = (nw - size) // 2
        return img[y1:y1 + size, x1:x1 + size]

    @staticmethod
    def _hsv_jitter(img, h=0.015, s=0.7, v=0.4):
        """HSV 颜色扰动（官方 detect/classify 同款量级）"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        dh = random.uniform(-h, h) * 180.0
        ds = random.uniform(1 - s, 1 + s)
        dv = random.uniform(1 - v, 1 + v)
        hsv[..., 0] = (hsv[..., 0] + dh) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * ds, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * dv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __call__(self, img_rgb):
        """img_rgb: HWC RGB uint8 → CHW float32 (0-1)"""
        if self.train and self.aug_mode == "full":
            img = self._random_resized_crop(img_rgb, self.size)
            if random.random() < 0.5:
                img = img[:, ::-1]
            img = self._hsv_jitter(img)
        elif self.train and self.aug_mode == "dpe":
            # DPE：固定等比缩放+中心裁剪（不做随机缩放），只做几何变换，不动颜色、不擦除
            img = self._resize_center_crop(img_rgb, self.size)
            r = random.random()
            if r < 0.2:
                pass  # 原样
            elif r < 0.4:
                img = img[:, ::-1]  # 水平翻转
            elif r < 0.6:
                img = img[::-1, :]  # 上下翻转
            elif r < 0.8:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 90° 旋转
            else:
                img = cv2.rotate(img, cv2.ROTATE_180)  # 180° 旋转
        else:
            img = self._resize_center_crop(img_rgb, self.size)
            if self.train and self.aug_mode == "flip_only" and random.random() < 0.5:
                img = img[:, ::-1]
        img = img.astype(np.float32) / 255.0
        x = torch.from_numpy(img.transpose(2, 0, 1))
        if self.train and self.aug_mode == "full" and random.random() < 0.1:
            x = self._random_erasing(x, 0.02, 0.33, (0.3, 3.3))
        return x

    @staticmethod
    def _random_erasing(x, scale_min, scale_max, ratio_range):
        """torch 版 RandomErasing（p 由外部控制，这里在 p 命中后调用）"""
        c, h, w = x.shape
        area = h * w
        for _ in range(10):
            target_area = random.uniform(scale_min, scale_max) * area
            aspect = random.uniform(*ratio_range)
            tw = int(round(math.sqrt(target_area * aspect)))
            th = int(round(math.sqrt(target_area / aspect)))
            if 0 < tw <= w and 0 < th <= h:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                x[:, y1:y1 + th, x1:x1 + tw] = 0.0
                break
        return x


class ClassificationDataset(Dataset):
    """ImageFolder 风格分类数据集：root/<split>/<class>/ 下的图片"""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, split="train", img_size=224, train=True, max_samples=None, aug_mode="full"):
        self.split_root = Path(root) / split
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"找不到数据子集目录: {self.split_root}")
        self.classes = sorted([d.name for d in self.split_root.iterdir() if d.is_dir()])
        if not self.classes:
            raise FileNotFoundError(f"{self.split_root} 下没有类别子目录")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            cdir = self.split_root / c
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((str(f), self.class_to_idx[c]))
        if max_samples is not None:
            # 按类别均匀截取，避免只取到排序靠前的类（调试采样也有代表性）
            per = max(max_samples // len(self.classes), 1)
            self.samples = [s for c in self.classes for s in
                            [x for x in self.samples if x[1] == self.class_to_idx[c]][:per]]
        if not self.samples:
            raise FileNotFoundError(f"{self.split_root} 下没有图片")
        self.transforms = ClassTransforms(img_size=img_size, train=train, aug_mode=aug_mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = imread_unicode(path)
        if img_bgr is None:
            return self[(idx + 1) % len(self.samples)]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transforms(img_rgb), label

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# 损失：原生 Focal Loss（G0-G2）与 CSFL 类特异 Focal Loss（G3）
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """原生 Focal Loss（分类 CE 版）：(1 - pt)^gamma * CE，gamma 对所有类一致"""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可选：每类权重 tensor

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


class ClassSpecificFocalLoss(nn.Module):
    """CSFL：每类独立 gamma 的 Focal Loss。
    闪络类用高 gamma（更强地压低简单样本、聚焦难样本），
    污渍/正常类用低 gamma（减少简单样本训练开销）。"""

    def __init__(self, gamma_class=(1.0, 2.5, 1.0), alpha=None):
        super().__init__()
        self.gamma_class = gamma_class
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        g = torch.as_tensor(self.gamma_class, dtype=ce.dtype, device=ce.device)[targets]  # 逐样本取所在类的 gamma
        loss = (1 - pt) ** g * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


# ---------------------------------------------------------------------------
# 训练器（CE/Focal/CSFL + SGD + warmup + cosine + EMA + top1/top5 验证）
# ---------------------------------------------------------------------------
class ClassificationTrainer:
    """YOLO26n-cls 训练器，流程对齐官方 classify：CE 损失、SGD(mom=0.937)、warmup、cosine、EMA"""

    def __init__(self, data_root, epochs=100, batch_size=256, img_size=224, lr0=0.01, lrf=0.01,
                 momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, num_workers=4,
                 device=None, seed=0, max_samples=None, out_prefix="cls_trained",
                 train_split="train", val_split="test", aug_mode="full", pretrained=None,
                 lmp=False, loss_type="ce", gamma=2.0, gamma_flashover=2.5, gamma_other=1.0,
                 flashover_class="jyzsl", alpha=None):
        self.data_root = str(data_root)
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr0 = lr0
        self.lrf = lrf
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.num_workers = num_workers
        self.seed = seed
        self.max_samples = max_samples
        self.out_prefix = out_prefix
        self.train_split = train_split
        self.val_split = val_split
        self.aug_mode = aug_mode
        self.pretrained = pretrained
        self.lmp = lmp
        self.loss_type = loss_type
        self.gamma = gamma
        self.gamma_flashover = gamma_flashover
        self.gamma_other = gamma_other
        self.flashover_class = flashover_class
        self.alpha = alpha
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(self):
        self._set_seed()
        print(f"使用设备: {self.device}")
        print(f"数据集: {self.data_root}")

        train_ds = ClassificationDataset(self.data_root, self.train_split, self.img_size, train=True, max_samples=self.max_samples, aug_mode=self.aug_mode)
        val_ds = ClassificationDataset(self.data_root, self.val_split, self.img_size, train=False)
        self.names = train_ds.classes
        self.nc = len(self.names)
        print(f"类别数: {self.nc}，类别: {self.names}")
        print(f"训练集({self.train_split}): {len(train_ds)} 张，测试集({self.val_split}): {len(val_ds)} 张")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True, drop_last=False,
                                  collate_fn=ClassificationDataset.collate_fn)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True,
                                collate_fn=ClassificationDataset.collate_fn)

        model = ClassificationModel(nc=self.nc, lmp=self.lmp).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {n_params / 1e6:.3f}M" + ("（含 LMP 多尺度池化头）" if self.lmp else ""))
        if self.pretrained:
            ckpt = torch.load(self.pretrained, map_location="cpu", weights_only=False)
            m = ckpt.get("model", ckpt)
            sd = m.state_dict() if hasattr(m, "state_dict") else m
            loaded, missing = model.load_state_dict(sd)
            print(f"预训练权重: {self.pretrained}（成功加载 {len(loaded)}/{len(model.state_dict())} 键，"
                  f"未加载 {len(missing)} 键——分类头维度不同则自动随机初始化）")

        # 损失选择：ce / focal（原生 Focal Loss，G0-G2）/ csfl（类特异 Focal Loss，G3）
        if self.loss_type == "csfl":
            gamma_class = [self.gamma_other] * self.nc
            if self.flashover_class in self.names:
                gamma_class[self.names.index(self.flashover_class)] = self.gamma_flashover
            criterion = ClassSpecificFocalLoss(gamma_class=tuple(gamma_class), alpha=self.alpha)
            print(f"CSFL 类特异 γ: {gamma_class}（闪络类={self.flashover_class}, γ={self.gamma_flashover}；其余 γ={self.gamma_other}）")
        elif self.loss_type == "focal":
            criterion = FocalLoss(gamma=self.gamma, alpha=self.alpha)
            print(f"原生 Focal Loss: γ={self.gamma}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("损失: CrossEntropyLoss")
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr0, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        total_iters = max(len(train_loader), 1) * self.epochs
        warmup_iters = int(self.warmup_epochs * len(train_loader))
        # 官方：cosine 从 lr0 衰减到 lr0*lrf
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.epochs - self.warmup_epochs, 1),
                                                               eta_min=self.lr0 * self.lrf)

        ema = ModelEMA(model, decay=0.9999)

        best_top1 = 0.0
        best_state = None
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            t0 = time.time()
            loss_sum = 0.0
            n_batches = 0
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                # warmup：前 warmup_iters 步 lr 从 ~0 线性升到 lr0
                if global_step < warmup_iters:
                    lr = self.lr0 * (global_step + 1) / max(warmup_iters, 1)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                elif global_step == warmup_iters:
                    for pg in optimizer.param_groups:
                        pg["lr"] = self.lr0
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                ema.update(model)
                loss_sum += loss.item()
                n_batches += 1
                global_step += 1
            if global_step >= warmup_iters and (global_step - warmup_iters) % len(train_loader) == 0:
                scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]
            avg_loss = loss_sum / max(n_batches, 1)
            top1, top5 = self._validate(ema.ema if ema.ema is not None else model, val_loader)
            print(f"Epoch [{epoch}/{self.epochs}] Loss: {avg_loss:.4f} Top1: {top1:.4f} Top5: {top5:.4f} "
                  f"LR: {cur_lr:.6f} 用时: {time.time() - t0:.1f}s")
            if top1 > best_top1:
                best_top1 = top1
                best_state = {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}
                print(f"  新最佳 top1: {top1:.4f}")

        # 保存最佳与最终
        torch.save({"epoch": self.epochs, "top1": best_top1, "names": self.names,
                    "state_dict": best_state}, f"{self.out_prefix}_best.pt")
        final_state = {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}
        torch.save({"epoch": self.epochs, "top1": top1, "names": self.names,
                    "state_dict": final_state}, f"{self.out_prefix}_last.pt")
        print(f"训练完成！最佳 top1: {best_top1:.4f}")
        print(f"已保存: {self.out_prefix}_best.pt / {self.out_prefix}_last.pt")
        return best_top1

    @torch.no_grad()
    def _validate(self, model, val_loader):
        model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        for imgs, labels in val_loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            out = model(imgs)
            logits = out[1] if isinstance(out, (tuple, list)) else out  # eval 返回 (softmax, logits)
            probs = torch.softmax(logits, dim=-1)
            _, pred5 = probs.topk(min(5, probs.shape[-1]), dim=-1)
            correct1 += (pred5[:, 0] == labels).sum().item()
            correct5 += (pred5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            total += labels.numel()
        model.train()
        return (correct1 / total) if total else 0.0, (correct5 / total) if total else 0.0


class ModelEMA:
    """EMA（官方 ModelEMA 语义: decay=0.9999, 前 2000 步线性 ramp）"""

    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        import copy
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay * min(1.0, (self.updates + 1) / (self.warmup_steps + 1))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # 跳过 num_batches_tracked 等 Long 张量
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
class Classifier:
    """加载 checkpoint 推理单图/目录，输出 top-k 类别与置信度"""

    def __init__(self, weights_path, device=None, img_size=224):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        self.names = ckpt.get("names")
        if isinstance(self.names, dict):
            self.names = [self.names[i] for i in sorted(self.names)]
        if not self.names:
            n_out = sd["model.10.linear.weight"].shape[0]
            self.names = ["class_" + str(i) for i in range(n_out)]
        self.nc = len(self.names)
        # 动态识别权重是否为 LMP 结构（存在 model.10.lmp.* 键）
        is_lmp = any(k.startswith("model.10.lmp") for k in sd)
        self.model = ClassificationModel(nc=self.nc, lmp=is_lmp).to(self.device)
        loaded, missing = self.model.load_state_dict(sd)
        print(f"已加载权重: {weights_path}（{len(loaded)}/{len(self.model.state_dict())} 个参数"
              + ("，LMP 结构" if is_lmp else "") + "）")
        self.model.eval()
        self.transforms = ClassTransforms(img_size=img_size, train=False)

    @torch.no_grad()
    def predict(self, img_path, topk=5):
        img_bgr = imread_unicode(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transforms(img_rgb).unsqueeze(0).to(self.device)
        out = self.model(x)
        logits = out[1] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=-1)[0]
        k = min(topk, self.nc)
        p, idx = probs.topk(k)
        p = p.cpu().numpy()
        idx = idx.cpu().numpy()
        return [(self.names[i], float(p[j])) for j, i in enumerate(idx)]


def run_infer(path, weights, topk=5, img_size=224, out=None):
    """推理单个文件或整个目录"""
    clf = Classifier(weights, img_size=img_size)
    p = Path(path)
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.suffix.lower() in ClassificationDataset.IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"路径下没有图片: {path}")
    correct = 0
    for i, f in enumerate(files):
        preds = clf.predict(f, topk=topk)
        line = f"  {preds[0][0]} {preds[0][1]:.4f}" + "".join(f" | {n} {s:.3f}" for n, s in preds[1:])
        if f.parent.name in clf.names and preds[0][0] == f.parent.name:
            correct += 1
        print(f"[{i + 1}/{len(files)}] {f.name}: {line}")
    if all(f.parent.name in clf.names for f in files):
        print(f"目录精度 top1: {correct / len(files):.4f}（{correct}/{len(files)}）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO26n 分类 standalone（不依赖 ultralytics）")
    sub = parser.add_subparsers(dest="mode")

    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--data", default=r"D:\datasets\jyz\jyz_sl_v6")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch", type=int, default=256)
    p_train.add_argument("--imgsz", type=int, default=224)
    p_train.add_argument("--lr", type=float, default=0.01)
    p_train.add_argument("--lrf", type=float, default=0.01)
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--max-samples", type=int, default=None, help="调试用：限制训练样本数")
    p_train.add_argument("--train-split", default="train", help="训练数据子目录名")
    p_train.add_argument("--val-split", default="test", help="验证数据子目录名")
    p_train.add_argument("--out", default="cls_trained")
    p_train.add_argument("--aug", default="full", choices=["full", "flip_only", "dpe"],
                         help="full=完整增强(crop+翻转+HSV+擦除)；flip_only=仅水平翻转；dpe=DPE 保缺陷几何增强(翻转+旋转,无擦除/颜色/缩放)")
    p_train.add_argument("--pretrained", default=None, help="官方预训练权重路径（如 yolo26n-cls.pt），默认随机初始化")
    p_train.add_argument("--lmp", action="store_true", help="启用 LMP 多尺度池化分类头（并联分支）")
    p_train.add_argument("--loss", default="ce", choices=["ce", "focal", "csfl"],
                         help="ce=CrossEntropy；focal=原生 Focal Loss(γ 全类一致)；csfl=类特异 Focal Loss")
    p_train.add_argument("--gamma", type=float, default=2.0, help="原生 Focal Loss 的 γ")
    p_train.add_argument("--gamma-flashover", type=float, default=2.5, help="CSFL：闪络类 γ")
    p_train.add_argument("--gamma-other", type=float, default=1.0, help="CSFL：其余类 γ")
    p_train.add_argument("--flashover-class", default="jyzsl", help="CSFL：闪络类名（在数据子目录中的名称）")

    p_infer = sub.add_parser("infer", help="推理图片或目录")
    p_infer.add_argument("path")
    p_infer.add_argument("--weights", default="cls_trained_best.pt")
    p_infer.add_argument("--topk", type=int, default=5)
    p_infer.add_argument("--imgsz", type=int, default=224)

    args = parser.parse_args()
    if args.mode == "train":
        ClassificationTrainer(
            data_root=args.data, epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz,
            lr0=args.lr, lrf=args.lrf, num_workers=args.workers, seed=args.seed,
            max_samples=args.max_samples, out_prefix=args.out,
            train_split=args.train_split, val_split=args.val_split, aug_mode=args.aug,
            pretrained=args.pretrained, lmp=args.lmp, loss_type=args.loss,
            gamma=args.gamma, gamma_flashover=args.gamma_flashover, gamma_other=args.gamma_other,
            flashover_class=args.flashover_class,
        ).fit()
    elif args.mode == "infer":
        run_infer(args.path, args.weights, topk=args.topk, img_size=args.imgsz)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
