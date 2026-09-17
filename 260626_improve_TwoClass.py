"""YOLO12-Chroma (CA-Head + CAC-Branch) 二类微调。.

创新点：类感知色度支路 (Class-Aware Chromatic Branch)
  - backbone / neck 与 YOLO12 完全一致 → 单类 v12s 预训练完整对齐
  - box 分支不变，P3 cls logit 叠加每类独立色度残差
  - 共享 HSV 编码 + 每类零初始化 head/scale → 训练起点等价 baseline
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import Counter

from ultralytics import YOLO

CHROMA_YAML = "yolo12s-chroma.yaml"
PRETRAINED = "runs/train/v12s/weights/best.pt"

# 参数量参考（scale=s, nc=80 初始化时）
BASELINE_PARAMS = 9_284_096
CHROMA_EXTRA_MIN = 5_000  # CA-Head 新增参数量下限


def verify_chroma_architecture(yolo: YOLO) -> None:
    """训练前校验 CA-Head 架构。."""
    types = Counter(type(m).__name__ for m in yolo.model.modules())
    n_params = sum(p.numel() for p in yolo.model.parameters())

    has_chroma = types.get("ChromaDetect", 0) >= 1
    has_old = types.get("ColorStem", 0) + types.get("P3ColorEnhance", 0)

    print(f"\n{'=' * 55}")
    print("CA-Head + CAC-Branch 架构校验")
    print(f"  参数量: {n_params:,}  (baseline 约 {BASELINE_PARAMS:,})")
    print(f"  ChromaDetect: {types.get('ChromaDetect', 0)}")
    print(f"  旧 Color 模块: ColorStem={types.get('ColorStem', 0)}, P3ColorEnhance={types.get('P3ColorEnhance', 0)}")
    print(f"{'=' * 55}\n")

    if not has_chroma:
        print("错误: 未检测到 ChromaDetect，请使用 yolo12s-chroma.yaml")
        sys.exit(1)
    if has_old > 0:
        print("错误: 检测到旧版 ColorStem/P3ColorEnhance，请改用 yolo12-chroma.yaml")
        sys.exit(1)
    if n_params < BASELINE_PARAMS + CHROMA_EXTRA_MIN:
        print("警告: 参数量接近 baseline，CA-Head 可能未正确加载")


def main():
    data_yaml = "datasets/new_auto_annotations/dataset.yaml"
    model = YOLO(CHROMA_YAML)
    verify_chroma_architecture(model)

    results = model.train(
        data=data_yaml,
        pretrained=PRETRAINED,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        cos_lr=True,
        patience=20,
        close_mosaic=20,
        cls_pw=0.75,
        copy_paste=0.1,
        seed=0,
        project="runs/train",
        name="new_improv_TwoClass_ColorYOLO12s_v2",
        exist_ok=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    print("训练结束")
    print(results.save_dir)


if __name__ == "__main__":
    main()
