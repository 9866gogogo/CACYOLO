"""水稻白穗检测模型本地化部署（命令行批量推理）。.

支持两个二类检测模型（healthy / whitehead）的本地批量推理：
  - base : new_TwoClass_YOLO12s           (标准 YOLO12s 基线)
  - color: new_improv_TwoClass_ColorYOLO12s (CAC-YOLO12, 色度增强)

功能：
  - 指定图片文件夹，批量推理
  - 通过 --model 切换选择其中一个模型
  - 保存标注可视化图片
  - 输出逐图计数与白穗率 CSV，并打印整体汇总

用法示例：
  python deploy_detect_cli.py --model color --source datasets/baisui_datas/images
  python deploy_detect_cli.py --model base  --source path/to/imgs --output runs/deploy/base --conf 0.25
  python deploy_detect_cli.py --model runs/detect/runs/train/xxx/weights/best.pt --source imgs
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# 预置模型别名 → 权重路径
MODEL_ZOO = {
    "base": "runs/detect/runs/train/new_TwoClass_YOLO12s/weights/best.pt",
    "color": "runs/detect/runs/train/new_improv_TwoClass_ColorYOLO12s/weights/best.pt",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_weight(model: str) -> Path:
    """将模型别名或路径解析为权重文件绝对路径。."""
    if model in MODEL_ZOO:
        weight = ROOT / MODEL_ZOO[model]
    else:
        weight = Path(model)
        if not weight.is_absolute():
            weight = ROOT / weight
    if not weight.is_file():
        raise FileNotFoundError(f"未找到模型权重: {weight}")
    return weight


def collect_images(source: Path) -> list[Path]:
    """收集文件夹（或单张图片）中的所有图片路径。."""
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    if not source.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {source}")
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return images


def main():
    parser = argparse.ArgumentParser(description="水稻白穗检测本地批量推理")
    parser.add_argument(
        "--model",
        type=str,
        default="color",
        help="模型别名(base/color)或权重路径，默认 color(CAC-YOLO12)",
    )
    parser.add_argument("--source", type=str, required=True, help="待检测图片文件夹或单张图片")
    parser.add_argument("--output", type=str, default=None, help="输出目录，默认 runs/deploy/<model>")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--device", type=str, default=None, help="设备，如 0 或 cpu，默认自动选择")
    parser.add_argument("--max-det", type=int, default=1000, help="单图最大检测框数")
    parser.add_argument("--no-save-img", action="store_true", help="仅统计，不保存标注图片")
    args = parser.parse_args()

    weight = resolve_weight(args.model)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    images = collect_images(source)
    if not images:
        raise FileNotFoundError(f"未在 {source} 找到图片文件")

    model_tag = args.model if args.model in MODEL_ZOO else weight.stem
    output_dir = Path(args.output) if args.output else ROOT / "runs" / "deploy" / model_tag
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    vis_dir = output_dir / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_img:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"模型      : {model_tag}")
    print(f"权重      : {weight}")
    print(f"输入      : {source}  (共 {len(images)} 张)")
    print(f"输出目录  : {output_dir}")
    print(f"设备      : {device}  |  conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")
    print("=" * 60)

    model = YOLO(str(weight))
    names = model.names  # {0: 'healthy', 1: 'whitehead'}

    # 逐图推理（stream 模式，节省显存）
    results = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )

    rows: list[dict] = []
    total_healthy = 0
    total_whitehead = 0

    for res in results:
        img_path = Path(res.path)
        cls_ids = res.boxes.cls.int().tolist() if res.boxes is not None else []
        healthy = sum(1 for c in cls_ids if names.get(c) == "healthy")
        whitehead = sum(1 for c in cls_ids if names.get(c) == "whitehead")
        total = healthy + whitehead
        rate = (whitehead / total * 100.0) if total > 0 else 0.0

        total_healthy += healthy
        total_whitehead += whitehead

        rows.append(
            {
                "image": img_path.name,
                "healthy": healthy,
                "whitehead": whitehead,
                "total": total,
                "whitehead_rate(%)": round(rate, 2),
            }
        )

        if not args.no_save_img:
            annotated = res.plot()  # BGR ndarray
            cv2.imwrite(str(vis_dir / img_path.name), annotated)

    # 保存逐图结果 CSV
    csv_path = output_dir / "detection_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "healthy", "whitehead", "total", "whitehead_rate(%)"])
        writer.writeheader()
        writer.writerows(rows)

    # 整体汇总
    grand_total = total_healthy + total_whitehead
    overall_rate = (total_whitehead / grand_total * 100.0) if grand_total > 0 else 0.0

    print("\n" + "=" * 60)
    print("批量推理完成")
    print(f"图片数量       : {len(rows)}")
    print(f"健康穗总数     : {total_healthy}")
    print(f"白穗总数       : {total_whitehead}")
    print(f"稻穗总数       : {grand_total}")
    print(f"整体白穗率     : {overall_rate:.2f}%")
    print(f"逐图结果 CSV   : {csv_path}")
    if not args.no_save_img:
        print(f"标注图片目录   : {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
