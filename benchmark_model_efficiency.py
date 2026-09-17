"""统计 runs/train 下各模型在 RicePanicleDemoDataset test 集上的 GFLOPs、参数量与 FPS。."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "datasets/RicePanicleDemoDataset.yaml"
TEST_IMAGES = ROOT / "datasets/RicePanicleDemoDataset/images/test"
OUTPUT_DIR = ROOT / "runs/model_efficiency"
IMGSZ = 640
BATCH = 1  # FPS 以 batch=1 衡量
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# YOLOv8/v10/v12 (n/s) + RT-DETR 系列
MODELS = {
    "YOLOv8n": "runs/train/v8n/weights/best.pt",
    "YOLOv8s": "runs/train/v8s/weights/best.pt",
    "YOLOv10n": "runs/train/v10n/weights/best.pt",
    "YOLOv10s": "runs/train/v10s/weights/best.pt",
    "YOLOv12n": "runs/train/v12n/weights/best.pt",
    "YOLOv12s": "runs/train/v12s/weights/best.pt",
    "RT-DETR-l": "runs/train/rtdetr-l/weights/best.pt",
    "RT-DETR-R50": "runs/train/rtdetr-resnet50/weights/best.pt",
}


def count_test_images() -> int:
    """统计 test 图像数量。."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(1 for p in TEST_IMAGES.iterdir() if p.suffix.lower() in exts)


def speed_to_fps(speed: dict) -> tuple[float, float, dict]:
    """将 val 返回的 speed(ms/img) 转为 FPS 与总延迟。."""
    preprocess = float(speed.get("preprocess", 0.0) or 0.0)
    inference = float(speed.get("inference", 0.0) or 0.0)
    postprocess = float(speed.get("postprocess", 0.0) or 0.0)
    latency_ms = preprocess + inference + postprocess
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    detail = {
        "preprocess_ms": round(preprocess, 2),
        "inference_ms": round(inference, 2),
        "postprocess_ms": round(postprocess, 2),
    }
    return round(fps, 2), round(latency_ms, 2), detail


def measure_fps_on_test(model: YOLO) -> dict:
    """在 test 划分上通过 val 统计端到端推理速度。."""
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        verbose=False,
        plots=False,
    )
    speed = getattr(metrics, "speed", {}) or {}
    fps, latency_ms, detail = speed_to_fps(speed)
    return {"FPS": fps, "latency_ms": latency_ms, **detail}


def benchmark_one(name: str, weight_rel: str, n_images: int) -> dict:
    """单模型：GFLOPs、参数量、FPS。."""
    weight = ROOT / weight_rel
    if not weight.exists():
        raise FileNotFoundError(f"权重不存在: {weight}")

    print(f"\n{'=' * 60}")
    print(f"模型: {name}")
    print(f"权重: {weight}")

    model = YOLO(str(weight))
    params = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=IMGSZ)
    speed = measure_fps_on_test(model)

    row = {
        "model": name,
        "weight": str(weight_rel),
        "Parameters": int(params),
        "Parameters(M)": round(params / 1e6, 3),
        "GFLOPs": round(float(gflops), 2),
        "test_images": n_images,
        **speed,
        "device": str(DEVICE),
        "imgsz": IMGSZ,
        "batch": BATCH,
    }

    print(
        f"  Parameters: {row['Parameters']:,} ({row['Parameters(M)']} M)  |  "
        f"GFLOPs: {row['GFLOPs']}  |  FPS: {row['FPS']}  |  "
        f"Latency: {row['latency_ms']} ms/img"
    )
    return row


def save_results(rows: list[dict]) -> None:
    """增量保存结果，避免长时间运行中断后丢失。."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "model_efficiency_test.csv"
    xlsx_path = OUTPUT_DIR / "model_efficiency_test.xlsx"
    json_path = OUTPUT_DIR / "model_efficiency_test.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if not TEST_IMAGES.is_dir():
        raise FileNotFoundError(f"test 图像目录不存在: {TEST_IMAGES}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"数据集配置不存在: {DATA_YAML}")

    n_images = count_test_images()
    print(f"test 图像数量: {n_images}")
    print(f"设备: {DEVICE}  |  imgsz={IMGSZ}  |  batch={BATCH}")

    rows = []
    for name, weight in MODELS.items():
        rows.append(benchmark_one(name, weight, n_images))
        save_results(rows)

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print("汇总表:")
    display_cols = ["model", "Parameters(M)", "GFLOPs", "FPS", "latency_ms"]
    print(df[display_cols].to_string(index=False))
    print(f"\n已保存: {OUTPUT_DIR / 'model_efficiency_test.csv'}")
    print(f"已保存: {OUTPUT_DIR / 'model_efficiency_test.xlsx'}")


if __name__ == "__main__":
    main()
