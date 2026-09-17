"""统计 TwoClass 模型在 auto_annotations test 集上的 GFLOPs、参数量与 FPS。"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "datasets/auto_annotations/dataset.yaml"
TEST_IMAGES = ROOT / "datasets/auto_annotations/images/test"
OUTPUT_DIR = ROOT / "runs/model_efficiency"
IMGSZ = 640
BATCH = 1
DEVICE = 0 if torch.cuda.is_available() else "cpu"

MODELS = {
    "TwoClass_YOLO12s": "runs/detect/runs/train/new_TwoClass_YOLO12s/weights/best.pt",
    "improv_TwoClass_ColorYOLO12s_v2": (
        "runs/detect/runs/train/new_improv_TwoClass_ColorYOLO12s/weights/best.pt"
    ),
}


def count_test_images() -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(1 for p in TEST_IMAGES.iterdir() if p.suffix.lower() in exts)


def speed_to_fps(speed: dict) -> tuple[float, float, dict]:
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "twoclass_model_efficiency_test"
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_excel(OUTPUT_DIR / f"{stem}.xlsx", index=False)
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
    print(df[["model", "Parameters(M)", "GFLOPs", "FPS", "latency_ms"]].to_string(index=False))
    print(f"\n已保存: {OUTPUT_DIR / 'twoclass_model_efficiency_test.csv'}")


if __name__ == "__main__":
    main()
