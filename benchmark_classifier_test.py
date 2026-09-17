"""评估 runs/classify/runs/classify 下二分类模型在 classifier_datas test 集上的指标。."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets/classifier_datas"
TEST_DIR = DATA_DIR / "test"
MODEL_ROOT = ROOT / "runs/classify/runs/classify"
OUTPUT_DIR = ROOT / "runs/classify/compare_test"
IMGSZ = 224
BATCH = 1
# 独立测速：充分预热后多次整段计时，取中位数，避免 val() 单次墙钟抖动
FPS_WARMUP = 50
FPS_ITERS = 100
FPS_TRIALS = 7

MODEL_DISPLAY = {
    "yolov8n-cls": "YOLOv8n-cls",
    "yolov8s-cls": "YOLOv8s-cls",
    "yolo11n-cls": "YOLO11n-cls",
    "yolo11s-cls": "YOLO11s-cls",
    "yolov12n-cls": "YOLOv12n-cls",
    "yolov12s-cls": "YOLOv12s-cls",
}


def resolve_device() -> str | int:
    """选择推理设备，CUDA 不可用时静默回退 CPU。."""
    return 0 if torch.cuda.is_available() else "cpu"


DEVICE = resolve_device()


def discover_models() -> dict[str, Path]:
    """扫描分类训练目录，收集 best.pt 权重。."""
    models: dict[str, Path] = {}
    if not MODEL_ROOT.is_dir():
        raise FileNotFoundError(f"未找到模型目录: {MODEL_ROOT}")
    for run_dir in sorted(MODEL_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        weight = run_dir / "weights" / "best.pt"
        if weight.is_file():
            models[run_dir.name] = weight
    if not models:
        raise FileNotFoundError(f"未在 {MODEL_ROOT} 下找到任何 best.pt 权重")
    return models


def count_test_images() -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(
        1 for cls_dir in TEST_DIR.iterdir() if cls_dir.is_dir() for p in cls_dir.iterdir() if p.suffix.lower() in exts
    )


def _torch_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int) or (isinstance(device, str) and str(device).isdigit()):
        return torch.device(f"cuda:{int(device)}")
    return torch.device(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sigma_clip(data: np.ndarray, sigma: float = 2.0, max_iters: int = 3) -> np.ndarray:
    """剔除偏离均值超过 sigma 的异常试次。."""
    data = np.asarray(data, dtype=float)
    for _ in range(max_iters):
        mean, std = data.mean(), data.std()
        if std < 1e-9 or len(data) < 3:
            break
        clipped = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
        if len(clipped) < max(3, len(data) // 2) or len(clipped) == len(data):
            break
        data = clipped
    return data


@torch.inference_mode()
def measure_stable_fps(
    model: YOLO,
    imgsz: int = IMGSZ,
    batch: int = BATCH,
    device=DEVICE,
    warmup: int = FPS_WARMUP,
    iters: int = FPS_ITERS,
    trials: int = FPS_TRIALS,
) -> tuple[float, float, float]:
    """用假数据测纯推理 FPS，返回 (中位FPS, 中位延迟ms, 延迟标准差ms)。.

    不走 val() 的 preprocess/磁盘 I/O；整段计时摊薄 Python 开销，多次试次取中位数。
    """
    torch_device = _torch_device(device)
    net = model.model.to(torch_device).eval()
    try:
        model.fuse()
        net = model.model.to(torch_device).eval()
    except Exception:
        pass

    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=torch_device)
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for _ in range(warmup):
        net(dummy)
    _synchronize(torch_device)

    trial_ms = []
    for _ in range(trials):
        _synchronize(torch_device)
        t0 = time.perf_counter()
        for _ in range(iters):
            net(dummy)
        _synchronize(torch_device)
        trial_ms.append((time.perf_counter() - t0) * 1000.0 / iters / batch)

    trial_ms = _sigma_clip(np.array(trial_ms))
    latency_ms = float(np.median(trial_ms))
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return round(fps, 2), round(latency_ms, 3), round(float(np.std(trial_ms)), 3)


def metrics_from_confusion(matrix: np.ndarray, class_names: dict, top1: float | None = None) -> dict:
    """根据混淆矩阵计算宏平均 Precision / Recall / F1。.

    Ultralytics 分类验证的混淆矩阵索引为 matrix[预测类, 真实类]；默认按 detect 任务 初始化时尺寸为 (nc+1)x(nc+1)，需截取前 nc 类，避免 background 行/列拉低宏平均。
    """
    nc = len(class_names)
    matrix = np.asarray(matrix[:nc, :nc], dtype=float)

    tp = np.diag(matrix)
    fp = matrix.sum(1) - tp  # 预测为 i 的样本数 - TP
    fn = matrix.sum(0) - tp  # 真实为 i 的样本数 - TP
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # 仅对 test 集中实际出现的类做宏平均
    present = (matrix.sum(1) + matrix.sum(0)) > 0
    macro = lambda arr: round(float(arr[present].mean()), 4) if present.any() else 0.0

    if top1 is None:
        total = matrix.sum()
        top1 = float(tp.sum() / total) if total > 0 else 0.0

    return {
        "Precision": macro(precision),
        "Recall": macro(recall),
        "F1-Score": macro(f1),
        "top1_acc": round(float(top1), 4),
        "per_class": {
            str(i): {
                "name": class_names[i],
                "Precision": round(float(precision[i]), 4),
                "Recall": round(float(recall[i]), 4),
                "F1-Score": round(float(f1[i]), 4),
                "support": int(matrix.sum(0)[i]),
            }
            for i in range(nc)
        },
    }


def benchmark_one(run_name: str, weight: Path, n_images: int) -> dict:
    display_name = MODEL_DISPLAY.get(run_name, run_name)
    print(f"\n{'=' * 60}")
    print(f"模型: {display_name}")
    print(f"权重: {weight}")

    model = YOLO(str(weight))
    params = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=IMGSZ)

    metrics = model.val(
        data=str(DATA_DIR),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        verbose=False,
        plots=False,
    )

    class_names = metrics.confusion_matrix.names
    cls_metrics = metrics_from_confusion(
        metrics.confusion_matrix.matrix,
        class_names=class_names,
        top1=float(metrics.top1),
    )
    fps, latency_ms, latency_std = measure_stable_fps(model)

    row = {
        "model": display_name,
        "run_name": run_name,
        "weight": str(weight.relative_to(ROOT)).replace("\\", "/"),
        "Precision": cls_metrics["Precision"],
        "Recall": cls_metrics["Recall"],
        "F1-Score": cls_metrics["F1-Score"],
        "top1_acc": cls_metrics["top1_acc"],
        "top5_acc": round(float(metrics.top5), 4),
        "Parameters": int(params),
        "Params(M)": round(params / 1e6, 3),
        "GFLOPs": round(float(gflops), 2),
        "FPS": fps,
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std,
        "test_images": n_images,
        "class_names": class_names,
        "per_class": cls_metrics["per_class"],
        "device": str(DEVICE),
        "imgsz": IMGSZ,
        "batch": BATCH,
    }

    print(
        f"  Precision: {row['Precision']:.4f}  Recall: {row['Recall']:.4f}  "
        f"F1: {row['F1-Score']:.4f}  top1: {row['top1_acc']:.4f}"
    )
    print(
        f"  Params(M): {row['Params(M)']}  GFLOPs: {row['GFLOPs']}  "
        f"FPS: {row['FPS']}  Latency: {row['latency_ms']}±{row['latency_std_ms']} ms/img"
    )
    return row


def save_results(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in {"per_class", "class_names"}}
        flat_rows.append(flat)

    stem = "classifier_test_metrics"
    pd.DataFrame(flat_rows).to_csv(OUTPUT_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flat_rows).to_excel(OUTPUT_DIR / f"{stem}.xlsx", index=False)
    (OUTPUT_DIR / f"{stem}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"test 目录不存在: {TEST_DIR}")

    models = discover_models()
    n_images = count_test_images()
    print(f"发现模型数: {len(models)}")
    print(f"test 图像数量: {n_images}")
    print(f"设备: {DEVICE}  |  imgsz={IMGSZ}  |  batch={BATCH}")
    print(f"FPS 测速: warmup={FPS_WARMUP}  iters={FPS_ITERS}  trials={FPS_TRIALS}（假数据纯推理中位数）")

    rows = []
    for run_name, weight in models.items():
        rows.append(benchmark_one(run_name, weight, n_images))
        save_results(rows)

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"per_class", "class_names"}} for r in rows])
    print(f"\n{'=' * 60}")
    print("汇总表:")
    cols = ["model", "top1_acc", "Precision", "Recall", "F1-Score", "Params(M)", "GFLOPs", "FPS", "latency_ms"]
    print(df[cols].to_string(index=False))
    print(f"\n已保存: {OUTPUT_DIR / 'classifier_test_metrics.csv'}")


if __name__ == "__main__":
    main()
