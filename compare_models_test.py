"""在 auto_annotations 测试集上对比 Baseline YOLO12s 与 CAC-YOLO12 检测性能。"""

import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DATA_YAML = "datasets/auto_annotations/dataset.yaml"
SPLIT = "test"
TEST_IMAGES = "datasets/auto_annotations/images/test"

MODELS = {
    "Baseline_YOLO12s": "runs/detect/runs/train/TwoClass_YOLO12s/weights/best.pt",
    "CAC_YOLO12_v2": "runs/detect/runs/train/improv_TwoClass_ColorYOLO12s_v2/weights/best.pt",
}

OUTPUT_DIR = Path("runs/compare_test")
BATCH = 16
DEVICE = 0


def eval_model(name: str, weight: str) -> dict:
    """在带标注的 test 集上评估，返回整体与逐类指标。"""
    weight_path = Path(weight)
    if not weight_path.exists():
        raise FileNotFoundError(f"权重不存在: {weight_path}")

    print(f"\n{'=' * 60}")
    print(f"评估模型: {name}")
    print(f"权重: {weight_path}")
    print(f"数据: {DATA_YAML}  |  split: {SPLIT}")
    print(f"测试图像目录: {TEST_IMAGES}")

    model = YOLO(str(weight_path))
    metrics = model.val(
        data=DATA_YAML,
        split=SPLIT,
        batch=BATCH,
        device=DEVICE,
        verbose=False,
    )

    p, r, map50, map5095 = metrics.mean_results()
    summary = {row["Class"]: row for row in metrics.summary()}

    # 推理速度（val 结束时 metrics 中通常含 speed）
    speed = getattr(metrics, "speed", {}) or {}

    return {
        "model": name,
        "weight": str(weight_path),
        "all": {
            "Precision": p,
            "Recall": r,
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map5095,
        },
        "classes": summary,
        "speed": speed,
    }


def results_to_rows(result: dict) -> list[dict]:
    """将单次评估结果展平为表格行。"""
    rows = []
    base = {"Model": result["model"]}
    for k, v in result["all"].items():
        rows.append({**base, "Class": "all", "Metric": k, "Value": v})

    for cls_name, cls_row in result["classes"].items():
        for metric in ("Box-P", "Box-R", "mAP50", "mAP50-95"):
            display = metric.replace("Box-", "").replace("mAP50", "mAP@0.5").replace(
                "mAP50-95", "mAP@0.5:0.95"
            )
            rows.append(
                {
                    **base,
                    "Class": cls_name,
                    "Metric": display,
                    "Value": cls_row[metric],
                }
            )
    return rows


def print_metrics(result: dict) -> None:
    """控制台打印指标。"""
    a = result["all"]
    print(f"\n  [Overall]")
    print(f"    Precision   : {a['Precision']:.4f}")
    print(f"    Recall      : {a['Recall']:.4f}")
    print(f"    mAP@0.5     : {a['mAP@0.5']:.4f}")
    print(f"    mAP@0.5:0.95: {a['mAP@0.5:0.95']:.4f}")

    print(f"\n  [Per-class]")
    for cls_name, row in result["classes"].items():
        print(
            f"    {cls_name:10s}  P={row['Box-P']:.4f}  R={row['Box-R']:.4f}  "
            f"mAP@0.5={row['mAP50']:.4f}  mAP@0.5:0.95={row['mAP50-95']:.4f}"
        )

    if result["speed"]:
        print(f"\n  [Speed]  {result['speed']}")


def print_comparison(baseline: dict, improve: dict) -> None:
    """打印两模型差值。"""
    print(f"\n{'=' * 60}")
    print("差值 (CAC_YOLO12_v2 - Baseline_YOLO12s)")
    print(f"{'=' * 60}")

    for key in baseline["all"]:
        d = improve["all"][key] - baseline["all"][key]
        print(f"  all  Δ{key:14s} = {d:+.4f}")

    for cls in baseline["classes"]:
        if cls not in improve["classes"]:
            continue
        d50 = improve["classes"][cls]["mAP50"] - baseline["classes"][cls]["mAP50"]
        d95 = improve["classes"][cls]["mAP50-95"] - baseline["classes"][cls]["mAP50-95"]
        dr = improve["classes"][cls]["Box-R"] - baseline["classes"][cls]["Box-R"]
        dp = improve["classes"][cls]["Box-P"] - baseline["classes"][cls]["Box-P"]
        print(
            f"  {cls:10s}  ΔP={dp:+.4f}  ΔR={dr:+.4f}  "
            f"ΔmAP@0.5={d50:+.4f}  ΔmAP@0.5:0.95={d95:+.4f}"
        )


def save_tables(results: list[dict], output_dir: Path) -> None:
    """保存对比表为 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 宽表：Overall + 逐类
    wide_rows = []
    for res in results:
        row = {"Model": res["model"]}
        row.update({f"all_{k}": v for k, v in res["all"].items()})
        for cls_name, cls_row in res["classes"].items():
            prefix = cls_name.replace(" ", "_")
            row[f"{prefix}_P"] = cls_row["Box-P"]
            row[f"{prefix}_R"] = cls_row["Box-R"]
            row[f"{prefix}_mAP50"] = cls_row["mAP50"]
            row[f"{prefix}_mAP50-95"] = cls_row["mAP50-95"]
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_path = output_dir / "test_metrics_comparison.csv"
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {wide_path}")

    # 长表
    long_rows = []
    for res in results:
        long_rows.extend(results_to_rows(res))
    long_df = pd.DataFrame(long_rows)
    long_path = output_dir / "test_metrics_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {long_path}")


def main():
    results = []
    for name, weight in MODELS.items():
        res = eval_model(name, weight)
        print_metrics(res)
        results.append(res)

    print_comparison(results[0], results[1])
    save_tables(results, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("评估完成。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
