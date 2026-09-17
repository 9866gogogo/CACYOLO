from __future__ import annotations

import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd

from ultralytics import YOLO

DATA_YAML = "datasets/new_auto_annotations/dataset.yaml"
SPLIT = "test"
PROJECT = "runs/new_compare_test"

BASELINE = "runs/detect/runs/train/new_TwoClass_YOLO12s/weights/best.pt"
IMPROVE = "runs/detect/runs/train/new_improv_TwoClass_ColorYOLO12s/weights/best.pt"
CLASS_NAMES = ["healthy", "whitehead", "background"]


def eval_one(name: str, path: str, run_name: str) -> dict:
    """在 test 集上评估单个模型。."""
    print(f"\n>>> {name}: {path}")
    metrics = YOLO(path).val(
        data=DATA_YAML,
        split=SPLIT,
        batch=16,
        device=0,
        verbose=False,
        plots=True,
        project=PROJECT,
        name=run_name,
        exist_ok=True,
    )
    p, r, map50, map5095 = metrics.mean_results()
    rows = {r["Class"]: r for r in metrics.summary()}
    confusion = getattr(metrics, "confusion_matrix", None)
    matrix = confusion.matrix if confusion is not None else None
    return {
        "name": name,
        "all": {
            "Box-P": p,
            "Box-R": r,
            "mAP50": map50,
            "mAP50-95": map5095,
        },
        "classes": rows,
        "save_dir": Path(metrics.save_dir),
        "confusion_matrix": matrix,
    }


def print_result(res: dict) -> None:
    """打印整体与逐类指标。."""
    a = res["all"]
    print(f"  all       P={a['Box-P']:.4f}  R={a['Box-R']:.4f}  mAP50={a['mAP50']:.4f}  mAP50-95={a['mAP50-95']:.4f}")
    for cls, row in res["classes"].items():
        print(
            f"  {cls:10s}  P={row['Box-P']:.4f}  R={row['Box-R']:.4f}  "
            f"mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50-95']:.4f}"
        )
    print(f"  结果目录: {res['save_dir']}")
    print_confusion_matrix(res)


def print_confusion_matrix(res: dict) -> None:
    """打印并保存 test 集混淆矩阵。."""
    matrix = res.get("confusion_matrix")
    if matrix is None:
        print("  未读取到混淆矩阵。")
        return

    df = pd.DataFrame(matrix.astype(int), index=CLASS_NAMES, columns=CLASS_NAMES)
    out_path = res["save_dir"] / "confusion_matrix_values.csv"
    df.to_csv(out_path, encoding="utf-8-sig")

    print("\n  Confusion matrix (rows=Predicted, columns=True):")
    print(df.to_string())
    print(f"  混淆矩阵数值已保存: {out_path}")
    print(f"  混淆矩阵图片: {res['save_dir'] / 'confusion_matrix.png'}")


def paper_rows_from_cm(name: str, matrix) -> list[dict]:
    """由混淆矩阵生成论文用统计行。 Ultralytics: rows=Predicted, columns=True。 Correctly detected / Misclassified as other class /
    Missed as background。.
    """
    m = matrix.astype(int)
    # 前景类索引: 0=healthy, 1=whitehead；2=background
    observed = [("Healthy Panicles", 0, 1), ("White Heads", 1, 0)]
    rows = []
    for obs_name, true_i, other_i in observed:
        rows.append(
            {
                "Model": name,
                "Observed class": obs_name,
                "Correctly detected": int(m[true_i, true_i]),
                "Misclassified as other class": int(m[other_i, true_i]),
                "Missed as background": int(m[2, true_i]),
            }
        )
    return rows


def print_paper_table(results: list[dict]) -> None:
    """打印并保存论文格式混淆统计表。."""
    rows = []
    for res in results:
        rows.extend(paper_rows_from_cm(res["name"], res["confusion_matrix"]))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 55)
    print("论文用混淆统计表 (test)")
    print(df.to_string(index=False))

    out = Path(PROJECT) / "confusion_paper_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存: {out.resolve()}")


def main():
    print(f"数据集: {DATA_YAML}  |  split: {SPLIT}")

    baseline = eval_one("Baseline YOLOv12s", BASELINE, "Baseline_test")
    improve = eval_one("CAC-YOLOv12s", IMPROVE, "CAC_YOLOv12s_test")

    print("\n" + "=" * 55)
    print("Baseline YOLOv12s")
    print_result(baseline)
    print("\nCAC-YOLOv12s")
    print_result(improve)

    print_paper_table([baseline, improve])

    b, i = baseline["all"], improve["all"]
    print("\n" + "=" * 55)
    print("差值 (CAC-YOLOv12s - Baseline)")
    print(f"  all  ΔmAP50={i['mAP50'] - b['mAP50']:+.4f}  ΔmAP50-95={i['mAP50-95'] - b['mAP50-95']:+.4f}")
    for cls in baseline["classes"]:
        if cls in improve["classes"]:
            d = improve["classes"][cls]["mAP50"] - baseline["classes"][cls]["mAP50"]
            print(f"  {cls:10s}  ΔmAP50={d:+.4f}")


if __name__ == "__main__":
    main()
