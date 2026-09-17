"""使用 Baseline(YOLO12s) 与 改进模型(CAC-YOLO12/ColorYOLO12s_v2) 分别对
test 集图像进行推理标注，并挑选出改进模型相比 Baseline 明显提升的图像用于论文对比。

输出目录 runs/compare_test/annotate_compare/ 下包含:
  baseline_YOLO12s/          Baseline 逐图标注结果
  improved_ColorYOLO12s_v2/  改进模型逐图标注结果
  per_image_metrics.csv      逐图指标(TP/FP/FN/F1)及提升量
  top3_improvement/          明显提升的 Top3 三联对比图(GT | Baseline | Improved)
"""

import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TEST_IMAGES = Path("datasets/auto_annotations/images/test")
TEST_LABELS = Path("datasets/auto_annotations/labels/test")
CLASS_NAMES = {0: "healthy", 1: "whitehead"}

MODELS = {
    "baseline_YOLO12s": "runs/detect/runs/train/TwoClass_YOLO12s/weights/best.pt",
    "improved_ColorYOLO12s_v2": "runs/detect/runs/train/improv_TwoClass_ColorYOLO12s_v2/weights/best.pt",
}
BASELINE_KEY = "baseline_YOLO12s"
IMPROVED_KEY = "improved_ColorYOLO12s_v2"

OUTPUT_DIR = Path("runs/compare_test/annotate_compare")
CONF = 0.25          # 置信度阈值
NMS_IOU = 0.5        # 预测 NMS IoU
MATCH_IOU = 0.5      # 与 GT 匹配的 IoU 阈值
DEVICE = 0
MIN_GT = 5           # 挑选对比图时要求 GT 目标数不少于该值(避免选到近乎空图)
TOP_K = 3
DPI = 350            # 输出图像的 DPI

# 画框颜色 (BGR)
COLOR_GT = (0, 255, 0)        # 绿: 真值
COLOR_HEALTHY = (0, 200, 255) # 橙黄: healthy 预测
COLOR_WHITE = (0, 0, 255)     # 红: whitehead 预测


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def save_image(path: Path, bgr: np.ndarray, dpi: int = DPI) -> None:
    """以指定 DPI 保存图像(BGR->RGB), 将 DPI 写入文件元数据。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(path), dpi=(dpi, dpi))


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    """读取 YOLO 归一化标签, 返回 [N,5] -> (cls, x1, y1, x2, y2) 像素坐标。"""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        rows.append([c, x1, y1, x2, y2])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两组框的 IoU 矩阵, a:[N,4] b:[M,4] (x1,y1,x2,y2)。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def match_metrics(gt: np.ndarray, pred_boxes: np.ndarray, pred_cls: np.ndarray) -> dict:
    """按类别做 IoU 贪心匹配, 统计整体及 whitehead 的 TP/FP/FN/F1。"""
    stats = {}
    for scope, cls_filter in (("all", None), ("whitehead", 1)):
        if cls_filter is None:
            g = gt
            pb, pc = pred_boxes, pred_cls
        else:
            g = gt[gt[:, 0] == cls_filter]
            mask = pred_cls == cls_filter
            pb, pc = pred_boxes[mask], pred_cls[mask]

        tp = 0
        matched_gt = set()
        if len(pb) and len(g):
            ious = iou_matrix(pb, g[:, 1:5])
            # 按预测顺序贪心匹配同类且 IoU 最大的 GT
            for pi in range(len(pb)):
                best_iou, best_gi = MATCH_IOU, -1
                for gi in range(len(g)):
                    if gi in matched_gt:
                        continue
                    if g[gi, 0] != pc[pi]:
                        continue
                    if ious[pi, gi] >= best_iou:
                        best_iou, best_gi = ious[pi, gi], gi
                if best_gi >= 0:
                    tp += 1
                    matched_gt.add(best_gi)
        fp = len(pb) - tp
        fn = len(g) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        stats[scope] = {
            "gt": len(g), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return stats


def draw_boxes(img: np.ndarray, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """在图像副本上按类别着色绘制检测框(healthy/whitehead 分色)。"""
    out = img.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        c = int(classes[i])
        color = COLOR_WHITE if c == 1 else COLOR_HEALTHY
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in MODELS:
        (OUTPUT_DIR / key).mkdir(parents=True, exist_ok=True)
    top_dir = OUTPUT_DIR / "top3_improvement"
    top_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in TEST_IMAGES.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    print(f"测试图像数量: {len(images)}")

    # 缓存每个模型对每张图的预测(框/类别/置信度)
    preds = {key: {} for key in MODELS}

    for key, weight in MODELS.items():
        wp = Path(weight)
        if not wp.exists():
            raise FileNotFoundError(f"权重不存在: {wp}")
        print(f"\n{'=' * 60}\n加载模型 [{key}]: {wp}")
        model = YOLO(str(wp))

        results = model.predict(
            source=[str(p) for p in images],
            conf=CONF, iou=NMS_IOU, device=DEVICE,
            verbose=False, stream=True,
        )
        for img_path, res in zip(images, results):
            annotated = res.plot()  # BGR ndarray, 标准 YOLO 风格
            save_image(OUTPUT_DIR / key / img_path.name, annotated)
            b = res.boxes
            if b is not None and len(b):
                preds[key][img_path.name] = (
                    b.xyxy.cpu().numpy(),
                    b.cls.cpu().numpy().astype(int),
                    b.conf.cpu().numpy(),
                )
            else:
                preds[key][img_path.name] = (
                    np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,))
                )
        print(f"  已标注 {len(images)} 张 -> {OUTPUT_DIR / key}")

    # 逐图计算指标与提升量
    print(f"\n{'=' * 60}\n逐图计算指标 ...")
    rows = []
    for img_path in images:
        name = img_path.name
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = load_gt(TEST_LABELS / f"{img_path.stem}.txt", w, h)

        bb, bc, _ = preds[BASELINE_KEY][name]
        ib, ic, _ = preds[IMPROVED_KEY][name]
        m_base = match_metrics(gt, bb, bc)
        m_imp = match_metrics(gt, ib, ic)

        rows.append({
            "image": name,
            "gt_all": m_base["all"]["gt"],
            "gt_whitehead": m_base["whitehead"]["gt"],
            "base_f1": m_base["all"]["f1"],
            "imp_f1": m_imp["all"]["f1"],
            "delta_f1": m_imp["all"]["f1"] - m_base["all"]["f1"],
            "base_wh_f1": m_base["whitehead"]["f1"],
            "imp_wh_f1": m_imp["whitehead"]["f1"],
            "delta_wh_f1": m_imp["whitehead"]["f1"] - m_base["whitehead"]["f1"],
            "base_fn": m_base["all"]["fn"],
            "imp_fn": m_imp["all"]["fn"],
            "base_fp": m_base["all"]["fp"],
            "imp_fp": m_imp["all"]["fp"],
        })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "per_image_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"逐图指标已保存: {csv_path}")

    # 挑选明显提升的图像:
    #   综合评分 = 整体 F1 提升 + 0.5 * whitehead F1 提升
    #   要求 GT 目标数 >= MIN_GT, 且整体 F1 确有提升
    df["score"] = df["delta_f1"] + 0.5 * df["delta_wh_f1"].fillna(0)
    cand = df[(df["gt_all"] >= MIN_GT) & (df["delta_f1"] > 0)].copy()
    cand = cand.sort_values("score", ascending=False)
    top = cand.head(TOP_K)

    print(f"\n{'=' * 60}\n明显提升 Top{TOP_K}:")
    for _, r in top.iterrows():
        print(f"  {r['image']:24s}  F1 {r['base_f1']:.3f}->{r['imp_f1']:.3f} "
              f"(Δ{r['delta_f1']:+.3f})  漏检 {int(r['base_fn'])}->{int(r['imp_fn'])} "
              f"误检 {int(r['base_fp'])}->{int(r['imp_fp'])}")

    # 生成三联对比图: GT | Baseline | Improved
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        name = r["image"]
        img_path = TEST_IMAGES / name
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = load_gt(TEST_LABELS / f"{Path(name).stem}.txt", w, h)

        bb, bc, _ = preds[BASELINE_KEY][name]
        ib, ic, _ = preds[IMPROVED_KEY][name]

        panel_gt = draw_boxes(img, gt[:, 1:5], gt[:, 0])
        panel_base = draw_boxes(img, bb, bc)
        panel_imp = draw_boxes(img, ib, ic)

        gap = np.full((panel_gt.shape[0], 8, 3), 255, dtype=np.uint8)
        combo = np.hstack([panel_gt, gap, panel_base, gap, panel_imp])
        out_path = top_dir / f"top{rank}_{Path(name).stem}.jpg"
        save_image(out_path, combo)
        print(f"  已生成对比图: {out_path}")

    print(f"\n{'=' * 60}\n完成。全部结果位于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
