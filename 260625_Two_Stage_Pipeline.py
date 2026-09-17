# import os
# import cv2
# import numpy as np

# from ultralytics import YOLO

# # ==========================
# # 加载模型
# # ==========================

# detector = YOLO(
#     "runs/train/v12s/weights/best.pt"
# )

# classifier = YOLO(
#     "runs/classify/rice_whitehead_cls/weights/best.pt"
# )

# # ==========================
# # 输入图片
# # ==========================

# img_path = "test.jpg"

# img = cv2.imread(img_path)

# # ==========================
# # 稻穗检测
# # ==========================

# det_results = detector.predict(
#     source=img,
#     conf=0.25,
#     verbose=False
# )

# result = det_results[0]

# boxes = result.boxes.xyxy.cpu().numpy()

# healthy_count = 0
# whitehead_count = 0

# # ==========================
# # 遍历每个检测框
# # ==========================

# for box in boxes:

#     x1, y1, x2, y2 = map(int, box)

#     crop = img[y1:y2, x1:x2]

#     if crop.size == 0:
#         continue

#     # ======================
#     # 分类
#     # ======================

#     cls_result = classifier.predict(
#         source=crop,
#         verbose=False
#     )

#     cls_id = int(
#         cls_result[0].probs.top1
#     )

#     conf = float(
#         cls_result[0].probs.top1conf
#     )

#     # ======================
#     # healthy
#     # ======================

#     if cls_id == 0:

#         healthy_count += 1

#         color = (0,255,0)

#         label = f"Healthy {conf:.2f}"

#     # ======================
#     # whitehead
#     # ======================

#     else:

#         whitehead_count += 1

#         color = (0,0,255)

#         label = f"Whitehead {conf:.2f}"

#     # ======================
#     # 画框
#     # ======================

#     cv2.rectangle(
#         img,
#         (x1,y1),
#         (x2,y2),
#         color,
#         2
#     )

#     cv2.putText(
#         img,
#         label,
#         (x1,y1-5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         color,
#         2
#     )

# # ==========================
# # 白穗率
# # ==========================

# total = healthy_count + whitehead_count

# if total > 0:

#     whitehead_rate = (
#         whitehead_count / total
#     ) * 100

# else:

#     whitehead_rate = 0

# print(f"Healthy : {healthy_count}")
# print(f"Whitehead : {whitehead_count}")
# print(f"Whitehead Rate : {whitehead_rate:.2f}%")

# # ==========================
# # 保存结果
# # ==========================

# cv2.imwrite(
#     "result.jpg",
#     img
# )

# print("结果保存为 result.jpg")

import os
import cv2
import csv
import shutil
import random
from pathlib import Path

from ultralytics import YOLO

# =====================================================
# 配置
# =====================================================

DETECT_MODEL = "runs/train/v12s/weights/best.pt"
CLS_MODEL = "runs/classify/runs/classify/yolo11n-cls/weights/best.pt"

# 原始图片目录
INPUT_DIR = "datasets/baisui_datas/images"

# 输出目录
OUTPUT_DIR = "datasets/new_auto_annotations_1"

# 阈值
DET_CONF = 0.35
CLS_CONF = 0.90

# 数据集划分
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子
RANDOM_SEED = 42

# =====================================================
# 加载模型
# =====================================================

detector = YOLO(DETECT_MODEL)
classifier = YOLO(CLS_MODEL)

# =====================================================
# 创建目录
# =====================================================

dirs = [
    "images/train",
    "images/val",
    "images/test",

    "labels/train",
    "labels/val",
    "labels/test",

    "visualize/train",
    "visualize/val",
    "visualize/test",
]

for d in dirs:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

# =====================================================
# 搜索图片
# =====================================================

suffix = [".jpg", ".jpeg", ".png", ".bmp"]

images = []

for root, _, files in os.walk(INPUT_DIR):

    for f in files:

        if Path(f).suffix.lower() in suffix:
            images.append(os.path.join(root, f))

random.seed(RANDOM_SEED)
random.shuffle(images)

total_images = len(images)

train_end = int(total_images * TRAIN_RATIO)
val_end = int(total_images * (TRAIN_RATIO + VAL_RATIO))

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

print("=" * 50)
print(f"Total Images : {total_images}")
print(f"Train        : {len(train_imgs)}")
print(f"Val          : {len(val_imgs)}")
print(f"Test         : {len(test_imgs)}")
print("=" * 50)

# =====================================================
# CSV
# =====================================================

csv_path = os.path.join(OUTPUT_DIR, "annotation.csv")

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "image",
    "class",
    "detector_conf",
    "classifier_conf",
    "x1",
    "y1",
    "x2",
    "y2"
])

# =====================================================
# 统计变量
# =====================================================

total_boxes = 0
healthy_boxes = 0
whitehead_boxes = 0
filtered_boxes = 0


# =====================================================
# 处理函数
# =====================================================

def process(image_list, mode):

    global total_boxes
    global healthy_boxes
    global whitehead_boxes
    global filtered_boxes

    for img_path in image_list:

        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        vis = img.copy()

        result = detector.predict(
            source=img,
            conf=DET_CONF,
            verbose=False
        )[0]

        txt_lines = []

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        det_scores = result.boxes.conf.cpu().numpy()

        for box, det_score in zip(boxes, det_scores):

            x1, y1, x2, y2 = box.astype(int)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            cls_result = classifier.predict(
                source=crop,
                verbose=False
            )[0]

            cls_id = int(cls_result.probs.top1)
            cls_score = float(cls_result.probs.top1conf)

            # 双重置信度过滤
            if det_score < DET_CONF or cls_score < CLS_CONF:
                filtered_boxes += 1
                continue

            total_boxes += 1

            if cls_id == 0:

                healthy_boxes += 1

                color = (0, 255, 0)
                cls_name = "Healthy"

            else:

                whitehead_boxes += 1

                color = (0, 0, 255)
                cls_name = "Whitehead"

            # YOLO格式
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            txt_lines.append(
                f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

            # 保存CSV
            writer.writerow([
                os.path.basename(img_path),
                cls_name,
                float(det_score),
                float(cls_score),
                x1,
                y1,
                x2,
                y2
            ])

            # 可视化
            label = f"{cls_name} {cls_score:.2f}"

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                vis,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # 保存图片
        img_name = os.path.basename(img_path)

        shutil.copy(
            img_path,
            os.path.join(
                OUTPUT_DIR,
                "images",
                mode,
                img_name
            )
        )

        # 保存可视化

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                "visualize",
                mode,
                img_name
            ),
            vis
        )

        # 保存txt

        txt_name = Path(img_name).stem + ".txt"

        with open(
            os.path.join(
                OUTPUT_DIR,
                "labels",
                mode,
                txt_name
            ),
            "w"
        ) as f:

            for line in txt_lines:
                f.write(line + "\n")


# =====================================================
# 开始生成
# =====================================================

print("Generating Train Labels...")
process(train_imgs, "train")

print("Generating Val Labels...")
process(val_imgs, "val")

print("Generating Test Labels...")
process(test_imgs, "test")

csv_file.close()

# =====================================================
# 生成dataset.yaml
# =====================================================

yaml_text = f"""path: {OUTPUT_DIR}

train: images/train
val: images/val
test: images/test

names:
  0: healthy
  1: whitehead
"""

with open(
    os.path.join(
        OUTPUT_DIR,
        "dataset.yaml"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml_text)

# =====================================================
# 输出统计信息
# =====================================================

print("\n")
print("=" * 60)
print("Automatic Annotation Finished")
print("=" * 60)

print(f"Images               : {total_images}")
print(f"Total Boxes          : {total_boxes}")
print(f"Healthy              : {healthy_boxes}")
print(f"Whitehead            : {whitehead_boxes}")
print(f"Filtered             : {filtered_boxes}")

if total_boxes > 0:
    print(f"Whitehead Rate       : {whitehead_boxes / total_boxes * 100:.2f}%")

print("=" * 60)
print("Dataset Saved To:")
print(OUTPUT_DIR)
print("=" * 60)