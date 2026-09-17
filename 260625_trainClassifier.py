# import os
# import random
# import shutil
# from pathlib import Path

# """
# 按照比例抽取patches文件夹中的图片到classifier_datas文件夹中
# 70% train, 15% val, 15% test
# """


# def move_random_images(source_dir, target_dir, ratio=0.2, extensions=None):
#     """根据指定比例随机抽取图片并移动到指定文件夹

#     :param source_dir: 源图片目录路径
#     :param target_dir: 目标图片目录路径
#     :param ratio: 抽取比例 (0.0 到 1.0 之间)
#     :param extensions: 过滤的图片后缀名，默认为常见图片格式
#     """
#     if extensions is None:
#         extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

#     source_path = Path(source_dir)
#     target_path = Path(target_dir)

#     # 1. 创建目标文件夹（如果不存在的话）
#     target_path.mkdir(parents=True, exist_ok=True)

#     # 2. 获取源目录下所有符合条件的图片
#     image_files = [
#         f for f in source_path.iterdir() if f.suffix.lower() in extensions and f.is_file()
#     ]

#     total_count = len(image_files)
#     if total_count == 0:
#         print(f"❌ 在指定目录 '{source_dir}' 中没有找到图片。")
#         return

#     # 3. 计算需要抽取的数量
#     sample_count = int(total_count * ratio)
#     if sample_count == 0 and total_count > 0:
#         sample_count = 1  # 确保只要有图片，哪怕比例再小也至少抽一张

#     print(f"📊 总图片数: {total_count} 张，抽取比例: {ratio*100}%，预计移动: {sample_count} 张")

#     # 4. 随机抽取
#     selected_images = random.sample(image_files, sample_count)

#     # 5. 开始移动文件
#     moved_count = 0
#     for img_path in selected_images:
#         try:
#             # 拼接目标路径
#             dest_path = target_path / img_path.name
#             # 移动文件
#             shutil.move(str(img_path), str(dest_path))
#             moved_count += 1
#         except Exception as e:
#             print(f"⚠️ 移动文件 {img_path.name} 失败: {e}")

#     print(f"✅ 成功移动了 {moved_count} 张图片到 '{target_dir}'")

# # --- 使用示例 ---
# if __name__ == "__main__":
#     # 💡 在这里修改你的文件夹路径和抽样比例
#     SOURCE_DIRECTORY = r"datasets\Patches\whitehead\target_domain"  # 源图片文件夹
#     TARGET_DIRECTORY = r"datasets\classifier_datas\val\whitehead"  # 目标文件夹
#     SAMPLE_RATIO = 1  # 抽取比例

#     move_random_images(SOURCE_DIRECTORY, TARGET_DIRECTORY, ratio=SAMPLE_RATIO)


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO


def main():
    path_list = [
        "pretrain_cls_weights/yolov8n-cls.pt",
        "pretrain_cls_weights/yolov8s-cls.pt",
        "pretrain_cls_weights/yolo11n-cls.pt",
        "pretrain_cls_weights/yolo11s-cls.pt",
        "pretrain_cls_weights/yolov12n-cls.pt",
        "pretrain_cls_weights/yolov12s-cls.pt",
    ]
    for path in path_list:
        model = YOLO(path)
        # model = YOLO("yolov8n-cls.pt")

        results = model.train(
            data="datasets/classifier_datas",
            epochs=100,
            imgsz=224,
            batch=64,
            device=0,
            workers=8,
            project="runs/classify",
            name=path.split("/")[-1].split(".")[0],
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            patience=20,
            seed=42,
        )

        print("训练完成")
        print("结果目录：", results.save_dir)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------
# import os
# import torch
# from ultralytics import YOLO
# from sklearn.metrics import precision_score, recall_score
# from collections import Counter

# def eval_dataset(model, datadir, classnames):
#     from torchvision import datasets, transforms
#     import numpy as np

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     ds = datasets.ImageFolder(datadir, transform=transform)
#     loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
#     all_labels = []
#     all_preds = []
#     for batch_imgs, batch_labels in loader:
#         preds = model.predict(source=batch_imgs, imgsz=224, device=0, stream=True)
#         for pred, gt in zip(preds, batch_labels):
#             pred_cls = int(pred.probs.top1)
#             all_preds.append(pred_cls)
#             all_labels.append(gt.item())
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)

#     prec = precision_score(all_labels, all_preds, average=None, labels=range(len(classnames)))
#     rec = recall_score(all_labels, all_preds, average=None, labels=range(len(classnames)))
#     counts = Counter(all_labels)

#     print(f"\n数据集: {datadir} (样本总数: {len(ds)})")
#     for idx, cname in enumerate(classnames):
#         n = counts[idx]
#         p = prec[idx]
#         r = rec[idx]
#         print(f"  类别: {cname:<16} 数量: {n:<4}  Precision: {p:.4f}  Recall: {r:.4f}")
#     print("")

# def main():
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     model = YOLO("runs/classify/rice_whitehead_cls/weights/best.pt")
#     # 数据集路径
#     data_root = "datasets/classifier_datas"
#     classnames = sorted([d for d in os.listdir(os.path.join(data_root, "train")) if os.path.isdir(os.path.join(data_root, "train", d))])
#     for split in ["train", "val", "test"]:
#         split_dir = os.path.join(data_root, split)
#         if os.path.exists(split_dir):
#             eval_dataset(model, split_dir, classnames)
#         else:
#             print(f"[警告] 未找到 split: {split_dir}")

# if __name__ == "__main__":
#     main()
