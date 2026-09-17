import os
from collections import Counter

# ---------------- 配置参数 ----------------
# 替换为你的 .txt 标注文件所在的文件夹路径
folder_path = r"datasets\RicePanicleDemoDataset\labels\val"  
# ------------------------------------------

# 初始化统计器
class_counts = Counter()
total_boxes = 0
total_files = 0
files_with_boxes = 0

# 遍历文件夹下的所有 .txt 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        
        has_box = False
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                parts = line.split()
                if len(parts) >= 2:  # YOLO格式通常包含：class_id x_center y_center width height
                    class_id = parts[0]
                    class_counts[class_id] += 1
                    total_boxes += 1
                    has_box = True
        
        if has_box:
            files_with_boxes += 1

# 输出统计结果
print("=" * 40)
print(" 📊 YOLO 标注框统计结果")
print("=" * 40)
print(f"读取的文件总数: {total_files}")
print(f"包含标注的文件数: {files_with_boxes}")
print(f"标注框总数量 (Total Boxes): {total_boxes}\n")

print("--- 各类别 (Class ID) 数量统计 ---")
# 按类别编号排序输出
for class_id in sorted(class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x):
    count = class_counts[class_id]
    percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
    print(f"类别 ID {class_id:>3s} : {count:>8d} 个框  ({percentage:6.2f}%)")
print("=" * 40)