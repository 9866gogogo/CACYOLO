import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO


def main():
    # 指定数据集的YAML配置路径
    data_yaml = "datasets/RicePanicleDemoDataset.yaml"
    # 选择模型类型
    model = YOLO('yolov8n.yaml') 

    # 训练参数
    results = model.train(
        data=data_yaml,        
        epochs=100,            
        imgsz=640,             
        batch=16,             
        device=0,              
        workers=4,             
        project="runs/train",
        name="yolov8n_ricepanicle", 
        exist_ok=True,         
        pretrained=True        
    )

    print("训练结束，结果保存在:", results.save_dir)

if __name__ == "__main__":
    main()