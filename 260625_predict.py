import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO


def main():
    model_weights = "runs/train/v12s/weights/best.pt"
    model = YOLO(model_weights)

    baisui_images_dir = "datasets/baisui_datas/images"

    results = model.predict(
        source=baisui_images_dir, save=True, save_txt=True, save_conf=True, imgsz=640, conf=0.25, device=0
    )

    print("预测完成，结果已保存在: ", results[0].save_dir if results else "无结果")


if __name__ == "__main__":
    main()
