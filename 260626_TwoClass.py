# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from ultralytics import YOLO


# def main():
#     data_yaml = "datasets/auto_annotations/dataset.yaml"
#     model = YOLO("runs/train/v12s/weights/best.pt")
#     results = model.train(
#         data=data_yaml,
#         epochs=100,
#         imgsz=640,
#         batch=16,
#         device=0,
#         workers=4,
#         optimizer="AdamW",
#         lr0=0.001,
#         weight_decay=5e-4,
#         cos_lr=True,
#         patience=30,
#         project="runs/train",
#         name="TwoClass_YOLOv12s",
#         exist_ok=True,
#         pretrained=True,
#         save=True,
#         save_period=10,
#         verbose=True
#     )

#     print("训练结束")
#     print(results.save_dir)


# if __name__ == "__main__":
#     main()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO



def main():

    data_yaml = "datasets/new_auto_annotations/dataset.yaml"
    # model = YOLO("ultralytics/cfg/models/12/yolo12s.yaml")
    # model = YOLO("runs/detect/runs/train/TwoClass_YOLO12s/weights/last.pt")
    model = YOLO("runs/train/v12s/weights/best.pt")

    results = model.train(
        data=data_yaml,
        pretrained=True,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4, 
        amp=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        cos_lr=True,
        patience=20,
        close_mosaic=20,
        cls_pw=0.75,  # whitehead 分类 loss 适度加权
        copy_paste=0.1,
        seed=0,
        project="runs/train",
        name="new_TwoClass_YOLO12s",
        # name="TwoClass_YOLO12s",
        exist_ok=True,
        save=True,
        save_period=10,
        verbose=True
    )



    print("训练结束")
    print(results.save_dir)





if __name__ == "__main__":

    main()


