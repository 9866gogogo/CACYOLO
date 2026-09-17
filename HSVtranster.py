import cv2

# 读取原始图片
img_bgr = cv2.imread("WhitePanicle_4320.jpg")
if img_bgr is None:
    raise FileNotFoundError("未找到指定图片：WhitePanicle_4320.jpg")

# 转换为HSV色彩空间
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 保存HSV图片（注意：HSV空间通常用于处理，不直接用于显示，此处仅演示转换和保存）
cv2.imwrite("WhitePanicle_4320_hsv.png", img_hsv)
print("已保存HSV图片：WhitePanicle_4320_hsv.png")
