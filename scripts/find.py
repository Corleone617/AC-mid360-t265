import cv2
import numpy as np

def find_circles(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("图像读取失败")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edged = cv2.Canny(blurred, 30, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 检测圆形
    for contour in contours:
        # 轮廓近似
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # 计算轮廓的边界框
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # 圆形检测条件：使用近似点少于10且半径在指定范围内
        if len(approx) < 10 and 90 <= radius <= 140:
            print(f"找到一个圆形，中心点为 ({x}, {y})，半径为 {radius:.2f} 像素")
            # 绘制圆形轮廓
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Circles Found", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用实际图像路径替换 'path_to_image.jpg'
find_circles('path_to_image.jpg')