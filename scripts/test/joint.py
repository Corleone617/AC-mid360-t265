import cv2
import os
import numpy as np

# 定义内切矩形的坐标
top_left = (676, 676)
bottom_right = (1408, 1408)

# 计算内切矩形的宽度和高度
rect_width = bottom_right[0] - top_left[0]
rect_height = bottom_right[1] - top_left[1]

# 源图片目录和目标背景图片路径
image_dir = '/home/corleone/ROS1/realsense/src/fcu_serial/scripts/image'
background_path = '/home/corleone/ROS1/realsense/src/fcu_serial/scripts/square.jpg'
output_dir = '/home/corleone/ROS1/realsense/src/fcu_serial/scripts/result'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载背景图片
background = cv2.imread(background_path)
if background is None:
    print(f"无法加载背景图片: {background_path}")
else:
    # 读取并处理每一张图片
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图片: {image_path}")
            continue

        # 计算缩放比例
        scale = min(rect_width / image.shape[1], rect_height / image.shape[0])

        # 计算新的尺寸
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

        # 缩放图片
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # 在背景图片上创建一个副本来嵌入图片
        embedded_image = background.copy()

        # 计算嵌入的起始位置
        start_x = top_left[0] + (rect_width - new_size[0]) // 2
        start_y = top_left[1] + (rect_height - new_size[1]) // 2

        # 将缩放后的图片嵌入到背景中的指定位置
        embedded_image[start_y:start_y+new_size[1], start_x:start_x+new_size[0]] = resized_image

        # 保存处理后的图片
        output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_image_path, embedded_image)
        print(f"处理后的图片已保存为: {output_image_path}")