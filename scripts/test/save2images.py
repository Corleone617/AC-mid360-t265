import os
import pickle
import numpy as np
from PIL import Image

# CIFAR-100数据集的位置
data_dir = '/home/corleone/ROS1/realsense/src/data/cifar-100-python'
output_dir = '/home/corleone/ROS1/realsense/src/fcu_serial/scripts/image'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载CIFAR-100的类别标签
with open(os.path.join(data_dir, 'meta'), 'rb') as file:
    meta = pickle.load(file, encoding='latin1')
    class_names = meta['fine_label_names']

# 加载训练数据
with open(os.path.join(data_dir, 'train'), 'rb') as file:
    data = pickle.load(file, encoding='latin1')
    images = data['data']
    labels = data['fine_labels']

# 将图片数据转换为3D数组 (颜色通道, 高度, 宽度)
images = images.reshape((-1, 3, 32, 32))

# 创建一个字典来记录每个类别保存的图片数量
class_counts = {class_name: 0 for class_name in class_names}

# 限制每个类别保存的图片数量
limit_per_class = 1

for img, label in zip(images, labels):
    class_name = class_names[label]
    class_count = class_counts[class_name]

    # 如果已经达到限制，则跳过当前类别
    if class_count == limit_per_class:
        continue

    # 转换图像形状为(H, W, C)
    img = img.transpose((1, 2, 0))

    # 保存图像
    output_filename = os.path.join(output_dir, f'{class_name}_{class_count}.jpg')
    Image.fromarray(img).save(output_filename)

    # 更新字典中的计数
    class_counts[class_name] += 1

    # 如果所有类别都已经达到限制，则停止循环
    if all(count == limit_per_class for count in class_counts.values()):
        break

print("Images have been saved successfully.")