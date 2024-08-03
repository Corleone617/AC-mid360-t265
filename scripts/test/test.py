import cv2
import numpy as np
import onnxruntime as ort
import json
import rospy
from std_msgs.msg import String
from torchvision import transforms

# 初始化ONNX运行时会话
ort_session = ort.InferenceSession('/home/corleone/ROS1/realsense/src/fcu_serial/scripts/model/cifar100_resnet50.onnx')

# 修改此处的预处理流程，以匹配训练模型时的预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # 调整为与训练时一致的分辨率
    transforms.ToTensor(),
    # 此处应当与训练模型时使用的归一化参数一致
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_with_onnx(ort_session, image):
    # 应用预处理
    image = transform(image).unsqueeze(0).numpy()

    # 运行ONNX模型
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)

    # 返回预测结果
    return ort_outs

# 定义ONNX推理函数
def predict_with_onnx(ort_session, image):
    # 将图像转换为模型所需的格式和大小
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 应用预处理
    image = transform(image).unsqueeze(0).numpy()
    
    # 运行ONNX模型
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # 返回预测结果
    return ort_outs

# 假设class_to_label是从类别索引到英文名称的映射
class_to_label = {
    0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver',
    5: 'bed', 6: 'bee', 7: 'beetle',    8: 'bicycle', 9: 'bottle',
    10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly',
    15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle',
    20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach',
    25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dinosaur',
    30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox',
    35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'computer_keyboard',
    40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard',
    45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain',
    50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid',
    55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree',
    60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum',
    65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket',
    70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew',
    75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider',
    80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table',
    85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor',
    90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe',
    95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'
}

# 初始化摄像头
cap = cv2.VideoCapture(0)

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 应用模型预测
    preds = predict_with_onnx(ort_session, frame)
    # 取得最大概率的预测结果
    pred_class = np.argmax(preds[0])
    
    # 获取类别名称
    class_label = class_to_label.get(pred_class, "Unknown")
    
    # 显示类别标签和图像
    cv2.putText(frame, class_label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('CIFAR-100 Recognition', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
