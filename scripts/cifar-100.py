import cv2
import onnxruntime as ort
from torchvision import transforms
import rospy
import numpy as np
from fcu_serial.msg import cifar,AC_staue
from collections import deque
import pyzbar.pyzbar as pyzbar

# 建立从类别索引到英文名称的映射
class_to_label = {
    0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver',
    5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle',
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
# 建立从英文名称到类别索引的映射
label_to_class = {v: k for k, v in class_to_label.items()}

# 边距大小,根据需要设置
margin = 0.2929
# 全局变量,存储上一帧的ROI坐标
last_roi = None
# 初始化一个变量来存储上一次的预测结果,用于检测跳动
last_prediction = None
# 用于计数相同预测结果的帧数
prediction_stability_count = 0
# 用于记录上一帧圆心的坐标
last_center = None
# 设定圆心位置和半径变化的阈值
MAX_RADIUS_DIFF = 20  # 允许的最大半径变化
MAX_CENTER_DIFF = 20  # 允许的最大圆心位置变化
# 保存上一次识别到的圆的信息
last_circle_center = None
last_circle_radius = None
# 初始化一个计数器用于跟踪自上次成功识别圆后的迭代次数
no_circles_detected_count = 0
# 设置一个阈值,用于判断何时重置last_circle_center和last_circle_radius
RESET_THRESHOLD = 5  # 根据实际情况设置适当的阈值
# 定义用于记录连续识别不同预测结果的计数器
continuous_different_count = 0
MAX_CONTINUOUS_DIFFERENT = 10  # 允许连续识别不同预测结果的最大次数

AVERAGE_WINDOW = 3  # 平均窗口的大小
center_history = deque(maxlen=AVERAGE_WINDOW)
radius_history = deque(maxlen=AVERAGE_WINDOW)

# 任务
task = 1

global waypoint_,upORdown_,LeftorRight,digital1,digital2 # 在主逻辑部分也声明这些全局变量
waypoint_ = 0
upORdown_ = 0
LeftorRight = 800.0
digital1 = 800.0
digital2 = 800.0
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# 初始化ONNX运行时会话
ort_session = ort.InferenceSession('/home/haique1/haique04/realsense/src/fcu_serial/scripts/model/seresnet152-170-best.onnx')

def average_circles(centers, radii):
    avg_center = np.mean(centers, axis=0)
    avg_radius = np.mean(radii)
    return (int(avg_center[0]), int(avg_center[1])), int(avg_radius)

# 计算偏移量
def calculate_displacement(center_x, center_y, frame_width, frame_height):
    displacement_x = center_x - frame_width // 2
    displacement_y = center_y - frame_height // 2
    return displacement_x, displacement_y

# 定义ONNX推理函数
def predict_with_onnx(ort_session, roi_np):
    ort_inputs = {ort_session.get_inputs()[0].name: roi_np}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

# 定义预处理变换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # 根据CIFAR-100的图像大小调整
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)  # 根据需要调整归一化参数
])

# 设置ROS的发布者和订阅者
def setup_ros_communications():
    rospy.init_node('cifar100_py', anonymous=True)
    pub = rospy.Publisher('camera_data', cifar, queue_size=10)
    sub = rospy.Subscriber("AC_data", AC_staue, AC_Callback, queue_size=10)
    return pub

def AC_Callback(AC_staue):
    global waypoint_,upORdown_  
    waypoint_ = AC_staue.waypoint
    upORdown_ = AC_staue.upORdown

def detect_horizontal_lines(Canny):

    # 使用霍夫变换检测线条
    lines = cv2.HoughLinesP(Canny, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # 筛选出横线并计算中心坐标
    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # 筛选斜率接近0的线条
                horizontal_lines.append((x1, y1, x2, y2))

        # 计算横线的中心坐标
        if len(horizontal_lines) > 1:
            centers = []
            for line in horizontal_lines:
                x1, y1, x2, y2 = line
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centers.append((center_x, center_y))
            
            # 计算两条横线的总体中心坐标
            total_center_x = sum(center[0] for center in centers) // len(centers)
            total_center_y = sum(center[1] for center in centers) // len(centers)
            return total_center_x, total_center_y
        else:
            print("没有检测到足够的横线")
    else:
        print("没有检测到线条")

    return None


# 初始化ROS通信
pub = setup_ros_communications()

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 摄像头分辨率
width, height = 640, 480

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)     
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)

try:
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        # 读取摄像头帧
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 垂直翻转
        publish_msg = cifar()

        if not ret:
            break

        # 识别二维码
        if task == 1:
            decoded_objects = pyzbar.decode(frame)
            for obj in decoded_objects:
                # 提取二维码的数据和位置
                data = obj.data.decode("utf-8")
                print(data)
                parts = data.split(',')  # 分割数据
                if len(parts) == 3:  # 确保数据包含三个部分
                    label1, label2, direction = parts

                    digital1 = label_to_class.get(label1, 0)  # 将第一个标签转换为数字
                    digital2 = label_to_class.get(label2, 0)  # 将第二个标签转换为数字
                    
                    # 处理降落方向
                    if direction == "left":
                        LeftorRight =  100.0
                    elif direction == "right":
                        LeftorRight =  200.0
                    else:
                        LeftorRight = 0  # 如果不是左或右，则设置为0
                    #打印到终端示意
                    # print(digital1,digital2,LeftorRight)

                    task = 2

                    # 构造要发送的消息
                    publish_msg.flag = 800.0
                    publish_msg.x_err = 800.0
                    publish_msg.y_err = 800.0
                    publish_msg.leftORright = LeftorRight
                    pub.publish(publish_msg)

                points = obj.polygon
                # 如果二维码多边形点不是四个，我们就用凸包
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    points = hull
                # 标记二维码在图像中的位置
                n = len(points)
                for j in range(n):
                    cv2.line(frame, tuple(points[j]), tuple(points[(j + 1) % n]), (255,0,0), 3)

        #识别圆形并预测图案
        if task == 2:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 应用中值滤波来去除椒盐噪声
            median_filtered_img = cv2.medianBlur(gray, 5)
            # 应用高斯模糊
            Gauss = cv2.GaussianBlur(median_filtered_img, (5, 5), 0)
            # 应用Canny边缘检测
            Canny = cv2.Canny(Gauss, 120, 200)
            # 应用霍夫圆变换检测圆形
            
            if waypoint_ == 11 and upORdown_ == 0:#来到降落点
                # #1.5m
                kernel = np.ones((3, 3), np.uint8)
                closing = cv2.morphologyEx(Canny, cv2.MORPH_CLOSE, kernel, iterations=1)
                dilate = cv2.dilate(closing, kernel, 1)
                dilate = cv2.blur(dilate,(4,4)) #均值滤波！！！ 滤除背景噪声
                circles = cv2.HoughCircles(dilate, cv2.HOUGH_GRADIENT, 1, minDist=40, param1=80, param2=40, minRadius=65, maxRadius=80)
                # detect_horizontal_lines(Canny)

            elif waypoint_ == 11 and upORdown_ == 1:
                #0.8m
                kernel = np.ones((3, 3), np.uint8)
                closing = cv2.morphologyEx(Canny, cv2.MORPH_CLOSE, kernel, iterations=1)
                dilate = cv2.dilate(closing, kernel, 1)
                dilate = cv2.blur(dilate,(4,4)) #均值滤波！！！ 滤除背景噪声
                circles = cv2.HoughCircles(dilate, cv2.HOUGH_GRADIENT, 1, minDist=40, param1=80, param2=40, minRadius=135, maxRadius=165)

            elif upORdown_ == 0:
                #1.6m: 
                kernel = np.ones((3, 3), np.uint8)

                closing = cv2.morphologyEx(Canny, cv2.MORPH_CLOSE, kernel, iterations=1)
                dilate = cv2.dilate(closing, kernel, 1)
                dilate = cv2.blur(dilate,(4,4)) #均值滤波！！！ 滤除背景噪声
                
                circles = cv2.HoughCircles(dilate, cv2.HOUGH_GRADIENT, 1, minDist=40, param1=80, param2=40, minRadius=40, maxRadius=70)
            
            elif upORdown_ == 1:
                #0.8m: 
                kernel = np.ones((3, 3), np.uint8)

                closing = cv2.morphologyEx(Canny, cv2.MORPH_CLOSE, kernel, iterations=1)
                dilate = cv2.dilate(closing, kernel, 2)
                dilate = cv2.blur(dilate,(2,2)) #均值滤波！！！ 滤除背景噪声
                # cv2.imshow('dilate',dilate)

                circles = cv2.HoughCircles(dilate, cv2.HOUGH_GRADIENT, 1, minDist=30, param1=80, param2=45, minRadius=80, maxRadius=120)
            # cv2.imshow()
            # 识别到圆形
            if circles is not None:
                circles = np.int32(np.around(circles))
                if circles.size > 0:
                    max_circle = max(circles[0, :], key=lambda x: x[2])
                    center = (max_circle[0], max_circle[1])
                    radius = max_circle[2]
                    if last_circle_center is not None and last_circle_radius is not None:
                        radius_diff = np.abs(radius - last_circle_radius)
                        center_diff = np.linalg.norm(np.array(center) - np.array(last_circle_center))
                        if radius_diff > MAX_RADIUS_DIFF or center_diff > MAX_CENTER_DIFF:
                            no_circles_detected_count += 1
                            if no_circles_detected_count >= RESET_THRESHOLD:
                                last_circle_center = None
                                last_circle_radius = None   
                                center_history.clear()
                                radius_history.clear()
                                no_circles_detected_count = 0
                            continue  # 忽略这个圆形
                    
                    center_history.append(center)
                    radius_history.append(radius)

                    if len(center_history) == AVERAGE_WINDOW:
                        smooth_center, smooth_radius = average_circles(np.array(center_history), np.array(radius_history))
                        # print("center:",smooth_center)
                        # print(smooth_radius)
                        # 更新最后检测到的圆的信息
                        last_circle_center = center
                        last_circle_radius = radius
                        no_circles_detected_count = 0

                        # 使用平滑后的中心和半径计算ROI区域
                        x1 = max(0, smooth_center[0] - smooth_radius)
                        y1 = max(0, smooth_center[1] - smooth_radius)
                        x2 = min(width, smooth_center[0] + smooth_radius)
                        y2 = min(height, smooth_center[1] + smooth_radius)
                        # 调整坐标以裁剪边距
                        x1_crop = x1 + int(margin * smooth_radius)
                        y1_crop = y1 + int(margin * smooth_radius)
                        x2_crop = x2 - int(margin * smooth_radius)
                        y2_crop = y2 - int(margin * smooth_radius)

                        roi_with_margin = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                        cv2.imshow("roi",roi_with_margin)
              
                        if roi_with_margin.size > 0:

                            # 这里计算x_err和y_err,然后进行平滑处理
                            x_err, y_err = calculate_displacement(smooth_center[0], smooth_center[1], frame.shape[1], frame.shape[0])
                            
                            #仅在1.6m时识别
                            if upORdown_ == 0:

                                # 将图像转换为模型的输入格式
                                roi_pil = transform(roi_with_margin).unsqueeze(0)
                                roi_np = np.array(roi_pil, dtype=np.float32)

                                # 使用ONNX模型进行预测
                                onnx_prediction = predict_with_onnx(ort_session, roi_np)

                                # 获取预测的结果
                                predicted_class = np.argmax(onnx_prediction[0], axis=1)[0]

                                # 使用预测的类别索引来获取英文名称
                                predicted_class_label = class_to_label[predicted_class]

                                # 在图像上显示预测的结果
                                cv2.putText(frame, f'Predicted Class: {predicted_class_label}', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            if digital1 == predicted_class or digital2 == predicted_class:
                                publish_msg.flag = 111.0  #AC实施投放
                                publish_msg.x_err = x_err
                                publish_msg.y_err = y_err
                                publish_msg.leftORright = LeftorRight
                                pub.publish(publish_msg)
                            else:
                                publish_msg.flag = 800.0  #AC不实施投放
                                publish_msg.x_err = x_err
                                publish_msg.y_err = y_err
                                publish_msg.leftORright = LeftorRight
                                pub.publish(publish_msg)

                            # 在摄像头帧上绘制最小的圆形和中心点
                            # cv2.circle(frame, smooth_center, 1, (0, 100, 100), 2)
                            cv2.circle(frame, smooth_center, smooth_radius, (255, 0, 255), 2)
            # 没有识别到圆
            else:
                publish_msg.flag = 800.0
                publish_msg.x_err = 800.0
                publish_msg.y_err = 800.0
                publish_msg.leftORright = LeftorRight
                pub.publish(publish_msg)

        cv2.imshow('Frame', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rate.sleep()

except rospy.ROSInterruptException:
    pass

finally:
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()