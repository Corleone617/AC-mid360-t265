# import cv2
# import numpy as np

# def undistort_fisheye(img, K, D, dim, balance=0.0):
#     """ 矫正鱼眼图像的函数 """
#     dim1 = img.shape[:2][::-1]  # 图像尺寸，(width, height)
#     assert dim1[0]/dim1[1] == dim[0]/dim[1], "图像和目标尺寸必须有相同的宽高比"

#     # 计算新的内参矩阵
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, dim, cv2.CV_16SC2, balance=balance)
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     return undistorted_img

# def detect_circles(img):
#     """ 检测并绘制圆形的函数 """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    
#     return img

# # 摄像头参数
# K = np.array([[285.722, 0, 421.996], [0, 285.722, 403.696], [0, 0, 1]])  # 内参矩阵
# D = np.array([-0.003482, 0.001687, -0.003277, 0.000730])  # 畸变系数
# DIM = (848, 800)  # 图像尺寸

# # 捕获视频流
# cap = cv2.VideoCapture(1)  # 可能需要调整设备索引

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("无法获取图像")
#         break

#     # 矫正鱼眼
#     frame_undistorted = undistort_fisheye(frame, K, D, DIM)

#     # 检测圆形
#     frame_with_circles = detect_circles(frame_undistorted)

#     # 显示图像
#     cv2.imshow('Detected Circles', frame_with_circles)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import pyrealsense2 as rs 
import cv2
import numpy as np

# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.pose)

# 启动管道
pipeline.start(config)

try:
    while True:
        # 等待一对帧：深度和彩色
        frames = pipeline.wait_for_frames()
        pose_frame = frames.get_pose_frame()

        if pose_frame:
            data = pose_frame.get_pose_data()
            print("Frame #{}".format(pose_frame.frame_number))
            print("Position: {}".format(data.translation))
            print("Velocity: {}".format(data.velocity))
            print("Acceleration: {}".format(data.acceleration))

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()