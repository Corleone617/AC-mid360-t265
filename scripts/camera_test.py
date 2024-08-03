import cv2 as cv

if __name__ == "__main__":
    # cap = cv.VideoCapture()
    # cap.open(0, cv.CAP_DSHOW)       # 我这里0为电脑自带摄像头，1为外接相机
    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1900)      # 解决问题的关键！！！
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv.CAP_PROP_FPS, 30)

    while True:
        if not cap.isOpened():
            print('can not open camera')
            break
        ret, frame = cap.read()     # 读取图像
        if not ret:                 # 图像读取失败则直接进入下一次循环
            continue
        cv.namedWindow("cv_test")
        cv.imshow('cv_test', frame)
        my_key = cv.waitKey(1)
        # 按q退出循环，0xFF是为了排除一些功能键对q的ASCII码的影响
        if my_key & 0xFF == ord('q'):
            break

    #释放资源
    cap.release()
    cv.destroyAllWindows()