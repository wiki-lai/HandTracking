import cv2
import mediapipe as mp
import time

'''

使用mediapipe库自带的解决方案，实现手的动作识别，并描绘关节点及其连线

'''


#彩色摄像头下的识别准确率较高
cap = cv2.VideoCapture(0)
if cap.isOpened():
    cap.set(3,1280)
    cap.set(4,720)

    # 使用mediapipe库自带的解决方案，solutions.hands.Hands()返回一个解决问题的对象
    # 检测置信度大于0.5就显示关节点，跟踪置信度有0.5以上就更新位置
    handsProcessor = mp.solutions.hands.Hands(max_num_hands=2,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils

    pastTime = 0
    currentTime = 0

    while True:
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landMarks = handsProcessor.process(imgRGB)
            # print(results.multi_hand_landmarks)

            #如果检测到手（即 返回的landmark不为空）
            if landMarks.multi_hand_landmarks:
                # 遍历每只手的landmark
                for handLms in landMarks.multi_hand_landmarks:
                    #遍历 landmark的每个点
                    for id, lm in enumerate(handLms.landmark):

                        # landmark 的点坐标代表相对位置，而不是像素点
                        print(lm.x,lm.y)
                        # cx，cy 圆的中心点
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        print(id, cx, cy)

                        # 如果需要追踪特定的手指
                        # f id == 4:

                        # 给每个关节画个圆，能更明显地显示关节
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    #根据landmark画图
                    drawer.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)

            # 显示帧率
            currentTime = time.time()
            fps = 1 / (currentTime - pastTime)
            pastTime = currentTime

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            # 显示。按 Esc 键退出
            cv2.imshow("Image", img)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        else:
            print("读取画面失败")

    # 释放内存
    cap.release()
    cv2.destroyAllWindows()

else:
    print("摄像头初始化失败")