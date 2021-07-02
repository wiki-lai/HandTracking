import cv2
import mediapipe as mp
import time

# 存在性能问题，待修复

class handDetector:

    def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

        self.mode=static_image_mode
        self.maxHands=max_num_hands
        self.minHands=min_detection_confidence
        self.trackingCon = min_tracking_confidence

        # 使用mediapipe库自带的解决方案，solutions.hands.Hands()返回一个解决问题的对象
        self.handsProcessor = mp.solutions.hands.Hands(self.mode, self.maxHands, self.minHands, self.trackingCon)
        self.drawer = mp.solutions.drawing_utils



    def findHands(self,img,draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        landMarks = self.handsProcessor.process(imgRGB)
        # print(results.multi_hand_landmarks)
         #如果检测到手（即 返回的landmark不为空）
        if landMarks.multi_hand_landmarks:
              # 遍历每只手的landmark
             for handLms in landMarks.multi_hand_landmarks:
               if draw:
                       #根据landmark画图
                      self.drawer.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        return img

"""
#遍历 landmark的每个点
for id, lm in enumerate(handLms.landmark):
    # landmark 的点坐标代表相对位置，而不是像素点
    print(lm.x, lm.y)
    # cx，cy 圆的中心点
    h, w, c = img.shape
    cx, cy = int(lm.x * w), int(lm.y * h)
    print(id, cx, cy)
    # if id == 4:

    # 给每个关节画个圆，能更明显地显示关节
    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
"""


def main():

    pTime = 0

    # 比较适合彩色摄像头
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    if cap.isOpened():
        while True:
            success,img  = cap.read()
            if success:
                detector = handDetector()
                img_result = detector.findHands(img)
                # 显示帧率
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                cv2.putText(img_result, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
                cv2.imshow("Image", img_result)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                print('读取画面失败')
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('摄像头初始化失败')


if __name__ == '__main__':
    main()

