import cv2 as cv
import numpy as np
import mediapipe 
import autopy 

cap = cv.VideoCapture(0)
initHand = mediapipe.solutions.hands
mainHand = initHand.Hands()

draw = mediapipe.solutions.drawing_utils
wScr, hScr = autopy.screen.size()

pX , pY = 0, 0
cX , cY = 0, 0
def handlandmarks(colorImg):
    landmarkList = []
    landmarkPosition = mainHand.process(colorImg)
    landmarkCheck = landmarkPosition.multi_hand_landmarks

    if  landmarkCheck: 
       for hand in landmarkCheck:
              for index, landmark in enumerate(hand.landmark):
                  draw.draw_landmarks(colorImg, hand, initHand.HAND_CONNECTIONS)
                  h, w, c = colorImg.shape
                  cx, cy = int(landmark.x * w), int(landmark.y * h)
                  landmarkList.append([index, cx, cy])
    return landmarkList

def fingers(landmarks):
    fingersTips = []
    tipIds = [4, 8, 12, 16, 20]
    if landmarks[tipIds[0][1]] < landmarks[tipIds[0] -1[2]]:
        fingersTips.append(1)
    else:
        fingersTips.append(0)


    for id in range(1, 5):
           if landmarks[tipIds[id][2]] < landmarks[tipIds[id] -3[2]]:
                fingersTips.append(1)
           else:
                fingersTips.append(0)  
    return fingersTips 
    
while True:
   chek, colorImg = cap.read()
   imgRGB = cv.cvtColor(colorImg, cv.COLOR_BGR2RGB)

   lmList = handlandmarks(imgRGB)
   print(lmList)

   if len(lmList) != 0:

      x1, y1 = lmList[8][1:]
      x2, y2 = lmList[12][1:]
      finger = fingers(lmList)
         
      if finger[1] == 1 and finger[2] == 0:
            x3 = np.interp(x1, (0, 640), (0, wScr))
            y3 = np.interp(y1, (0, 480), (0, hScr))
            cX = pX + (x3 - pX) / 7
            cY = pY + (y3 - pY) / 7
            autopy.mouse.move(wScr - cX, cY)
            pX, pY = cX, cY

      if finger[1] == 1 and finger[2] == 1:
            autopy.mouse.click()

      cv.imshow("Image", colorImg)
      if cv.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv.destroyAllWindows()
 

     
   




