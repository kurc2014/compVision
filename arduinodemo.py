import cv2
import arduinodemo as cnt
from cvzone.HandTrackingModule import HandDetector

detector=HandDetector(detectionCon=0.8,maxHands=1)

video=cv2.VideoCapture(0)
finger_count_text = ''

while True:
    ret,frame=video.read()
    frame = cv2.resize(frame, (1000, 750))  # Resize

  
    hands,img=detector.findHands(frame)
    if hands:
        lmList=hands[0]
        fingerUp=detector.fingersUp(lmList)
        print(fingerUp)
            
        if fingerUp == [0, 0, 0, 0, 0]:
            finger_count_text = 'Finger count: 0'
        elif fingerUp == [0, 1, 0, 0, 0]:
            finger_count_text = 'Finger count: 1'
        elif fingerUp == [0, 1, 1, 0, 0]:
            finger_count_text = 'Finger count: 2'
        elif fingerUp == [0, 1, 1, 1, 0]:
            finger_count_text = 'Finger count: 3'
        elif fingerUp == [0, 1, 1, 1, 1]:
            finger_count_text = 'Finger count: 4'
        elif fingerUp == [1, 1, 1, 1, 1]:
            finger_count_text = 'Finger count: 5'

    # Display the text on the frame
    cv2.putText(frame, finger_count_text, (30, 720), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    if k==ord("q"):
        break

video.release()
cv2.destroyAllWindows()
