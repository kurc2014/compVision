import cv2
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use camera index 0, change to 1 if needed
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Resize the display window
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1000, 750)  # Set the window size

detector = htm.handDetector()

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Warning: Failed to capture frame from the camera.")
        break
    
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])  # Example: Print the coordinates of landmark 4 (thumb tip)

    cTime = time.time()
    fps = 1 / max(cTime - pTime, 1e-5)  # Avoid division by zero
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3,
            (0, 0, 255), 3)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
