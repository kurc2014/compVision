from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PoseModule import PoseDetector
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam, or adjust for external cameras

# Initialize detectors
hand_detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
face_detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)
pose_detector = PoseDetector(staticMode=False, modelComplexity=1, smoothLandmarks=True, 
                              enableSegmentation=False, smoothSegmentation=True, 
                              detectionCon=0.5, trackCon=0.5)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    # Resize for consistent processing
    img = cv2.resize(img, (800, 600))

    # Detect hand landmarks
    hands, img = hand_detector.findHands(img, draw=True, flipType=True)

    # Detect face mesh
    img, faces = face_detector.findFaceMesh(img, draw=True)

    # Detect human pose
    img = pose_detector.findPose(img)
    lm_list, bbox_info = pose_detector.findPosition(img, draw=True, bboxWithHands=False)

    # Process face mesh detection
    if faces:
        for face in faces:
            left_eye_up = face[159]
            left_eye_down = face[23]

            # Calculate distance between eye landmarks
            eye_distance, _, _ = hand_detector.findDistance(left_eye_up, left_eye_down)
            print(f"Left Eye Vertical Distance: {eye_distance}")

    # Process hand detection
    if hands:
        hand1 = hands[0]
        lm_list1 = hand1["lmList"]
        fingers1 = hand_detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")

        # Distance between landmarks on the first hand
        length, _, img = hand_detector.findDistance(lm_list1[8][0:2], lm_list1[12][0:2], img, color=(255, 0, 255), scale=10)

        if len(hands) == 2:
            hand2 = hands[1]
            lm_list2 = hand2["lmList"]
            fingers2 = hand_detector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

            # Distance between index fingers of both hands
            length, _, img = hand_detector.findDistance(lm_list1[8][0:2], lm_list2[8][0:2], img, color=(255, 0, 0), scale=10)

        print()

    # Process pose detection
    if lm_list:
        # Get the center of the bounding box
        center = bbox_info["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Calculate distance between shoulder and wrist (landmarks 11 and 15)
        length, img, _ = pose_detector.findDistance(lm_list[11][0:2], lm_list[15][0:2], img=img, color=(255, 0, 0), scale=10)

        # Calculate angle between shoulder, elbow, and wrist (landmarks 11, 13, and 15)
        angle, img = pose_detector.findAngle(lm_list[11][0:2], lm_list[13][0:2], lm_list[15][0:2], img=img, color=(0, 0, 255), scale=10)

        # Check if the angle is approximately 50 degrees
        is_close_to_50 = pose_detector.angleCheck(myAngle=angle, targetAngle=50, offset=10)
        print(f"Angle ~50: {is_close_to_50}")

    # Display the final output
    cv2.imshow("Image", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
