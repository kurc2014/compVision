import cv2
from cvzone.HandTrackingModule import HandDetector
import pyfirmata

# === Arduino Setup ===
board = pyfirmata.Arduino('COM3')  # Replace with your port
servos = {
    "index": board.get_pin('d:2:s'),
    "middle": board.get_pin('d:3:s'),
    "ring": board.get_pin('d:4:s'),
    "little": board.get_pin('d:5:s'),
}

# Initial positions (90° is center)
for s in servos.values():
    s.write(90)

# === OpenCV Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.5)

# Threshold and state
pinch_threshold = 50
current_finger = None
color = (60, 60, 60)

# === Main Loop ===
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        lm = hands[0]["lmList"]
        thumb = lm[4][:2]
        fingers = {
            name: lm[i][:2]
            for name, i in zip(["index", "middle", "ring", "little"], [8, 12, 16, 20])
        }

        # Detect new pinch
        for name, tip in fingers.items():
            length, *_ = detector.findDistance(thumb, tip, img)
            if length < pinch_threshold:
                if name != current_finger:
                    current_finger = name
                    print(f"Switched to tracking: {current_finger}")
                break

        # If actively tracking a finger
        if current_finger:
            pt2 = fingers[current_finger]
            length, *_ = detector.findDistance(thumb, pt2, img)
            ref_len, *_ = detector.findDistance(lm[5][:2], lm[0][:2], img)

            # Draw connection
            cv2.line(img, thumb, pt2, color, 3)
            cv2.circle(img, thumb, 8, color, cv2.FILLED)
            cv2.circle(img, pt2, 8, color, cv2.FILLED)

            # Normalize and send to servo
            ratio = length / ref_len
            clamped_ratio = max(0.5, min(ratio, 1.5))
            angle = 20 + (clamped_ratio - 0.5) * 80  # 20°–100° range
            servos[current_finger].write(int(angle))

            # Display angle using ASCII degree symbol for better compatibility
            cv2.putText(
                img,
                f"{current_finger.capitalize()} angle: {int(angle)} deg",
                (pt2[0] - 50, pt2[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
