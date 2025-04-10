import cv2
import time
from cvzone.HandTrackingModule import HandDetector
import pyfirmata2 as pyfirmata

# === Arduino Setup ===
board = pyfirmata.Arduino('COM11')  # Replace with your port
servos = {
    "index": board.get_pin('d:10:s'),
    "middle": board.get_pin('d:11:s'),
    "ring": board.get_pin('d:13:s'),
    "little": board.get_pin('d:12:s'),
}

# Initialize servo angles
servo_angles = {name: 90 for name in servos}
for s in servos.values():
    s.write(90)

# === OpenCV & Hand Detector ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.5)

pinch_threshold = 50
current_finger = None
color = (30, 30, 30)

# === Display Timer ===
last_display_time = time.time()
display_text = ""

# === Main Loop ===
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        lm = hands[0]["lmList"]

        if len(lm) >= 21:
            thumb = lm[4][:2]
            fingers = {
                name: lm[i][:2]
                for name, i in zip(["index", "middle", "ring", "little"], [8, 12, 16, 20])
            }

            for name, tip in fingers.items():
                length, *_ = detector.findDistance(thumb, tip, img)
                if length < pinch_threshold:
                    if name != current_finger:
                        current_finger = name
                        print(f"Switched to tracking: {current_finger}")
                    break

            if current_finger:
                pt2 = fingers[current_finger]
                length, *_ = detector.findDistance(thumb, pt2, img)
                ref_len, *_ = detector.findDistance(lm[5][:2], lm[0][:2], img)

                # Draw tracking visuals
                cv2.line(img, thumb, pt2, color, 3)
                cv2.circle(img, thumb, 8, color, cv2.FILLED)
                cv2.circle(img, pt2, 8, color, cv2.FILLED)

                # Calculate target angle
                ratio = length / ref_len
                clamped_ratio = max(0.5, min(ratio, 1.5))
                target_angle = int(20 + (clamped_ratio - 0.5) * 80)

                # Smoothly update servo angle
                current_angle = servo_angles[current_finger]
                if abs(current_angle - target_angle) > 0:
                    step = 1 if target_angle > current_angle else -1
                    current_angle += step
                    servo_angles[current_finger] = current_angle
                    servos[current_finger].write(current_angle)


                # Update display every 0.3s
                current_time = time.time()
                if current_time - last_display_time >= 0.3:
                    display_text = f"{current_finger.capitalize()} angle: {target_angle} deg"
                    last_display_time = current_time

                if display_text:
                    cv2.putText(img, display_text, (pt2[0] - 50, pt2[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
