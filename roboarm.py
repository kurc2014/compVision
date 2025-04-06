import cv2
import time
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
detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.5)

# Threshold and state
pinch_threshold = 50
current_finger = None
color = (60, 60, 60)
last_display_time = time.time()
display_text = ""

# === Main Loop ===
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    img = cv2.resize(img, (800, 600))
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        lm = hands[0]["lmList"]

        if len(lm) >= 21:
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

                current_time = time.time()
                if current_time - last_display_time >= 1:
               # Normalize and send to servo
                    ratio = length / ref_len

                    # Clamp ratio to the range [0.5, 1.5]
                    clamped_ratio = max(0.5, min(ratio, 1.5))

                    # Convert to angle:
                    # Below 0.5 ratio → 20 degrees, above 1.5 → 100 degrees
                    angle = 20 + (clamped_ratio - 0.5) * 80  # Linearly map the ratio to 20-100 degrees

                    # Send the angle to the corresponding servo
                    servos[current_finger].write(int(angle))

                    # Update the display text with the angle value
                    display_text = f"{current_finger.capitalize()} angle: {int(angle)} Degrees"

                    # Show the most recent text
                    if display_text:
                        cv2.putText(img, display_text, (pt2[0] - 50, pt2[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
