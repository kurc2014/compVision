import cv2
import time
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.5)

# Pinch threshold in pixels
pinch_threshold = 50

# Track the currently active pinched finger
current_finger = None

# Color used for drawing (same for all fingers)
#color = (30, 30, 30)
color = (255, 250, 250)


# Timer for controlling display update
last_display_time = time.time()
display_text = ""

while True:
    # Capture frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    # Resize for consistent display
    img = cv2.resize(img, (800, 600))

    # Detect hand(s)
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Get landmark list of the first hand
        lm = hands[0]["lmList"]

        if len(lm) >= 21:
            # Get thumb tip coordinates
            thumb = lm[4][:2]

            # Get finger tip coordinates
            fingers = {
                name: lm[i][:2]
                for name, i in zip(["index", "middle", "ring", "little"], [8, 12, 16, 20])
            }

            # Check if a new pinch is happening
            for name, tip in fingers.items():
                length, *_ = detector.findDistance(thumb, tip, img)
                if length < pinch_threshold:
                    # If pinch detected and it's a new finger, update
                    if name != current_finger:
                        current_finger = name
                        print(f"Switched to tracking: {current_finger}")
                    break

            # If there is an active finger being tracked
            if current_finger:
                pt2 = fingers[current_finger]
                length, *_ = detector.findDistance(thumb, pt2, img)
                ref_len, *_ = detector.findDistance(lm[5][:2], lm[0][:2], img)  # Reference hand size

                # Draw connection
                cv2.line(img, thumb, pt2, color, 3)
                cv2.circle(img, thumb, 8, color, cv2.FILLED)
                cv2.circle(img, pt2, 8, color, cv2.FILLED)

                # Update text once per second
                current_time = time.time()
                if current_time - last_display_time >= 1:
                    ratio = length / ref_len

                    # Clamp ratio to the range [0.5, 1.5]
                    clamped_ratio = max(0.5, min(ratio, 1.5))

                    # Convert to angle:
                    # Below 0.5 ratio → 20 degrees, above 1.5 → 100 degrees
                    angle = 20 + (clamped_ratio - 0.5) * 80 
                    display_text = f"{current_finger.capitalize()} angle: {angle:.2f} Degrees"
                    last_display_time = current_time

                # Show the most recent text
                if display_text:
                    cv2.putText(img, display_text, (pt2[0] - 50, pt2[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the image
    cv2.imshow("Image", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
