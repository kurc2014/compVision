import cv2
import time
from cvzone.HandTrackingModule import HandDetector

# Input and output video paths
input_path = "input.mp4"
output_path = "output.mp4"

# Initialize video capture from file
cap = cv2.VideoCapture(input_path)

# Get input video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize hand detector
detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.5)

# Pinch threshold in pixels
pinch_threshold = 50
current_finger = None
color = (255, 250, 250)

# Timer for controlling display update
last_display_time = time.time()
display_text = ""

while True:
    success, img = cap.read()
    if not success:
        print("Finished processing video.")
        break

    # No resizing; keep original resolution
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        lm = hands[0]["lmList"]
        if len(lm) >= 21:
            thumb = lm[4][:2]
            fingers = {
                name: lm[i][:2]
                for name, i in zip(["index", "middle", "ring", "little"], [8, 12, 16, 20])
            }

            # Check if a new pinch is happening
            for name, tip in fingers.items():
                length, *_ = detector.findDistance(thumb, tip, img)
                if length < pinch_threshold:
                    if name != current_finger:
                        current_finger = name
                        print(f"Switched to tracking: {current_finger}")
                    break

            # If there is an active finger being tracked
            if current_finger:
                pt2 = fingers[current_finger]
                length, *_ = detector.findDistance(thumb, pt2, img)
                ref_len, *_ = detector.findDistance(lm[5][:2], lm[0][:2], img)

                cv2.line(img, thumb, pt2, color, 3)
                cv2.circle(img, thumb, 8, color, cv2.FILLED)
                cv2.circle(img, pt2, 8, color, cv2.FILLED)

                current_time = time.time()
                if current_time - last_display_time >= 1:
                    ratio = length / ref_len
                    clamped_ratio = max(0.5, min(ratio, 1.5))
                    angle = 20 + (clamped_ratio - 0.5) * 80
                    display_text = f"{current_finger.capitalize()} angle: {angle:.2f} Degrees"
                    last_display_time = current_time

                if display_text:
                    cv2.putText(img, display_text, (pt2[0] - 50, pt2[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Write to output video
    out.write(img)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
