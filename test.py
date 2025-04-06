import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Set custom window size
window_name = "Camera Test"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow window resizing
cv2.resizeWindow(window_name, 1000,700)  # Set the window size

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from the camera.")
        break

    cv2.imshow(window_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
