import cv2
import numpy as np
from picarx import Picarx
from time import sleep

# Initialize PiCar-X
px = Picarx()

# Define constants
STEERING_NEUTRAL = 0
STEERING_LEFT = -45  
STEERING_RIGHT = 30
FORWARD_SPEED = 20 
MAX_TURN_SPEED = 40 


px.set_cam_tilt_angle(-15)  

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Move forward at start
#px.forward(FORWARD_SPEED)
sleep(1)  # Allow motors to start

def adjust_camera_steering(steering_angle):
    """Adjust the camera's tilt based on the steering angle."""
    if steering_angle == STEERING_LEFT:
        # Tilt the camera slightly to the left when turning left
        px.set_cam_tilt_angle(-20)
    elif steering_angle == STEERING_RIGHT:
        # Tilt the camera slightly to the right when turning right
        px.set_cam_tilt_angle(-10) 
    else:
        # Keep the camera neutral when going straight
        px.set_cam_tilt_angle(-15) 

def process_frame(frame):
    """Process frame to detect lane lines and adjust steering."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use binary thresholding to highlight dark (black) tape
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # Invert to highlight black tape

    # Optionally apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    
    # Define region of interest (ROI)
    mask = np.zeros_like(blur)
    roi_vertices = np.array([[ 
        (0, 480),
        (220, 240),
        (420, 240),
        (640, 480)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi_vertices, 255)
    masked_blur = cv2.bitwise_and(blur, mask)

    # Detect lane lines
    lines = cv2.HoughLinesP(masked_blur, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)

    left_lines, right_lines = [], []
    center_x = 320 

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
            if slope < -0.2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.2:
                right_lines.append((x1, y1, x2, y2))

    # Stop the car if no lanes are detected
    if not left_lines and not right_lines:
        print("No lanes detected! Stopping car.")
        px.stop()
        return frame

    # Determine lane center
    if left_lines and right_lines:
        left_x = np.mean([x1 for x1, _, x2, _ in left_lines])
        right_x = np.mean([x1 for x1, _, x2, _ in right_lines])
        center_x = (left_x + right_x) // 2

    # Calculate the error between the center of the image and the lane center
    error = center_x - 320

    # Steering logic with more refined adjustments
    if error < -20:  # Lane shifted too far left, turn right
        steering_angle = min(STEERING_RIGHT, abs(error) / 10)
        px.set_dir_servo_angle(steering_angle)
        px.forward(min(FORWARD_SPEED + abs(steering_angle) * 2, MAX_TURN_SPEED))  # Slow down when turning
        adjust_camera_steering(STEERING_RIGHT)
    elif error > 20:  # Lane shifted too far right, turn left
        steering_angle = max(STEERING_LEFT, -abs(error) / 10)
        px.set_dir_servo_angle(steering_angle)
        px.forward(min(FORWARD_SPEED + abs(steering_angle) * 2, MAX_TURN_SPEED))  # Slow down when turning
        adjust_camera_steering(STEERING_LEFT)
    else:  # Lane is centered, go straight
        px.set_dir_servo_angle(STEERING_NEUTRAL)
        px.forward(FORWARD_SPEED)
        adjust_camera_steering(STEERING_NEUTRAL)

    # Draw detected lanes
    for x1, y1, x2, y2 in left_lines + right_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Lane Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    px.stop()
    cap.release()
    cv2.destroyAllWindows()
