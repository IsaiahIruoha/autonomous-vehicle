import cv2
import numpy as np
from picarx import Picarx
from time import sleep

px = Picarx()

STEERING_LEFT_LIMIT = -45
STEERING_RIGHT_LIMIT = 45
STEERING_NEUTRAL = 0

BASE_SPEED = 10     # Baseline forward speed
MAX_TURN_SPEED = 25 # Speed limit while turning

CAMERA_TILT_DEFAULT = -15
CAMERA_TILT_LEFT = -20
CAMERA_TILT_RIGHT = -10

has_started = False  # To track if the car just started moving

# Will tune these later
KP = 0.15  # Proportional gain
KD = 0.10  # Derivative gain

# For PD Control
last_error = 0.0

px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

sleep(0.5)  # Let camera/wheels settle

# Steering Help
def clamp(val, min_val, max_val):
    """Clamp val between [min_val, max_val]."""
    return max(min_val, min(val, max_val))

def adjust_camera_tilt(current_steering_angle):
    """Tilt the camera slightly left/right depending on steering."""
    if current_steering_angle < -10:  # turning left
        px.set_cam_tilt_angle(CAMERA_TILT_LEFT)
    elif current_steering_angle > 10: # turning right
        px.set_cam_tilt_angle(CAMERA_TILT_RIGHT)
    else:
        px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
        
     
def steering_control(error):
    """
    PD control for smoother steering:
    steering = KP * error + KD * (error - last_error)
    """
    global last_error

    derivative = error - last_error
    steering = KP * error + KD * derivative
    
    # Update last_error
    last_error = error

    # Clamp to left/right limits
    steering = clamp(steering, STEERING_LEFT_LIMIT, STEERING_RIGHT_LIMIT)
    return steering

def region_of_interest(img):
    """
    Define a polygon to mask out everything but
    the region where we expect lane lines.
    """
    mask = np.zeros_like(img)
    
    # Example: trapezoid from the bottom up ~halfway as a test
    roi_vertices = np.array([[
        (0, 480),
        (0, 300),
        (640, 300),
        (640, 480)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(img, mask)

def process_frame(frame):
    """
    1) Convert to HSV
    2) Threshold for white and yellow lines
    3) Morphological ops
    4) Canny edge
    5) ROI
    6) Hough transform
    7) Classify lines, compute center
    8) PD steering
    """
    
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # -------------------------------------------------
    # Color Thresholding
    # White lines (high V, low S)
    # -------------------------------------------------
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 40, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # -------------------------------------------------
    # Yellow lines
    # Adjust these depending on your lighting conditions
    # -------------------------------------------------
    lower_yellow = np.array([18, 94, 140], dtype=np.uint8)
    upper_yellow = np.array([48, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # -------------------------------------------------
    # Morphological Ops (tweak kernel as needed)
    # -------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # -------------------------------------------------
    # Edge Detection
    # -------------------------------------------------
    edges = cv2.Canny(combined_mask, 50, 150)

    # ROI
    roi_edges = region_of_interest(edges)
    
    # -------------------------------------------------
    # Hough Lines
    # -------------------------------------------------
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=50)

    if lines is None:
        # No lines --> possibly stop or keep going
        print("No lane lines found; stopping for safety.")
        px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)  # face forward
        px.stop()
        return frame
    
    # -------------------------------------------------
    # Separate lines into left vs right
    # -------------------------------------------------
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # avoid /0
        # Filter near-horizontal
        if abs(slope) < 0.2:
            continue
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))
    
    if not left_lines and not right_lines:
        print("No valid left/right lines; stopping.")
        px.stop()
        px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT) 
        return frame

    # Approximate lane center
    frame_center_x = frame.shape[1] // 2  # ~320
    lane_center_x = frame_center_x  # default
    
    if left_lines and right_lines:
        left_x_avg = np.mean([ (x1 + x2) / 2.0 for (x1,y1,x2,y2) in left_lines ])
        right_x_avg = np.mean([ (x1 + x2) / 2.0 for (x1,y1,x2,y2) in right_lines ])
        lane_center_x = (left_x_avg + right_x_avg) / 2.0
    elif left_lines:
        left_x_avg = np.mean([ (x1 + x2) / 2.0 for (x1,y1,x2,y2) in left_lines ])
        lane_center_x = left_x_avg + 80  # shift from left line
    elif right_lines:
        right_x_avg = np.mean([ (x1 + x2) / 2.0 for (x1,y1,x2,y2) in right_lines ])
        lane_center_x = right_x_avg - 80  # shift from right line
        
     # -------------------------------------------------
    # Steering via PD
    # -------------------------------------------------
    # Negative error means lane is left of center (need to turn left).
    # Positive error means lane is right of center (need to turn right).
    error = lane_center_x - frame_center_x
    steer_angle = steering_control(error)

    px.set_dir_servo_angle(steer_angle)
    adjust_camera_tilt(steer_angle)

    # Slow down more if turning
    if abs(steer_angle) > 10:
        speed = clamp(BASE_SPEED + 0.5 * abs(steer_angle), BASE_SPEED, MAX_TURN_SPEED)
    else:
        speed = BASE_SPEED
        
    # Let the car start smoothly the first time
    global has_started
    px.forward(speed)

    if not has_started:
        sleep(0.2)  # Small delay to let camera adjust before motion
        has_started = True
    
    # Left lines in blue
    for (x1,y1,x2,y2) in left_lines:
        cv2.line(frame, (x1,y1), (x2,y2), (255, 0, 0), 3)
    # Right lines in red
    for (x1,y1,x2,y2) in right_lines:
        cv2.line(frame, (x1,y1), (x2,y2), (0, 0, 255), 3)
        
    # Draw lane center vs frame center
    cv2.circle(frame, (int(lane_center_x), 400), 5, (0,255,0), -1)
    cv2.circle(frame, (int(frame_center_x), 400), 5, (0,255,255), -1)
    cv2.line(frame, (int(frame_center_x), 400), (int(lane_center_x), 400), (0,255,0), 2)

    return frame, mask_white, mask_yellow, combined_mask 

# ============================
#    Main Loop
# ============================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera feed detected.")
            break
        
        processed_frame, mask_white, mask_yellow, combined_mask = process_frame(frame)
        
        cv2.imshow("Lane Detection", processed_frame)
        
        cv2.imshow("White Mask", mask_white)
        cv2.imshow("Yellow Mask", mask_yellow)
        cv2.imshow("Combined Mask", combined_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt: stopping...")

finally:
    px.stop()
    px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
    cap.release()
    cv2.destroyAllWindows()