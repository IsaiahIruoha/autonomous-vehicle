import cv2
import numpy as np
from picarx import Picarx
from time import sleep

px = Picarx()

# Steering limits
STEERING_LEFT_LIMIT = -45
STEERING_RIGHT_LIMIT = 45

# Speed config
BASE_SPEED = 5
MAX_TURN_SPEED = 15

# PD Gains
KP = 0.10
KD = 0.05

# Steering Smoothing
ALPHA = 0.7  # smoothing factor for final steering angle

# Camera tilt angles
CAMERA_TILT_DEFAULT = -15
CAMERA_TILT_LEFT = -20
CAMERA_TILT_RIGHT = -10

# Memory for lines
last_left_avg = None   # (slope, intercept) or x-position
last_right_avg = None
lost_frames_count = 0
MAX_LOST_FRAMES = 5  # after these many frames of no lines, reduce speed drastically

# For PD control
last_error = 0.0
last_steering_angle = 0.0

has_started = False

px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

sleep(0.5)  # Let camera/wheels settle

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def adjust_camera_tilt(current_steering_angle):
    if current_steering_angle < -10:  # turning left
        px.set_cam_tilt_angle(CAMERA_TILT_LEFT)
    elif current_steering_angle > 10: # turning right
        px.set_cam_tilt_angle(CAMERA_TILT_RIGHT)
    else:
        px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)

def steering_control(error):
    global last_error
    derivative = error - last_error
    steering = KP * error + KD * derivative
    last_error = error
    return clamp(steering, STEERING_LEFT_LIMIT, STEERING_RIGHT_LIMIT)

def adaptive_roi(img, speed):
    height, width = img.shape[:2]
    min_top = 200
    max_top = 320
    dynamic_top = int(max_top - (speed / MAX_TURN_SPEED) * (max_top - min_top))
    dynamic_top = clamp(dynamic_top, 100, height-1)
    
    mask = np.zeros_like(img)
    roi_vertices = np.array([[
        (0, height),
        (0, dynamic_top),
        (width, dynamic_top),
        (width, height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(img, mask)

def get_line_params(x1, y1, x2, y2):
    """
    Return slope and intercept of the line in y=mx+b form.
    Handle vertical lines carefully by returning None if slope is infinite.
    """
    if (x2 - x1) == 0:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (slope, intercept)

def running_average_line(new_line, old_line, alpha=0.9):
    if old_line is None:
        return new_line
    slope_new = alpha * old_line[0] + (1-alpha) * new_line[0]
    intercept_new = alpha * old_line[1] + (1-alpha) * new_line[1]
    return (slope_new, intercept_new)

def estimate_line_x_at_y(line_params, y):
    """
    For line_params=(slope, intercept),
    return x at that y. i.e. x = (y - b)/m
    """
    slope, intercept = line_params
    if abs(slope) < 1e-8:
        return None
    x = (y - intercept) / slope
    return x

def process_frame(frame):
    global has_started
    global last_steering_angle
    global last_left_avg, last_right_avg
    global lost_frames_count

    # Step 1: Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Step 2: Color thresholding
    # White: high V, low S
    lower_white_1 = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_1 = np.array([255, 40, 255], dtype=np.uint8)
    mask_white_1 = cv2.inRange(hsv, lower_white_1, upper_white_1)

    # Additional "whitish-gray" mask in case lighting is different
    lower_white_2 = np.array([0, 0, 160], dtype=np.uint8)
    upper_white_2 = np.array([255, 60, 255], dtype=np.uint8)
    mask_white_2 = cv2.inRange(hsv, lower_white_2, upper_white_2)

    # Combine the two white masks
    mask_white = cv2.bitwise_or(mask_white_1, mask_white_2)

    # Yellow lines
    lower_yellow = np.array([18, 94, 140], dtype=np.uint8)
    upper_yellow = np.array([48, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine white + yellow
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)

    # Step 3: Morphological Ops
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Edge detection
    edges = cv2.Canny(combined_mask, 50, 150)

    # Step 5: ROI (dynamic version)
    current_speed = px.GetSpeed()  # Or track it manually if needed
    roi_edges = adaptive_roi(edges, current_speed)

    # Step 6: Hough transform
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 40,
                            minLineLength=30, maxLineGap=50)
    height, width = frame.shape[:2]
    frame_center_x = width // 2
    # We'll choose some reference y (close to bottom) for line x-position
    # e.g. y_ref = 400
    y_ref = int(height * 0.8)

    left_detected = False
    right_detected = False
    left_params_list = []
    right_params_list = []

    # Step 7: Parse lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            params = get_line_params(x1, y1, x2, y2)
            if params is None:
                continue

            slope, intercept = params

            # Filter near-horizontal lines
            if abs(slope) < 0.2:
                continue

            # Decide left or right by slope sign
            # Negative slope => left side, positive => right side
            if slope < 0:
                left_params_list.append(params)
            else:
                right_params_list.append(params)

    # Combine left lines if we have them
    if len(left_params_list) > 0:
        avg_slope = np.mean([p[0] for p in left_params_list])
        avg_int = np.mean([p[1] for p in left_params_list])
        line_avg = (avg_slope, avg_int)
        # Running average
        last_left_avg = running_average_line(line_avg, last_left_avg, alpha=0.9)
        left_detected = True

    # Combine right lines
    if len(right_params_list) > 0:
        avg_slope = np.mean([p[0] for p in right_params_list])
        avg_int = np.mean([p[1] for p in right_params_list])
        line_avg = (avg_slope, avg_int)
        # Running average
        last_right_avg = running_average_line(line_avg, last_right_avg, alpha=0.9)
        right_detected = True

    # If both lines are missing, increment lost_frames_count
    if not left_detected and not right_detected:
        lost_frames_count += 1
        if lost_frames_count > MAX_LOST_FRAMES:
            print("No lane lines for too long, stopping.")
            px.stop()
            return frame, mask_white, mask_yellow, combined_mask
    else:
        # If at least one line is found, reset lost_frames_count
        lost_frames_count = 0

    # Step 8: Estimate lane center
    # Use reference y= y_ref
    left_x = None
    right_x = None

    if last_left_avg is not None:
        left_x = estimate_line_x_at_y(last_left_avg, y_ref)

    if last_right_avg is not None:
        right_x = estimate_line_x_at_y(last_right_avg, y_ref)

    # If both lines are available, center = average
    if left_x is not None and right_x is not None:
        lane_center_x = (left_x + right_x) / 2.0
    elif left_x is not None:
        lane_center_x = left_x + 80  # default offset
    elif right_x is not None:
        lane_center_x = right_x - 80
    else:
        # fallback: no lines in memory => center is frame center
        lane_center_x = frame_center_x

    error = lane_center_x - frame_center_x

    # PD Steering
    raw_steering_angle = steering_control(error)

    # Steering smoothing
    steer_angle = ALPHA * last_steering_angle + (1 - ALPHA) * raw_steering_angle
    last_steering_angle = steer_angle

    px.set_dir_servo_angle(steer_angle)
    adjust_camera_tilt(steer_angle)

    # Speed control
    turn_factor = 0.2
    speed = clamp(BASE_SPEED - turn_factor * abs(steer_angle), 0, MAX_TURN_SPEED)

    # If we haven't seen lines in a while, reduce speed drastically or stop
    if lost_frames_count > 2:
        speed = min(speed, 1.0)  # slow way down

    px.forward(speed)

    # Let the car start smoothly the first time
    if not has_started:
        sleep(0.2)
        has_started = True

    # Debug logging
    print(f"Error: {error:.2f}  Steer: {steer_angle:.2f}  Speed: {speed:.2f}")

    # - If last_left_avg is not None, reconstruct a line in the ROI
    if last_left_avg is not None:
        slope, intercept = last_left_avg
        # pick two points for drawing
        y1_draw = height
        y2_draw = int(height/2)
        x1_draw = int((y1_draw - intercept) / slope)
        x2_draw = int((y2_draw - intercept) / slope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (255,0,0), 3)

    # - If last_right_avg is not None, do similarly
    if last_right_avg is not None:
        slope, intercept = last_right_avg
        y1_draw = height
        y2_draw = int(height/2)
        x1_draw = int((y1_draw - intercept) / slope)
        x2_draw = int((y2_draw - intercept) / slope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0,0,255), 3)

    # Draw lane center
    cv2.circle(frame, (int(lane_center_x), y_ref), 5, (0,255,0), -1)
    cv2.circle(frame, (frame_center_x, y_ref), 5, (0,255,255), -1)
    cv2.line(frame, (frame_center_x, y_ref), (int(lane_center_x), y_ref), (0,255,0), 2)

    return frame, mask_white, mask_yellow, combined_mask

# ============================
#          Main Loop
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
