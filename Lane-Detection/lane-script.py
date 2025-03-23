import cv2
import numpy as np
from picarx import Picarx
from time import sleep

px = Picarx()

# Steering limits
STEERING_LEFT_LIMIT = -45
STEERING_RIGHT_LIMIT = 45

# Speed settings
BASE_SPEED = 5        # Lower base speed for safety
MAX_TURN_SPEED = 15   # Max speed on straight or gentle curves

# PD Gains (tune these!)
KP = 0.10
KD = 0.05

# Steering angle smoothing (0 < ALPHA < 1)
# Higher ALPHA => more smoothing
ALPHA = 0.7

# Camera tilt angles
CAMERA_TILT_DEFAULT = -15
CAMERA_TILT_LEFT = -20
CAMERA_TILT_RIGHT = -10

# If lines are lost for too many frames, slow/stop
MAX_LOST_FRAMES = 5

# Lane offset if only one line is detected
LANE_HALF_WIDTH_PX = 80

last_error = 0.0
last_steering_angle = 0.0
has_started = False

# Running average lines for left/right
# store as (slope, intercept) in y=mx+b form
last_left_avg = None
last_right_avg = None

# Count how many consecutive frames we haven't seen lines
lost_frames_count = 0

# Initialize
px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

sleep(0.5)  # Let camera/wheels settle

def clamp(val, min_val, max_val):
    """Clamp 'val' between 'min_val' and 'max_val'."""
    return max(min_val, min(val, max_val))

def adjust_camera_tilt(steering_angle):
    """Point camera left/right depending on steering angle."""
    if steering_angle < -10:
        px.set_cam_tilt_angle(CAMERA_TILT_LEFT)
    elif steering_angle > 10:
        px.set_cam_tilt_angle(CAMERA_TILT_RIGHT)
    else:
        px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)

def steering_control(error):
    """
    PD control for steering:
    steering = KP * error + KD * (error - last_error)
    """
    global last_error
    derivative = error - last_error
    raw_steering = KP * error + KD * derivative
    last_error = error
    # Clamp to mechanical limits
    return clamp(raw_steering, STEERING_LEFT_LIMIT, STEERING_RIGHT_LIMIT)

def region_of_interest(img):
    """
    Define a static polygon mask that keeps only the
    bottom portion of the image, where we expect to see lane lines.
    """
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    # Example: trapezoid from bottom up to ~60% height
    roi_vertices = np.array([[
        (0, height),
        (0, int(0.6 * height)),
        (width, int(0.6 * height)),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(img, mask)

def get_line_params(x1, y1, x2, y2):
    """
    Return (slope, intercept) for the line in y=mx+b form,
    or None if line is vertical.
    """
    if (x2 - x1) == 0:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (slope, intercept)

def running_average_line(new_line, old_line, alpha=0.9):
    """
    Exponential smoothing of line parameters:
    old_line, new_line = (slope, intercept)
    Returns updated line = alpha*old_line + (1-alpha)*new_line
    """
    if old_line is None:
        return new_line
    old_slope, old_int = old_line
    new_slope, new_int = new_line
    smoothed_slope = alpha * old_slope + (1 - alpha) * new_slope
    smoothed_int   = alpha * old_int   + (1 - alpha) * new_int
    return (smoothed_slope, smoothed_int)

def estimate_line_x_at_y(line_params, y):
    """
    For line_params = (slope, intercept),
    returns x where line crosses 'y'.
    i.e., x = (y - b)/m
    """
    slope, intercept = line_params
    if abs(slope) < 1e-9:
        return None
    x = (y - intercept) / slope
    return x

def process_frame(frame):
    global has_started
    global last_steering_angle
    global last_left_avg, last_right_avg
    global lost_frames_count

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White lines (two ranges to handle different brightness)
    lower_white_1 = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_1 = np.array([255, 40, 255], dtype=np.uint8)
    mask_white_1 = cv2.inRange(hsv, lower_white_1, upper_white_1)

    lower_white_2 = np.array([0, 0, 160], dtype=np.uint8)
    upper_white_2 = np.array([255, 60, 255], dtype=np.uint8)
    mask_white_2 = cv2.inRange(hsv, lower_white_2, upper_white_2)

    mask_white = cv2.bitwise_or(mask_white_1, mask_white_2)

    # Yellow lines
    lower_yellow = np.array([18, 94, 140], dtype=np.uint8)
    upper_yellow = np.array([48, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine white + yellow
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(combined_mask, 50, 150)

    roi_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=50
    )

    # reference the bottom portion for lane center
    height, width = frame.shape[:2]
    frame_center_x = width // 2
    y_ref = int(height * 0.8)  # near bottom

    left_params_list = []
    right_params_list = []
    left_detected = False
    right_detected = False

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            params = get_line_params(x1, y1, x2, y2)
            if params is None:
                continue
            slope, intercept = params
            # Filter near-horizontal
            if abs(slope) < 0.2:
                continue
            # Negative slope => left side, positive => right side
            if slope < 0:
                left_params_list.append((slope, intercept))
            else:
                right_params_list.append((slope, intercept))

    # If we found left lines, average them => exponential smoothing
    if len(left_params_list) > 0:
        avg_slope = np.mean([p[0] for p in left_params_list])
        avg_int   = np.mean([p[1] for p in left_params_list])
        new_left_line = (avg_slope, avg_int)
        last_left_avg = running_average_line(new_left_line, last_left_avg, alpha=0.9)
        left_detected = True

    # If we found right lines, average them => exponential smoothing
    if len(right_params_list) > 0:
        avg_slope = np.mean([p[0] for p in right_params_list])
        avg_int   = np.mean([p[1] for p in right_params_list])
        new_right_line = (avg_slope, avg_int)
        last_right_avg = running_average_line(new_right_line, last_right_avg, alpha=0.9)
        right_detected = True

    # Handle line-lost logic
    if not left_detected and not right_detected:
        # Neither line found
        lost_frames_count += 1
        print("No lane lines found this frame.")
        if lost_frames_count > MAX_LOST_FRAMES:
            # If we lose lines for too many frames => stop for safety
            print("No lane lines for too long, stopping.")
            px.stop()
            return frame, mask_white, mask_yellow, combined_mask
    else:
        # At least one line was found => reset lost_frames_count
        lost_frames_count = 0

    left_x = None
    right_x = None

    if last_left_avg is not None:
        left_x = estimate_line_x_at_y(last_left_avg, y_ref)
    if last_right_avg is not None:
        right_x = estimate_line_x_at_y(last_right_avg, y_ref)

    # If both lines are present, center is average
    if left_x is not None and right_x is not None:
        lane_center_x = (left_x + right_x) / 2.0
    elif left_x is not None:
        # Only left line
        lane_center_x = left_x + LANE_HALF_WIDTH_PX
    elif right_x is not None:
        # Only right line
        lane_center_x = right_x - LANE_HALF_WIDTH_PX
    else:
        # No line memory at all => default to frame center
        lane_center_x = frame_center_x

    error = lane_center_x - frame_center_x

    raw_steer_angle = steering_control(error)
    steer_angle = ALPHA * last_steering_angle + (1 - ALPHA) * raw_steer_angle
    last_steering_angle = steer_angle

    px.set_dir_servo_angle(steer_angle)
    adjust_camera_tilt(steer_angle)

    # The sharper the turn, the slower we go
    turn_factor = 0.2
    speed = clamp(BASE_SPEED - turn_factor * abs(steer_angle), 0, MAX_TURN_SPEED)

    # If lines have been missing recently, slow further
    if lost_frames_count > 2:
        speed = min(speed, 1.0)  # slow down a lot

    px.forward(speed)

    if not has_started:
        sleep(0.2)
        has_started = True

    # Debug info
    print(f"[DEBUG] error={error:.2f}, steer={steer_angle:.2f}, speed={speed:.2f}")

    # Draw the lane center vs frame center at y_ref
    cv2.circle(frame, (int(frame_center_x), y_ref), 5, (0, 255, 255), -1)
    cv2.circle(frame, (int(lane_center_x), y_ref), 5, (0, 255, 0), -1)
    cv2.line(frame, (int(frame_center_x), y_ref), (int(lane_center_x), y_ref),
             (0, 255, 0), 2)

    # Optionally draw the stored left/right lines in the frame
    if last_left_avg is not None:
        slope, intercept = last_left_avg
        y1_draw = height
        y2_draw = int(height/2)
        x1_draw = int((y1_draw - intercept) / slope)
        x2_draw = int((y2_draw - intercept) / slope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 0), 3)

    if last_right_avg is not None:
        slope, intercept = last_right_avg
        y1_draw = height
        y2_draw = int(height/2)
        x1_draw = int((y1_draw - intercept) / slope)
        x2_draw = int((y2_draw - intercept) / slope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 0, 255), 3)

    # Return frames for display
    return frame, mask_white, mask_yellow, combined_mask

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
