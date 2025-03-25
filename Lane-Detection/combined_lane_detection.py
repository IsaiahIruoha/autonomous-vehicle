import cv2
import numpy as np
import time
from picarx import Picarx

px = Picarx()

# =========== Steering / Speed / PD Gains ============
STEERING_LEFT_LIMIT = -45
STEERING_RIGHT_LIMIT = 45

BASE_SPEED = 1.5      # Base speed for normal operation
MAX_TURN_SPEED = 15  # Max speed on straight or gentle curves

KP = 0.20
KD = 0.10

ALPHA = 0.3  # Steering angle smoothing

CAMERA_TILT_DEFAULT = -10
CAMERA_TILT_LEFT    = -10
CAMERA_TILT_RIGHT   =  0

MAX_LOST_FRAMES = 5
LANE_HALF_WIDTH_PX = 80

last_error = 0.0
last_steering_angle = 0.0
has_started = False

last_left_avg  = None
last_right_avg = None
lost_frames_count = 0

# =========== Grayscale Sensor Settings ============
WHITE_THRESHOLD = 700  # Tune for your environment
GRAYSCALE_OVERRIDE_TURN = 30  # Hard turn angle if boundary is detected
STOP_LINE_TIMEOUT = 3.0  # Stop for 3 seconds if all sensors see white

# =========== Initialization ============
px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

time.sleep(0.5)  # Let camera/wheels settle

def clamp(val, min_val, max_val):
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
    """PD control for steering."""
    global last_error
    derivative = error - last_error
    raw_steering = KP * error + KD * derivative
    last_error = error
    return clamp(raw_steering, STEERING_LEFT_LIMIT, STEERING_RIGHT_LIMIT)

def draw_roi_polygon(frame, roi_vertices, color=(0, 255, 0), thickness=2):
    pts = roi_vertices.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

def region_of_interest(img, visualize_on_frame=None):
    """Keep only bottom portion of the image."""
    height, width = img.shape[:2]
    
    top_y = int(0.35 * height)
    roi_vertices = np.array([[
        (0,   height),
        (0,   top_y),
        (width, top_y),
        (width, height)
    ]], dtype=np.int32)

    if visualize_on_frame is not None:
        draw_roi_polygon(visualize_on_frame, roi_vertices)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(img, mask)

def apply_gaussian_blur(frame, ksize=5):
    return cv2.GaussianBlur(frame, (ksize, ksize), 0) 

def get_line_params(x1, y1, x2, y2):
    """Return (slope, intercept) in y=mx+b form, or None if vertical."""
    if (x2 - x1) == 0:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (slope, intercept)

def running_average_line(new_line, old_line, alpha=0.3):
    """Exponential smoothing of line parameters."""
    if old_line is None:
        return new_line
    old_slope, old_int = old_line
    new_slope, new_int = new_line
    smoothed_slope = alpha * old_slope + (1 - alpha) * new_slope
    smoothed_int   = alpha * old_int   + (1 - alpha) * new_int
    return (smoothed_slope, smoothed_int)

def estimate_line_x_at_y(line_params, y):
    """x = (y - b)/m for line_params = (slope, intercept)."""
    slope, intercept = line_params
    if abs(slope) < 1e-9:
        return None
    return (y - intercept) / slope

def process_frame(frame):
    """Run lane detection on a frame, return a *camera-based* steering angle."""
    global has_started, last_steering_angle
    global last_left_avg, last_right_avg
    global lost_frames_count

    # ----- Camera-based Lane Detection -----
    frame_blur = apply_gaussian_blur(frame, ksize=5)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # White range masks
    lower_white_1 = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_1 = np.array([255, 40, 255], dtype=np.uint8)
    mask_white_1 = cv2.inRange(hsv, lower_white_1, upper_white_1)

    lower_white_2 = np.array([0, 0, 160], dtype=np.uint8)
    upper_white_2 = np.array([255, 60, 255], dtype=np.uint8)
    mask_white_2 = cv2.inRange(hsv, lower_white_2, upper_white_2)

    mask_white = cv2.bitwise_or(mask_white_1, mask_white_2)

    # Yellow range masks
    lower_yellow = np.array([15, 70, 100], dtype=np.uint8)
    upper_yellow = np.array([50, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine white + yellow
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)

    # Morphological ops
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(combined_mask, 50, 150)
    roi_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=25
    )

    height, width = frame.shape[:2]
    frame_center_x = width // 2
    y_ref = int(height * 0.8)

    left_params_list = []
    right_params_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            params = get_line_params(x1, y1, x2, y2)
            if params is None:
                continue
            slope, intercept = params
            if abs(slope) < 0.2:
                continue
            if slope < 0:
                left_params_list.append((slope, intercept))
            else:
                right_params_list.append((slope, intercept))

    # Average & Smooth lines
    if len(left_params_list) > 0:
        avg_slope = np.mean([p[0] for p in left_params_list])
        avg_int   = np.mean([p[1] for p in left_params_list])
        new_left_line = (avg_slope, avg_int)
        last_left_avg = running_average_line(new_left_line, last_left_avg, alpha=0.3)

    if len(right_params_list) > 0:
        avg_slope = np.mean([p[0] for p in right_params_list])
        avg_int   = np.mean([p[1] for p in right_params_list])
        new_right_line = (avg_slope, avg_int)
        last_right_avg = running_average_line(new_right_line, last_right_avg, alpha=0.3)

    # Check if lines were detected
    left_detected = (len(left_params_list) > 0)
    right_detected = (len(right_params_list) > 0)
    if not left_detected and not right_detected:
        lost_frames_count += 1
        print("[DEBUG] No lane lines found.")
        if lost_frames_count > MAX_LOST_FRAMES:
            px.stop()
            return 0.0
    else:
        lost_frames_count = 0

    # Compute the lane center
    left_x = estimate_line_x_at_y(last_left_avg, y_ref) if last_left_avg else None
    right_x = estimate_line_x_at_y(last_right_avg, y_ref) if last_right_avg else None

    if (left_x is not None) and (right_x is not None):
        lane_center_x = (left_x + right_x) / 2.0
    elif left_x is not None:
        lane_center_x = left_x + LANE_HALF_WIDTH_PX
    elif right_x is not None:
        lane_center_x = right_x - LANE_HALF_WIDTH_PX
    else:
        lane_center_x = frame_center_x  # fallback

    error = lane_center_x - frame_center_x
    raw_steer_angle = steering_control(error)
    steer_angle = ALPHA * last_steering_angle + (1 - ALPHA) * raw_steer_angle
    last_steering_angle = steer_angle

    # Draw for debugging
    cv2.circle(frame, (int(frame_center_x), y_ref), 5, (0, 255, 255), -1)
    cv2.circle(frame, (int(lane_center_x), y_ref), 5, (0, 255, 0), -1)
    cv2.line(frame, (int(frame_center_x), y_ref), (int(lane_center_x), y_ref), (0, 255, 0), 2)

    # Draw lines if we have them
    if last_left_avg is not None:
        lslope, lint = last_left_avg
        y1_draw = height
        y2_draw = int(height/2)
        x1_draw = int((y1_draw - lint) / lslope)
        x2_draw = int((y2_draw - lint) / lslope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 0), 3)

    if last_right_avg is not None:
        rslope, rint = last_right_avg
        y1_draw = height
        y2_draw = int(height/2)
        x1_draw = int((y1_draw - rint) / rslope)
        x2_draw = int((y2_draw - rint) / rslope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 0, 255), 3)

    return steer_angle

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera feed detected.")
            break

        # ========== 1) Camera-based Lane Detection =============
        camera_steer_angle = process_frame(frame)

        # ========== 2) Grayscale Sensor Check =============
        sensor_values = px.get_grayscale_data()
        left_sensor   = sensor_values[0]
        middle_sensor = sensor_values[1]
        right_sensor  = sensor_values[2]

        # ========== 2a) Detect "Stop Line" if ALL sensors see white ==========
        if (left_sensor > WHITE_THRESHOLD and 
            middle_sensor > WHITE_THRESHOLD and 
            right_sensor > WHITE_THRESHOLD):
            print("All sensors see white => Stop sign / crosswalk detected!")
            px.stop()
            time.sleep(STOP_LINE_TIMEOUT)  # Wait 3 seconds (or your chosen time)

            # After 3 seconds, let's move straight briefly to clear the line.
            # Here we set a near-neutral steering angle (e.g., -13).
            px.set_dir_servo_angle(-13)
            px.forward(2)   # Move forward at speed=2 for a short distance
            time.sleep(1)   # Move for 1 second
            px.stop()
            # Then continue the loop, which goes back into normal logic.
            continue

        # ========== 2b) Boundary Override Logic ==========
        # If left sensor is bright => turn right
        # If right sensor is bright => turn left
        final_steer_angle = camera_steer_angle
        if left_sensor > WHITE_THRESHOLD:
            print("Left grayscale triggered => HARD right turn!")
            final_steer_angle = GRAYSCALE_OVERRIDE_TURN
        elif right_sensor > WHITE_THRESHOLD:
            print("Right grayscale triggered => HARD left turn!")
            final_steer_angle = -GRAYSCALE_OVERRIDE_TURN

        # ========== 3) Send Final Steering Angle ==========
        final_steer_angle = clamp(final_steer_angle, STEERING_LEFT_LIMIT, STEERING_RIGHT_LIMIT)
        px.set_dir_servo_angle(final_steer_angle)
        adjust_camera_tilt(final_steer_angle)

        # ========== 4) Forward Speed Logic ==========
        # The sharper the turn, the slower we go
        turn_factor = 0.5
        MIN_SPEED   = 1.0
        raw_speed   = BASE_SPEED - turn_factor * abs(final_steer_angle)
        speed       = clamp(raw_speed, MIN_SPEED, MAX_TURN_SPEED)

        # If lines missing recently, slow further
        if lost_frames_count > 2:
            speed = min(speed, 1.0)

        px.forward(speed)

        # Show debug windows
        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt: stopping...")

finally:
    px.stop()
    px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
    cap.release()
    cv2.destroyAllWindows()
