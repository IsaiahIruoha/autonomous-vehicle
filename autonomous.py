import cv2
import numpy as np
import time
import os
import threading

# ========== PiCar-X & Lights ==========
from picarx import Picarx
from robot_hat import PWM

# ========== AIY MakerKit / Coral Object Detection ==========
from aiymakerkit import vision
from aiymakerkit import utils
from pycoral.utils.dataset import read_label_file

# ------------------------------------------------------------
#   A) CONSTANTS & SETTINGS
# ------------------------------------------------------------
# Steering & Speed
STEERING_LEFT_LIMIT   = -45
STEERING_RIGHT_LIMIT  =  45
BASE_SPEED            =  1.5
MAX_TURN_SPEED        = 15
MIN_SPEED             =  1.0

KP = 0.20
KD = 0.10

ALPHA_LANE_SMOOTH   = 0.8  # For line-parameter smoothing
ALPHA_STEER_SMOOTH  = 0.2  # For camera-based steering smoothing
SMOOTHING_ALPHA     = 0.7  # For blending grayscale override

CAMERA_TILT_DEFAULT = 0
CAMERA_TILT_LEFT    = -10
CAMERA_TILT_RIGHT   =  10

MAX_LOST_FRAMES   = 5
LANE_HALF_WIDTH_PX = 80

# Grayscale sensor + stop line
WHITE_THRESHOLD = 700
GRAYSCALE_OVERRIDE_TURN = 30
STOP_LINE_TIMEOUT = 3.0

# Object Detection (frame skip)
DETECTION_EVERY_N_FRAMES = 5  # run detection every 5 frames

# Obstacle/Emergency braking thresholds
OBSTACLE_SIZE_THRESHOLD     = 0.15  # % of frame area => "large"
OBSTACLE_CENTER_TOLERANCE_X = 0.3   # ± fraction of frame width from center

# Global Lane Detection
last_error           = 0.0
last_steering_angle  = 0.0
lost_frames_count    = 0
last_left_avg        = None
last_right_avg       = None
last_final_steer_angle = 0.0

# ------------------------------------------------------------
#   LANE DETECTION FUNCTIONS
# ------------------------------------------------------------
def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def adjust_camera_tilt(px_obj, steering_angle):
    """Point camera left/right depending on steering angle."""
    if steering_angle < -10:
        px_obj.set_cam_tilt_angle(CAMERA_TILT_LEFT)
    elif steering_angle > 10:
        px_obj.set_cam_tilt_angle(CAMERA_TILT_RIGHT)
    else:
        px_obj.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)

def steering_control(error):
    """PD control for steering = KP*error + KD*(error - last_error)."""
    global last_error
    derivative = error - last_error
    raw_steering = KP * error + KD * derivative
    last_error = error
    return clamp(raw_steering, STEERING_LEFT_LIMIT, STEERING_RIGHT_LIMIT)

def apply_gaussian_blur(frame, ksize=5):
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

def region_of_interest(img):
    """Keep only the bottom portion of the image (rough trapezoid)."""
    height, width = img.shape[:2]
    top_y = int(0.35 * height)
    roi_vertices = np.array([[
        (0,      height),
        (0,      top_y),
        (width,  top_y),
        (width,  height)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(img, mask)

def get_line_params(x1, y1, x2, y2):
    """Return (slope, intercept) in y=mx+b form, or None if vertical."""
    if (x2 - x1) == 0:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (slope, intercept)

def running_average_line(new_line, old_line, alpha=0.9):
    """Exponential smoothing of (slope, intercept)."""
    if old_line is None:
        return new_line
    old_slope, old_int = old_line
    new_slope, new_int = new_line
    smoothed_slope = alpha * old_slope + (1 - alpha) * new_slope
    smoothed_int   = alpha * old_int   + (1 - alpha) * new_int
    return (smoothed_slope, smoothed_int)

def estimate_line_x_at_y(line_params, y):
    """x = (y - intercept)/slope, for line_params=(slope, intercept)."""
    if line_params is None:
        return None
    slope, intercept = line_params
    if abs(slope) < 1e-9:
        return None
    return (y - intercept) / slope

def process_frame(px_obj, frame):
    """
    1) Lane detection on 'frame'
    2) Returns a steering angle
    3) Draws lines + center circles for debugging
    """
    global last_steering_angle, lost_frames_count
    global last_left_avg, last_right_avg

    # 1) Preprocessing
    blurred = apply_gaussian_blur(frame, ksize=5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2) White mask
    lower_white_1 = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_1 = np.array([255, 40, 255], dtype=np.uint8)
    mask_white_1  = cv2.inRange(hsv, lower_white_1, upper_white_1)

    lower_white_2 = np.array([0, 0, 160], dtype=np.uint8)
    upper_white_2 = np.array([255, 60, 255], dtype=np.uint8)
    mask_white_2  = cv2.inRange(hsv, lower_white_2, upper_white_2)

    mask_white    = cv2.bitwise_or(mask_white_1, mask_white_2)

    # 3) Yellow mask
    lower_yellow  = np.array([15, 70, 100], dtype=np.uint8)
    upper_yellow  = np.array([50, 255, 255], dtype=np.uint8)
    mask_yellow   = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine white + yellow
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)

    # Morphological ops (reduce noise)
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 4) Edges + ROI
    edges    = cv2.Canny(combined_mask, 50, 150)
    roi_edge = region_of_interest(edges)

    # 5) Hough lines
    lines = cv2.HoughLinesP(
        roi_edge,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=25
    )

    # 6) Split lines into left vs. right
    height, width = frame.shape[:2]
    frame_center_x = width // 2
    y_ref = int(height * 0.8)

    left_params_list  = []
    right_params_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            params = get_line_params(x1, y1, x2, y2)
            if params is None:
                continue
            slope, intercept = params
            # ignore near-horizontal lines
            if abs(slope) < 0.2:
                continue
            if slope < 0:
                left_params_list.append((slope, intercept))
            else:
                right_params_list.append((slope, intercept))

    # 7) Exponential smoothing for left/right lines
    if len(left_params_list) > 0:
        avg_slope = np.mean([p[0] for p in left_params_list])
        avg_int   = np.mean([p[1] for p in left_params_list])
        new_left_line = (avg_slope, avg_int)
        last_left_avg = running_average_line(new_left_line, last_left_avg, alpha=ALPHA_LANE_SMOOTH)

    if len(right_params_list) > 0:
        avg_slope = np.mean([p[0] for p in right_params_list])
        avg_int   = np.mean([p[1] for p in right_params_list])
        new_right_line = (avg_slope, avg_int)
        last_right_avg = running_average_line(new_right_line, last_right_avg, alpha=ALPHA_LANE_SMOOTH)

    # Check detection presence
    left_detected  = (len(left_params_list) > 0)
    right_detected = (len(right_params_list) > 0)
    if not left_detected and not right_detected:
        lost_frames_count += 1
        print("[DEBUG] No lane lines found.")
        if lost_frames_count > MAX_LOST_FRAMES:
            px_obj.stop()
            return 0.0
    else:
        lost_frames_count = 0

    # 8) Compute lane center
    left_x  = estimate_line_x_at_y(last_left_avg,  y_ref) if last_left_avg  else None
    right_x = estimate_line_x_at_y(last_right_avg, y_ref) if last_right_avg else None

    if (left_x is not None) and (right_x is not None):
        lane_center_x = (left_x + right_x) / 2.0
    elif left_x is not None:
        lane_center_x = left_x + LANE_HALF_WIDTH_PX
    elif right_x is not None:
        lane_center_x = right_x - LANE_HALF_WIDTH_PX
    else:
        lane_center_x = frame_center_x

    # 9) PD control for steering
    error = lane_center_x - frame_center_x
    raw_steer_angle = steering_control(error)

    # Smooth final angle
    steer_angle = (ALPHA_STEER_SMOOTH * last_steering_angle
                   + (1 - ALPHA_STEER_SMOOTH) * raw_steer_angle)
    last_steering_angle = steer_angle

    # 10) Draw lines + debug centers
    # Frame center (yellow)
    cv2.circle(frame, (int(frame_center_x), y_ref), 5, (0, 255, 255), -1)
    # Lane center (green)
    cv2.circle(frame, (int(lane_center_x), y_ref), 5, (0, 255, 0), -1)
    cv2.line(frame, (int(frame_center_x), y_ref),
             (int(lane_center_x), y_ref), (0, 255, 0), 2)

    # Draw left line in BLUE
    if last_left_avg is not None:
        lslope, lint = last_left_avg
        y1_draw = height
        y2_draw = int(height * 0.5)
        x1_draw = int((y1_draw - lint) / lslope)
        x2_draw = int((y2_draw - lint) / lslope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 0), 3)

    # Draw right line in RED
    if last_right_avg is not None:
        rslope, rint = last_right_avg
        y1_draw = height
        y2_draw = int(height * 0.5)
        x1_draw = int((y1_draw - rint) / rslope)
        x2_draw = int((y2_draw - rint) / rslope)
        cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 0, 255), 3)

    return steer_angle

# ------------------------------------------------------------
#   OBJECT DETECTION
# ------------------------------------------------------------
def path(name):
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, "DetectionModel", name)

ROAD_SIGN_DETECTION_MODEL_EDGETPU = path('efficientdet-lite-road-signs_edgetpu.tflite')
ROAD_SIGN_DETECTION_LABELS        = path('road-signs-labels.txt')

detector = vision.Detector(ROAD_SIGN_DETECTION_MODEL_EDGETPU)
labels   = read_label_file(ROAD_SIGN_DETECTION_LABELS)

def detect_objects(frame, threshold=0.6):
    """Run object detection on 'frame' with threshold=0.6, return objects."""
    objs = detector.get_objects(frame, threshold=threshold)
    return objs

# ------------------------------------------------------------
#   LIGHTS SETUP WITH THREADING
# ------------------------------------------------------------
leds = {
    'brake_left': PWM('P6'),   # Brake lights
    'brake_right': PWM('P11'),
    'left_signal_1': PWM('P10'),  # Left turn signals
    'left_signal_2': PWM('P9'),
    'right_signal_1': PWM('P8'),  # Right turn signals
    'right_signal_2': PWM('P7'),
}

for pwm_obj in leds.values():
    pwm_obj.freq(50)
    pwm_obj.prescaler(1)
    pwm_obj.period(100)

def turn_on_brake_lights():
    leds['brake_left'].pulse_width_percent(100)
    leds['brake_right'].pulse_width_percent(100)
    
def turn_off_brake_lights():
    leds['brake_left'].pulse_width_percent(0)
    leds['brake_right'].pulse_width_percent(0)

def _blink_signals(pwm_list, duration, on_time, off_time):
    end_time = time.time() + duration
    while time.time() < end_time:
        for p in pwm_list:
            p.pulse_width_percent(100)
        time.sleep(on_time)
        for p in pwm_list:
            p.pulse_width_percent(0)
        time.sleep(off_time)

def blink_left_signal(duration=3, on_time=0.3, off_time=0.3):
    pwm_list = [leds['left_signal_1'], leds['left_signal_2']]
    t = threading.Thread(target=_blink_signals, args=(pwm_list, duration, on_time, off_time), daemon=True)
    t.start()

def blink_right_signal(duration=3, on_time=0.3, off_time=0.3):
    pwm_list = [leds['right_signal_1'], leds['right_signal_2']]
    t = threading.Thread(target=_blink_signals, args=(pwm_list, duration, on_time, off_time), daemon=True)
    t.start()

def blink_hazard_signal(duration=3, on_time=0.3, off_time=0.3):
    pwm_list = [
        leds['left_signal_1'], leds['left_signal_2'],
        leds['right_signal_1'], leds['right_signal_2']
    ]
    t = threading.Thread(target=_blink_signals, args=(pwm_list, duration, on_time, off_time), daemon=True)
    t.start()

# ------------------------------------------------------------
#   MAIN LOOP
# ------------------------------------------------------------
def main():
    px = Picarx()
    px.set_cam_tilt_angle(CAMERA_TILT_DEFAULT)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    time.sleep(0.5)

    global last_final_steer_angle

    frame_count = 0
    last_objects = []  # store last detection result for continuous bounding-box display

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No camera feed detected.")
                break

            # ========== A) Check for STOP line via grayscale sensors ==========
            sensor_values = px.get_grayscale_data()
            left_sensor, middle_sensor, right_sensor = sensor_values

            # If all 3 see white => stop line
            if (left_sensor > WHITE_THRESHOLD and
                middle_sensor > WHITE_THRESHOLD and
                right_sensor > WHITE_THRESHOLD):
                print("[INFO] STOP line detected! Stopping for 3 sec.")
                px.stop()
                turn_on_brake_lights()
                time.sleep(STOP_LINE_TIMEOUT)
                turn_off_brake_lights()

                # Re-check the camera after stopping
                ret2, frame2 = cap.read()
                if ret2:
                    new_camera_angle = process_frame(px, frame2)
                    last_final_steer_angle = new_camera_angle
                    px.set_dir_servo_angle(new_camera_angle)

                # Optional move to clear line:
                px.set_dir_servo_angle(-13)
                px.forward(1.5)
                time.sleep(0.5)
                px.stop()

                continue

            # ========== B) Lane Detection => camera steer angle ==========
            camera_steer_angle = process_frame(px, frame)

            # Grayscale boundary override (left/right)
            grayscale_correction = 0
            if left_sensor > WHITE_THRESHOLD:
                grayscale_correction = GRAYSCALE_OVERRIDE_TURN   # Turn right
            elif right_sensor > WHITE_THRESHOLD:
                grayscale_correction = -GRAYSCALE_OVERRIDE_TURN  # Turn left

            # Combine camera angle + grayscale override
            target_steer = camera_steer_angle + grayscale_correction

            # Smooth final steering
            raw_final = ((1 - SMOOTHING_ALPHA) * target_steer 
                         + SMOOTHING_ALPHA * last_final_steer_angle)
            final_steer_angle = clamp(raw_final, STEERING_LEFT_LIMIT, STEERING_RIGHT_LIMIT)
            last_final_steer_angle = final_steer_angle

            px.set_dir_servo_angle(final_steer_angle)
            adjust_camera_tilt(px, final_steer_angle)

            # Speed logic
            turn_factor = 0.5
            raw_speed = BASE_SPEED - turn_factor * abs(final_steer_angle)
            speed = clamp(raw_speed, MIN_SPEED, MAX_TURN_SPEED)
            px.forward(speed)

            # ========== C) Object Detection (skip frames) ==========
            frame_count += 1
            detect_frame = frame.copy()  # We'll draw bounding boxes on this copy

            # Draw last known bounding boxes
            if last_objects:
                vision.draw_objects(detect_frame, last_objects, labels)

            # If it's a detection frame, do fresh detection
            if (frame_count % DETECTION_EVERY_N_FRAMES) == 0:
                new_objs = detect_objects(detect_frame, threshold=0.6)  # consistent 0.6 threshold
                detect_frame = frame.copy()
                vision.draw_objects(detect_frame, new_objs, labels)
                last_objects = new_objs

                # Check for sign or obstacle
                for obj in new_objs:
                    cls_id    = obj.id
                    cls_label = labels.get(cls_id, "unknown")
                    x1, y1, x2, y2 = obj.bbox
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area   = box_w * box_h
                    frame_area = frame.shape[0] * frame.shape[1]
                    center_x   = (x1 + x2) / 2.0

                    # Specific signs
                    if cls_label == "sign_stop":
                        print("[DETECT] STOP sign => blink brake lights.")
                        turn_on_brake_lights()
                        time.sleep(1)
                        turn_off_brake_lights()
                    elif cls_label == "sign_oneway_left":
                        print("[DETECT] One-way left => blink left signal.")
                        blink_left_signal(duration=2)
                    elif cls_label == "sign_oneway_right":
                        print("[DETECT] One-way right => blink right signal.")
                        blink_right_signal(duration=2)

                    # Obstacle Handling
                    if cls_label in ["duck_regular", "duck_specialty", "vehicle"]:
                        if (box_area / frame_area) > OBSTACLE_SIZE_THRESHOLD:
                            frame_width = frame.shape[1]
                            dist_from_center = abs(center_x - (frame_width / 2.0))
                            if dist_from_center < (OBSTACLE_CENTER_TOLERANCE_X * frame_width):
                                print("[EMERGENCY] Large obstacle in lane => STOP + hazard!")
                                px.stop()
                                blink_hazard_signal(duration=2)
                                time.sleep(2)
                                # px.forward(speed) # optional continue

            # Show lane lines
            cv2.imshow("Lane Detection", frame)
            # Show bounding boxes
            cv2.imshow("Object Detection", detect_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt => Stopping...")

    finally:
        px.stop()
        turn_off_brake_lights()
        cap.release()
        cv2.destroyAllWindows()

# Entry Point
if __name__ == "__main__":
    main()
