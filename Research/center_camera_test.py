"""
camera_center_test.py

Moves both the tilt and pan servos to 0 degrees
so you can check if your camera is visually pointing straight ahead.
"""

import time
from picarx import Picarx

def main():
    px = Picarx()
    
    # === Set both tilt and pan to 0 ===
    pan_angle = -10
    tilt_angle = -10

    print("Setting both pan and tilt to 0° (forward-facing)...")
    px.set_cam_pan_angle(pan_angle)
    px.set_cam_tilt_angle(tilt_angle)

    # Let you observe the result
    print("[INFO] Pan = 0°, Tilt = 0°")
    print("[INFO] Visually confirm that the camera is facing straight ahead.")
    time.sleep(5)
    
    DEFAULT_TILT = 0
    DEFAULT_PAN = 0

    print(f"Resetting to default angles: Pan = {DEFAULT_PAN}°, Tilt = {DEFAULT_TILT}°")
    px.set_cam_pan_angle(DEFAULT_PAN)
    px.set_cam_tilt_angle(DEFAULT_TILT)

    print("Camera center test complete.")

if __name__ == "__main__":
    main()