"""
camera_tilt_test.py

Script to cycle through different tilt angles on the PiCar-X camera
so you can visually check the positions and confirm proper movement.
"""

import time
from picarx import Picarx

def main():
    px = Picarx()

    # List of angles you want to test
    test_angles = [-30, -20, -10, 0, 10, 20, 30]

    print("Starting camera tilt test...")
    for angle in test_angles:
        px.set_cam_tilt_angle(angle)
        print(f"[INFO] Camera tilt set to {angle} degrees.")
        time.sleep(2)  # Give yourself time to observe

    # Optionally reset to a default angle
    default_angle = -10
    px.set_cam_tilt_angle(default_angle)
    print(f"[INFO] Reset camera tilt to default angle ({default_angle} degrees).")

    print("Camera tilt test complete.")

if __name__ == "__main__":
    main()