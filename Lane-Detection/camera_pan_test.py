"""
camera_pan_test.py

Script to cycle through different pan (left-right) angles on the PiCar-X camera
so you can visually confirm the positions.
"""

import time
from picarx import Picarx
def main():
    px = Picarx()

    # List of pan angles you want to test (left to right)
    # Adjust these ranges if your servo allows more or fewer degrees
    test_pan_angles = [-45, -30, -15, 0, 15, 30, 45]

    print("Starting camera pan test...")
    for angle in test_pan_angles:
        px.set_cam_pan_angle(angle)
        print(f"[INFO] Camera pan set to {angle} degrees.")
        time.sleep(2)  # Give time to observe each position

    default_pan_angle = 0
    px.set_cam_pan_angle(default_pan_angle)
    print(f"[INFO] Reset camera pan to default angle ({default_pan_angle} degrees).")

    print("Camera pan test complete.")

if __name__ == "__main__":
    main()