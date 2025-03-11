from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.resolution = (1280, 720)  # Set resolution
camera.start_preview()

for i in range(5):  # Capture 10 images
    sleep(3)  # Wait before taking a picture
    camera.capture(f'calib_image_{i+1}.jpg')
    print(f"Captured calib_image_{i+1}.jpg")

camera.stop_preview()
print("All calibration images captured!")
