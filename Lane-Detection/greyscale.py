from picarx import Picarx
import time

px = Picarx()
WHITE_THRESHOLD = 700  # Adjust this based on your environment

def adjust_direction():
    """Adjust the car's direction based on grayscale sensor values."""
    sensor_values = px.get_grayscale_data()
    left_sensor = sensor_values[0]
    right_sensor = sensor_values[2]

    if left_sensor > 200:
        print("Left sensor detected high value! Turning right.")
        px.set_dir_servo_angle(50)  
    elif right_sensor > 200:
        print("Right sensor detected high value! Turning left.")
        px.set_dir_servo_angle(-76)
    else:
        print("Following straight.")
        px.set_dir_servo_angle(-13)  # Neutral for straight movement

def main():
    """
    Main function that moves the car forward and continuously checks
    the grayscale sensor values to adjust direction.
    """
    # Start driving forward at a moderate speed
    px.forward(30)

    try:
        while True:
            adjust_direction()
            time.sleep(0.1)  # Small delay to avoid flooding the servo
    except KeyboardInterrupt:
        print("Stopping the robot.")
        px.stop()

if __name__ == "__main__":
    main()
