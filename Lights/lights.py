#!/usr/bin/env python3
from robot_hat import PWM
import time

# Create PWM objects for the specified pins
leds = {
    'brake_left': PWM('P6'),   # Brake lights
    'brake_right': PWM('P11'),
    'left_signal_1': PWM('P10'),  # Left turn signals
    'left_signal_2': PWM('P9'),
    'right_signal_1': PWM('P8'),  # Right turn signals
    'right_signal_2': PWM('P7'),
}

for pwm in leds.values():
    pwm.freq(50)
    pwm.prescaler(1)
    pwm.period(100)

def turn_on_brake_lights():
    """Turns on the brake lights."""
    leds['brake_left'].pulse_width_percent(100)
    leds['brake_right'].pulse_width_percent(100)
    
def turn_off_brake_lights():
    """Turns off the brake lights."""
    leds['brake_left'].pulse_width_percent(0)
    leds['brake_right'].pulse_width_percent(0)

def blink_left_signal(duration=5, on_time=0.5, off_time=0.5):
    """Blinks the left turn signals for a specified duration."""
    end_time = time.time() + duration
    while time.time() < end_time:
        leds['left_signal_1'].pulse_width_percent(100)
        leds['left_signal_2'].pulse_width_percent(100)
        time.sleep(on_time)
        leds['left_signal_1'].pulse_width_percent(0)
        leds['left_signal_2'].pulse_width_percent(0)
        time.sleep(off_time)

def blink_right_signal(duration=5, on_time=0.5, off_time=0.5):
    """Blinks the right turn signals for a specified duration."""
    end_time = time.time() + duration
    while time.time() < end_time:
        leds['right_signal_1'].pulse_width_percent(100)
        leds['right_signal_2'].pulse_width_percent(100)
        time.sleep(on_time)
        leds['right_signal_1'].pulse_width_percent(0)
        leds['right_signal_2'].pulse_width_percent(0)
        time.sleep(off_time)

try:
    turn_on_brake_lights()
    #turn_off_brake_lights()
    #blink_left_signal()
    #blink_right_signal()
    pass

except KeyboardInterrupt:
    print("Exiting gracefully.")
    for pwm in leds.values():
        pwm.pulse_width_percent(0)  # Ensure all LEDs are off before exiting