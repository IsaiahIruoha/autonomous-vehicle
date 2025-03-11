import time
import threading
from queue import Queue
import readchar
from picarx import Picarx 
px = Picarx()


target_speed = 0
current_speed = 0
target_angle = 0
current_angle = 0


SPEED_STEP = 10
ANGLE_STEP = 5 
MAX_SPEED = 100
MAX_ANGLE = 30 
ACCEL_RATE = 5 
ANGLE_ACCEL_RATE = 5 


key_queue = Queue()

def read_keys_in_thread():
    while True:
        k = readchar.readkey()
        key_queue.put(k)
        
        if k == chr(27) or k == readchar.key.CTRL_C:
            break
        
def accelerate(current, target, rate):
    if current < target:
        return min(current + rate, target)
    elif current > target:
        return max(current - rate, target)
    else:
        return current
    
def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def handle_key(k):
    global target_speed, target_angle
    
    k_lower = k.lower()
    
    if k_lower == 'w':
        target_speed += SPEED_STEP
    elif k_lower == 's':
        target_speed -= SPEED_STEP
    elif k_lower == 'a':
        target_angle -= ANGLE_STEP
    elif k_lower == 'd':
        target_angle += ANGLE_STEP
    elif k_lower == ' ':
        target_speed = 0
        
    target_speed = clamp(target_speed, -MAX_SPEED, MAX_SPEED)
    target_angle = clamp(target_angle, -MAX_ANGLE, MAX_ANGLE)
    
reader_thread = threading.Thread(target=read_keys_in_thread, daemon=True)
reader_thread.start()

try:
    print("Use W/S to move forward/backward, A/D to turn left/right.")
    print("Press ESC or Ctrl+C to stop.")
    
    while True:
        while not key_queue.empty():
            k = key_queue.get()
            
            if k == chr(27) or k == readchar.key.CTRL_C:
                raise KeyboardInterrupt
            else: handle_key(k)
            
            current_speed = accelerate(current_speed, target_speed, ACCEL_RATE)
            current_angle = accelerate(current_angle, target_angle, ANGLE_ACCEL_RATE)
            
            px.set_dir_servo_angle(int(current_angle))
            
            if current_speed > 0:
                px.forward(int(current_speed))
            elif current_speed < 0:
                px.backward(abs(int(current_speed)))
            else: px.stop()
            
            print(f"Target Speed: {target_speed}, Current Speed: {current_speed:.1f}, " f"Target Angle: {target_angle}, Current Angle: {current_angle:.1f}")
            
            time.sleep(0.1)
            
except KeyboardInterrupt:
    pass
finally:
    px.stop()
    print("Stopped PiCar-X.")
