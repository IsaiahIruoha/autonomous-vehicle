import time
import threading
from detect_road_signs import vision, detector, labels
from path_planning import a_star, graph, nearest_node, get_current_position
from some_lane_detection_module import LaneDetector 
from picar import front_wheels, back_wheels  

TEAM = 40 

fw = front_wheels.Front_Wheels()
bw = back_wheels.Back_Wheels()
bw.speed = 30 

def detect_objects():
    """Continuously detects objects and reacts accordingly."""
    for frame in vision.get_frames():
        objects = detector.get_objects(frame, threshold=0.3)
        for obj in objects:
            label = labels.get(obj.id, "Unknown")
            print(f"Detected: {label}")

            if label == "Stop Sign":
                print("Stopping for a stop sign!")
                bw.stop()
                time.sleep(3)
                bw.forward()

def lane_following():
    """Adjusts car direction based on lane detection."""
    lane_detector = LaneDetector()
    
    while True:
        direction = lane_detector.get_steering_angle()
        fw.turn(direction)
        time.sleep(0.1)

def path_navigation():
    """Plans and follows the best path."""
    pos = get_current_position(TEAM)
    if pos is None:
        print("Failed to get car position.")
        return

    start_node = nearest_node(*pos)
    goal_node = "N"

    path = a_star(start_node, goal_node)
    if path is None:
        print("No valid path found.")
        return

    print(f"Following path: {path}")
    for node in path:
        print(f"Moving to {node}")
        bw.forward()
        time.sleep(1) 

def main():
    """Runs all subsystems concurrently."""
    object_thread = threading.Thread(target=detect_objects)
    lane_thread = threading.Thread(target=lane_following)
    path_thread = threading.Thread(target=path_navigation)

    object_thread.start()
    lane_thread.start()
    path_thread.start()

    object_thread.join()
    lane_thread.join()
    path_thread.join()

if __name__ == "__main__":
    main()
