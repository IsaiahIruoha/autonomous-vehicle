# research script for Object detection
# UNTESTED as of reading week*****

# use this command to install:
# pip install ultralytics opencv-python

# *****with the labeled dataset once we get it, fine-tune YOLO using transfer learning:******
# yolo train data=your_dataset.yaml model=yolov5s.pt epochs=50 imgsz=640


import cv2
from ultralytics import YOLO
from picarx import Picarx

# load the fine-tuned YOLOv5 model
# REPLACE WITH FIND TUNED MODEL PATH*****
model = YOLO('fine_tuned_traffic_signs.pt') 

px = Picarx()

# starting the video capture:
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

try:
    while True:
        # capture frame by frame
        ret, frame = cap.read()
        if not ret:
            break

        #run object detection on the frame:
        results = model.predict(frame, conf=0.5)  # confidence threshold set to 50% (not sure if correct vaue but we can change it)

        # Extract detected sign names
        detected_signs = [result['name'] for result in results[0].boxes.data]

        #Decision logic for traffic signs:
        if "stop" in detected_signs:
            print("Stop sign detected. Stopping car.")
            px.stop()
          elif "yield" in detected_signs:
            print("Yield sign detected. Slowing down.")
            px.forward(10)  # Slow down
        elif "do-not-enter" in detected_signs:
            print("Do Not Enter sign detected. Reversing.")
            px.backward(10)
        elif "one-way" in detected_signs:
            print("One-Way sign detected. Adjusting direction.")
            px.forward(15)  # Normal speed
        else:
            print("Continuing forward (no signs detected).")
            px.forward(20)

        # Display the frame with detections
        annotated_frame = results[0].plot()
      cv2.imshow("Traffic Sign Detection", annotated_frame)

        # Exit loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping the car and shutting down.")

finally:
    px.stop()
    cap.release()
    cv2.destroyAllWindows()


