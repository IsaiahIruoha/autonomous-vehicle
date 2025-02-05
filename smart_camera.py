# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performs continuous face detection with the camera.

Simply run the script and it will draw boxes around detected faces:

    python3 detect_faces.py

For more instructions, see g.co/aiy/maker
"""

import os
from datetime import datetime
from pycoral.adapters.detect import BBox
from aiymakerkit import vision

# Define image save directory
PICTURE_DIR = "/home/rachelblainey/Pictures/" 
os.makedirs(PICTURE_DIR, exist_ok=True)

# Load face detection model
MODEL_PATH = "/home/rachelblainey/aiy-maker-kit/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
detector = vision.Detector(MODEL_PATH)

# Define the auto shutter detection zone
width, height = vision.VIDEO_SIZE
xmin = int(width * 0.25)
xmax = int(width - (width * 0.25))
ymin = int(height * 0.2)
ymax = int(height - (height * 0.2))
camera_bbox = BBox(xmin, ymin, xmax, ymax)

def box_is_in_box(inner: BBox, outer: BBox) -> bool:
    """Checks if the 'inner' BBox is fully contained within 'outer' BBox."""
    return (
        inner.xmin >= outer.xmin and
        inner.xmax <= outer.xmax and
        inner.ymin >= outer.ymin and
        inner.ymax <= outer.ymax
    )


for frame in vision.get_frames():
    faces = detector.get_objects(frame, threshold=0.5)
    print(faces) 
    vision.draw_objects(frame, faces)
    vision.draw_rect(frame, camera_bbox)
    
    if faces and box_is_in_box(faces[0].bbox, camera_bbox):
        print("Face is inside the box!")
        timestamp = datetime.now()
        filename = f"SMART_CAM_{timestamp.strftime('%Y-%m-%d_%H%M%S')}.jpg"
        filename = os.path.join(PICTURE_DIR, filename)
        vision.save_frame(filename, frame)

    
    
