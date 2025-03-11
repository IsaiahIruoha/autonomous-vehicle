#!/usr/bin/env python

"""camera_calibration.py: Performs camera calibration.

Usage: camera_calibration.py
Ensure images of the chessboard are contained in the same folder. A 
camera_calibration.json file will be generated which will contain the results 
of the calibration.
"""
__author__      = "Matthew Pan"
__copyright__   = "Copyright 2024, Matthew Pan"

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Define checkerboard size
CHESSBOARD = (7, 10)

# Optimization termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points and image points

# Arrays to store object points and image points from all the images.
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane

objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

# Load calibration images
images = glob.glob('*.jpg') # Gets images in the same folder as this script with the .jpg extension

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret:
        obj_points.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_points.append(corners2)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()      
# Perform calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Finds Re-projection Error
mean_error = 0
for i in range(len(obj_points)):
    img_points_2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(img_points[i], img_points_2, cv2.NORM_L2)/len(img_points_2)
    mean_error += error
    rep_error = mean_error/len(obj_points)
 # Prints results of calibration
print("Intrinsic Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist)
print( "Total Re-Projection Error (Pixels): {}".format(mean_error/len(obj_points)) )

# write results to json file
with open('camera_calibration.json', 'w') as f:
    data_dump = { 
        "repError" : rep_error,
        "instrinsicMatrix" : K,
        "distCoeff" : dist,
        "rvecs" : rvecs,
        "tvecs" : tvecs }
    json.dump(data_dump, f, cls=NumpyEncoder, separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
f.close()


# Plots the 3D position of the chessboards in all photos for 10 second before closing the program
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
  R, _ = cv2.Rodrigues(rvec)
  points_3d = np.dot(R, obj_points[i].T).T + tvec.T
  ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], label=f'Image {i+1}')

plt.title("3D Chessboard Positions")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='upper left')
plt.show(block=False)
plt.pause(10) 
plt.close()