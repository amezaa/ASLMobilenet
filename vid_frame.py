import cv2
import os

vidcap = cv2.VideoCapture("accept3.mov")
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("/Users/bowenite/Desktop/tensorflow-for-poets-2/frames/" +  "frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

