import numpy as np
import cv2
import os

boundaries = [
    ([0, 120, 0], [140, 255, 100]),
    ([25, 0, 75], [180, 38, 255])
]

#[
#    ([110,0,20],[130,5,40]),
#    ([60, 0, 10], [225, 100, 255])
#]

def hand_segment(frame):

    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask1 = cv2.inRange(frame, lower, upper)

    lower, upper = boundaries[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper)

    mask = cv2.bitwise_or(mask1, mask2)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    return gray

image = cv2.imread("/Users/bowenite/Desktop/tensorflow-for-poets-2/frames/frame32.jpg")
seg_frame = hand_segment(image)
cv2.imwrite("/Users/bowenite/Desktop/tensorflow-for-poets-2/processed_frames/frame32.jpg", seg_frame)

#if __name__ == '__main__':
#    for frame in ["framesframe27.HEIC"]: #os.listdir("/Users/bowenite/Desktop/tensorflow-for-poets-2/frames"):
#        image = cv2.imread("/Users/bowenite/Desktop/tensorflow-for-poets-2/frames/" + frame)
#        print(image)
#        seg_frame = hand_segment(image)
#        cv2.imwrite("/Users/bowenite/Desktop/tensorflow-for-poets-2/processed_frames" + frame, seg_frame)

