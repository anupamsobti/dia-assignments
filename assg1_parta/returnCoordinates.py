#!/usr/bin/python3
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

swatches = []

def mouseCallback(event,x,y,flags,param):
    global swatches,img
    flag = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        print ("Starts at ",x,y)
        (startX,startY) = (x,y)
        flag = 1
    elif flag == 1 and event == cv2.EVENT_MOUSEMOVE:
        cv2.rectangle(img,(startX,startY),(x,y),(255,0,0),5)
    elif event == cv2.EVENT_LBUTTONUP:
        print ("Ends at ",x,y)
        endX,endY = x,y
        #swatches.append((startX,startY,endX,endY))
        flag = 0

# Create a black image, a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouseCallback)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
