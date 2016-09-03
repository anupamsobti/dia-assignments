#!/usr/bin/python3
import cv2
import numpy as np
import sys

minRegionX = 10
minRegionY = 10

img = cv2.imread(sys.argv[1])

swatches = []

def mouseCallback(event,x,y,flags,param):
    global swatches,img,startX,startY,endX,endY,minRegionX,minRegionY
    flag = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        (startX,startY) = (x,y)
        print ("Starts at ",startX,startY)
    elif event == cv2.EVENT_LBUTTONUP:
        endX,endY = x,y
        print ("Ends at ",x,y)
        cv2.rectangle(img,(startX,startY),(endX,endY),(255,0,0),3)
        if ((endX - startX > minRegionX) and (endY - startY > minRegionY)):
            swatches.append((startX,startY,endX,endY))

# Create a black image, a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouseCallback)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

print (swatches)
cv2.destroyAllWindows()
