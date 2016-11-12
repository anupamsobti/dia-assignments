#!/usr/bin/python3
import cv2
import sys
import colorsys

img = cv2.imread(sys.argv[1])/255
y,i,q = colorsys.rgb_to_yiq(img[:,:,2],img[:,:,1],img[:,:,0])

cv2.imshow("Y",y)
cv2.imshow("I",i)
cv2.imshow("Q",q)
cv2.waitKey(0)
