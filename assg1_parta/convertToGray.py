#!/usr/bin/python3
import cv2
import sys

img = cv2.imread(sys.argv[1],0)
cv2.imwrite("output.jpg",img)
