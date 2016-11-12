#!/usr/bin/python3
import cv2
import sys

img = cv2.imread(sys.argv[1])

L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))

cv2.imshow("L",L)
cv2.imshow("A",A)
cv2.imshow("B",B)
cv2.waitKey(0)
