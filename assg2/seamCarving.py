#!/usr/bin/python3
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])
energyFunction = cv2.Scharr(img,ddepth = -1,dx = 1,dy = 0)
cv2.imshow("Image",energyFunction)
cv2.waitKey(0)
cv2.destroyAllWindows()
