#!/usr/bin/python3
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

inputImage = cv2.imread(sys.argv[1],0)

#'Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst'
inputImage = cv2.Laplacian(inputImage,cv2.CV_8U,5)
#inputImage = cv2.Laplacian(inputImage,cv2.CV_16S,5)

ret,inputImage2 = cv2.threshold(inputImage,25,255,cv2.THRESH_BINARY_INV)
ret,inputImage = cv2.threshold(inputImage,25,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#cv2.convertScaleAbs(inputImage,inputImage)
#inputImage = cv2.cvtColor(inputImage,cv2.COLOR_BGR2RGB)
plt.subplot(121),plt.imshow(inputImage2,"gray"),plt.title("Binary Inverse Thresholding")
plt.subplot(122),plt.imshow(inputImage,"gray"),plt.title("Binary Inverse + Otsu Thresholding")
plt.show()
