#!/usr/bin/python3
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])
L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))

def printSeamFrom(img,x,y):
    seam = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    while x < img.shape[1] - 1 and y < img.shape[0] - 1:
        img[y,x,0] = 255
        seam[y,x] = 255
        #array = (xenergyFunction[y,x-1],xenergyFunction[y+1,x-1],xenergyFunction[y-1,x-1])
        array = (xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])
        minPoint = np.argmin(array)
        #minPoint = np.argmin(xenergyFunction[x-1,y],xenergyFunction[x-1,y+1],xenergyFunction[x-1,y-1])
        if minPoint == 0:
            x,y = x+1,y
        elif minPoint == 1:
            x,y = x+1,y+1
        else:
            x,y = x+1,y-1

xenergyFunction = cv2.Scharr(L,ddepth = -1,dx = 0,dy = 1)

seamWeights = np.float64(xenergyFunction[:,0])    #Vertical Seam
for x in range(L.shape[1]-1):
    for y in range(1,L.shape[0]-1):  #Gives Ymax
        seamWeights += min(xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])

minSeam = np.argmin(seamWeights)

#Color Seam in original image
seamEndPoint = (L.shape[1]-1,minSeam)
printSeamFrom(img,0,minSeam)

cv2.imshow("Image",img)
cv2.imshow("Energy Function",xenergyFunction)
cv2.waitKey(0)
cv2.destroyAllWindows()
