#!/usr/bin/python3
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])
L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))

def printSeamFrom(img,x,y):
    while x < img.shape[1] - 1 and y < img.shape[0] - 1:
        img[y,x,0] = 255
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
        #print("X : ",x," Y: ",y)
        seamWeights += min(xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])
        #seamWeights += min(xenergyFunction[x+1,y],xenergyFunction[x+1,y+1],xenergyFunction[x+1,y-1])

print(seamWeights)
minSeam = np.argmin(seamWeights)
#Color Seam in original image
seamEndPoint = (L.shape[1]-1,minSeam)
printSeamFrom(img,0,minSeam)
print("Min : ",seamWeights[minSeam]," 0th ",seamWeights[0])
#x,y = seamEndPoint
#while x >= 1 and y >=1:
#    img[y,x,2] = 255
#    array = (xenergyFunction[y,x-1],xenergyFunction[y+1,x-1],xenergyFunction[y-1,x-1])
#    minPoint = np.argmin(array)
#    #minPoint = np.argmin(xenergyFunction[x-1,y],xenergyFunction[x-1,y+1],xenergyFunction[x-1,y-1])
#    if minPoint == 0:
#        x,y = x-1,y
#    elif minPoint == 1:
#        x,y = x-1,y+1
#    else:
#        x,y = x-1,y-1

cv2.imshow("Image",img)
cv2.imshow("Energy Function",xenergyFunction)
cv2.waitKey(0)
cv2.destroyAllWindows()
