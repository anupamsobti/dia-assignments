#!/usr/bin/python3
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])


def deleteHorizontalSeam(img):
    IMGY,IMGX = img.shape[0],img.shape[1]

    def printSeamFrom(img,x,y):
        IMGY,IMGX = img.shape[0],img.shape[1]
        seam = np.zeros((IMGY,IMGX),np.uint8)
        while x < IMGX - 1 and y < IMGY - 1:
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
            if x > IMGX or y > IMGY:
                print("The corner case")
        return seam

    L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    xenergyFunction = cv2.Scharr(L,ddepth = -1,dx = 0,dy = 1)

    seamWeights = np.float64(xenergyFunction[:,0])    #Horizontal Seam
    for x in range(L.shape[1]-1):
        for y in range(1,L.shape[0]-1):  #Gives Ymax
            seamWeights += min(xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])

    minSeam = np.argmin(seamWeights)

    #Color Seam in original image
    seam = printSeamFrom(img,0,minSeam)
    cv2.imshow("Seam",seam)
    cv2.waitKey(0)
    deletedSeamImage = np.zeros((IMGY -1 ,IMGX,3),np.uint8)
    print ("New Image Size : ", deletedSeamImage.shape)
    count = 0
    for x in range(IMGX):
        skip=0
        for y in range(IMGY-1):
            if seam[y,x] == 255:
                skip=1
                #print (y)
                count+=1
            #deletedSeamImage.itemset((y,x,0),L[y+skip,x])
            #deletedSeamImage.itemset((y,x,1),A[y+skip,x])
            #deletedSeamImage.itemset((y,x,2),B[y+skip,x])
            deletedSeamImage[y,x,0] = L[y+skip,x]
            deletedSeamImage[y,x,1] = A[y+skip,x]
            deletedSeamImage[y,x,2] = B[y+skip,x]
    print("Count : ",count)

    return cv2.cvtColor(deletedSeamImage,cv2.COLOR_Lab2BGR)

newImage = img
for i in range(10):
    newImage = deleteHorizontalSeam(newImage)

#newImage = deleteHorizontalSeam(img)
cv2.imwrite("ResizedImage.png",newImage)
cv2.imshow("Image",img)
#cv2.imshow("Energy Function",xenergyFunction)
cv2.waitKey(0)
cv2.destroyAllWindows()
