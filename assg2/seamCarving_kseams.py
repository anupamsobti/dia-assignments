#!/usr/bin/python3
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

def deleteVerticalSeam(img,noOfSeams):
    IMGY,IMGX = img.shape[0],img.shape[1]

    def printVerticalSeamFrom(img,minSeams):
        IMGY,IMGX = img.shape[0],img.shape[1]
        seam = np.zeros((IMGY,IMGX),np.uint8)
        for seamStartX in minSeams:
            x=seamStartX
            y=0
            while x < IMGX - 1 and y < IMGY - 1:
                img[y,x,:] = 0
                seam[y,x] += 1
                #array = (yEnergyFunction[y,x-1],yEnergyFunction[y+1,x-1],yEnergyFunction[y-1,x-1])
                array = (yEnergyFunction[y+1,x],yEnergyFunction[y+1,x-1],yEnergyFunction[y+1,x+1])
                minPoint = np.argmin(array)
                #minPoint = np.argmin(yEnergyFunction[x-1,y],yEnergyFunction[x-1,y+1],yEnergyFunction[x-1,y-1])
                if minPoint == 0:
                    x,y = x,y+1
                    yEnergyFunction[y,x] = 255  #Added to handle duplication of lines
                elif minPoint == 1:
                    x,y = x-1,y+1
                    yEnergyFunction[y,x] = 255
                else:
                    x,y = x+1,y+1
                    yEnergyFunction[y,x] = 255
        return seam

    L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    yEnergyFunction = cv2.Scharr(L,ddepth = -1,dx = 0,dy = 1)
    cv2.imshow("Energy Function",yEnergyFunction)
    seamWeights = np.float64(yEnergyFunction[:,0])    #Horizontal Seam
    for y in range(L.shape[0]-1):
        for x in range(1,L.shape[1]-1):  #Gives Xmax
            seamWeights += min(yEnergyFunction[y+1,x],yEnergyFunction[y+1,x-1],yEnergyFunction[y+1,x+1])

    #minSeams = np.argpartition(seamWeights,noOfSeams)[:noOfSeams]
    minSeams = np.argsort(seamWeights)[:noOfSeams]
    #print(minSeams)

    #Color Seam in original image
    seam = printVerticalSeamFrom(img,minSeams)
    #cv2.imshow("Seam",seam)
    #cv2.waitKey(0)
    deletedSeamImage = np.zeros((IMGY,IMGX - noOfSeams,3),np.uint8)
    print ("New Image Size : ", deletedSeamImage.shape)
    for y in range(IMGY):
        seamSum = 0
        for x in range(IMGX - noOfSeams):
            seamSum += seam[y,x]
            deletedSeamImage[y,x,0] = L[y,x+seamSum]
            deletedSeamImage[y,x,1] = A[y,x+seamSum]
            deletedSeamImage[y,x,2] = B[y,x+seamSum]

    return cv2.cvtColor(deletedSeamImage,cv2.COLOR_Lab2BGR)


def deleteHorizontalSeam(img,noOfSeams):
    IMGY,IMGX = img.shape[0],img.shape[1]

    def printHorizontalSeamFrom(img,minSeams):
        IMGY,IMGX = img.shape[0],img.shape[1]
        seam = np.zeros((IMGY,IMGX),np.uint8)
        for seamStartY in minSeams:
            x=0
            y=seamStartY
            while x < IMGX - 1 and y < IMGY - 1:
                img[y,x,:] = 0
                seam[y,x] += 1
                #array = (xenergyFunction[y,x-1],xenergyFunction[y+1,x-1],xenergyFunction[y-1,x-1])
                array = (xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])
                minPoint = np.argmin(array)
                #minPoint = np.argmin(xenergyFunction[x-1,y],xenergyFunction[x-1,y+1],xenergyFunction[x-1,y-1])
                if minPoint == 0:
                    x,y = x+1,y
                    xenergyFunction[y,x] = 255
                elif minPoint == 1:
                    x,y = x+1,y+1
                    xenergyFunction[y,x] = 255
                else:
                    x,y = x+1,y-1
                    xenergyFunction[y,x] = 255
        return seam

    L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    xenergyFunction = cv2.Scharr(L,ddepth = -1,dx = 1,dy = 0)
    cv2.imshow("Energy Function",xenergyFunction)
    seamWeights = np.float64(xenergyFunction[:,0])    #Horizontal Seam
    for x in range(L.shape[1]-1):
        for y in range(1,L.shape[0]-1):  #Gives Ymax
            seamWeights += min(xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])

    #minSeams = np.argpartition(seamWeights,noOfSeams)[:noOfSeams]
    minSeams = np.argsort(seamWeights)[:noOfSeams]
    #print(minSeams)

    #Color Seam in original image
    seam = printHorizontalSeamFrom(img,minSeams)
    #cv2.imshow("Seam",seam)
    #cv2.waitKey(0)
    deletedSeamImage = np.zeros((IMGY - noOfSeams,IMGX,3),np.uint8)
    print ("New Image Size : ", deletedSeamImage.shape)
    for x in range(IMGX):
        seamSum = 0
        for y in range(IMGY-noOfSeams):
            seamSum += seam[y,x]
            deletedSeamImage[y,x,0] = L[y+seamSum,x]
            deletedSeamImage[y,x,1] = A[y+seamSum,x]
            deletedSeamImage[y,x,2] = B[y+seamSum,x]
        print ("X = ",x,"Seam Sum : ",seamSum)

    return cv2.cvtColor(deletedSeamImage,cv2.COLOR_Lab2BGR)

#newImage = deleteHorizontalSeam(img,50)
#cv2.imshow("After deleting horizontal",img)
finalImage = deleteHorizontalSeam(img,150)

#newImage = deleteHorizontalSeam(img)
#cv2.imwrite("ResizedImage.png",newImage)
cv2.imshow("ResizedImage.png",finalImage)
cv2.imshow("Image",img)
#cv2.imshow("Energy Function",xenergyFunction)
cv2.waitKey(0)
cv2.destroyAllWindows()
