#!/usr/bin/python3
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1])

def appendVerticalSeam(img,noOfSeams):
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
    cv2.imshow("Vertical Energy Function",yEnergyFunction)
    seamWeights = np.float64(yEnergyFunction[0,:])    #Vertical Seam

    #Code for minWeights calculation
    x=0;y=1
    seamX = []
    for i in range(IMGX):
        seamX.append(i)
    while x < IMGX - 1:
        while y < IMGY - 1:
            neighbourArray = (yEnergyFunction[y+1,seamX[x]],yEnergyFunction[y+1,seamX[x]],yEnergyFunction[y+1,seamX[x]])
            minPoint = np.argmin(neighbourArray)
            if minPoint == 0:
                seamX[x] += 0
            elif minPoint == 1:
                seamX[x] += 1
            else:
                seamX[x] -= 1
            y+=1
            seamWeights[x] += neighbourArray[minPoint]
        y=1;x+=1

    minSeams = np.argpartition(seamWeights,noOfSeams)[:noOfSeams]
    #minSeams = np.argsort(seamWeights)[:noOfSeams]
    #plt.plot(seamWeights)
    #print(seamWeights)
    #print(minSeams)

    #Color Seam in original image
    seam = printVerticalSeamFrom(img,minSeams)
    #cv2.imshow("Seam",seam)
    #cv2.waitKey(0)
    appendedSeamImage = np.zeros((IMGY,IMGX + noOfSeams,3),np.uint8)
    print ("New Image Size : ", appendedSeamImage.shape)
    for y in range(IMGY):
        origX = 0
        for x in range(IMGX + noOfSeams):
            if origX < IMGX:
                if seam[y,origX] == 0:
                    appendedSeamImage[y,x,0] = L[y,origX]
                    appendedSeamImage[y,x,1] = A[y,origX]
                    appendedSeamImage[y,x,2] = B[y,origX]
                    origX+=1 
                    #print("orig:",origX,"New X:",x)
                elif seam[y,origX] >= 1:
                    appendedSeamImage[y,x,0] = L[y,origX]
                    appendedSeamImage[y,x,1] = A[y,origX]
                    appendedSeamImage[y,x,2] = B[y,origX]
                    seam[y,origX] -=1
            else:
                appendedSeamImage[y,x,0] = L[y,origX-1]
                appendedSeamImage[y,x,1] = A[y,origX-1]
                appendedSeamImage[y,x,2] = B[y,origX-1]

    return cv2.cvtColor(appendedSeamImage,cv2.COLOR_Lab2BGR)


def appendHorizontalSeam(img,noOfSeams):
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
                neighbourArray = (xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])
                minPoint = np.argmin(neighbourArray)
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
    cv2.imshow("Horizontal Energy Function",xenergyFunction)
    seamWeights = np.float64(xenergyFunction[:,0])    #Horizontal Seam

    #Code for minWeights calculation
    x=1;y=0
    seamY = []
    for i in range(IMGY):
        seamY.append(i)
    while y < IMGY - 1:
        while x < IMGX - 1:
            neighbourArray = (xenergyFunction[seamY[y],x+1],xenergyFunction[seamY[y],x+1],xenergyFunction[seamY[y],x+1])
            minPoint = np.argmin(neighbourArray)
            if minPoint == 0:
                seamY[y] += 0
            elif minPoint == 1:
                seamY[y] += 1
            else:
                seamY[y] -= 1
            x+=1
            seamWeights[y] += neighbourArray[minPoint]
        x=1;y+=1

    minSeams = np.argpartition(seamWeights,noOfSeams)[:noOfSeams]
    #minSeams = np.argsort(seamWeights)[:noOfSeams]
    #plt.plot(seamWeights)

    #Color Seam in original image
    seam = printHorizontalSeamFrom(img,minSeams)
    #cv2.imshow("Seam",seam)
    #cv2.waitKey(0)
    appendedSeamImage = np.zeros((IMGY + noOfSeams,IMGX,3),np.uint8)
    print ("New Image Size : ", appendedSeamImage.shape)
    for x in range(IMGX):
        origY = 0
        for y in range(IMGY + noOfSeams):
            if origY < IMGY:
                if seam[origY,x] == 0:
                    appendedSeamImage[y,x,0] = L[origY,x]
                    appendedSeamImage[y,x,1] = A[origY,x]
                    appendedSeamImage[y,x,2] = B[origY,x]
                    origY +=1
                elif seam[origY,x] >= 1:
                    appendedSeamImage[y,x,0] = L[origY,x]
                    appendedSeamImage[y,x,1] = A[origY,x]
                    appendedSeamImage[y,x,2] = B[origY,x]
                    seam[origY,x] -=1
            else:
                appendedSeamImage[y,x,0] = L[origY-1,x]
                appendedSeamImage[y,x,1] = A[origY-1,x]
                appendedSeamImage[y,x,2] = B[origY-1,x]

    return cv2.cvtColor(appendedSeamImage,cv2.COLOR_Lab2BGR)

#newImage = deleteHorizontalSeam(img,50)
#cv2.imshow("After deleting horizontal",img)
finalImage = appendHorizontalSeam(img,200)

#newImage = deleteHorizontalSeam(img)
#cv2.imwrite("ResizedImage.png",newImage)
cv2.imshow("ResizedImage.png",finalImage)
cv2.imshow("Image",img)
#cv2.imshow("Energy Function",xenergyFunction)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
