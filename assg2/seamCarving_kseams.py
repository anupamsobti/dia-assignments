#!/usr/bin/python3
import cv2
import numpy as np
#from matplotlib import pyplot as plt
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
    deletedSeamImage = np.zeros((IMGY,IMGX - noOfSeams,3),np.uint8)
    print ("New Image Size : ", deletedSeamImage.shape)
    for y in range(IMGY):
        newImgX = 0
        flag = 0
        for x in range(IMGX):
            if seam[y,x] == 0 and flag == 0:
                deletedSeamImage[y,newImgX,0] = L[y,x]
                deletedSeamImage[y,newImgX,1] = A[y,x]
                deletedSeamImage[y,newImgX,2] = B[y,x]
                if newImgX < IMGX - noOfSeams -1:
                    newImgX+=1 
            elif seam[y,x] >= 1:
                flag+=seam[y,x] - 1
            else:
                flag -=1
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
                neighbourArray = (xenergyFunction[y,x+1],xenergyFunction[y+1,x+1],xenergyFunction[y-1,x+1])
                minPoint = np.argmin(neighbourArray)
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
    #print(seamWeights)
    #plt.plot(seamWeights)

    #Color Seam in original image
    seam = printHorizontalSeamFrom(img,minSeams)
    #cv2.imshow("Seam",seam)
    #cv2.waitKey(0)
    deletedSeamImage = np.zeros((IMGY - noOfSeams,IMGX,3),np.uint8)
    print ("New Image Size : ", deletedSeamImage.shape)
    for x in range(IMGX):
        newImgY = 0
        flag = 0
        for y in range(IMGY):
            if seam[y,x] == 0 and flag ==0:
                deletedSeamImage[newImgY,x,0] = L[y,x]
                deletedSeamImage[newImgY,x,1] = A[y,x]
                deletedSeamImage[newImgY,x,2] = B[y,x]
                if newImgY < IMGY - noOfSeams -1:
                    newImgY+=1 
            elif seam[y,x] >= 1:
                flag += seam[y,x]-1
            else:
                flag -=1

    return cv2.cvtColor(deletedSeamImage,cv2.COLOR_Lab2BGR)

#newImage = deleteHorizontalSeam(img,50)
#cv2.imshow("After deleting horizontal",img)
finalImage = deleteVerticalSeam(img,100)

#newImage = deleteHorizontalSeam(img)
#cv2.imwrite("ResizedImage.png",newImage)
cv2.imshow("ResizedImage.png",finalImage)
cv2.imshow("Image",img)
#cv2.imshow("Energy Function",xenergyFunction)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
