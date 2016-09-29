#!/usr/bin/python3
import cv2
import numpy as np
#from matplotlib import pyplot as plt
from collections import namedtuple
import sys

img = cv2.imread(sys.argv[1])

def updateEnergyFunction(energyFunction,houghPoints):
    for (x,y) in houghPoints:
        energyFunction[y,x] = 255
    return energyFunction

def appendVerticalSeam(img,noOfSeams,houghPoints):
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
                    yEnergyFunction[y,x] = 127 #Added to handle duplication of lines
                elif minPoint == 1:
                    x,y = x-1,y+1
                    yEnergyFunction[y,x] = 127
                else:
                    x,y = x+1,y+1
                    yEnergyFunction[y,x] = 127
        cv2.imshow("Appending Vertical Seams",img)
        return seam

    L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    yEnergyFunction = cv2.Scharr(L,ddepth = -1,dx = 0,dy = 1)

    yEnergyFunction = updateEnergyFunction(yEnergyFunction,houghPoints)

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

def appendHorizontalSeam(img,noOfSeams,houghPoints):
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
                    xenergyFunction[y,x] = 127 
                elif minPoint == 1:
                    x,y = x+1,y+1
                    xenergyFunction[y,x] = 127
                else:
                    x,y = x+1,y-1
                    xenergyFunction[y,x] = 127
        cv2.imshow("Appending Horizontal Seams",img)
        return seam

    L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    xenergyFunction = cv2.Scharr(L,ddepth = -1,dx = 1,dy = 0)

    xenergyFunction = updateEnergyFunction(xenergyFunction,houghPoints)

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



def deleteVerticalSeam(img,noOfSeams,houghPoints):
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
                    yEnergyFunction[y,x] = 127 #Added to handle duplication of lines
                elif minPoint == 1:
                    x,y = x-1,y+1
                    yEnergyFunction[y,x] = 127
                else:
                    x,y = x+1,y+1
                    yEnergyFunction[y,x] = 127
        cv2.imshow("Deleting Vertical Seams",img)
        return seam

    L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    yEnergyFunction = cv2.Scharr(L,ddepth = -1,dx = 0,dy = 1)

    yEnergyFunction = updateEnergyFunction(yEnergyFunction,houghPoints)

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


def deleteHorizontalSeam(img,noOfSeams,houghPoints):
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
                    xenergyFunction[y,x] = 127
                elif minPoint == 1:
                    x,y = x+1,y+1
                    xenergyFunction[y,x] = 127
                else:
                    x,y = x+1,y-1
                    xenergyFunction[y,x] = 127
        cv2.imshow("Deleting Horizontal Seams",img)
        return seam

    L,A,B = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    xenergyFunction = cv2.Scharr(L,ddepth = -1,dx = 1,dy = 0)

    xenergyFunction = updateEnergyFunction(xenergyFunction,houghPoints)
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

finalX,finalY = (int(sys.argv[2]),int(sys.argv[3]))
hough = int(sys.argv[4])

IMGY,IMGX = img.shape[0],img.shape[1]
houghPoints = []

if hough == 1:
    houghInputImg = cv2.imread(sys.argv[5],0)
    print("Hough : Using Reference object as :",sys.argv[5])
    
    sobel64f_x = cv2.Sobel(houghInputImg,cv2.CV_64F,1,0,ksize=5)
    sobel64f_y = cv2.Sobel(houghInputImg,cv2.CV_64F,0,1,ksize=5)
    
    a,b = houghInputImg.shape[:2]
    
    #computation of gradient angle phi
    phiAngle  = np.zeros((a,b),np.double)
    phiAngle = cv2.phase(sobel64f_x,sobel64f_y,angleInDegrees=True)
    
    #detect the edges
    edges = cv2.Canny(houghInputImg,150,200)
    
    #choose a centroid
    xc = np.round(a/2+0.5)
    yc = np.round(b/2+0.5)
    print("centeroid in source: (",xc,",",yc,")")
    
    #construction of RTable using gradient angle (phiAngle) computed above
    rAlpha = namedtuple('rAlpha', ['dist', 'cosAlpha', 'sinAlpha'])
    RTable = [[] for i in range(361)]
    
    
    for x in range(a):
        for y in range(b):
            if edges[x][y] != 0:
               xdiff = xc - x
               ydiff = yc - y
               r = np.sqrt((xdiff * xdiff) + (ydiff * ydiff))
               sinAlpha = ydiff/r
               cosAlpha = xdiff/r
               Node = rAlpha(dist=r, cosAlpha=cosAlpha,sinAlpha=sinAlpha)
               phi = int(np.round(phiAngle[x][y],0))
               RTable[phi].append(Node)
               #print("x:",x,"y:",y,"r:",Node.dist,"alpha:",Node.angle*180.0/np.pi,"phi:",phiAngle[x][y],phi)
    
    houghTargetImg = cv2.imread(sys.argv[1],0)
    
    a_target,b_target = houghTargetImg.shape[:2]
    phiAngle_target  = np.zeros((a_target,b_target),np.double)
    print("Target image size:(",a_target,b_target,")")
    
    #computation of gradient angle phi
    sobel64f_x = cv2.Sobel(houghTargetImg,cv2.CV_64F,1,0,ksize=5)
    sobel64f_y = cv2.Sobel(houghTargetImg,cv2.CV_64F,0,1,ksize=5)
    phiAngle_target = cv2.phase(sobel64f_x,sobel64f_y,angleInDegrees=True)
    
    #detect the edges in target
    edges_target = cv2.Canny(houghTargetImg,150,200)
    
    #Hough Parameters
    SizeFactor = 1
    scalingRes = 2
    thetaMax = 1
    Threshold = 30
    
    VoteXY = namedtuple('VoteXY', ['xCor', 'yCor'])
    Q = [ [ [ [ [] for i in range(SizeFactor*(scalingRes+1)) ] for j in range(thetaMax) ] for k in range(scalingRes*b_target) ] for l in range(scalingRes*a_target)]
    
    #P  = np.zeros((scalingRes*a_target+100,scalingRes*b_target+100,thetaMax,SizeFactor*(scalingRes+1)),np.int)
    #for each edge pixel in target image , search for nearest gradient angle in RTable
    #for each entry (r,alpha) at that index compute xc,yc and vote at P[xc][yc]
    for x in range(a_target):
        for y in range(b_target):
            if edges_target[x][y] != 0:
                phi = int(np.round(phiAngle_target[x][y],0))
                phaseListLen = len(RTable[phi])
                if phaseListLen != 0:
                    for p in range(phaseListLen):
                        Node = RTable[phi][p]
                        r = Node.dist
                        xDash = -(r*Node.cosAlpha)
                        yDash = (r*Node.sinAlpha)
                        for theta in range(thetaMax):
                            thetaRad = (np.pi/180 ) * theta
                            for l in range(SizeFactor):
                                for s in range(1,scalingRes+1):
                                    x_c = int(np.round(x - ((xDash * np.cos(thetaRad) - yDash * np.sin(thetaRad))*(l+s/scalingRes))))
                                    y_c = int(np.round(y + ((xDash * np.sin(thetaRad) + yDash *np.cos(thetaRad))*(l+s/scalingRes))))
                                    voteXY_Node = VoteXY(xCor=x,yCor=y)
                                    Q[x_c][y_c][theta][l*(scalingRes)+s].append(voteXY_Node)
                                    #P[x_c][y_c][theta][l*(scalingRes)+s] = P[x_c][y_c][theta][l*scalingRes+s] + 1
                                    #print(x_c,y_c,theta,l+s/scalingRes,l*scalingRes+s)
    
    
    
    detectedContourImg = np.zeros((a_target,b_target),np.uint8)
    for xc in range(scalingRes*a_target):
        for yc in range(scalingRes*b_target):
            for theta in range(thetaMax):
                for l in range(SizeFactor):
                    for s in range(scalingRes+1):
                        vote = len(Q[xc][yc][theta][l*(scalingRes)+s])
                        if vote > Threshold:
                            print("Vote:",vote,"centeroid detected:(",xc,",",yc,")",s,l)
                            for v in range(vote):
                                voteXY_Node =  Q[xc][yc][theta][l*(scalingRes)+s][v]
                                detectedContourImg[voteXY_Node.xCor][voteXY_Node.yCor] = 255
                                houghPoints.append((voteXY_Node.yCor,voteXY_Node.xCor))

    cv2.imshow("Detected Contour",detectedContourImg)


if finalY > IMGY:
    outputImage = appendHorizontalSeam(img,finalY-IMGY,houghPoints)
else:
    outputImage = deleteHorizontalSeam(img,IMGY - finalY,houghPoints)

cv2.imshow("After Y Modification",outputImage)

if finalX > IMGX:
    outputImage = appendVerticalSeam(outputImage,finalX - IMGX,houghPoints)
else:
    outputImage = deleteVerticalSeam(outputImage,IMGX-finalX,houghPoints)

cv2.imshow("After X Modification",outputImage)


cv2.waitKey(0)
cv2.destroyAllWindows()
