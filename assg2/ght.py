#!/usr/bin/python3
import cv2
import numpy as np
import sys
import math
import struct
from collections import namedtuple

img = cv2.imread(sys.argv[1],0)
print("Using input image as ",sys.argv[1])
#cv2.imshow("Input Image",img)
#cv2.waitKey(0)


sobel64f_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobel64f_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

a,b = img.shape[:2]

#computation of gradient angle phi
phiAngle  = np.zeros((a,b),np.double)
phiAngle = cv2.phase(sobel64f_x,sobel64f_y,angleInDegrees=True)

#detect the edges
edges = cv2.Canny(img,150,200)

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
           r = math.sqrt((xdiff * xdiff) + (ydiff * ydiff))
           sinAlpha = ydiff/r
           cosAlpha = xdiff/r
           Node = rAlpha(dist=r, cosAlpha=cosAlpha,sinAlpha=sinAlpha)
           phi = int(np.round(phiAngle[x][y],0))
           RTable[phi].append(Node)
           #print("x:",x,"y:",y,"r:",Node.dist,"alpha:",Node.angle*180.0/np.pi,"phi:",phiAngle[x][y],phi)






targetImg = cv2.imread(sys.argv[2],0)
#cv2.imshow("Target Image",targetImg)
#cv2.waitKey(0)


a_target,b_target = targetImg.shape[:2]
phiAngle_target  = np.zeros((a_target,b_target),np.double)
print("target image size:(",a_target,b_target,")")

#computation of gradient angle phi
sobel64f_x = cv2.Sobel(targetImg,cv2.CV_64F,1,0,ksize=5)
sobel64f_y = cv2.Sobel(targetImg,cv2.CV_64F,0,1,ksize=5)
phiAngle_target = cv2.phase(sobel64f_x,sobel64f_y,angleInDegrees=True)

#detect the edges in target
edges_target = cv2.Canny(targetImg,150,200)



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



FinalImage = np.zeros((a_target,b_target),np.uint8)
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
                            FinalImage[voteXY_Node.xCor][voteXY_Node.yCor] = 255
                            print("X,Y:",voteXY_Node.xCor,voteXY_Node.yCor)
        

#draw the edge image of source and target
cv2.imshow('source edges',edges)
cv2.imshow('target edges',edges_target)
cv2.imshow('detected edges',FinalImage)
cv2.waitKey(0)
                               
#idx = np.argmax(P)
#tup = np.unravel_index(idx,P.shape)
#print(tup,P[tup])




cv2.destroyAllWindows()
