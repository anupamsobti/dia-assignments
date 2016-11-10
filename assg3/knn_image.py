#!/usr/bin/python3
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

imgA = cv2.imread(sys.argv[1],0)
imgB = cv2.imread(sys.argv[2],0)

trainData = []
responses = []

noOfRows = imgA.shape[0]
noOfColumns = imgA.shape[1]

for i in range(noOfRows):
    for j in range(noOfColumns):
        trainData.append(imgA[i][j])
        responses.append(noOfRows*i+j)


trainData = np.asarray(trainData).astype(np.float32)
responses = np.asarray(responses).astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
newcomer = np.array(0).astype(np.float32)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 1)

print ("result: ", results,"\n")
print ("Resulting Coordinates (y,x) : ",divmod(results,noOfRows))
(yNearest,xNearest) = divmod(results,noOfRows)
yNearest = int(yNearest)
xNearest = int(xNearest)

print(yNearest,xNearest)
print ("neighbours: ", neighbours,"\n")
print ("distance: ", dist)
