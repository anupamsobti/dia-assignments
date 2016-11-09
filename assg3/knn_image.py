#!/usr/bin/python3
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

imgA = cv2.imread(sys.argv[1],0)
imgB = cv2.imread(sys.argv[2],0)

trainData = []
responses = []

for i in range(imgA.shape[0]):
    for j in range(imgA.shape[1]):
        trainData.append(imgA[i][j])
        responses.append(i*j)

#trainData = [1,2,3,4,5,6]
#responses = ["A","B","C","D","E","F"]
#responses = [100,101,102,103,104,105]

trainData = np.asarray(trainData).astype(np.float32)
responses = np.asarray(responses).astype(np.float32)
#newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
#plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
#
knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
newcomer = np.array(255).astype(np.float32)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)

print ("result: ", results,"\n")
print ("neighbours: ", neighbours,"\n")
print ("distance: ", dist)
#plt.show()
