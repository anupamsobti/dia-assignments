#!/usr/bin/python3
import cv2
import numpy as np
import random
import sys
from matplotlib import pyplot as plt

##Functions for the main program

#Function for luminance remapping - Returns image with remapped histogram
def luminanceRemap(sourceImage,inputImage):
    sourceImageMean = sourceImage.mean()
    sourceImageStd = sourceImage.std()
    inputImageMean = inputImage.mean()
    inputImageStd = inputImage.std()
    #print("sourceImageMean : ",sourceImageMean, " Input Image Mean : ",inputImageMean)
    (Xmax,Ymax) = inputImage.shape
    for x in range(Xmax):
        for y in range(Ymax):
            #newValue = ((sourceImage.item(x,y) - sourceImageMean) * (inputImageStd/sourceImageStd)) + inputImageMean
            newValue = (inputImageStd * (sourceImage.item(x,y) - sourceImageMean) /sourceImageStd) + inputImageMean
            sourceImage.itemset((x,y),newValue)


#Function returns the index of the array where the closest value is present
def find_nearest(value,array):
    array = np.array(array)
    idx = (np.abs(array-value)).argmin()
    return idx

#Resize input images to 640x480
print("Using input image as ",sys.argv[1])
print("Using colored input source image as ",sys.argv[2])
#inputGrayImage = cv2.imread("inputGray.jpg",0)
inputGrayImage = cv2.imread(sys.argv[1],0)
inputGrayImage = cv2.resize(inputGrayImage,(640,480),0,0,interpolation = cv2.INTER_CUBIC)
#inputColorImage = cv2.imread("sourceImage.png")
inputColorImage = cv2.imread(sys.argv[2])
inputColorImage = cv2.resize(inputColorImage,(640,480),0,0,interpolation = cv2.INTER_CUBIC)

#Convert to Lab and separate components
convertedInputColorImage = cv2.cvtColor(inputColorImage,cv2.COLOR_BGR2Lab)
(sourceL,sourceA,sourceB) = cv2.split(convertedInputColorImage)

#Remap Luminance of source image to gray image
#print("Input Image : ",inputGrayImage.shape)
#print("Source Image: ",sourceL.shape)
luminanceRemap(sourceL,inputGrayImage)

#Currently using mean instead of std deviation for calculation of neightborhood statistics
averagingKernel = np.ones((3,3),np.uint8)/9
neighborhoodInfo = cv2.filter2D(inputGrayImage,-1,averagingKernel)

#Choose samples from source Image
pointsFromSource = [(random.randrange(480),random.randrange(640)) for x in range(200)]
intensitiesFromSource = []
for point in pointsFromSource:
    (x,y) = point
    intensitiesFromSource.append(sourceL[x,y])

#Find nearest match for each pixel and update output image with it's alpha/beta values
outputImage = np.zeros((480,640,3),np.uint8)
(Xmax,Ymax) = inputGrayImage.shape
#print(Xmax,Ymax)
for x in range(Xmax):
    for y in range(Ymax):
        weightedIntensity = (inputGrayImage.item(x,y) + neighborhoodInfo.item(x,y))/2
        index = find_nearest(weightedIntensity,intensitiesFromSource)
        outputImage.itemset((x,y,0),inputGrayImage.item(x,y))
        (sourceX,sourceY) = pointsFromSource[index]
        outputImage.itemset((x,y,1),sourceA.item(sourceX,sourceY))
        outputImage.itemset((x,y,2),sourceB.item(sourceX,sourceY))

outputImage = cv2.cvtColor(outputImage,cv2.COLOR_Lab2RGB)
#Calculate Histograms
inputGrayHist = cv2.calcHist([inputGrayImage],[0],None,[256],[0,255])
sourceHist = cv2.calcHist([sourceL],[0],None,[256],[0,255])

#Plot Images
plt.subplot(231),plt.imshow(inputGrayImage,'gray'),plt.title('Gray Input')
plt.subplot(233),plt.imshow(outputImage),plt.title('Colored Output')
inputColorImage = cv2.cvtColor(inputColorImage,cv2.COLOR_BGR2RGB)
plt.subplot(232),plt.imshow(inputColorImage),plt.title('Color Input')
plt.subplot(234),plt.plot(inputGrayHist),plt.title('Gray Image Histogram')
plt.subplot(235),plt.plot(sourceHist),plt.title('Source Luminance Histogram - after remapping')
plt.subplot(236),plt.imshow(sourceL,'gray'),plt.title('Luminance Remapped Source')
plt.show()
