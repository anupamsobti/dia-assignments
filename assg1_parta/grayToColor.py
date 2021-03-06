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
    (Xmax,Ymax) = sourceImage.shape
    for x in range(Xmax):
        for y in range(Ymax):
            #newValue = ((sourceImage.item(x,y) - sourceImageMean) * (inputImageStd/sourceImageStd)) + inputImageMean
            newValue = (inputImageStd * (sourceImage.item(x,y) - sourceImageMean) /sourceImageStd) + inputImageMean
            sourceImage.itemset((x,y),newValue)

def stdDeviationFilter(img):
    #Standard deviation = E[X^2] - E^2[X]
    #averagingKernel = np.ones((5,5),np.uint8)/25
    averagingKernel = np.ones((10,10),np.uint8)/100
    meanImage = cv2.filter2D(img,-1,averagingKernel)
    meanImageSquare = meanImage * meanImage
    meanOfImageSquare = cv2.filter2D(img*img,-1,averagingKernel)
    stdDeviationImage = np.sqrt(np.abs(meanOfImageSquare - meanImageSquare))
    return stdDeviationImage

#Function returns the index of the array where the closest value is present
def find_nearest(value,array):
    array = np.array(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def transferColor(inputGrayImage,sourceImage,outputImage,grayBoundary,sourceBoundary):
    (sourceL,sourceA,sourceB) = sourceImage
    (gray_xmin,gray_xmax,gray_ymin,gray_ymax) = grayBoundary
    (sourcexmin,sourcexmax,sourceymin,sourceymax) = sourceBoundary
    #remap luminance for the swatch
    luminanceRemap(sourceL[sourcexmin:sourcexmax,sourceymin:sourceymax],inputGrayImage[gray_xmin:gray_xmax,gray_ymin:gray_ymax])
    #calculate std deviation for neighborhood of input gray image
    neighborhoodInfo = stdDeviationFilter(inputGrayImage[gray_xmin:gray_xmax,gray_ymin:gray_ymax])
    #Choose samples from source Image
    pointsFromSource = [(random.randrange(sourcexmin,sourcexmax),random.randrange(sourceymin,sourceymax)) for x in range(200)]
    intensitiesFromSource = []
    #calculate std deviation of samples in source
    stdDeviationOfSource = stdDeviationFilter(sourceL[sourcexmin:sourcexmax,sourceymin:sourceymax])
    #populate a list of intensitiesFromSource which contains weighted sum of std deviation and luminance of randomly chosen points
    for point in pointsFromSource:
        (x,y) = point
        intensitiesFromSource.append(sourceL[x,y]/2 + stdDeviationOfSource[x,y]/2)
    #Find matching points and assign colors
    for x in range(gray_xmin,gray_xmax):
        for y in range(gray_ymin,gray_ymax):
            weightedIntensity = (inputGrayImage.item(x,y) + neighborhoodInfo.item(x,y))/2
            index = find_nearest(weightedIntensity,intensitiesFromSource)
            outputImage.itemset((x,y,0),inputGrayImage.item(x,y))
            (sourceX,sourceY) = pointsFromSource[index]
            outputImage.itemset((x,y,1),sourceA.item(sourceX,sourceY))
            outputImage.itemset((x,y,2),sourceB.item(sourceX,sourceY))




#Resize input images to 640x480
print("Using input image as ",sys.argv[1])
print("Using colored input source image as ",sys.argv[2])
inputGrayImage = cv2.imread(sys.argv[1],0)
inputGrayImage = cv2.resize(inputGrayImage,(640,480),0,0,interpolation = cv2.INTER_CUBIC)
inputColorImage = cv2.imread(sys.argv[2])
inputColorImage = cv2.resize(inputColorImage,(640,480),0,0,interpolation = cv2.INTER_CUBIC)

#Convert to Lab and separate components
convertedInputColorImage = cv2.cvtColor(inputColorImage,cv2.COLOR_BGR2Lab)
(sourceL,sourceA,sourceB) = cv2.split(convertedInputColorImage)

#Define a black outputImage
outputImage = np.zeros((480,640,3),np.uint8)

transferColor(inputGrayImage,(sourceL,sourceA,sourceB),outputImage,(0,479,0,639),(0,479,0,639))


outputImage = cv2.cvtColor(outputImage,cv2.COLOR_Lab2RGB)
#Calculate Histograms
inputGrayHist = cv2.calcHist([inputGrayImage],[0],None,[256],[0,255])
sourceHist = cv2.calcHist([sourceL],[0],None,[256],[0,255])

#Plot Images
#plt.subplot(231),plt.imshow(stdDeviationFilter(inputGrayImage),'gray'),plt.title('Gray Input')
plt.subplot(231),plt.imshow(inputGrayImage,'gray'),plt.title('Gray Input')
plt.subplot(233),plt.imshow(outputImage),plt.title('Colored Output')
inputColorImage = cv2.cvtColor(inputColorImage,cv2.COLOR_BGR2RGB)
plt.subplot(232),plt.imshow(inputColorImage),plt.title('Color Input')
plt.subplot(234),plt.plot(inputGrayHist),plt.title('Gray Image Histogram')
plt.subplot(235),plt.plot(sourceHist),plt.title('Source Luminance Histogram - after remapping')
plt.subplot(236),plt.imshow(sourceL,'gray'),plt.title('Luminance Remapped Source')
plt.show()
