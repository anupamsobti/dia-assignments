#!/usr/bin/python3
import cv2
import numpy as np
import random
import sys
#from matplotlib import pyplot as plt

#Constants/Global variables
minRegionX = 10
minRegionY = 10
graySwatches = []
colorSwatches = []

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

#Makes the output image colored based on inputs from sample image
def transferColor(inputGrayImage,sourceImage,outputImage,grayBoundary,sourceBoundary):
    global neighborhoodInfo,stdDeviationOfSource
    (sourceL,sourceA,sourceB) = sourceImage
    (gray_xmin,gray_xmax,gray_ymin,gray_ymax) = grayBoundary
    (sourcexmin,sourcexmax,sourceymin,sourceymax) = sourceBoundary
    print("Gray Boundary : ",grayBoundary) #Debug
    print("Source Boundary: ",sourceBoundary)   #Debug
    #remap luminance for the swatch
    luminanceRemap(sourceL[sourcexmin:sourcexmax,sourceymin:sourceymax],inputGrayImage[gray_xmin:gray_xmax,gray_ymin:gray_ymax])

    #Choose samples from source Image
    pointsFromSource = [(random.randrange(sourcexmin,sourcexmax),random.randrange(sourceymin,sourceymax)) for x in range(200)]
    #print(pointsFromSource[1:10])
    intensitiesFromSource = []

    #populate a list of intensitiesFromSource which contains weighted sum of std deviation and luminance of randomly chosen points
    for point in pointsFromSource:
        (x,y) = point
        #print(point)    #Debug
        #print("Shape of source : ",sourceL.shape," Source of Std deviation : ",stdDeviationOfSource.shape)
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

#Callback for selecting swatches
def grayMouseCallback(event,x,y,flags,param):
    global graySwatches,inputGrayImage_swatched,startX,startY,endX,endY,minRegionX,minRegionY
    if event == cv2.EVENT_LBUTTONDOWN:
        (startX,startY) = (x,y)
        print ("Grayscale: Starts at ",startX,startY)
    elif event == cv2.EVENT_LBUTTONUP:
        endX,endY = x,y
        print ("Grayscale: Ends at ",x,y)
        if ((endX - startX > minRegionX) and (endY - startY > minRegionY)):
            #graySwatches.append((startX,endX,startY,endY))
            graySwatches.append((startY,endY,startX,endX))  #Corrected to swap x and y since mouse callback returns swapped x and y
            cv2.rectangle(inputGrayImage_swatched,(startX,startY),(endX,endY),0,3)
        else:
            print("Swatch Rejected")

def colorMouseCallback(event,x,y,flags,param):
    global colorSwatches,inputColorImage_swatched,startX,startY,endX,endY,minRegionX,minRegionY
    if event == cv2.EVENT_LBUTTONDOWN:
        (startX,startY) = (x,y)
        print ("Colored Input: Starts at ",startX,startY)
    elif event == cv2.EVENT_LBUTTONUP:
        endX,endY = x,y
        print ("Colored Input: Ends at ",x,y)
        if ((endX - startX > minRegionX) and (endY - startY > minRegionY)):
            #colorSwatches.append((startX,endX,startY,endY))
            colorSwatches.append((startY,endY,startX,endX)) #Corrected to swap x and y since mouse callback returns swapped x and y
            cv2.rectangle(inputColorImage_swatched,(startX,startY),(endX,endY),(255,0,0),3)
        else:
            print("Swatch rejected")


#Resize input images to 640x480
print("Using input image as ",sys.argv[1])
print("Using colored input source image as ",sys.argv[2])
inputGrayImage = cv2.imread(sys.argv[1],0)
inputGrayImage = cv2.resize(inputGrayImage,(640,480),0,0,interpolation = cv2.INTER_CUBIC)
inputColorImage = cv2.imread(sys.argv[2])
inputColorImage = cv2.resize(inputColorImage,(640,480),0,0,interpolation = cv2.INTER_CUBIC)

inputGrayImage_swatched = inputGrayImage.copy()
inputColorImage_swatched = inputColorImage.copy()

#Convert to Lab and separate components
convertedInputColorImage = cv2.cvtColor(inputColorImage,cv2.COLOR_BGR2Lab)
(sourceL,sourceA,sourceB) = cv2.split(convertedInputColorImage)

#calculate std deviation for neighborhood of input gray image
neighborhoodInfo = stdDeviationFilter(inputGrayImage)

#calculate std deviation of samples in source
stdDeviationOfSource = stdDeviationFilter(sourceL)

#Define a black outputImage
outputImage = np.zeros((480,640,3),np.uint8)

#Creating windows to display input image and source image
cv2.namedWindow("Input Grayscale Image")
cv2.setMouseCallback("Input Grayscale Image",grayMouseCallback)

cv2.namedWindow("Input Colored Image")
cv2.setMouseCallback("Input Colored Image",colorMouseCallback)

#Get swatches
while(1):
    cv2.imshow("Input Grayscale Image",inputGrayImage_swatched)
    cv2.imshow("Input Colored Image",inputColorImage_swatched)
    if cv2.waitKey(20) & 0xFF == 27:
        if len(colorSwatches) != len(graySwatches):
            print("Number of swatches is unequal")
            cv2.destroyAllWindows()
            exit()
        break

#transferColor(inputGrayImage,(sourceL,sourceA,sourceB),outputImage,(0,479,0,639),(0,479,0,639))
for index,swatch in enumerate(graySwatches):
    print("Calling transferColor(inputGrayImage,(sourceL,sourceA,sourceB),outputImage,",swatch,colorSwatches[index],")")
    transferColor(inputGrayImage,(sourceL,sourceA,sourceB),outputImage,graySwatches[index],colorSwatches[index])

outputImage = cv2.cvtColor(outputImage,cv2.COLOR_Lab2BGR)
cv2.imshow("Output Image",outputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
