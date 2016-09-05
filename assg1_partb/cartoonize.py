#!/usr/bin/python3
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import sys

inputImage = cv2.imread(sys.argv[1],0)

def myThreshold(img,eps,sai):
    Xmax,Ymax = img.shape
    outputImage = np.zeros((Xmax,Ymax),np.uint8)
    for x in range(Xmax):
        for y in range(Ymax):
            if img[x,y] < eps:
                outputImage[x,y] = 255
            else:
                outputImage[x,y] = (1 + np.tanh(sai * img[x,y]))*256
    return outputImage
                
def invert(img):
    return (255-img)

def nothing(value):
    #print(value)
    pass


#'Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst'
#inputImage = cv2.Laplacian(inputImage,cv2.CV_8U,5)
#inputImage = cv2.Laplacian(inputImage,cv2.CV_16S,5)
sigma = 10
sizeOfKernel=5
k = 3
tao = 0.95
eps = 0
sai = 1

#Create new window
cv2.namedWindow('image')

# create trackbars for different parameters
cv2.createTrackbar('sigma','image',0,255,nothing)
cv2.createTrackbar('eps','image',0,255,nothing)
cv2.createTrackbar('sai','image',0,255,nothing)
cv2.createTrackbar('k','image',0,255,nothing)
cv2.createTrackbar('tao','image',0,255,nothing)

while(1):
    # get current positions of four trackbars
    sigma = cv2.getTrackbarPos('sigma','image')/15
    k = cv2.getTrackbarPos('k','image')/15
    eps = cv2.getTrackbarPos('eps','image') - 128
    sai = cv2.getTrackbarPos('sai','image')/50
    tao = cv2.getTrackbarPos('tao','image')/50

    gaussian1 = cv2.GaussianBlur(inputImage,(sizeOfKernel,sizeOfKernel),sigmaX = sigma,sigmaY = sigma)
    gaussian2 = cv2.GaussianBlur(inputImage,(sizeOfKernel,sizeOfKernel),sigmaX = k*sigma,sigmaY = k*sigma)
    
    DoG_image = gaussian1 - tao*gaussian2
    
    outputImage = myThreshold(DoG_image,eps,sai)
    outputImage = invert(outputImage)
    cv2.imshow('image',outputImage)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break







##MATPLOTLIB Plotting
##ret,inputImage2 = cv2.threshold(inputImage,25,255,cv2.THRESH_BINARY_INV)
##ret,inputImage = cv2.threshold(inputImage,25,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
##cv2.convertScaleAbs(inputImage,inputImage)
##inputImage = cv2.cvtColor(inputImage,cv2.COLOR_BGR2RGB)
#plt.subplot(221),plt.imshow(inputImage,"gray"),plt.title("Input Image")
#plt.subplot(222),plt.imshow(gaussian1,"gray"),plt.title("Gaussian 1")
#plt.subplot(223),plt.imshow(gaussian2,"gray"),plt.title("Gaussian 2")
#plt.subplot(224),plt.imshow(outputImage,"gray"),plt.title("Output (DoG + Thresholding)")
#plt.show()
