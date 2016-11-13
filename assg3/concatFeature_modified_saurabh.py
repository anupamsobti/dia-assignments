#!/Users/saurbhtewari/.virtualenvs/cv3/bin/python
import cv2
import numpy as np
import sys
import colorsys


gaussian_5x5 = np.array([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.
gaussian_12  = np.array([1,4,6,4,1,4,16,24,16,4,6,24])/256.
gaussian_3x3 = np.array([0.077847,0.123317,0.077847,0.123317,0.195346,0.123317,0.077847,0.123317,0.077847,0.077847,0.123317,0.077847,0.123317,0.195346,0.123317,0.077847,0.123317,0.077847])
coherence_param_k = 5

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
    
def convertToYIQ(img):
    img = img/255
    Y,I,Q = colorsys.rgb_to_yiq(img[:,:,2],img[:,:,1],img[:,:,0])
    return Y,I,Q

def train_PyramidData(A_pyramid,l):
    imgA = A_pyramid[l]

    noOfRows,noOfColumns = imgA.shape
    trainData = []
    responses = []
    for i in range(noOfRows):
        for j in range(noOfColumns):
            trainData.append(imgA[i][j])
            responses.append(noOfRows*i+j)
    
    trainData = np.asarray(trainData).astype(np.float32)
    responses = np.asarray(responses).astype(np.float32)
    
    knn = cv2.ml.KNearest_create()
    knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
    
    return knn


def best_approximate_match_knn(knn,A_pyramid,B_pyramid,l,i,j):
    imgA = A_pyramid[l]

    noOfRows,noOfColumns = imgA.shape
    pixelValue = B_pyramid[l][i][j]
    newcomer = np.array(pixelValue).astype(np.float32)
    ret, results, neighbours ,dist = knn.findNearest(newcomer, 1)
    
    (yNearest,xNearest) = divmod(results,noOfRows)
 
    yNearest = int(yNearest)
    xNearest = int(xNearest)

    if(xNearest >= noOfColumns):
        xNearest = noOfColumns-1
    if(yNearest >= noOfRows):
        yNearest = noOfRows-1

    return yNearest,xNearest
    
def best_approximate_match(A_featurePyramid,B_featurePyramid,l,i,j):
#concerned with only A_feature_l and B_feature_l at level l. 
#Find the nearest distance neighbour and return its coordinates

    F_q = B_featurePyramid[l][i,j]
    h,w = A_featurePyramid[l].shape[0:2]
    minDistance = 100000

    for x in range(2,h-2):
        for y in range(2,w-2):
            F_p = A_featurePyramid[l][x,y]
            distance =  np.linalg.norm(F_p-F_q)
            if(distance < minDistance):
                minDistance = distance
                minDist_i,minDist_j = x,y
    return minDist_i,minDist_j

def concat_feature_new(X_pyramid,X_prime_pyramid,l,i,j,L):
   #feature_vector = X_l(5x5) + X_prime_l_F(12) + X_l-1_F(3x3) + X_prime_l-1_F(3x3)
   #l=L-1 is the coarsest resolution and l= 0 is the highest resolution level
   #l=current resolution level, l+1 will be coarser level.
   #concatenate l and l+1 level feature only, if l < L-1 OR l <= L-2, otherwise only one level need to be returned
   #total element = 55 elements

    feature_vector = np.zeros(25+12+9+9,np.float32)
    index = 0
    #i,j is the pixel being synthesized. So neigbourhood in X_prime will be valid upto i,j-1
    
    w,h = X_pyramid[l].shape
   #for main image at level = l, 25 pixels
    for x in range(i-2,i+3):
        for y in range(j-2,j+3):
            if x>=0 and x<=w-1 and y>=0 and y<=h-1:
                feature_vector[index] = X_pyramid[l][x,y]
            index = index + 1

    feature_vector[0:25] = feature_vector[0:25]*gaussian_5x5

    w,h = X_prime_pyramid[l].shape
    #For prime image at level = l, 12 pixels
    for x in range(i-2,i+1):
        for y in range(j-2,j+3):
            if(x == i and y >= j):
               break 
            if x>=0 and x<=w-1 and y>=0 and y<=h-1:
               feature_vector[index] = X_prime_pyramid[l][x,y]
            index = index + 1

    feature_vector[25:37] = feature_vector[25:37]*gaussian_12

   #for level l+1 (coarser level), since it is synthesised completely we will consider all 3x3 neighbourhood of i/2,j/2
    i_prev,j_prev = np.floor(i/2).real.astype(int),np.floor(j/2).real.astype(int)
    if(l<L-1):
        w,h = X_pyramid[l+1].shape
        for x in range(i_prev-1,i_prev+2):
            for y in range(j_prev-1,j_prev+2):
                if x>=0 and x<=w-1 and y>=0 and y<=h-1:
                    feature_vector[index] = X_pyramid[l+1][x,y]
                    feature_vector[index+1] = X_prime_pyramid[l+1][x,y]
                index = index + 2

    feature_vector[37:55] = feature_vector[37:55] * gaussian_3x3

    return feature_vector

def best_coherence_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,S_pyramid,l,i,j,L,pi_app,pj_app):
    h,w = A_pyramid[l].shape
    
    #intialize with some value
    final_x,final_y = pi_app,pj_app

    #feature vector near pixel q
    F_q = concat_feature_new(B_pyramid, B_prime_pyramid,l, i, j, L)
    minDistance = 100000
    for r_x in range(i-2,i+1):
        for r_y in range(j-2,j+3):
            if(r_x == i and r_y >= j):
                break
            src_x,src_y = S_pyramid[l][r_x,r_y]
            #if neigbour pixel r is not synthesised i.e. source(r) is not available, don't process further
            if(src_x == 0 and src_y == 0):
                continue
            else:
                src_x = src_x + (i - r_x)
                src_y = src_y + (j - r_y)

            if(src_x < 2 or src_x>(h-3) or src_y<2  or src_y>(w-3)):
                continue

            #print("nbr r:",r_x,r_y,"srcMapping s(r):",src_x,src_y)
            F_p = concat_feature_new(A_pyramid, A_prime_pyramid,l,src_x, src_y, L)
            distance = np.linalg.norm(F_p - F_q)
            if(distance < minDistance):
                minDistance = distance
                rstar_x,rstar_y = r_x,r_y
                final_x,final_y = src_x,src_y
                 
    return final_x,final_y


def best_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,S_pyramid,l,i,j,L,knn):

    #print("**   for pixel q:    ***",i,j)
    #pi_app,pj_app = best_approximate_match(A_featurePyramid,B_featurePyramid,l,i,j)
   
    pi_app,pj_app = best_approximate_match_knn(knn,A_pyramid,B_pyramid,l,i,j)
    h,w = B_pyramid[l].shape
    #print("best_approx_match p_app:",pi_app,pj_app)

    #for coarsest level and for border 2 rows and 2 columns only ANN is considered, 
    #as we don't have S_pyramid ready for that
    if( (i<2) or (i>(h-3)) or (j<2) or (j>(w-3)) or (l == L-1)):
        #print("best ANN ret",pi_app,pj_app)
        return pi_app,pj_app
    

    pi_coh,pj_coh = best_coherence_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,S_pyramid,l,i,j,L,pi_app,pj_app)
    #print("best_coherent match: r_star:",pi_coh,pj_coh)

    #concatenation of features from source or target images at level l and level l-1
    app_Fp = concat_feature_new(A_pyramid,A_prime_pyramid,l,pi_app,pj_app,L)
    coh_Fp = concat_feature_new(A_pyramid,A_prime_pyramid,l,pi_coh,pj_coh,L)
    #need to prepare B_prime_feature 
    Fq = concat_feature_new(B_pyramid,B_prime_pyramid,l,i,j,L)
    
    #AFeature(p_app) - BFeature(q)

    d_app = np.linalg.norm(app_Fp - Fq)
    d_coh = np.linalg.norm(coh_Fp - Fq)
     
    if d_coh <= d_app * (1 + (2**(-l))*coherence_param_k):
        #print("best_match: coh",pi_coh,pj_coh)
        return(pi_coh,pj_coh)
    else:
        #print("best_match: approx",pi_app,pj_app)
        return(pi_app,pj_app)

    
def createImageAnalogy(A_Y,A_PRIME_Y,B_Y,B_PRIME_Y,S_Y):
    
    
############ pyramid construction from Y images i.e one dimensional ####################
#pyramids for A_Y, B_Y, A_PRIME_Y, B_PRIME_Y , S

    G_A = A_Y.copy()
    G_A_prime = A_PRIME_Y.copy()
    G_B = B_Y.copy()
    G_B_prime = B_PRIME_Y.copy()
    G_s = S_Y.copy()

    A_pyramid = [G_A]
    A_prime_pyramid = [G_A_prime]
    B_pyramid = [G_B]
    B_prime_pyramid = [G_B_prime]
    S_pyramid = [G_s]

    hA,wA = G_A.shape
   #for i in range(6):a
    while ((hA >= 20) and (wA >= 20)):
        G_A = cv2.pyrDown(G_A)
        A_pyramid.append(G_A)

        G_A_prime = cv2.pyrDown(G_A_prime)
        A_prime_pyramid.append(G_A_prime)

        hB,wB = G_B.shape
        G_B = cv2.pyrDown(G_B)
        B_pyramid.append(G_B)

        G_B_prime = cv2.pyrDown(G_B_prime)
        B_prime_pyramid.append(G_B_prime)

        G_s = cv2.pyrDown(G_s)
        S_pyramid.append(G_s)
  
        hA,wA = G_A.shape

    L = len(A_pyramid)

    #from l = (L-1) to 0
    #from coarsest resolution (L-1) to finest resolution (L=0)

    for l in range(L-1,-1,-1):
        h,w = B_pyramid[l].shape
        knn = train_PyramidData(A_pyramid,l)
        for i in range(0,h):
            for j in range(0,w):
                best_i,best_j = best_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,S_pyramid,l,i,j,L,knn)
                S_pyramid[l][i,j] = (best_i,best_j)
                B_prime_pyramid[l][i,j] = A_prime_pyramid[l][best_i,best_j]
    return S_pyramid[0]


def imageAnalogy(A_RGB,A_PRIME_RGB,B_RGB):



    ### return values ####
    B_PRIME_RGB = np.zeros(B_RGB.shape,dtype=np.float32)
    B_PRIME_RGB_SAVE = np.zeros(B_RGB.shape,dtype=np.float32)
    
    ######### convert each image to YIQ format ###########
    A_Y = np.zeros(A_RGB.shape[0:2], dtype=np.float32)
    A_PRIME_Y = np.zeros(A_PRIME_RGB.shape[0:2], dtype=np.float32)
    B_Y = np.zeros(B_RGB.shape[0:2], dtype=np.float32)

    A_I = np.zeros(A_RGB.shape[0:2], dtype=np.float32)
    A_PRIME_I = np.zeros(A_PRIME_RGB.shape[0:2], dtype=np.float32)
    B_I = np.zeros(B_RGB.shape[0:2], dtype=np.float32)

    A_Q = np.zeros(A_RGB.shape[0:2], dtype=np.float32)
    A_PRIME_Q = np.zeros(A_PRIME_RGB.shape[0:2], dtype=np.float32)
    B_Q = np.zeros(B_RGB.shape[0:2], dtype=np.float32)

    B_PRIME_Y = np.zeros(B_RGB.shape[0:2],dtype=np.float32)
    
    h,w = B_RGB.shape[0:2]
    S_Y = np.zeros((h,w,2),dtype=np.uint16)
             

    A_Y,A_I,A_Q = convertToYIQ(A_RGB)
    B_Y,B_I,B_Q = convertToYIQ(B_RGB)
    A_PRIME_Y,A_PRIME_I,A_PRIME_Q=convertToYIQ(A_PRIME_RGB)

    luminanceRemap(A_Y,B_Y)
    luminanceRemap(A_PRIME_Y,B_Y)

    s_map = createImageAnalogy(A_Y,A_PRIME_Y,B_Y,B_PRIME_Y,S_Y)
    #from last stage of pyramid,copy the calculated feature
    h,w = B_RGB.shape[0:2]
    for i in range(h):
        for j in range(w):
            source = s_map[i,j]
            y = A_PRIME_Y[source[0],source[1]]
            ii = B_I[i,j]
            q = B_Q[i,j]
            #i = A_PRIME_I[source[0],source[1]]
            #q = A_PRIME_Q[source[0],source[1]]            
            r,g,b = colorsys.yiq_to_rgb(y,ii,q)
            B_PRIME_RGB[i,j] = b,g,r
            B_PRIME_RGB_SAVE[i,j] = B_PRIME_RGB[i,j]*255

            
    return B_PRIME_RGB,B_PRIME_RGB_SAVE


 
#Callback for selecting swatches
def inputFaceMouseCallback(event,x,y,flags,param):
    global inputSwatches,inputFaceImage_swatched,startX,startY,endX,endY,minRegionX,minRegionY
    if event == cv2.EVENT_LBUTTONDOWN:
        (startX,startY) = (x,y)
        print ("Input Face: Starts at ",startX,startY)
    elif event == cv2.EVENT_LBUTTONUP:
        endX,endY = x,y
        print ("Input Face: Ends at ",x,y)
        if ((endX - startX > minRegionX) and (endY - startY > minRegionY)):
            #inputSwatches.append((startX,endX,startY,endY))
            inputSwatches.append((startY,endY,startX,endX))  #Corrected to swap x and y since mouse callback returns swapped x and y
            cv2.rectangle(inputFaceImage_swatched,(startX,startY),(endX,endY),0,3)
        else:
            print("Swatch Rejected")

def modifyingFaceMouseCallback(event,x,y,flags,param):
    global modifyingSwatches,modifyingFaceImage_swatched,startX,startY,endX,endY,minRegionX,minRegionY
    if event == cv2.EVENT_LBUTTONDOWN:
        (startX,startY) = (x,y)
        print ("Modifying Face: Starts at ",startX,startY)
    elif event == cv2.EVENT_LBUTTONUP:
        endX,endY = x,y
        print ("Modifying Face: Ends at ",x,y)
        if ((endX - startX > minRegionX) and (endY - startY > minRegionY)):
            #modifyingSwatches.append((startX,endX,startY,endY))
            modifyingSwatches.append((startY,endY,startX,endX)) #Corrected to swap x and y since mouse callback returns swapped x and y
            cv2.rectangle(modifyingFaceImage_swatched,(startX,startY),(endX,endY),(255,0,0),3)
        else:
            print("Swatch rejected")


minRegionX = 10
minRegionY = 10
inputSwatches = []
modifyingSwatches = []

######### reading images in RGB format ###############
#read images in RGB format
A = cv2.imread(sys.argv[1])
A_PRIME = cv2.imread(sys.argv[2])
B = cv2.imread(sys.argv[3])


inputFaceImage = A
modifyingFaceImage = B

inputFaceImage_swatched = inputFaceImage.copy()
modifyingFaceImage_swatched = modifyingFaceImage.copy()

#Creating windows to display input image and source image
cv2.namedWindow("Face to be modified")
cv2.setMouseCallback("Face to be modified",modifyingFaceMouseCallback)

cv2.namedWindow("Example face")
cv2.setMouseCallback("Example face",inputFaceMouseCallback)

while(1):
    cv2.imshow("Face to be modified",modifyingFaceImage_swatched)
    cv2.imshow("Example face",inputFaceImage_swatched)
    if cv2.waitKey(20) & 0xFF == 27:
        if len(modifyingSwatches) != len(inputSwatches):
            print("Number of swatches is unequal")
            cv2.destroyAllWindows()
            exit()
        break

sampleOutputImage = A_PRIME
outputImage = B.copy()

for i,swatch in enumerate(inputSwatches):
    startY,endY,startX,endX = swatch
    startmY,endmY,startmX,endmX = modifyingSwatches[i]

    outputImage[startmY:endmY,startmX:endmX,:] = imageAnalogy(inputFaceImage[startY:endY,startX:endX,:],sampleOutputImage[startY:endY,startX:endX,:],modifyingFaceImage[startmY:endmY,startmX:endmX,:])[1]


cv2.imwrite("finalOutput.png",outputImage)
cv2.imshow('Created Image',outputImage)
cv2.imshow("Sample Output",A_PRIME)

cv2.waitKey(0)

