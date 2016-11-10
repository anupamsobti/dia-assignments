#!/home/saurabh/.virtualenvs/cv/bin/python
import cv2
import numpy as np
import sys
import colorsys


gaussian_5x5 = np.array([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.
gaussian_12  = np.array([1,4,6,4,1,4,16,24,16,4,6,24])/256.
gaussian_3x3 = np.array([0.077847,0.123317,0.077847,0.123317,0.195346,0.123317,0.077847,0.123317,0.077847,0.077847,0.123317,0.077847,0.123317,0.195346,0.123317,0.077847,0.123317,0.077847])
coherence_param_k = 15

######### reading images in RGB format ###############
#read images in RGB format
A_RGB = cv2.imread(sys.argv[1])
A_PRIME_RGB = cv2.imread(sys.argv[2])
B_RGB = cv2.imread(sys.argv[3])
B_PRIME_RGB = np.zeros(B_RGB.shape,dtype=np.float32)

######### convert each image to YIQ format and create Y images ###########
A_YIQ = np.zeros(A_RGB.shape, dtype=np.float32)
A_PRIME_YIQ = np.zeros(A_PRIME_RGB.shape, dtype=np.float32)
B_YIQ = np.zeros(B_RGB.shape, dtype=np.float32)

A_Y = np.zeros(A_RGB.shape[0:2], dtype=np.float32)
A_PRIME_Y = np.zeros(A_PRIME_RGB.shape[0:2], dtype=np.float32)
B_Y = np.zeros(B_RGB.shape[0:2], dtype=np.float32)


Aheight,Awidth = A_RGB.shape[0:2]
for i in range(Aheight):
    for j in range(Awidth):
        colors = A_RGB[i,j]/255.
        YIQ = colorsys.rgb_to_yiq(colors[0],colors[1],colors[2])
        A_YIQ[i,j] = YIQ
        A_Y[i,j] = YIQ[0]


APrimeHeight,APrimeWidth = A_PRIME_RGB.shape[0:2]
for i in range(APrimeHeight):
    for j in range(APrimeWidth):
        colors = A_PRIME_RGB[i,j]/255.
        YIQ = colorsys.rgb_to_yiq(colors[0],colors[1],colors[2])
        A_PRIME_YIQ[i,j] = YIQ
        A_PRIME_Y[i,j] = YIQ[0]

Bheight,Bwidth = B_RGB.shape[0:2]
for i in range(Bheight):
    for j in range(Bwidth):
        colors = B_RGB[i,j]/255.
        YIQ = colorsys.rgb_to_yiq(colors[0],colors[1],colors[2])
        B_YIQ[i,j] = YIQ
        B_Y[i,j] = YIQ[0]


def best_approximate_match_knn(A_pyramid,B_pyramid,l,i,j):
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
    newcomer = np.array(0).astype(np.float32)
    ret, results, neighbours ,dist = knn.findNearest(newcomer, 1)
    
    (yNearest,xNearest) = divmod(results,noOfRows)
    yNearest = int(yNearest)
    xNearest = int(xNearest)

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
    
   #for main image at level = l, 25 pixels
    for x in range(i-2,i+3):
        for y in range(j-2,j+3):
            #print(x,y)
            feature_vector[index] = X_pyramid[l][x,y]
            index = index + 1
    feature_vector[0:25] = feature_vector[0:25]*gaussian_5x5
    #For prime image at level = l, 12 pixels
    for x in range(i-2,i+1):
        for y in range(j-2,j+3):
            if(x == i and y >= j):
               break 
            feature_vector[index] = X_prime_pyramid[l][x,y]
            index = index + 1

    feature_vector[25:37] = feature_vector[25:37]*gaussian_12

   #for level l+1 (coarser level), since it is synthesised completely we will consider all 3x3 neighbourhood of i/2,j/2
    i_prev,j_prev = np.floor(i/2).real.astype(int),np.floor(j/2).real.astype(int)
    if(l<L-1):
        for x in range(i_prev-1,i_prev+2):
            for y in range(j_prev-1,j_prev+2):
                feature_vector[index] = X_pyramid[l+1][x,y]
                index = index + 1
                
                feature_vector[index] = X_prime_pyramid[l+1][x,y]
                index = index + 1
    feature_vector[37:55] = feature_vector[37:55] * gaussian_3x3

    return feature_vector

def best_coherence_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,S_pyramid,l,i,j,L):
    h,w = A_pyramid[l].shape
    
    #intialize with some value
    final_x,final_y = i,j

    #feature vector near pixel q
    F_q = concat_feature_new(B_pyramid, B_prime_pyramid,l, i, j, L)
    minDistance = 100000
    print("considering q",i,j)
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

            print("nbr r:",r_x,r_y,"srcMapping s(r):",src_x,src_y)
            F_p = concat_feature_new(A_pyramid, A_prime_pyramid,l,src_x, src_y, L)
            distance = np.linalg.norm(F_p - F_q)
            if(distance < minDistance):
                minDistance = distance
                rstar_x,rstar_y = r_x,r_y
                final_x,final_y = src_x,src_y
                 
    return final_x,final_y


def best_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,A_featurePyramid,
                                           Aprime_featurePyramid,B_featurePyramid,S_pyramid,l,i,j,L):

    print("for pixel q:",i,j)

    #pi_app,pj_app = best_approximate_match(A_featurePyramid,B_featurePyramid,l,i,j)
    pi_app,pj_app = best_approximate_match_knn(A_pyramid,B_pyramid,l,i,j)

    h,w = B_pyramid[l].shape
    print("best_approx_match p_app:",pi_app,pj_app)

    #for coarsest level only ANN to be considered, as we don't have S_pyramid ready for that 
    if( (i<2) or (i>(h-3)) or (j<2) or (j>(w-3)) or (l == L-1)):
        return pi_app,pj_app
    

    pi_coh,pj_coh = best_coherence_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,S_pyramid,l,i,j,L)
    print("best_coherent match: r_star:",pi_coh,pj_coh)

    #concatenation of features from source or target images at level l and level l-1
    app_Fp = concat_feature_new(A_pyramid,A_prime_pyramid,l,pi_app,pj_app,L)
    coh_Fp = concat_feature_new(A_pyramid,A_prime_pyramid,l,pi_coh,pj_coh,L)
    #need to prepare B_prime_feature 
    Fq = concat_feature_new(B_pyramid,B_prime_pyramid,l,i,j,L)
    
    #AFeature(p_app) - BFeature(q)
    d_app = np.linalg.norm(A_featurePyramid[l][pi_app,pj_app] - B_featurePyramid[l][i,j])
    d_coh = np.linalg.norm(A_featurePyramid[l][pi_coh,pj_coh] - B_featurePyramid[l][i,j])

    d_app = np.linalg.norm(app_Fp - Fq)
    d_coh = np.linalg.norm(coh_Fp - Fq)
     
    if d_coh <= d_app * (1 + (2**(l - L))*coherence_param_k):
        print("best_match: coh",pi_coh,pj_coh)
        return(pi_coh,pj_coh)
    else:
        print("best_match: approx",pi_app,pj_app)
        return(pi_app,pj_app)

    
def createImageAnalogy(A_Y,A_prime_Y,B_Y):

############ pyramid construction from Y images i.e one dimensional ####################
#pyramids for A_Y, B_Y, A_PRIME_Y, B_PRIME_Y , S

#first create blank image for Bprime and S
    B_PRIME_Y = np.zeros((Bheight,Bwidth),np.float32)
    S_Y = np.zeros((Bheight,Bwidth,2),np.uint16)


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


######### pre-computing feature for each pixel, at every level for A, Aprime and B 
######### creating a list of length L, for each level l. Each list have corresponding number of pixels as in pyramid above
######### with each pixel there will be a 5x5 = 25 neighbouring pixels feature data associated.
    A_featurePyramid = []
    Aprime_featurePyramid = []
    B_featurePyramid = []
    pixelNbrhoodFeatureSet = np.zeros(25,np.float32)

    for l in range(L):
        h,w = A_pyramid[l].shape
        level_feature = np.zeros([h,w,25],np.float32)
        for i in range(2,h-2):
            for j in range(2,w-2):
                #select the 5x5 neighbour around (i,j)
                index = 0
                for x in range(-2,3,1):
                    for y in range(-2,3,1):
                        #store in a linear array of 25 values
                        pixelNbrhoodFeatureSet[index] = A_pyramid[l][i+x,j+y]
                        index = index + 1
                #store this feature set at index (i,j)
                level_feature[i,j] = pixelNbrhoodFeatureSet * gaussian_5x5
        #attach in the feature pyramid
        A_featurePyramid.append(level_feature)

    for l in range(L):
        h,w = A_prime_pyramid[l].shape
        level_feature = np.zeros([h,w,25],np.float32)
        for i in range(2,h-2):
            for j in range(2,w-2):
                #select the 5x5 neighbour around (i,j)
                index = 0
                for x in range(-2,3,1):
                    for y in range(-2,3,1):
                        #store in a linear array
                        pixelNbrhoodFeatureSet[index] = A_prime_pyramid[l][i+x,j+y]
                        index = index + 1
                level_feature[i,j] = pixelNbrhoodFeatureSet * gaussian_5x5
        #attach in the feature pyramid
        Aprime_featurePyramid.append(level_feature)

    for l in range(L):
        h,w = B_pyramid[l].shape
        level_feature = np.zeros([h,w,25],np.float32)
        for i in range(2,h-2):
            for j in range(2,w-2):
                #select the 5x5 neighbour around (i,j)
                index = 0
                for x in range(-2,3,1):
                    for y in range(-2,3,1):
                        #store in a linear array
                        pixelNbrhoodFeatureSet[index] = B_pyramid[l][i+x,j+y]
                        index = index + 1
                level_feature[i,j] = pixelNbrhoodFeatureSet * gaussian_5x5
        #attach in the feature pyramid list
        B_featurePyramid.append(level_feature)

    #from l = (L-1) to 0
    #from coarsest resolution to finest resolution
    #for l in range(L-1,-1,-1):
    for l in range(L-1,L-5,-1):
        h,w = B_pyramid[l].shape
        #for i in range(2,h-2):
        #    for j in range(2,w-2):
        for i in range(0,h):
            for j in range(0,w):
                best_i,best_j = best_match(A_pyramid,A_prime_pyramid,B_pyramid,B_prime_pyramid,A_featurePyramid,
                                           Aprime_featurePyramid,B_featurePyramid,S_pyramid,l,i,j,L)
                S_pyramid[l][i,j] = (best_i,best_j)
                #print("sourceMapping",i,j,S_pyramid[l][i,j])
                B_prime_pyramid[l][i,j] = A_prime_pyramid[l][best_i,best_j]
        #print(S_pyramid[l])

    #from last stage of pyramid,copy the calculated feature
    h,w = B_pyramid[L-1].shape
    for i in range(h):
        for j in range(w):
            B_PRIME_Y[i,j] = B_prime_pyramid[L-1][i,j]

    return B_PRIME_Y



B_PRIME_Y = createImageAnalogy(A_Y,A_PRIME_Y,B_Y)

for i in range(Bheight):
    for j in range(Bwidth):
        B_PRIME_RGB[i,j] = colorsys.yiq_to_rgb(B_PRIME_Y[i,j],B_YIQ[i,j,1],B_YIQ[i,j,2])


cv2.imshow('TPrime',B_PRIME_RGB)
cv2.waitKey(0)

