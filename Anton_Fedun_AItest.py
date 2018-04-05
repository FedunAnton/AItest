
#Problem ¹2 for AItest
#Image Compression
#By Anton Fedun

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
from scipy import ndimage

fname = "nature.jpg" #works with 512x512 images
N = 7                #number of colors in compressed image
max_iters = 7        #the bigger - the most likely data fit
num_px = 512

def initCentroids(X, N):
    ran = np.random.permutation(X)
    centroids = np.zeros((N, 3))
    for i in range(N):
        centroids[i] = ran[i]
    return centroids

def load_data(fname):
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px,3))
    X = my_image / 255    
    return X

def findClosestCentroids(X, centroids):
    m = X.shape[0]
    idx = np.zeros((m, 1))
    dist = 0
    for i in range(m):
        minim = np.inf
        for k in range(N):
            diff = np.ones(3)
            diff = X[i, :] - centroids[k, :]
            dist = np.dot(diff.T, diff)
            if minim > dist:
                minim = dist
                idx[i] = k
    return idx

def computeCentroids(X, idx):
    m = X.shape[0]
    n = X.shape[1]
    centroids = np.zeros((N, n))
    for k in range(N):
        summ = np.zeros((n, 1))
        numk = 0
        for i in range(m):
            if k == idx[i]:
                summ = summ + X[i, :].T
                numk = numk + 1      
        centroids[k, :] = (summ[1, :] / numk).T        
    return centroids

def runKMeans(X, init_centroids):
    m = X.shape[0]
    n = X.shape[1]
    centroids = init_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    for i in range(max_iters):
        print("Iteration ", i + 1, " of ", max_iters)
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx)
    return centroids, idx

def getImage(X, centroids):
    idx = findClosestCentroids(X, centroids)
    X_recov = np.zeros((X.shape))
    for i in range(X.shape[0]):
        X_recov[i, :] = centroids[idx[i].astype(int), :]
    X_recov = np.reshape(X_recov, (num_px, num_px, 3))
    X_recov = X_recov * 255
    X_recov = X_recov.astype(int)
    return X_recov

print("Loading data")
X = load_data(fname)

print("Get initial centroids")
init_centroids = initCentroids(X, N)

print("Running K-Means")
centroids, idx = runKMeans(X, init_centroids)

print("Compressing image")
X_recov = getImage(X, centroids)
scipy.misc.toimage(X_recov, cmin=0.0).save('nature_compressed.jpg')
print("Done!")