import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import pickle

#saving the np array
from tempfile import TemporaryFile



detector = cv2.SIFT()        #initialize detector
path = 'C:\\Users\\NISHANT\\Desktop\\Sem 2\\MP\\Project\\Code\\dataset\\'

feat = []
yt = [] 
n = 0
dt ={}

#Getting SIFT features of the images and then appending all
print "Training image of images to get SIFT " 
Images = [ f for f in listdir(path) if isfile(join(path,f)) ]    #store all image of folder
for j in range(0, len(Images)):
	print join( path, Images[n])
	n = n + 1
	tempImage = cv2.imread( join( path, Images[j]) )
	kp, des = detector.detectAndCompute(tempImage,None)     #detect features 
	feat.append(des)

#dumping the SIFT features to pickle file	
filekmeans = open('imagename.pickle','wb')
pickle.dump(Images,filekmeans)
filedesc = open('sift.pickle','wb')
ft = np.array(feat)
print ft.shape
pickle.dump(feat,filedesc)
descriptor = feat[0]

#stacking all the descriptors into a single array
for i in range(0, len(feat)):
	print feat[i].shape
	descriptor = np.vstack((descriptor, feat[i]))

#Perform Kmeans on descriptor array to get the cluster centers
print "K Means clustering"
k = 500
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retval, bestLabels, centers= cv2.kmeans(descriptor,k,criteria,10,0)

#dumping into kmeans
filekmeans = open('kmeans.pickle','wb')
pickle.dump(centers,filekmeans)


#Creating Histogram of clustered features
print "Creating Histogram"
X = np.zeros((n, k),dtype = np.float32)
for i in xrange(n):
	print feat[i]
	words, distance = vq(feat[i],centers)       #vector quantization
	for w in words:
		X[i][w] += 1          #bag-of-visual-words representation[]
		
filedoc = open('doc-word.pickle','wb')
pickle.dump(X,filedoc)


