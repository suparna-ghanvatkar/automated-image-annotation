import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import pickle


detector = cv2.SIFT()        #initialize detector
 
path = 'C:\\Users\\NISHANT\\Desktop\\Sem 2\\MP\\Project\\Code\\iart\\00\\' 

feat = []
yt = [] 
n = 0


	#Getting images of class (i+1)
print "Training image of images to get SIFT " 
Images = [ f for f in listdir(path) if isfile(join(path,f)) ]    #store all image of folder
for j in range(0, len(Images)):
	print join( path, Images[n])
	n = n + 1
	tempImage = cv2.imread( join( path, Images[j]) )
	kp, des = detector.detectAndCompute(tempImage,None)     #detect features 
	feat.append(des)

descriptor = feat[0]
for i in range(0, len(feat)):
	descriptor = np.vstack((descriptor, feat[i]))
	#print descriptor.shape
#print descriptor.shape, y.shape, n
filedesc = open('sift.pickle','w')
pickle.dump(descriptor,filedesc)

#Perform Kmeans on descriptor array
print "K Means clustering"
k = 100
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retval, bestLabels, centers= cv2.kmeans(descriptor,k,criteria,10,0)

filekmeans = open('kmeans.pickle','w')
pickle.dump(centers,filekmeans)

#Creating Histogram of clustered features
print "Creating Histogram"
X = np.zeros((n, k),dtype = np.float32)
for i in xrange(n):
	print feat[i]
	words, distance = vq(feat[i],centers)       #vector quantization
	for w in words:
		X[i][w] += 1          #bag-of-visual-words representation[]
		
filedoc = open('doc-word.pickle','w')
pickle.dump(X,filedoc)


