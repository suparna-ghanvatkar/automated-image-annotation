import numpy,cv2
import pickle

#File to get the SIFT features for tesing purpose for a single image
path = 'C:\\Users\\NISHANT\\Desktop\\Sem 2\\MP\\Project\\Code\\iart222\\New folder\\739.jpg'

img = cv2.imread(path)
detector = cv2.SIFT()
kp, des = detector.detectAndCompute(img, None)

pfile = open('testimg.pickle','wb')
pickle.dump(des, pfile)
pfile.close()