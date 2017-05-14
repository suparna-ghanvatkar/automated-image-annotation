# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:38:38 2017

@author: sups
"""
import numpy as np
from scipy.cluster.vq import *
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import re
import lda
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming
import cv2
import sklearn.metrics.pairwise

filev = open('./train/kmeans.pickle','rb')
filed = open('./train/doc-word.pickle', 'rb')
fileimgnames = open('./train/imagename.pickle','rb')
imagenms = pickle.load(fileimgnames)
k = 500
vocab = pickle.load(filev)
docs = pickle.load(filed)
docs = docs.astype(int)
#print len(imagenms)

'''
Preprocessing of text comments
'''

comments = []
txt_vocab = []
path = './iart/annot/'
#annots = [f for f in os.listdir(path)]
stemmer = WordNetLemmatizer()
for comm in imagenms:
    comm_nm = comm.split('.')[0] + '.eng'
    #print comm_nm
#    with open(path+comm,'r') as xml_file:
#        tree = ET.parse(xml_file)
#    root = tree.getroot()
#    for child in root.findall('DESCRIPTION'):
#        print child.text
    comm_file = open(path+comm_nm,'r')
    desc = []
    for line in comm_file:
        if 'DESCRIPTION' in line:
            cline =  filter(lambda x:x!='', re.split('[ <; ,>()]',line))
            #print  " ".join(cline[2:-2])
            #print cline
            cline = cline[1:-2]
            desc_sent = pos_tag(cline)
            cline = map(lambda(word, tag): word, filter(lambda (word, tag): tag == 'NN' or tag=='NNP', desc_sent))
            #print cline
            stoplist = stopwords.words('english')
            more_words = [u'background', u'foreground', u'left', u'right', u'top', u'middle']
            stoplist += more_words
            desc_line = [word for word in cline if word not in stoplist]
            desc_line = [stemmer.lemmatize(word) for word in desc_line]
            desc.extend(desc_line)
    #print len(comments)
    comments.append(desc)
for doc_comm in comments:
    for word in doc_comm:
        if word not in txt_vocab:
            txt_vocab.append(word)
#print txt_vocab
for i in range(0,len(comments)):
    for j in range(0, len(comments[i])):
        comments[i][j] = txt_vocab.index(comments[i][j])
    print i
#print len(comments)
#raw_input()
cdoc = np.zeros(shape=len(txt_vocab), dtype = 'int64')
print type(cdoc)
for word in comments[0]:
    cdoc[int(word)] = cdoc[int(word)]+1
doc_comments = cdoc
#print (doc_comments)
        
for i in range(1,len(comments)):
    cdoc = np.zeros(shape = len(txt_vocab), dtype = 'int64')
    for word in comments[i]:
        cdoc[int(word)] = cdoc[int(word)]+1
    doc_comments = np.vstack((doc_comments,cdoc))
#print doc_comments

comm_model = lda.LDA(n_topics = 10, n_iter=3000)
comm_model.fit(doc_comments)
'''
filecomm = open('./comments.txt','w')
filecomm.write(str(len(comments))+'\n')
for i in range(0,len(comments)):
    #print map(str,comments[i])
    #filecomm.write('1\n')
    comm_str = map(str,comments[i])
    for wrd in comm_str:
        filecomm.write(wrd+' ')
    filecomm.write('\n')
filecomm.close()
'''
comm_top_p = comm_model.doc_topic_
comm_imp_top_doc = list()
i = 0
for doc in comm_top_p: 
    dmin = np.min(doc)
    dmax = np.max(doc)
    threshold = (dmin+dmax)/2
    comm_imp_top_doc.insert(i,list(map(lambda x:1 if x>threshold else 0, doc)))
    #print imp_top_doc[i]
    i = i+1

comm_kmeans = KMeans(n_clusters = 17).fit(comm_imp_top_doc)
#print comm_kmeans.labels_
        
'''
Document visual words in proper format
'''
'''
doc_word = open('./lda_doc.txt','w')
doc_word.write(str(len(docs))+'\n')
for doc in docs:
    doc_word.write('1\n')
    for i in range(0, len(doc)):
        for no in range(0,doc[i]):
            doc_word.write(str(i)+' ')
    doc_word.write('\n')
'''
model = lda.LDA(n_topics=8, n_iter=4000)
model.fit(docs)
top = model.ndz_
doc_topic={}
for i in range(0,len(docs)):
    doc_topic[imagenms[i]]=np.argmax(top[i])
#print doc_topic
clusters = {}
for key,val in doc_topic.iteritems():
    clusters.setdefault(val,[]).append(key)
for key,val in clusters.iteritems():
    print key,':',val
    
#print model.doc_topic_
  
'''
for each document..adaptive thReshold on the topic proportions...then a knn approach 
'''

top_p = model.doc_topic_
imp_top_doc = list()
i = 0
for doc in top_p: 
    dmin = np.min(doc)
    dmax = np.max(doc)
    threshold = (dmin+dmax)/2
    imp_top_doc.insert(i,list(map(lambda x:1 if x>threshold else 0, doc)))
    #print imp_top_doc[i]
    i = i+1

kmeans = KMeans(n_clusters = 10).fit(imp_top_doc)
img_labels = kmeans.labels_
img_clusters = dict()
for img_no,cluster_label in enumerate(img_labels):
    img_clusters.setdefault(cluster_label,[]).append(imagenms[img_no])
print img_clusters

'''
for new image, predict!
'''

file_test = open('./testimg.pickle','rb')
test_imgs = pickle.load(file_test)
file_nms = open('./test/imagename.pickle','rb')
test_nms = pickle.load(file_nms)
#determine the sift descriptors as belonging to which visual vocabulary
file_no = 0
test = test_imgs
#for test in test_imgs:
try:
    words, distance = vq(np.array(test),vocab)
    X = np.zeros(k, dtype = np.int64)
    for w in words:
        X[w] += 1
    #bag of words model ready, now predict in lda of visual topics
    topic_est = model.transform(X)
    print topic_est
    
    tmin = np.min(topic_est)
    tmax = np.max(topic_est)
    thresh = (tmin+tmax)/2
    imp_t = list(map(lambda x:1 if x>thresh else 0, topic_est[0]))
    pred_lab = kmeans.predict(imp_t)    #warning
    annot_clust = [0]*17
    for img in img_clusters[pred_lab[0]]:
        wt = 1-hamming(imp_t, imp_top_doc[imagenms.index(img)])
        annot_clust[comm_kmeans.labels_[imagenms.index(img)]] += wt
    print "annot",annot_clust
    annot_topic = comm_kmeans.cluster_centers_[annot_clust.index(max(annot_clust))]
    
    amin = np.min(annot_topic)
    amax = np.max(annot_topic)
    thres = (amin+amax)/2
    imp_t = list(map(lambda x:1 if x>thres else 0, annot_topic))
    imp_topic_indices = [ind for ind in range(len(imp_t)) if imp_t[ind]==1]
    print imp_topic_indices
    topic_word = comm_model.topic_word_  # model.components_ also works
    '''
    n_top_words = 5
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(txt_vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        #print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        if imp_t[i]==1:
            print test_nms[file_no]+' '.join(topic_words)
    #cv2.imshow('./iart/New\ Folder/'+test_nms[file_no])
    '''
    #K most similar images from each vocab
    K = 1
    similar_imgs = {} #index and siilarity of images which are similar acc to annot
    for i in range(len(txt_vocab)):
        imgs_annotated = np.nonzero(np.transpose(doc_comments)[i])
        img_no = 0
        similar = {}    #for each annotation, similarity quotient
        for img in imgs_annotated[0]:
            #print img
            similar[img_no] = sklearn.metrics.pairwise.cosine_similarity(X, docs[img]) #warning
            img_no =+ 1
        top_k = sorted(range(len(similar)), key=lambda j: similar[j])[-K:]
        for index in top_k:
            similar_imgs[imgs_annotated[0][index]] = similar[index]
        print "s",i,similar_imgs
    #vocab word probability
    raw_input()
    pr = [0]*len(txt_vocab)
    print pr
    for i in range(len(txt_vocab)):
        for img,similarity in similar_imgs.iteritems():
            #print img
            if doc_comments[img][i]!=0:
                pr[i] += similarity[0][0]    
    pr = [float(i)/sum(pr) for i in pr]
    print "pr",pr
    #raw_input()
    #new probability acc to both topic_word and pr
    P_words = [0]*len(txt_vocab)
    for i in range(len(txt_vocab)):
        P_words[i] = pr[i]
        for j in imp_topic_indices:
            P_words[i] *= topic_word[j][i]
    P_words = [float(i)/sum(pr) for i in P_words]
    #print annotated words
    n_top_words = 5
    topic_words = np.array(txt_vocab)[np.argsort(P_words)][:-n_top_words:-1]
    print test_nms[file_no]+' '.join(topic_words)
        

            
except:
    file_no += 1
