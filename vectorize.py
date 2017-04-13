# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:38:38 2017

@author: sups
"""
import numpy as np
import scipy
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import xml.etree.ElementTree as ET
import os
import re
import lda
from sklearn.cluster import KMeans

filev = open('./kmeans.pickle','rb')
filed = open('./doc-word.pickle', 'rb')
vocab = pickle.load(filev)
docs = pickle.load(filed)
docs = docs.astype(int)
print type(docs)

'''
Preprocessing of text comments
'''

comments = []
txt_vocab = []
path = './iart/annot/'
annots = [f for f in os.listdir(path)]
print annots
stemmer = WordNetLemmatizer()
for comm in annots:
    print comm
#    with open(path+comm,'r') as xml_file:
#        tree = ET.parse(xml_file)
#    root = tree.getroot()
#    for child in root.findall('DESCRIPTION'):
#        print child.text
    comm_file = open(path+comm,'r')
    for line in comm_file:
        if 'DESCRIPTION' in line:
            cline =  filter(lambda x:x!='', re.split('[ <; ,>()]',line))
            #print  " ".join(cline[2:-2])
            #print cline
            cline = cline[1:-2]
            desc_sent = pos_tag(cline)
            cline = map(lambda(word, tag): word, filter(lambda (word, tag): tag == 'NN' or tag=='NNP', desc_sent))
            #print cline
            desc = [word for word in cline if word not in stopwords.words('english')]
            desc = [stemmer.lemmatize(word) for word in desc]
            print desc
            comments.append(desc)
for doc_comm in comments:
    for word in doc_comm:
        if word not in txt_vocab:
            txt_vocab.append(word)
#print txt_vocab
for i in range(0,len(comments)):
    for j in range(0, len(comments[i])):
        comments[i][j] = txt_vocab.index(comments[i][j])
print comments

cdoc = np.zeros(shape=len(txt_vocab))
for word in comments[0]:
    cdoc[int(word)] = cdoc[int(word)]+1
doc_comments = cdoc
print doc_comments
        
for i in range(0,len(comments)):
    cdoc = np.zeros(len(txt_vocab))
    for word in comments[i]:
        cdoc[int(word)] = cdoc[int(word)]+1
    doc_comments.concatenate(cdoc)
print doc_comments

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

comm_kmeans = KMeans(n_clusters = 7).fit(comments)
print comm_kmeans

fileimgnames = open('./imagename.pickle','rb')
imagenms = pickle.load(fileimgnames)
        
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
model = lda.LDA(n_topics=5, n_iter=4000)
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
    
print model.doc_topic_
'''    
'''
for each document..adaptive thReshold on the topic proportions...then a knn approach 
'''
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

kmeans = KMeans(n_clusters = 7).fit(imp_top_doc)
'''