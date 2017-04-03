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
import xml.etree.ElementTree as ET
import os
import re

filev = open('./kmeans.pickle','rb')
filed = open('./doc-word.pickle', 'rb')
vocab = pickle.load(filev)
docs = pickle.load(filed)

'''
Preprocessing of text comments
'''
comments = []
txt_vocab = []
path = './iart/annot/'
annots = [f for f in os.listdir(path)]
#print annots
stemmer = WordNetLemmatizer()
for comm in annots:
    print comm
    #with open(path+comm,'r') as xml_file:
        #tree = ET.parse(xml_file)
    #root = tree.getroot()
    #for child in root.findall('DESCRIPTION'):
        #print child.text
    comm_file = open(path+comm,'r')
    for line in comm_file:
        if 'DESCRIPTION' in line:
            cline =  filter(lambda x:x!='', re.split('[ <; ,>()]',line))
            #print  " ".join(cline[2:-2])
            #print cline
            cline = cline[1:-2]
            desc = [word for word in cline if word not in stopwords.words('english')]
            desc = [stemmer.lemmatize(word) for word in desc]
            print desc
            comments.append(desc)
for doc_comm in comments:
    for word in doc_comm:
        if word not in txt_vocab:
            txt_vocab.append(word)
print txt_vocab
for i in range(0,len(comments)):
    for j in range(0, len(comments[i])):
        comments[i][j] = txt_vocab.index(comments[i][j])
#print comments
filecomm = open('./comments.txt','w')
filecomm.write(str(len(comments))+'\n')
for i in range(0,len(comments)):
    #print map(str,comments[i])
    filecomm.write('1\n')
    comm_str = map(str,comments[i])
    for wrd in comm_str:
        filecomm.write(wrd+' ')
    filecomm.write('\n')
filecomm.close()
            
'''
Document visual words in proper format
'''
doc_word = open('./lda_doc.txt','w')
doc_word.write(str(len(docs))+'\n')
for doc in docs:
    doc_word.write('1\n')
    for i in range(0, len(doc)):
        for no in range(0,doc[i]):
            doc_word.write(str(i)+' ')
    doc_word.write('\n')