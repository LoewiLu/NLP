#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:24:37 2018

@author: loewi
"""

import os,re    
from os.path import join
import numpy as np
from time import time


path_train =r"/Users/loewi/Documents/Pre_Learn/classification/20news-bydate/20news-bydate-train" 
path_test =r"/Users/loewi/Documents/Pre_Learn/classification/20news-bydate/20news-bydate-test" 


def remove_extras(data):

    header, blank, text = data.partition('\n\n') 
    body, blank, footer = text.rpartition('\n\n')
    
    remove = re.compile('(In article|From:|@|Subject:|Quoted from|writes in|writes:|wrote:|says:|said:|^\|_|\+)')    
    lines = [ _ for _ in body.split('\n') if not remove.search(_)]
    
    return ' '.join(lines)

def __get(path):    
    
    dirs = os.listdir( path )
    docs_names = [ _ for _ in dirs ] 

    #docs_original_names = []
    docs_class_names = []
    docs_classNr = []
    docs_path = []
    all_datas = []
    
    for i in range(len(docs_names)):
        path0 = join ( path, docs_names[i] )
        dirs0 = os.listdir( path0 )    
        for n in dirs0:
    #        docs_original_names.append(n) 
            docs_path.append(join(path0,n))
        for m in range(len( dirs0 )):            
            docs_class_names.append(docs_names[i])
            docs_classNr.append(i)
            
    for doc_path in docs_path :
        data = open(doc_path, errors="ignore").read()
        data_clean = remove_extras(data)
        all_datas.append(data_clean)  
         
    #seal all you need in one    
    data_package = dict(classes = docs_names, 
                        docs_path = np.array(docs_path), 
                        datas = all_datas, 
                        docs = np.array(docs_classNr) 
                        )  
    
    return data_package

t0 = time() 
train = __get(path_train)
test = __get(path_test)
duration = time() - t0

print('%0.2fs get data package ：）'%duration)

#%%
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
    
import random

def preprocessing(data):  
    
    remove = re.compile('[^\w+]|\d+') #remove punctuations & numbers
    data = remove.sub(' ', data)  
    
#    return data.lower() #alternative

   #remove stopwords
    token = data.lower().split()
    ps = PorterStemmer()
    token = [ ps.stem( _ ) for _ in token if not _ in stopwords.words('english')]
    
    return ' '.join(token) #alternative

if __name__ == "__main__":
    
    i = random.randint(0,len(train['datas']))
    t0 = time() 
    article = preprocessing(train['datas'][i])
    duration = time() - t0
    print(i, "%r cleaned in %fs " % (article, duration))