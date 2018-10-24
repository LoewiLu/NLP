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

import re

def preprocessing(data):  
    
    remove = re.compile('[^\w+]|\d+') #remove punctuations & numbers
    data = remove.sub(' ', data)  
    
    return data.lower() 

t0 = time()    
train_datas = [ preprocessing( _ ) for _ in train['datas']]
duration = time() - t0
print("get cleaned training datas list in %fs " % duration )

t0 = time()    
test_datas = [ preprocessing( _ ) for _ in test['datas']]
duration = time() - t0
print("get cleaned testing datas list in %fs " % duration )

#%%

f = []
for i in range(len(train_datas)):
    for words in train_datas[i].split():
        f.append(words)
        
t0 = time()      
features = np.array(list(set(f)))
duration = time() - t0

print('get features  in %fs' %  duration)

#%%
import math 

def tfidf_vector(datas):
    
    row_count = len(datas)
    col_count = len(features)
    matrix = np.zeros(row_count, col_count)
    matrix1 = matrix.copy()
    
    for index in range(row_count) : 
        
        data = datas[index] #string
        words = data.split() #list
        words_count = len(words)
        
        for i in range(col_count):
            
            l = 0
            for j in range(words_count):                 
                if features[i] == words[j]:
                    l += 1
                matrix[index,i] = l/words_count #term frequency
                
            if features[i] in data:
                matrix1[index, i] += 1
    
    ma = matrix1.sum (axis=0)   #docs count for each feature

    matrix0 = math.log(row_count/(ma+1)) #idf

    return matrix * matrix0            
    

if __name__ == "__main__":
    
   
    t0 = time() 
    X_train = tfidf_vector(train_datas)   
    duration = time() - t0
    print("done in %fs" % duration )
    print("%d documents, %d features" % X_train.shape)

    
    t0 = time() 
    X_test = tfidf_vector(test_datas)   
    duration = time() - t0
    print("done in %fs " % duration )
    print("%d documents, %d features" % X_test.shape)
