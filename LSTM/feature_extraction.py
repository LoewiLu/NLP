#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: loewi
"""

import os,re    
from os.path import join
import numpy as np

def remove_extras(data):

    header, blank, text = data.partition('\n\n') 
    body, blank, footer = text.rpartition('\n\n')
    
    remove = re.compile('(In article|From:|@|Subject:|Quoted from|writes in|writes:|wrote:|says:|said:|^\|_|\+|<)')
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
                        data = all_datas, 
                        docs = np.array(docs_classNr) 
                        )  
    
    return data_package
