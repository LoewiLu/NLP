#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 17:25:37 2018

@author: loewi
"""

import os
path = r'/Users/loewi/Documents/Pre_Learn/wordcloud'
os.chdir( path )
print ("current path is %s" % os.getcwd()) 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt


def word_cloud(txt, frequency = False):
        
    text = open('prep/'+txt, errors="ignore").read()
    
    backgroud_Image = plt.imread('prep/tupian.jpg') 
      
    wc = WordCloud(background_color = 'white', max_words = 1000, 
                   mask = backgroud_Image, stopwords = STOPWORDS,
                   max_font_size=150, random_state=30)
    #two ways 
    
    if frequency == False:
        wc.generate_from_text(text) #by count
    
    else:        
        remove = re.compile('[^\w+]|\d+') #remove punctuations & numbers
        data = remove.sub(' ', text)
        token = data.lower().split() 
        #remove stopwords 
        tokened = [ _ for _ in token if not _ in stopwords.words('english')]   
        dic = nltk.FreqDist(tokened)
        wc.generate_from_frequencies(dic)#by frequency

    
    img_colors = ImageColorGenerator(backgroud_Image)
    wc.recolor(color_func = img_colors)
        
    plt.axis("off") #not present axis
    ax = plt.imshow(wc)
    fig = ax.figure
    fig.set_size_inches(25,20) 
    plt.show()
    wc.to_file('plot/wc.png')
    
if __name__ == "__main__":
    
    word_cloud('I Can\'t Think Straight.txt' )
    #DrawWordcloud('I Can\'t Think Straight.txt',frequency = True )
