#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:19:14 2018

@author: loewi
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

newsgroups_train = fetch_20newsgroups(subset='train', remove = ('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test' ,remove = ('headers', 'footers', 'quotes'))
print('data loaded')

X_train, X_test = newsgroups_train.data, newsgroups_test.data
y_train, y_test = newsgroups_train.target, newsgroups_test.target
print('X & y prepared')

#%% Naive Bayes Classifier
#调参 choose vectorizer

nbc_1 = Pipeline([
                ('vect', CountVectorizer(
                        stop_words='english'
                        )),
                ('clf', MultinomialNB()),
                ])
                
nbc_2 = Pipeline([
                ('vect', TfidfVectorizer()),
                ('clf', MultinomialNB()),
                ])

nbc_3 = Pipeline([
                ('vect', TfidfVectorizer(
                        stop_words='english'
                )),
                ('clf', MultinomialNB()),
                ])          
                
nbc_4 = Pipeline([
                ('vect', TfidfVectorizer(
#                        analyzer='word', 
#                        sublinear_tf=True, 
#                        max_df=0.5, 
                        lowercase=True, 
                        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z_]+\b',
                        stop_words='english'                        
                )),
                ('clf', MultinomialNB()),
                ])    
              
#nbc_5 = Pipeline([
#                ('vect', HashingVectorizer(non_negative=True)),
#                ('clf', MultinomialNB()),
#                ])

nbcs = [nbc_1, nbc_2, nbc_3, nbc_4]


def vectorizer_comparing(clf, X, y, cv):
    
    print('-'*10)
    scores = cross_val_score(clf, X, y, cv = cv, scoring='accuracy')
    print (scores)
    score = scores.mean()
    print("Accuracy: %0.2f (+/- %0.2f)" % (score, scores.std() * 2))
        
    return score

    
for nbc in nbcs:
    vectorizer_comparing(nbc, X_train, y_train, 5)
    
#%% better when vectorizer => TfidfVectorizer
#调参 choose value of alpha

import matplotlib.pyplot as plt
import numpy as np


X = TfidfVectorizer(stop_words='english').fit_transform(X_train)

alpha_range = np.arange(0.01,1,0.2)

a_scores = []

for a in alpha_range:
    nbc = MultinomialNB(alpha=a)
    scores = cross_val_score(nbc,X, y_train, cv=5, scoring='accuracy') # for classification
    a_scores.append(scores.mean())

plt.plot(alpha_range, a_scores)
plt.xlabel('value of alpha for MultinomialNB')
plt.ylabel('cross-validated accuracy')
plt.show()

#%% MNNB has better performence when alpha = 0.01
# predict

from sklearn import metrics

clf = Pipeline([
                ('vect', TfidfVectorizer(                        
                        stop_words='english',
                )),
                ('clf', MultinomialNB(alpha = 0.01)),
                ])                           
                
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

