#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:03:39 2018
参考：Keras:基于Python的深度学习库 https://keras-cn.readthedocs.io/en/latest/
@author: loewi

"""
import os
from time import time
from feature_extraction import __get
import numpy as np

path = r'/Users/loewi/Documents/Pre_Learn/classification/20news-bydate/'
os.chdir(path)
#print(os.getcwd())

print('Preparing data...')

t0 = time() 

newsgroups_train = __get('20news-bydate-train')
newsgroups_test = __get('20news-bydate-test')

duration = time() - t0
print('%0.2fs get data package ：）'%duration)

data_train,data_test = newsgroups_train['data'], newsgroups_test['data'] #list of strings
label_train, label_test = newsgroups_train['docs'], newsgroups_test['docs'] #array
print('Data prepared ：）')
print()
 #%% 
print('Indexing word vectors...')

words_index = {}
f = open('glove.6B.100d.txt',encoding='utf-8')
for line in f:
	word_vector = line.split()
	word = word_vector[0]
	vector = np.asarray(word_vector[1:], dtype='float32')
	words_index[word] = vector
f.close()
 
print('%s word vectors prepared ：）'%len(words_index)) #400000
print()
#%%

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

#%%
print('Tokenizing...')

MAX_NUM_WORDS = 20000
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
#keras.preprocessing.text.Tokenizer(num_words=None, #None或整数(最常见的)
#                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
#                                   lower=True,
#                                   split=" ",
#                                   char_level=False #char_level: 如果为 True, 每个字符将被视为一个标记
#                                   )
tokenizer.fit_on_texts(data_train)
sequences = tokenizer.texts_to_sequences(data_train)#返回值：2Dlist，每个list对应于一段输入文本
tokenizer.fit_on_texts(data_test)
sequences_test = tokenizer.texts_to_sequences(data_test)

word_index = tokenizer.word_index #dict{key= word, value = 排名或者索引(从1开始)}
print('Found %s unique tokens.'%len(word_index))
#word_counts:字典，将单词（字符串）映射为它们在训练期间出现的次数。仅在调用fit_on_texts之后设置。
#word_docs: 字典，将单词（字符串）映射为它们在训练期间所出现的文档或文本的数量。仅在调用fit_on_texts之后设置。
#word_index: 字典，将单词（字符串）映射为它们的排名或者索引。仅在调用fit_on_texts之后设置。
#document_count: 整数。分词器被训练的文档（文本或者序列）数量。仅在调用fit_on_texts或fit_on_sequences之后设置。
print()
#%%
MAX_SEQUENCE_LENGTH = 1000
#keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32'
#                                           padding='pre', truncating='pre', value=0.)
#sequences：浮点数或整数构成的两层嵌套列表 
#maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0 
#padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
#truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断
#value：浮点数，此值将在填充时代替默认的填充值0
X_train = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)#长度不足1000的用0填充(前端填充)
X_test = pad_sequences(sequences_test, maxlen = MAX_SEQUENCE_LENGTH) 
#to_categorical(y, num_classes=None) 
#y: 类别向量，num_classes:总共类别数
y_train = to_categorical(label_train) #扩列，总类别20列
y_test = to_categorical(label_test)

print('shape of training data',X_train.shape)
print('shape of training labels',y_train.shape)
print('shape of testing data',X_test.shape)
print('shape of testing labels',y_test.shape)
print()
#%%把有效出现次数在前面的通过GloVe生成的字典，以及本身所有的Token串进行比对，得到出现在训练集中每个词的词向量
EMBEDDING_DIM = 100
num_words = min(MAX_NUM_WORDS,len(word_index))
embedding_matrix = np.zeros((num_words +1,EMBEDDING_DIM))
for word,i in word_index.items():
	if i>MAX_NUM_WORDS:
		continue
	embedding_vector = words_index.get(word) #array
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector        
print('shape of embedding matrix:',embedding_matrix.shape)
print()
#%%LSTM
#keras.layers.embeddings.Embedding(
#                                input_dim, output_dim, 
#                                embeddings_initializer='uniform', embeddings_regularizer=None, 
#                                activity_regularizer=None, embeddings_constraint=None, 
#                                mask_zero=False, input_length=None
#                                )
#Embedding层只能作为模型的第一层
embedding_layer = Embedding(num_words + 1, #input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
                            EMBEDDING_DIM,#output_dim：大于0的整数，代表全连接嵌入的维度
                            weights=[embedding_matrix], #(20001, 100)
                            input_length=MAX_SEQUENCE_LENGTH, 
#当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
                            )
print('Building model...')

model = Sequential() #序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”
model.add(embedding_layer)
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))  #100维
model.add(Dense(1))#dense层，大于0的整数，代表该层的输出维度
model.add(Activation('sigmoid')) #激活层是对一个层的输出施加激活函数
model.add(Dense(len(newsgroups_train['classes']), activation='softmax'))#Softmax将连续数值转化成相对概率
model.layers[1].trainable=False

print('Model completed ：）')
model.summary()
#%%
#compile(self, optimizer, loss, metrics=None, 
#        loss_weights=None, sample_weight_mode=None, 
#        weighted_metrics=None, target_tensors=None)
model.compile(
            optimizer='adam',#优化器
            loss='binary_crossentropy',#损失函数
            metrics=['accuracy'],#指标列表
            )

print('Training...')
#fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, 
#        callbacks=None, validation_split=0.0, validation_data=None, 
#        shuffle=True, class_weight=None, sample_weight=None, 
#        initial_epoch=0, steps_per_epoch=None, validation_steps=None)
#batch_size:整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
#epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置
#validation_data：形式为（X，y）或（X，y，sample_weights）的tuple，是指定的验证集。此参数将覆盖validation_spilt
batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_data=(X_test,y_test))

#evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
#x：输入数据，与fit一样，是numpy array或numpy array的list
#y：标签，numpy array
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Loss:',loss) #0.1973133358117183
print('Accuracy:',acc) #0.949999988079071


#%%
#model.save('my_model.h5')
#model = load_model('my_model.h5') 

'''
Train on 11314 samples, validate on 7532 samples
Epoch 1/5
11314/11314 [==============================] - 1354s 120ms/step - loss: 0.1984 - acc: 0.9500 - val_loss: 0.1983 - val_acc: 0.9500
Epoch 2/5
11314/11314 [==============================] - 400s 35ms/step - loss: 0.1981 - acc: 0.9500 - val_loss: 0.1981 - val_acc: 0.9500
Epoch 3/5
11314/11314 [==============================] - 400s 35ms/step - loss: 0.1977 - acc: 0.9500 - val_loss: 0.1980 - val_acc: 0.9500
Epoch 4/5
11314/11314 [==============================] - 393s 35ms/step - loss: 0.1968 - acc: 0.9500 - val_loss: 0.1977 - val_acc: 0.9500
Epoch 5/5
11314/11314 [==============================] - 382s 34ms/step - loss: 0.1954 - acc: 0.9500 - val_loss: 0.1973 - val_acc: 0.9500
7532/7532 [==============================] - 40s 5ms/step
'''

#%%
#from keras.utils import plot_model
#
#plot_model(model, to_file='model.png')
#print('plotted!')