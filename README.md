# NLP_Python (小白进阶篇)

## CLASSIFICATION 分类

### 【买菜】Dataset download:

官网：[20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/)

共20类别(Labels)，当然，每个类别下还有好多documents(.txt)：

|  | | |
|:------------- |:---------------| :-------------|
|comp.graphics  comp.os.ms-windows.misc  comp.sys.ibm.pc.hardware  comp.sys.mac.hardware comp.windows.x |  rec.autos  rec.motorcycles  rec.sport.baseball  rec.sport.hockey|sci.crypt  sci.electronics  sci.med  sci.space|
| misc.forsale|talk.politics.misc  talk.politics.guns  talk.politics.mideast|talk.religion.misc  alt.atheism  soc.religion.christian|


There are three versions of the data set.
And we are using the second one: [20news-bydate.tar.gz](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)`(training(60%) & test(40%) = 18846 documents)`


### 【磨刀】Prerequisite: 

* 平台:    
  [Anaconda](https://www.continuum.io/downloads)

* Libraries:    
  [scikit-learn](http://scikit-learn.org/stable/install.html)  
  [keras](https://keras.io/#installation)  
 

### 【备菜】Preparing the data：
 
I wrote a module [__get](https://github.com/LoewiLu/NLP/blob/master/LSTM/feature_extraction.py) the data I need:

~~~python
from feature_extraction import __get  
#enter your path of downloaded training data
newsgroups_train = __get('~/20news-bydate-train')  
#enter your path of downloaded test data
newsgroups_test = __get('~/20news-bydate-test')  
#list of strings
data_train, data_test = newsgroups_train['data'], newsgroups_test['data']   
#array
label_train, label_test = newsgroups_train['docs'], newsgroups_test['docs']   
~~~

TIP for those cannot download the data in the first step:  

~~~python
from sklearn.datasets import fetch_20newsgroups
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test') 

#data_train.data, data_test.data
#data_train.target, data_test.target  
~~~


### 【切菜】Extracting features:

Convert the text into numerical feature vectors.

#### <a name="WE"> -- Word Embedding (mainly for Keras): </a>

* [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)  

~~~python
from gensim.models import Word2Vec

vectorizer = Word2Vec(sentences, min_count=1)

embedding_matrix = np.zeros((vectorizer.wv.vocab, embedding_dim))
for i in range( vectorizer.wv.vocab ):
    embedding_vector = vectorizer.wv[vectorizer.wv.index2word[i]] 
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
print('Shape of the embedding matrix:',embedding_matrix.shape)
~~~
`sentences` is a list of lists containing tokenised string that had been [cleaned](https://github.com/LoewiLu/NLP/blob/master/LSTM/LSTM_Word2Vec.ipynb) before throw into the `Word2Vec`. 

* [GloVe](https://nlp.stanford.edu/projects/glove/)  
Download the files, and we use the `glove.6B.100d.txt`.  
Pretty the [same procedure](https://github.com/LoewiLu/NLP/blob/master/LSTM/LSTM_GloVe.ipynb).

#### <a name="BOW"> -- Bag of Words (from Sklearn):</a>

[Vectorizers](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)  | Description
------------- | -------------
CountVectorizer([…])|Convert a collection of text documents to a matrix of token counts
HashingVectorizer([…])	|Convert a collection of text documents to a matrix of token occurrences
TfidfTransformer([…])	|Transform a count matrix to a normalized tf or tf-idf representation
TfidfVectorizer([…])	|Convert a collection of raw documents to a matrix of TF-IDF features
~~~python
from sklearn.feature_extraction.text import CountVectorizer,...
vectorizer = HashingVectorizer( stop_words='english' )
X = vectorizer.fit_transform( dataset.data )
~~~


### 【炒菜】Building Models:

#### <a name="Prepare-X-y "> -- Prepare X, y: </a>
 
* For Sklearn, it's prepared in the first step:

~~~python
X_train, X_test = newsgroups_train['data'], newsgroups_test['data']   
y_train, y_test = newsgroups_train['docs'], newsgroups_test['docs'] 
~~~

* For Keras there is one more step:

~~~python
#Pads sequences to the same length.
#Return: Numpy array with shape (len(sequences), maxlen)
X_train = pad_sequences(sequences)
X_test = pad_sequences(sequences_test) 

#Converts a class vector (integers) to binary class matrix.
#Return: A binary matrix representation of the input. The classes axis is placed last.
y_train = to_categorical(label_train)
y_test = to_categorical(label_test)
~~~

#### <a name="Chose-a-Classifier"> -- Chose a Classifier:</a>

CAVEAT: So many different choices of ML models. （随便挑几个玩）

* Keras: (+[LSTM](https://github.com/LoewiLu/NLP/tree/master/LSTM))

~~~python
embedding_layer = Embedding([…])
model = Sequential() #序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”
model.add(embedding_layer)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))  #100维
model.add(Dense(1))#dense层，大于0的整数，代表该层的输出维度
model.add(Activation('sigmoid')) #激活层是对一个层的输出施加激活函数
model.add(Dense(len(newsgroups_train['classes']), activation='softmax'))#Softmax将连续数值转化成相对概率
print('Model completed ：）')
#Don't forget to compile before fit. 
~~~

* Sklearn: 

This time we just implement following [classifiers](https://github.com/LoewiLu/NLP/tree/master/cross_validation) :

~~~python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
~~~

#### <a name="CV"> -- Training:</a>

Just `fit(X, y)`

#### <a name="CV"> -- Cross Validation:</a>

The hardest part for me is to fine tune those parameters!   
It is really TIME COMSUMING!   
And sometimes I feel that my Mac might spontaneously combust :(

Here we use [Pipeline](http://scikit-learn.org/stable/modules/compose.html#combining-estimators) + [GrideSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)/[RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) + [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)

~~~python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

pipeline = Pipeline([
                ('vect', CountVectorizer(stop_words= 'english')),
                ('tfidf', TfidfTransformer(norm = 'l2')),
                ('clf', MultinomialNB()),
                ])
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.1, 0.01, 0.001, 0.0001),
}    
cv = StratifiedKFold(n_splits=5 ,random_state=0, shuffle=True)
clf = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv = cv)
clf.fit(X_train, y_train)
~~~

[Different models](https://github.com/LoewiLu/NLP/tree/master/cross_validation) have different parameters, just getting familiar with them...

~~~python
clf.best_score_
clf.best_params_
~~~

### 【盛盘】Evaluation:

* Keras: 

~~~python
loss, acc = model.evaluate(X_test, y_test)
~~~

* Sklearn: 

~~~python
pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
~~~

------------------  
------------------

## WORD CLOUD 词云图

visualization of word frequency in a given text as a weighted list 

<img src="https://github.com/LoewiLu/NLP/blob/master/wordcloud/result.png"  width="35%" /> 


