""""
https://www.geeksforgeeks.org/training-of-recurrent-neural-networks-rnn-in-tensorflow/

"""
import warnings
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.metrics import Precision, Recall

import re
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()

warnings.filterwarnings("ignore")

data = pd.read_csv("Week17/Clothing-Review.csv")
data.head(7)
print(data.shape)

data = data[data['Class Name'].isnull() == False]

sns.countplot(data=data, x='Class Name')
plt.xticks(rotation=90)
plt.show()

plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=data, x='Rating')

plt.subplot(1, 2, 2)
sns.countplot(data=data, x="Recommended IND")
plt.show()

# histogram is amount of frequency
fig = px.histogram(data, marginal='box',
                   x="Age", title="Age Group",
                   color="Recommended IND",
                   nbins=65-18,
                   color_discrete_sequence=['green', 'red'])
fig.update_layout(bargap=0.2)

fig = px.histogram(data,
                   x="Age",
                   marginal='box',
                   title="Age Group",
                   color="Rating",
                   nbins=65-18,
                   color_discrete_sequence\
                   =['black', 'green', 'blue', 'red', 'yellow'])
fig.update_layout(bargap=0.2)

"""
Prepare the Data to build Model
Since we are working on the NLP-based dataset it could be valid to use Text columns as the feature. So we select the features that are text and the Rating column is used for Sentiment Analysis. By the above Rating counterplot we can observe that there is too much of an imbalance between the rating. So all the rating above 3 is made as 1 and below 3 as 0. 

"""

def filter_score(rating):
    return int(rating > 3)

features = ['Class Name', 'Title', 'Review Text']

X = data[features]
y = data['Rating']
y = y.apply(filter_score)


"""Text Preprocessing
The text data we have comes with too much noise. This noise can be in form of repeated words or commonly used sentences. In text preprocessing we need the text in 
the same format so we first convert the entire text into lowercase. And then perform Lemmatization to remove the superposition of the words. Since we need clean text we also remove common words(aka Stopwords) and punctuation. 
"""


def toLower(data):
    if isinstance(data, float):
        return '<UNK>'
    else:
        return data.lower()

stop_words = stopwords.words("english")

def remove_stopwords(text):
    no_stop = []
    for word in text.split(' '):
        if word not in stop_words:
            no_stop.append(word)
    return " ".join(no_stop)

def remove_punctuation_func(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

X['Title'] = X['Title'].apply(toLower)
X['Review Text'] = X['Review Text'].apply(toLower)

X['Title'] = X['Title'].apply(remove_stopwords)
X['Review Text'] = X['Review Text'].apply(remove_stopwords)

X['Title'] = X['Title'].apply(lambda x: lemm.lemmatize(x))
X['Review Text'] = X['Review Text'].apply(lambda x: lemm.lemmatize(x))

X['Title'] = X['Title'].apply(remove_punctuation_func)
X['Review Text'] = X['Review Text'].apply(remove_punctuation_func)

X['Text'] = list(X['Title']+X['Review Text']+X['Class Name'])


X_train, X_test, y_train, y_test = train_test_split(
    X['Text'], y, test_size=0.25, random_state=42)

"""
Tokenization
In Tokenization we convert the text into Vectors. Keras API supports text pre-processing. This API consists of Tokenizer that takes in the total num_words to create the Word index. OOV stands for out of vocabulary this is triggered when new text is encountered.  Also remember that we fit_on_texts only on training data and not testing. 

"""


tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)


"""Padding the Text Data
Keras preprocessing helps in organizing the text. Padding helps in building models of the same size that further becomes easy to train neural network models.
 The padding adds extra zeros to satisfy the maximum length to feed a neural network. If the text length exceeds then it can be truncated from either the 
 beginning or end. By default it is pre, we can set it to post or leave it as it is.
"""

train_seq = tokenizer.texts_to_sequences(X_train)
test_seq = tokenizer.texts_to_sequences(X_test)

train_pad = pad_sequences(train_seq,
                          maxlen=40,
                          truncating="post",
                          padding="post")
test_pad = pad_sequences(test_seq,
                         maxlen=40,
                         truncating="post",
                         padding="post")

"""
Train a Recurrent Neural Network (RNN) in TensorFlow
Now that the data is ready, the next step is building a Simple Recurrent Neural network. Before training with SImpleRNN, the data is passed through the Embedding layer to perform the equal size of Word Vectors. 
"""

model = keras.models.Sequential()
model.add(keras.layers.Embedding(10000, 128))
model.add(keras.layers.SimpleRNN(64, return_sequences=True))
model.add(keras.layers.SimpleRNN(64))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1, activation="sigmoid"))

"""
Train a Recurrent Neural Network (RNN) in TensorFlow
Now that the data is ready, the next step is building a Simple Recurrent Neural network. Before training with SImpleRNN, the data is passed through the Embedding layer to perform the equal size of Word Vectors. 
"""

METRICS = metrics=['accuracy', 
                   Precision(name='precision'),
                   Recall(name='recall')]


model.compile("rmsprop",
              "binary_crossentropy",
               metrics = METRICS)
history = model.fit(train_pad,
                    y_train,
                    epochs=5)

print("Summary:                  \n" , model.summary())

