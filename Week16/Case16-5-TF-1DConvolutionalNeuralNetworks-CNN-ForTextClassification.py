"""
https://www.geeksforgeeks.org/text-classification-using-cnn/


"""

# importing the necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setting up the parameters
maximum_features = 5000  # Maximum number of words to consider as features
maximum_length = 100  # Maximum length of input sequences
word_embedding_dims = 50  # Dimension of word embeddings
no_of_filters = 250  # Number of filters in the convolutional layer
kernel_size = 3  # Size of the convolutional filters
hidden_dims = 250  # Number of neurons in the hidden layer
batch_size = 32  # Batch size for training
epochs = 2  # Number of training epochs
threshold = 0.5  # Threshold for binary classification


"""

https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb
https://keras.io/api/datasets/imdb/

classification dataset
[source]

load_data function
keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
    **kwargs
)

Loads the IMDB dataset.

This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a list
 of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent 
 word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode the pad token.

Returns

Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
x_train, x_test: lists of sequences, which are lists of indexes (integers). If the num_words argument was specific, the maximum possible index value is num_words - 1. If the maxlen argument was specified, the largest possible sequence length is maxlen.

y_train, y_test: lists of integer labels (1 or 0).

Note: The 'out of vocabulary' character is only used for words that were present in the training set but are not included because they're not making the num_words cut here. Words that were not seen in the training set but are in the test set have simply been skipped.

"""
# Loading the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maximum_features)

# Padding the sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=maximum_length)
x_test = pad_sequences(x_test, maxlen=maximum_length)

# Building the model
model = Sequential()

"""
https://keras.io/api/layers/core_layers/embedding/

Embedding layer
[source]

Embedding class
keras.layers.Embedding(
    input_dim,
    output_dim,
    embeddings_initializer="uniform",
    embeddings_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    weights=None,
    lora_rank=None,
    **kwargs
)
Turns nonnegative integers (indexes) into dense vectors of fixed size.

e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

This layer can only be used on nonnegative integer inputs of a fixed range.

"""
"""
Keras Conv1D Layer

The Conv1D layer in Keras is a one-dimensional convolution layer primarily used for temporal data. This layer creates a convolution kernel that is convolved with
 the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is set to True, a bias vector is created and added to the 
 outputs. Finally, if an activation function is specified, it is applied to the outputs as well.

 https://keras.io/api/layers/convolution_layers/convolution1d/

 https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D

"""

"""
GlobalMaxPooling1D layer
[source]

GlobalMaxPooling1D class
keras.layers.GlobalMaxPooling1D(data_format=None, keepdims=False, **kwargs)
Global max pooling operation for temporal data.

https://keras.io/api/layers/pooling_layers/global_max_pooling1d/

https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool1D

"""

# Adding the embedding layer to convert input sequences to dense vectors
model.add(Embedding(maximum_features, word_embedding_dims,
                    input_length=maximum_length))

# Adding the 1D convolutional layer with ReLU activation
model.add(Conv1D(no_of_filters, kernel_size, padding='valid',
                 activation='relu', strides=1))

# Adding the global max pooling layer to reduce dimensionality
model.add(GlobalMaxPooling1D())

# Adding the dense hidden layer with ReLU activation
model.add(Dense(hidden_dims, activation='relu'))

# Adding the output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compiling the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

# Predicting the probabilities for test data
y_pred_prob = model.predict(x_test)

# Converting the probabilities to binary classes based on threshold
y_pred = (y_pred_prob > threshold).astype(int)

# Calculating the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Printing the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

"""
See details about accuracy , precision , recall , f1
https://github.com/ShahzadSarwar10/AI-ML-Explorer/blob/main/Week5/Artificial%20Intelligence%20(Machine%20Learning%20%26%20Deep%20Learning)-Week5%20-%20Machine%20Learning%20-%20Important%20Concepts%20-%20Notes_Rev1.pdf

"""