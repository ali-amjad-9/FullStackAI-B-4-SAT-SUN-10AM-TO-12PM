"""
https://builtin.com/data-science/feedforward-neural-network-intro


Keras - run on Tensorflow

https://keras.io

https://pypi.org/project/keras/

Run: pip install keras --upgrade


Fix - Intelliegence - Bug - not show intelligence and help  
https://stackoverflow.com/questions/68860879/vscode-keras-intellisensesuggestion-not-working-properly


"""

import tensorflow as tf

keras = tf.keras

mnist = keras.datasets.mnist

""" 
https://keras.io/api/datasets/mnist/
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data


Loads the MNIST dataset.

This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. More info can be found at the MNIST homepage.

Arguments

path: path where to cache the dataset locally (relative to ~/.keras/datasets).
Returns

Tuple of NumPy arrays: (x_train, y_train), (x_test, y_test).
x_train: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. Pixel values range from 0 to 255.

y_train: uint8 NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.

x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. Pixel values range from 0 to 255.

y_test: uint8 NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data.


"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

"""

https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
https://www.tensorflow.org/guide/keras/sequential_model

tf.keras.models.Sequential
The tf.keras.models.Sequential class in TensorFlow is a simple and convenient way to build neural networks by stacking layers sequentially. This model is ideal for creating a plain stack of layers where each layer has exactly one input tensor and one output tensor.

"""


"""
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
https://keras.io/api/layers/reshaping_layers/flatten/
https://www.tutorialspoint.com/keras/keras_flatten_layers.htm

Flatten layer

Flatten class
keras.layers.Flatten(data_format=None, **kwargs)
Flattens the input. Does not affect the batch size.


============

Note: If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1).

Arguments

data_format: A string, one of "channels_last" (default) or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, ..., channels) while "channels_first" corresponds to inputs with shape (batch, channels, ...). When unspecified, uses image_data_format value found in your Keras config file at ~/.keras/keras.json (if exists). Defaults to "channels_last".

>>> x = keras.Input(shape=(10, 64))
>>> y = keras.layers.Flatten()(x)
>>> y.shape
(None, 640)

"""


"""
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
https://keras.io/api/layers/core_layers/dense/
https://www.geeksforgeeks.org/tf-keras-layers-dense-fully-connected-layer-in-tensorflow/

In TensorFlow, the tf.keras.layers.Dense layer represents a fully connected (or dense) layer, where every neuron in the layer is connected to every neuron in the previous layer. This layer is essential for building deep learning models, as it is used to learn complex patterns and relationships in data.

The Dense layer applies the following mathematical operation:

output
=
activation
(
dot
(
input
,
kernel
)
+
bias
)
output=activation(dot(input,kernel)+bias)

where:

activation: An element-wise activation function (if specified).
dot(input, kernel): A matrix multiplication between the input data and the weight matrix (kernel).
bias: A bias vector added to the computation (if use_bias=True).


"""

"""
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
https://keras.io/api/layers/regularization_layers/dropout/
https://www.geeksforgeeks.org/implementing-dropout-in-tensorflow/

Applies dropout to the input.

The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1 / (1 - rate) such that the sum over all inputs is unchanged.

Note that the Dropout layer only applies when training is set to True in call(), such that no values are dropped during inference. When using model.fit, training will be appropriately set to True automatically. In other contexts, you can set the argument explicitly to True when calling the layer.

(This is in contrast to setting trainable=False for a Dropout layer. trainable does not affect the layer's behavior, as Dropout does not have any variables/weights that can be frozen during training.)

"""

"""
TensorFlow Activation Functions

Activation functions are crucial components in neural networks as they introduce non-linearity, enabling the network to learn complex patterns. TensorFlow's tf.keras.activations module provides a variety of activation functions, each suitable for different scenarios and model architectures

.

Common Activation Functions in TensorFlow

ReLU (Rectified Linear Unit)

ReLU outputs the input directly if it is positive; otherwise, it outputs zero. This helps mitigate the vanishing gradient problem and improves training efficiency.

import tensorflow as tf

# ReLU activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Sigmoid

The Sigmoid function outputs values between 0 and 1, making it suitable for binary classification tasks. However, it is prone to the vanishing gradient problem.

import tensorflow as tf

# Sigmoid activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(32,)),
tf.keras.layers.Dense(1, activation='sigmoid') # Used for binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Tanh (Hyperbolic Tangent)

Tanh is similar to Sigmoid but outputs values between -1 and 1, making it better for hidden layers than Sigmoid.

import tensorflow as tf

# Tanh activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='tanh', input_shape=(32,)),
tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Softmax

Softmax is commonly used in the output layer for multi-class classification. It converts raw output (logits) into probabilities.

import tensorflow as tf

# Softmax activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
tf.keras.layers.Dense(10, activation='softmax') # For multi-class classification
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Leaky ReLU

Leaky ReLU allows a small, non-zero output for negative input values, helping to avoid the "dying ReLU" problem.

import tensorflow as tf

# Leaky ReLU activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, input_shape=(32,)),
tf.keras.layers.LeakyReLU(alpha=0.3), # Allows a small slope for negative values
tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ELU (Exponential Linear Unit)

ELU is similar to ReLU but has an exponential curve for negative values, helping to avoid the vanishing gradient problem.

import tensorflow as tf

# ELU activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='elu', input_shape=(32,)),
tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Swish

Swish, proposed by Google, has shown better performance than ReLU and its variants in many cases. It is defined as ( f(x) = x \cdot \text{sigmoid}(x) ).

import tensorflow as tf

# Swish activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='swish', input_shape=(32,)),
tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Softplus

Softplus is a smooth approximation of the ReLU function, continuous and differentiable, which helps avoid issues like dead neurons.

import tensorflow as tf

# Softplus activation
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='softplus', input_shape=(32,)),
tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Understanding and choosing the right activation function is crucial for the performance of deep learning models. Each activation function has its strengths and weaknesses, and the choice depends on the specific task and model architecture

"""
model =  tf.keras.models.Sequential( [ 
    tf.keras.layers.Flatten( input_shape = (28,28) ),
    tf.keras.layers.Dense(  128, activation='relu' ),
    tf.keras.layers.Dropout( 0.2 ),
    tf.keras.layers.Dense( 10, activation = "softmax" ),

])


print("model.compile(optimizer= 'adam' ,    loss='sparse_categorical_crossentropy' ,      metrics=['accuracy']) :    \n" , model.compile(optimizer= 'adam' ,
              loss='sparse_categorical_crossentropy' ,
              metrics=['accuracy'])
)

"""
An epoch in TensorFlow refers to a complete pass of the entire training dataset during the training process of a neural network. It is a unit of measurement used to
 track and control the number of times the model has seen and learned from the entire dataset. During each epoch, the training dataset is divided into smaller batches 
 to update the model's weights and biases. These batches allow the model to update its parameters based on the computed errors and the chosen optimization algorithm 
 (e.g., gradient descent). One epoch consists of iterating over all the training samples once, calculating the loss for each sample, and updating the model's parameters 
 accordingly. These iterations help the model gradually improve its accuracy and generate better predictions. The number of epochs is a hyperparameter that determines 
 the duration and quality of the training process. Setting the appropriate number of epochs requires finding a balance between underfitting (insufficient learning) 
 and overfitting (excessive learning). Underfitting occurs when the model does not learn enough from the data, resulting in poor performance. Overfitting occurs when 
 the model learns too much from the training data, leading to poor generalization on unseen data. Typically, training a model involves running multiple epochs until a 
 desired level of accuracy or convergence is achieved. However, using too many epochs can increase training time without significant improvement or even degrade the 
 model's performance on unseen data.

Read more at: 
https://almarefa.net/blog/what-is-an-epoch-in-tensorflow
https://www.geeksforgeeks.org/epoch-in-machine-learning/
https://nulldog.com/tensorflow-epochs-vs-steps-key-differences-explained

"""
print("model.fit(x_train,y_train , epochs=5) :         \n" , model.fit(x_train,y_train , epochs=3 , batch_size=20 ))



print("[INFO] Predict via network...")
predictions = model.predict(x_test)



print("model.evaluate(x_test,y_test) :    \n" , model.evaluate(x_test,y_test) )

print("Model summary:    \n " , model.summary() )

print()
