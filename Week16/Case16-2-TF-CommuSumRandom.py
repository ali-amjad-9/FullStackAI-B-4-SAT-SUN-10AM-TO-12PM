import tensorflow as tf
"""
https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum

This is the reduction operation for the elementwise tf.math.add op.

Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the tensor is reduced by 1 for each of the entries in axis, which must be unique. If keepdims is true, the reduced dimensions are retained with length 1.

If axis is None, all dimensions are reduced, and a tensor with a single element is returned.

Args
input_tensor	The tensor to reduce. Should have numeric type.
axis	The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range [-rank(input_tensor),rank(input_tensor)].
keepdims	If true, retains reduced dimensions with length 1.
name	A name for the operation (optional).

Returns
The reduced tensor, of the same dtype as the input_tensor.

================================================
tf.reduce_sum
1
2
3
The tf.reduce_sum function in TensorFlow computes the sum of elements across dimensions of a tensor. It reduces the input tensor along the specified dimensions and returns a tensor with the sum of elements.

Example

import tensorflow as tf

x = tf.constant([[1, 2, 3], [4, 5, 6]])
result = tf.reduce_sum(x)
print(result) # Output: 21
Usage

Sum all elements: If no axis is specified, it sums all elements in the tensor.

Sum along specific axis: You can specify an axis to sum along that dimension.

Sum along specific axis

result_axis_0 = tf.reduce_sum(x, axis=0)
print(result_axis_0) # Output: [5, 7, 9]


"""


""""
https://www.tensorflow.org/api_docs/python/tf/random/normal

Outputs random values from a normal distribution.

View aliases

tf.random.normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

Args
shape	A 1-D integer Tensor or Python array. The shape of the output tensor.
mean	A Tensor or Python value of type dtype, broadcastable with stddev. The mean of the normal distribution.
stddev	A Tensor or Python value of type dtype, broadcastable with mean. The standard deviation of the normal distribution.
dtype	The float type of the output: float16, bfloat16, float32, float64. Defaults to float32.
seed	A Python integer. Used to create a random seed for the distribution. See tf.random.set_seed for behavior.
name	A name for the operation (optional).
Returns
A tensor of the specified shape filled with random normal values.
==========================
tf.random.normal
1
2
3
The tf.random.normal function in TensorFlow generates random values from a normal distribution. This is useful for initializing weights in neural networks or creating random data for testing.

Example

import tensorflow as tf

# Generate a tensor with shape (3, 2) from a normal distribution with mean 10 and stddev 2
random_tensor = tf.random.normal(shape=(3, 2), mean=10.0, stddev=2.0)
print(random_tensor)
This will output a tensor similar to:

<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[ 8.537131 , 7.6625767],
[10.925293 , 11.804686 ],
[ 9.3763075, 6.701221 ]], dtype=float32)>
Parameters

shape: A 1-D integer Tensor or Python array defining the shape of the output tensor.

mean: The mean of the normal distribution.

stddev: The standard deviation of the normal distribution.

dtype: The data type of the output tensor (default is tf.float32).

seed: An optional integer seed for reproducibility.

name: An optional name for the operation.

Important Considerations

Reproducibility: To ensure reproducible results, set both the global and operation-level seeds using tf.random.set_seed.

Performance: Generating large tensors can be computationally expensive; ensure your system has adequate resources.

"""
print()
rsum=tf.random.normal([1000, 1000])
print(rsum)
print("sum as" , tf.reduce_sum(rsum))