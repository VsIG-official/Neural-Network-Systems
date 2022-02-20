
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
ops.reset_default_graph()
import tqdm
sess =tf.Session()
 
x_ = tf.placeholder(name="input", shape=[None, 2], dtype=tf.float32)
y_ = tf.placeholder(name= "output", shape=[None, 1], dtype=tf.float32)
 
hidden_neurons = 15
w1 = tf.Variable(tf.random_uniform(shape=[2,hidden_neurons ]))
b1 = tf.Variable(tf.constant(value=0.0, shape=[hidden_neurons ], dtype=tf.float32))
layer1 = tf.nn.relu(tf.add(tf.matmul(x_, w1), b1))
 
w2 = tf.Variable(tf.random_uniform(shape=[hidden_neurons ,1]))
b2 =  tf.Variable(tf.constant(value=0.0, shape=[1], dtype=tf.float32))
 
nn_output = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))
gd = tf.train.GradientDescentOptimizer(0.001)
loss =  tf.reduce_mean(tf.square(nn_output- y_))
train_step = gd.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[0]])
for _ in range(20000):
    sess.run(train_step, feed_dict={x_:x, y_:y})
    
print(sess.run(nn_output, feed_dict={x_:x}))