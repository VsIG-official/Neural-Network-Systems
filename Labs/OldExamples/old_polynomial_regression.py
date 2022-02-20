import os
import numpy as np
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


from tensorflow.python.framework import ops
ops.reset_default_graph()

order = 30
sess = tf.InteractiveSession()
# создадим выборку
x = np.linspace(0,10, 1000)
y = np.sin(x) + np.random.normal(size=len(x))

plt.plot(x,y)
plt.show()

poly_features = PolynomialFeatures(degree=order-1)
s = StandardScaler()
x_poly = s.fit_transform(poly_features.fit_transform(x.reshape(1000,1)))


# и разобьем её на тренировочную и контрольную части
shuffle_idxs =np.arange(len(x_poly))
np.random.shuffle(shuffle_idxs)


X_Train = x_poly[shuffle_idxs[:3*len(x)//4]]
Y_Train = y[shuffle_idxs[:3*len(x)//4]]

X_Test = x_poly[shuffle_idxs[3*len(x)//4:]]
Y_Test = y[shuffle_idxs[3*len(x)//4:]]

#Создадим граф
x_ = tf.placeholder(name="input", shape=[None, order], dtype=tf.float32)
y_ = tf.placeholder(name= "output", shape=[None, 1], dtype=tf.float32)

w = tf.Variable(tf.random_normal([order,1]), name='weights')

model_output = tf.matmul(x_,w)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                       100000, 0.96, staircase=True)

loss = tf.reduce_mean(tf.pow(y_ - model_output, 2)) + 0.85* tf.nn.l2_loss(w) + 0.15* tf.reduce_mean(tf.abs(w)) # функция потерь
gd = tf.train.GradientDescentOptimizer(learning_rate) #оптимизатор
train_step = gd.minimize(loss)
sess.run(tf.global_variables_initializer())
n_epochs = 1000

train_errors = []
test_errors = []
for i in tqdm.tqdm(range(n_epochs)): # 1000
    _, train_err = sess.run([train_step, loss ], feed_dict={x_:X_Train, y_: Y_Train.reshape(len(Y_Train),1)})
    train_errors.append(train_err)
    test_errors.append(sess.run(loss, feed_dict={x_:X_Test, y_: Y_Test.reshape((len(Y_Test),1))}))
    
plt.plot(list(range(n_epochs)), train_errors, label = 'train' )
plt.plot(list(range(n_epochs)), test_errors, label='test') 
plt.legend()
plt.savefig('lin_reg.png')
print(train_errors[:10])
print(test_errors[:10])

plt.plot(x, y)
plt.plot(x,sess.run(model_output, feed_dict={x_:x_poly.reshape((len(x),order))}))
#plt.savefig("poly_forward_pass.png")