import gzip
import _pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

train_x, train_ytmp = train_set
train_y = one_hot(train_ytmp,10)

valid_x, valid_ytmp = valid_set
valid_y = one_hot(valid_ytmp,10)

test_x, test_ytmp = test_set
test_y = one_hot(test_ytmp,10)

# ---------------- Visualizing some element of the MNIST dataset --------------
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print (train_y[57])
"""

# TODO: the neural net!!
inputLayer = 784
outputLayer = 10

x = tf.placeholder("float", [None, inputLayer])
y_ = tf.placeholder("float", [None, outputLayer])

W = tf.Variable(tf.zeros([inputLayer, outputLayer]))
b = tf.Variable(tf.zeros([outputLayer]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)




batch_size = 20
currentLoss = 9999
prevLoss = 9999
epoch = 0
while (currentLoss <= prevLoss):
    epoch += 1
    for jj in range(int(len(train_x) / batch_size)):
        batch_xsTrain = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ysTrain = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xsTrain, y_: batch_ysTrain})
    prevLoss = currentLoss
    currentLoss = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    print("Epoch#",epoch," Current misses:", currentLoss,"Previous misses:", prevLoss)

misses = 0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if (np.argmax(b) != np.argmax(r)):
        # print (b, "-->", r)
        misses += 1
print ("----------------------------------------------------------------------------------")
print ("Error:", misses/len(test_x),"% Total:",misses )
