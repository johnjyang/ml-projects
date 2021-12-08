import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
file = np.loadtxt('Origin/circledata.txt', delimiter=',')

np.random.shuffle(file)
hori = (file[:, :-1] - np.mean(file[:, :-1], axis=0)) / np.std(file[:, :-1], axis=0)
vert = file[:, -1]
t1 = tf.Variable(np.random.uniform(-2, 2, (3, 21)), dtype=tf.float32)
t2 = tf.Variable(np.random.uniform(-2, 2, (22, 49)), dtype=tf.float32)
t3 = tf.Variable(np.random.uniform(-2, 2, (50, 2)), dtype=tf.float32)

def create(type):
    if type == 'test':
        x = hori[94:118]
        y = vert[94:118]
    elif type == 'train':
        x = hori[:94]
        y = vert[:94]
    return np.array(x), np.array(y)

X_train = np.concatenate((np.array([[1] * 94]), create('train')[0].T), axis=0).T
y_train = create('train')[1]

X_test = np.concatenate((np.array([[1] * 24]), create('test')[0].T), axis=0).T
y_test = create('test')[1]
# ---------- graph ^ ----------

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))
y = tf.compat.v1.placeholder(tf.int32, shape=None)
y_hot = tf.one_hot(y, 2)

hidden1 = tf.sigmoid(tf.matmul(X, t1))
hidden1 = tf.transpose(tf.concat([tf.ones([1, tf.shape(X)[0]]), tf.transpose(hidden1)], 0))

hidden2 = tf.sigmoid(tf.matmul(hidden1, t2))
hidden2 = tf.transpose(tf.concat([tf.ones([1, tf.shape(X)[0]]), tf.transpose(hidden2)], 0))

pred = tf.sigmoid(tf.matmul(hidden2, t3))

loss = - tf.math.reduce_sum(y_hot*(tf.math.log(pred)) + (1-y_hot) * (tf.math.log(1-pred))) / tf.cast(tf.shape(X)[0], tf.float32)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()

sess.run(init)

def optimize():
    old = sess.run(loss, feed_dict={X: X_train, y: y_train})
    sess.run(optimizer, feed_dict={X: X_train, y: y_train})
    new = sess.run(loss, feed_dict={X: X_train, y: y_train})
    count = 0
    while old > new:
        old = new
        sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        new = sess.run(loss, feed_dict={X: X_train, y: y_train})
        count += 1
        print(new)
    print(count)
    return old

print("\nFinal Loss: ", optimize())

def accuracy(predict, actual):
    percent = 0
    p = np.argmax(sess.run(pred, feed_dict={X: predict}), axis=1)
    for i in range(len(actual)):
        if p[i] == actual[i]:
            percent += 1
    return percent / len(actual)

print('\nTraining accuracy: ', accuracy(X_train, y_train))
print('\nTesting accuracy: ', accuracy(X_test, y_test))
# ---------- calculation ^ ----------

yhat_train = np.argmax(sess.run(pred, feed_dict={X: X_train}), axis=1)
yhat_test = np.argmax(sess.run(pred, feed_dict={X: X_test}), axis=1)

plt.scatter(X_train[(yhat_train == 1).nonzero(), 1], X_train[(yhat_train == 1).nonzero(), 2], color='red')
plt.scatter(X_train[(yhat_train == 0).nonzero(), 1], X_train[(yhat_train == 0).nonzero(), 2], color='green')
plt.show()

plt.scatter(X_train[(y_train == 1).nonzero(), 1], X_train[(y_train == 1).nonzero(), 2], color='red')
plt.scatter(X_train[(y_train == 0).nonzero(), 1], X_train[(y_train == 0).nonzero(), 2], color='green')
plt.show()

plt.scatter(X_test[(yhat_test == 1).nonzero(), 1], X_train[(yhat_test == 1).nonzero(), 2], color='red')
plt.scatter(X_test[(yhat_test == 0).nonzero(), 1], X_train[(yhat_test == 0).nonzero(), 2], color='green')
plt.show()

plt.scatter(X_test[(y_test == 1).nonzero(), 1], X_train[(y_test == 1).nonzero(), 2], color='red')
plt.scatter(X_test[(y_test == 0).nonzero(), 1], X_train[(y_test == 0).nonzero(), 2], color='green')
plt.show()
# ---------- plotting ^ ----------

def show_pred():
    x_plot = np.arange(-2, 2, 0.005)
    y_plot = np.arange(-2, 2, 0.005)
    pairs = []

    for i in range(x_plot.size):
        for j in range(y_plot.size):
            pairs.append([x_plot[i], y_plot[j]])
    for i in pairs:
        i.insert(0, 1)

    pairs = np.asarray(pairs)
    grid_pred = sess.run(pred, feed_dict={X: pairs})[:, -1]
    img = np.zeros((x_plot.size, y_plot.size), dtype=np.float32)

    for i in range(x_plot.size):
        for j in range(y_plot.size):
            img[i, j] = grid_pred[x_plot.size-x_plot.size*i+j]

    plt.imshow(img, cmap='gray')
    plt.show()

show_pred()
