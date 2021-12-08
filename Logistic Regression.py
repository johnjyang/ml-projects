import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
file = np.loadtxt('Origin/collegedata.txt', delimiter=',')
np.random.shuffle(file)
scores = (file[:, :-1] - np.mean(file[:, :-1], axis=0))/np.std(file[:, :-1], axis=0)
result = file[:, -1]
theta = tf.Variable(np.random.uniform(-2, 2, (3, 2)), dtype=tf.float32)

def create(type):
    if type == 'test':
        x = scores[80:100]
        y = result[80:100]
    elif type == 'train':
        x = scores[0:80]
        y = result[0:80]
    return np.array(x), np.array(y)

X_train = np.concatenate((np.array([[1]*80]), create('train')[0].T), axis=0).T
y_train = create('train')[1]

X_test = np.concatenate((np.array([[1]*20]), create('test')[0].T), axis=0).T
y_test = create('test')[1]
# ---------- graph ^ ----------

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))
y = tf.compat.v1.placeholder(tf.int32, shape=(None))
y_hot = tf.one_hot(y, 2)

pred = tf.sigmoid(tf.matmul(X, theta))

loss = - tf.math.reduce_sum(y_hot*(tf.math.log(pred))+(1-y_hot)*(tf.math.log(1-pred)))/tf.cast(tf.shape(X)[0], tf.float32)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()

sess.run(init)

def accuracy(predict, actual):
    percent = 0
    p = np.argmax(sess.run(pred, feed_dict={X: predict}), axis=1)
    for i in range(len(actual)):
        if p[i] == actual[i]:
            percent += 1
    return percent/len(actual)

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
