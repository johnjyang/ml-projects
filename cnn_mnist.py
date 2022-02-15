import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

data = tf.keras.datasets.mnist.load_data(path='mnist.mpz')

X_train = np.reshape(data[0][0]/255, (-1, 28, 28, 1)) 
y_train = data[0][1]
    
X_test = np.reshape(data[1][0]/255, (-1, 28, 28, 1)) 
y_test = data[1][1]

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.compat.v1.placeholder(tf.int32, shape=None)
y_hot = tf.one_hot(y, 10)

conv1 = tf.nn.relu(tf.compat.v1.layers.conv2d(X, 16, 5))
conv2 = tf.nn.relu(tf.compat.v1.layers.conv2d(conv1, 32, 5))
conv3 = tf.nn.relu(tf.compat.v1.layers.conv2d(conv2, 64, 3))
conv3 = tf.compat.v1.layers.max_pooling2d(conv3, 2, 2)

full = tf.nn.dropout(tf.compat.v1.layers.flatten(conv3), 0.5)
h1 = tf.nn.relu(tf.compat.v1.layers.dense(full, 256))
pred = tf.compat.v1.layers.dense(h1, 10)

loss = tf.math.reduce_mean(tf.math.maximum(pred, 0) - pred*y_hot + tf.math.log(1+tf.math.exp(-tf.math.abs(pred))))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(1).minimize(loss)

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()

sess.run(init)

def optimize_times():
    batch_size = 60
    while accuracy(X_train, y_train) < 1.00:
        batch_list = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size]
            label = y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={X: batch, y: label})
            batch_list.append(sess.run(loss, feed_dict={X: batch, y: label}))
        print(min(batch_list), accuracy(X_test, y_test), accuracy(X_train, y_train))
    return min(batch_list)
    
def accuracy(predict, actual):
    p_list = []
    batch_size = 50
    for i in range(0, len(actual), batch_size):
        percent = 0
        batch = np.argmax(sess.run(pred, feed_dict={X: predict[i:i+batch_size]}), axis=1)
        label = actual[i:i+batch_size]
        for n in range(len(batch)):
            if batch[n] == label[n]:
                percent += 1
        p_list.append(percent/batch_size)
    return np.mean(p_list)

print("\nFinal Loss: ", optimize_times())

print('\nTraining accuracy: ', accuracy(X_train, y_train))
print('\nTest accuracy: ', accuracy(X_test, y_test))
