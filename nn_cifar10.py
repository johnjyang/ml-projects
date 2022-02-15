import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

data = np.load('data/mnist.npz')

X_train = np.concatenate((np.array([[1] * 60000]), (np.reshape(data['x_train'], (60000, 784))/255).T), axis=0).T
y_train = data['y_train']

X_test = np.concatenate((np.array([[1] * 10000]), (np.reshape(data['x_test'], (10000, 784))/255).T), axis=0).T
y_test = data['y_test']

nods = 1024
t1 = tf.compat.v1.get_variable(name="t1", shape=(785, nods), dtype=tf.float32)
t2 = tf.compat.v1.get_variable(name="t2", shape=(nods+1, 10), dtype=tf.float32)
# ---------- graph ^ ----------

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 785))
y = tf.compat.v1.placeholder(tf.int32, shape=None)
y_hot = tf.one_hot(y, 10)

hidden1 = tf.sigmoid(tf.matmul(X, t1))
hidden1 = tf.transpose(tf.concat([tf.ones([1, tf.shape(X)[0]]), tf.transpose(hidden1)], 0))

pred = tf.matmul(hidden1, t2)

loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_hot, logits=pred))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(1).minimize(loss)

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()

sess.run(init)

def accuracy(predict, actual):
    percent = 0
    p = np.argmax(sess.run(pred, feed_dict={X: predict}), axis=1)
    for i in range(len(actual)):
        if p[i] == actual[i]:
            percent += 1
    return percent / len(actual)

def optimize():
    e_list = []
    for i in range(1):
        e_list.append(sess.run(loss, feed_dict={X: X_train, y: y_train}))
        sess.run(optimizer, feed_dict={X: X_train, y: y_train})
    epoch = 0
    batch_size = 30
    best_yet = min(e_list)
    run = True
    while run:
        batch_list = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size]
            label = y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={X: batch, y: label})
            batch_list.append(sess.run(loss, feed_dict={X: batch, y: label}))
        e_list.insert(0, min(batch_list))
        epoch += 1
        if e_list[0] == min(e_list):
            best_yet = e_list[0]
        if min(e_list) > best_yet:
            run = False
        e_list.pop()
        print(e_list[0], accuracy(X_test, y_test))
    print(epoch)
    return best_yet

print("\nFinal Loss: ", optimize())

print('\nTraining accuracy: ', accuracy(X_train, y_train))
print('\nTesting accuracy: ', accuracy(X_test, y_test))

# Displaying the image
def image_display(x):
    plt.imshow(np.reshape(X_test[x][1:], (28, 28)), cmap='Greys_r')
    plt.show()
    return "    Actual: " + str(y_test[x]) + "\n    Prediction: " + str(np.argmax(sess.run(pred, feed_dict={X: X_test}), axis=1)[x])

for i in range(10):
    num = int(input("\nPlease enter a number between 0 and 10000:\n --> "))
    print(image_display(num))
