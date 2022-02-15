import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import gc
from sklearn.metrics import confusion_matrix

tf.compat.v1.disable_eager_execution()

def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

image_files = np.stack((unpickle('Origin/cifar/data_batch_1')[b'data'],
                        unpickle('Origin/cifar/data_batch_2')[b'data'],
                        unpickle('Origin/cifar/data_batch_3')[b'data'],
                        unpickle('Origin/cifar/data_batch_4')[b'data'],
                        unpickle('Origin/cifar/data_batch_5')[b'data']))

labels = np.append([unpickle('Origin/cifar/data_batch_1')[b'labels'],
                    unpickle('Origin/cifar/data_batch_2')[b'labels'],
                    unpickle('Origin/cifar/data_batch_3')[b'labels'],
                    unpickle('Origin/cifar/data_batch_4')[b'labels']],
                   unpickle('Origin/cifar/data_batch_5')[b'labels'])

words = unpickle('Origin/cifar/batches.meta')[b'label_names']

def create(dir):
    result = []
    if dir == 'train':
        for i in image_files:
            for im in i:
                result.append(np.rot90(np.stack((np.reshape(im[:1024], (32, 32)), np.reshape(im[1024:2048], (32, 32)), np.reshape(im[2048:3072], (32, 32)))).T/255, 3))
    elif dir == 'test':
        for im in unpickle('Origin/cifar/test_batch')[b'data']:
            result.append(np.rot90(np.stack((np.reshape(im[:1024], (32, 32)), np.reshape(im[1024:2048], (32, 32)), np.reshape(im[2048:3072], (32, 32)))).T/255, 3))
    return result

X_train = np.array(create('train')[:45000])
y_train = np.array(labels[:45000])

X_valid = np.array(create('train')[45000:])
y_valid = np.array(labels[45000:])

X_test = np.array(create('test'))
y_test = np.array(unpickle('Origin/cifar/test_batch')[b'labels'])

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.compat.v1.placeholder(tf.int32, shape=None)
y_hot = tf.one_hot(y, 10)

learning_rate = 0.00000005

conv1 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(X, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)

conv2 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv1, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv3 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv2, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv3 = conv3 + tf.compat.v1.layers.conv2d(conv1, 16, 1)
conv4 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv3, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv5 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv4, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv5 = conv5 + tf.compat.v1.layers.conv2d(conv3, 16, 1)
conv6 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv5, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv7 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv6, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv7 = conv7 + tf.compat.v1.layers.conv2d(conv5, 16, 1)
conv8 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv7, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv9 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv8, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv9 = conv9 + tf.compat.v1.layers.conv2d(conv7, 16, 1)
conv10 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv9, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv11 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv10, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv11 = conv11 + tf.compat.v1.layers.conv2d(conv9, 16, 1)

conv12 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv11, 32, 3, 2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv13 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv12, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv13 = conv13 + tf.compat.v1.layers.conv2d(conv11, 32, 1, 2)
conv14 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv13, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv15 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv14, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv15 = conv15 + tf.compat.v1.layers.conv2d(conv13, 32, 1)
conv16 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv15, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv17 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv16, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv17 = conv17 + tf.compat.v1.layers.conv2d(conv15, 32, 1)
conv18 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv17, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv19 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv18, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv19 = conv19 + tf.compat.v1.layers.conv2d(conv17, 32, 1)
conv20 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv19, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv21 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv20, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv21 = conv21 + tf.compat.v1.layers.conv2d(conv19, 32, 1)

conv22 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv21, 64, 3, 2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv23 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv22, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv23 = conv23 + tf.compat.v1.layers.conv2d(conv21, 64, 1, 2)
conv24 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv23, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv25 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv24, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv25 = conv25 + tf.compat.v1.layers.conv2d(conv23, 64, 1)
conv26 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv25, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv27 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv26, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv27 = conv27 + tf.compat.v1.layers.conv2d(conv25, 64, 1)
conv28 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv27, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv29 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv28, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv29 = conv29 + tf.compat.v1.layers.conv2d(conv27, 64, 1)
conv30 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv29, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv31 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv30, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv31 = conv31 + tf.compat.v1.layers.conv2d(conv29, 64, 1)

conv32 = tf.keras.layers.GlobalAveragePooling2D()(conv31)

full = tf.nn.dropout(tf.compat.v1.layers.flatten(conv32), 0.2)
pred = tf.compat.v1.layers.dense(full, 10)

saver = tf.compat.v1.train.Saver()

loss = tf.math.reduce_mean(tf.math.maximum(pred, 0) - pred * y_hot + tf.math.log(1 + tf.math.exp(- tf.math.abs(pred))))
loss = tf.math.add_n([loss] + tf.compat.v1.losses.get_regularization_losses())

optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()

sess.run(init)

def optimize_times():
    global results, learning_rate
    batch_size = 30
    results = []
    for e in range(1, 101):
        if e == 30:
            learning_rate /= 10
        elif e == 60:
            learning_rate /= 10
        elif e == 90:
            learning_rate /= 10
        '''
        if e == 25:
            learning_rate /= 10
        elif e == 50:
            learning_rate /= 10
        elif e == 75:
            learning_rate /= 10
        '''
        aug = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)
        new_imgs = aug.flow(X_train, y_train, batch_size)
        gc.collect()
        batch_list = []
        counter = 0
        for batch, label in new_imgs:
            counter += batch.shape[0]
            sess.run(optimizer, feed_dict={X: batch, y: label})
            batch_list.append(sess.run(loss, feed_dict={X: batch, y: label}))
            print(str(e), counter, batch_list[-1])
            if counter >= 60000:
                break
        results.append([e, sum(batch_list)/len(batch_list), accuracy(X_train, y_train), accuracy(X_valid, y_valid)])
        print(results[-1][0], results[-1][1], results[-1][2], results[-1][3], '(E, Loss, Training accuracy, Valid accuracy)')
        print("Valide confuse:\n", confusion_matrix(y_valid, np.argmax(sess.run(pred, feed_dict={X: X_valid}), axis=1)))
        print("Labels: ", words)
    return sum(batch_list)/len(batch_list)

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

def save_data():
    global results
    f = open('SimpLog.txt', 'w')
    for i in results:
        f.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(i[3]))
        f.write('\n')
    f.close()

print("\nFinal Loss: ", optimize_times())

print('\nTraining accuracy: ', accuracy(X_train, y_train))
print('\nTest accuracy: ', accuracy(X_test, y_test))

print("\nTest confuse:\n", confusion_matrix(y_test, np.argmax(sess.run(pred, feed_dict={X: X_test}), axis=1)))
print("\nLabels: ", words)
