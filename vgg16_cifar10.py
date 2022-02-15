import smtplib, ssl, getpass
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import gc

tf.compat.v1.disable_eager_execution()

def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

image_files = np.stack((unpickle('data/cifar/data_batch_1')[b'data'],
                        unpickle('data/cifar/data_batch_2')[b'data'],
                        unpickle('data/cifar/data_batch_3')[b'data'],
                        unpickle('data/cifar/data_batch_4')[b'data'],
                        unpickle('data/cifar/data_batch_5')[b'data']))

labels = np.append([unpickle('data/cifar/data_batch_1')[b'labels'],
                    unpickle('data/cifar/data_batch_2')[b'labels'],
                    unpickle('data/cifar/data_batch_3')[b'labels'],
                    unpickle('data/cifar/data_batch_4')[b'labels']],
                   unpickle('data/cifar/data_batch_5')[b'labels'])

words = unpickle('data/cifar/batches.meta')[b'label_names']

def create(dir):
    result = []
    if dir == 'train':
        for i in image_files:
            for im in i:
                result.append(np.rot90(np.stack((np.reshape(im[:1024], (32, 32)), np.reshape(im[1024:2048], (32, 32)), np.reshape(im[2048:3072], (32, 32)))).T/255, 3))
    elif dir == 'test':
        for im in unpickle('data/cifar/test_batch')[b'data']:
            result.append(np.rot90(np.stack((np.reshape(im[:1024], (32, 32)), np.reshape(im[1024:2048], (32, 32)), np.reshape(im[2048:3072], (32, 32)))).T/255, 3))
    return result

X_train = np.array(create('train')[:45000])
y_train = np.array(labels[:45000])

X_valid = np.array(create('train')[45000:])
y_valid = np.array(labels[45000:])

X_test = np.array(create('test'))
y_test = np.array(unpickle('data/cifar/test_batch')[b'labels'])

X = tf.compat.v1.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.compat.v1.placeholder(tf.int32, shape=None)
y_hot = tf.one_hot(y, 10)

learning_rate = 0.01

conv1 = tf.nn.relu(tf.compat.v1.layers.conv2d(X, 64, 3, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
conv1 = tf.compat.v1.layers.max_pooling2d(conv1, 2, 2)
conv2 = tf.nn.relu(tf.compat.v1.layers.conv2d(conv1, 128, 3, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
conv3 = tf.nn.relu(tf.compat.v1.layers.conv2d(conv2, 256, 3, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
conv3 = conv3 + tf.compat.v1.layers.conv2d(conv1, 256, 5, padding='valid')
conv4 = tf.nn.relu(tf.compat.v1.layers.conv2d(conv3, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
conv5 = tf.nn.relu(tf.compat.v1.layers.conv2d(conv4, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
conv5 = conv5 + tf.compat.v1.layers.conv2d(conv3, 512, 1, padding='valid')
conv6 = tf.nn.relu(tf.compat.v1.layers.conv2d(conv5, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
conv7 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv6, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
conv7 = conv7 + tf.compat.v1.layers.conv2d(conv5, 512, 1, padding='valid')
conv8 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv7, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
conv8 = tf.compat.v1.layers.max_pooling2d(conv8, 2, 2)

full = tf.compat.v1.layers.flatten(conv8)

def create_layers(str):
    global learning_rate, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, full
    if str == 'A':
        return
    elif str == 'B':
        learning_rate = 0.01
        conv1 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(tf.nn.dropout(X, 0.8), 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv1, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.compat.v1.layers.max_pooling2d(conv2, 2, 2)
        conv3 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv2, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv4 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv3, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv4 = tf.compat.v1.layers.max_pooling2d(conv4, 2, 2)
        conv5 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv4, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv6 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv5, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv6 = tf.compat.v1.layers.max_pooling2d(conv6, 2, 2)
        conv7 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv6, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv8 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv7, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv8 = tf.compat.v1.layers.max_pooling2d(conv8, 2, 2)
        conv9 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv8, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv10 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv9, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv10 = tf.compat.v1.layers.max_pooling2d(conv10, 2, 2)
        full = tf.compat.v1.layers.flatten(conv10)
    elif str == 'C':
        learning_rate = 0.001
        conv1 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(tf.nn.dropout(X, 0.8), 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv1, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.compat.v1.layers.max_pooling2d(conv2, 2, 2)
        conv3 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv2, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv4 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv3, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv4 = tf.compat.v1.layers.max_pooling2d(conv4, 2, 2)
        conv5 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv4, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv6 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv5, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv7 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv6, 256, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv7 = tf.compat.v1.layers.max_pooling2d(conv7, 2, 2)
        conv8 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv7, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv9 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv8, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv10 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv9, 512, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv10 = tf.compat.v1.layers.max_pooling2d(conv10, 2, 2)
        conv11 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv10, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv12 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv11, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv13 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv12, 512, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv13 = tf.compat.v1.layers.max_pooling2d(conv13, 2, 2)
        full = tf.compat.v1.layers.flatten(conv13)
    elif str == 'D':
        learning_rate = 0.0001
        conv1 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(tf.nn.dropout(X, 0.8), 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv1, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.compat.v1.layers.max_pooling2d(conv2, 2, 2)
        conv3 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv2, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv4 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv3, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv4 = tf.compat.v1.layers.max_pooling2d(conv4, 2, 2)
        conv5 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv4, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv6 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv5, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv7 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv6, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv7 = tf.compat.v1.layers.max_pooling2d(conv7, 2, 2)
        conv8 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv7, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv9 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv8, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv10 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv9, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv10 = tf.compat.v1.layers.max_pooling2d(conv10, 2, 2)
        conv11 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv10, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv12 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv11, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv13 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv12, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv13 = tf.compat.v1.layers.max_pooling2d(conv13, 2, 2)
        full = tf.compat.v1.layers.flatten(conv13)
    elif str == 'E':
        learning_rate = 0.00001
        conv1 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(tf.nn.dropout(X, 0.2), 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv1, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv2 = tf.compat.v1.layers.max_pooling2d(conv2, 2, 2)
        conv3 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv2, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4))))
        conv4 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv3, 128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv4 = tf.compat.v1.layers.max_pooling2d(conv4, 2, 2)
        conv5 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv4, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv6 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv5, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv7 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv6, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv8 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv7, 256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv8 = tf.compat.v1.layers.max_pooling2d(conv8, 2, 2)
        conv9 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv8, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.95))
        conv10 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv9, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv11 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv10, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv12 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv11, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv12 = tf.compat.v1.layers.max_pooling2d(conv12, 2, 2)
        conv13 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv12, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv14 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv13, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv15 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv14, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv16 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(tf.compat.v1.layers.conv2d(conv15, 512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(5e-4)), momentum=0.9))
        conv16 = tf.compat.v1.layers.max_pooling2d(conv16, 2, 2)
        full = tf.compat.v1.layers.flatten(conv16)

# fc1 = tf.nn.relu(tf.compat.v1.layers.dense(tf.nn.dropout(full, 0.5), 4096))
# fc2 = tf.nn.relu(tf.compat.v1.layers.dense(tf.nn.dropout(fc1, 0.5), 4096))
fc1 = tf.nn.relu(tf.compat.v1.layers.dense(tf.nn.dropout(full, 0.5), 2048))
pred = tf.compat.v1.layers.dense(fc1, 10)

loss = tf.math.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(y_hot, pred))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()

sess.run(init)

def optimize_times(model):
    global learning_rate, results
    batch_size = 120
    for e in range(1, 101):
        if e == 40:
            learning_rate /= 10
        elif e == 80:
            learning_rate /= 10
        aug = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.10,
            height_shift_range=0.15,
            shear_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest")
        new_imgs = aug.flow(X_train, y_train, batch_size)
        gc.collect()
        batch_list = []
        counter = 0
        for batch, label in new_imgs:
            counter += batch.shape[0]
            sess.run(optimizer, feed_dict={X: batch, y: label})
            batch_list.append(sess.run(loss, feed_dict={X: batch, y: label}))
            print(model+str(e), counter, batch_list[-1])
            if counter >= 60000:
                break
        results.append([e, min(batch_list), accuracy(X_train, y_train), accuracy(X_valid, y_valid)])
        print(results[-1][0], results[-1][1], results[-1][2], results[-1][3], '(E, Loss, Training accuracy, Valid accuracy)')
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

def save_data(file):
    global results
    f = open(file + '.txt', 'w')
    for i in results:
        f.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(i[3]))
        f.write('\n')
    f.close()

for m in ['E']:
    results = []
    print('\n---------------', m, '---------------')
    create_layers(m)
    print('Learning rate: ', learning_rate)
    print('\nFinal Loss: ', optimize_times(m))
    print('\nTraining accuracy: ', accuracy(X_train, y_train))
    print('\nTest accuracy: ', accuracy(X_test, y_test))
    save_data(m)
