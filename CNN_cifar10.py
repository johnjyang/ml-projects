import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import gc
from sklearn.metrics import confusion_matrix

'''
1. ssh -p 616 john@172.251.186.111
2. In a new terminal, sftp -oPort=616 john@172.251.186.111
    - cd /mnt/big 
3. tmux a -t JohnsCifar
4. source ~/envs/keras/bin/activate
5. Run the code
6. Ctr B + (then) D
'''
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

learning_rate = 0.001

conv1 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(X, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)

conv2 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv1, 16, 3, 2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv3 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv2, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv3 = conv3 + tf.compat.v1.layers.conv2d(conv1, 16, 1, 2)
conv4 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv3, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv5 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv4, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv5 = conv5 + tf.compat.v1.layers.conv2d(conv3, 16, 1)
conv6 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv5, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv7 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv6, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv7 = conv7 + tf.compat.v1.layers.conv2d(conv5, 16, 1)
conv8 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv7, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv9 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv8, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv9 = conv9 + tf.compat.v1.layers.conv2d(conv7, 16, 1)
conv10 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv9, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv11 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv10, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv11 = conv11 + tf.compat.v1.layers.conv2d(conv9, 16, 1)
conv12 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv11, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv13 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv12, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv13 = conv13 + tf.compat.v1.layers.conv2d(conv11, 16, 1)
conv14 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv13, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv15 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv14, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv15 = conv15 + tf.compat.v1.layers.conv2d(conv13, 16, 1)
conv16 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv15, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv17 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv16, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv17 = conv17 + tf.compat.v1.layers.conv2d(conv15, 16, 1)
conv18 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv17, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv19 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv18, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv19 = conv19 + tf.compat.v1.layers.conv2d(conv17, 16, 1)
conv20 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv19, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv21 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv20, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv21 = conv21 + tf.compat.v1.layers.conv2d(conv19, 16, 1)
conv22 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv21, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv23 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv22, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv23 = conv23 + tf.compat.v1.layers.conv2d(conv21, 16, 1)
conv24 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv23, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv25 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv24, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv25 = conv25 + tf.compat.v1.layers.conv2d(conv23, 16, 1)
conv26 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv25, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv27 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv26, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv27 = conv27 + tf.compat.v1.layers.conv2d(conv25, 16, 1)
conv28 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv27, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv29 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv28, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv29 = conv29 + tf.compat.v1.layers.conv2d(conv27, 16, 1)
conv30 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv29, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv31 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv30, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv31 = conv31 + tf.compat.v1.layers.conv2d(conv29, 16, 1)
conv32 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv31, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv33 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv32, 16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv33 = conv33 + tf.compat.v1.layers.conv2d(conv31, 16, 1)
conv34 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv33, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv35 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv34, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv35 = conv35 + tf.compat.v1.layers.conv2d(conv33, 16, 1)
conv36 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv35, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv37 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv36, 16, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv37 = conv37 + tf.compat.v1.layers.conv2d(conv35, 16, 1)

conv38 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv37, 32, 3, 2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv39 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv38, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv39 = conv39 + tf.compat.v1.layers.conv2d(conv37, 32, 1, 2)
conv40 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv39, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv41 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv40, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv41 = conv41 + tf.compat.v1.layers.conv2d(conv39, 32, 1)
conv42 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv41, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv43 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv42, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv43 = conv43 + tf.compat.v1.layers.conv2d(conv41, 32, 1)
conv44 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv43, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv45 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv44, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv45 = conv45 + tf.compat.v1.layers.conv2d(conv43, 32, 1)
conv46 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv45, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv47 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv46, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv47 = conv47 + tf.compat.v1.layers.conv2d(conv45, 32, 1)
conv48 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv47, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv49 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv48, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv49 = conv49 + tf.compat.v1.layers.conv2d(conv47, 32, 1)
conv50 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv49, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv51 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv50, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv51 = conv51 + tf.compat.v1.layers.conv2d(conv49, 32, 1)
conv52 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv51, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv53 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv52, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv53 = conv53 + tf.compat.v1.layers.conv2d(conv51, 32, 1)
conv54 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv53, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv55 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv54, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv55 = conv55 + tf.compat.v1.layers.conv2d(conv53, 32, 1)
conv56 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv55, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv57 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv56, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv57 = conv57 + tf.compat.v1.layers.conv2d(conv55, 32, 1)
conv58 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv57, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv59 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv58, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv59 = conv59 + tf.compat.v1.layers.conv2d(conv57, 32, 1)
conv60 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv59, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv61 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv60, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv61 = conv61 + tf.compat.v1.layers.conv2d(conv59, 32, 1)
conv62 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv61, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv63 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv62, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv63 = conv63 + tf.compat.v1.layers.conv2d(conv61, 32, 1)
conv64 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv63, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv65 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv64, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv65 = conv65 + tf.compat.v1.layers.conv2d(conv63, 32, 1)
conv66 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv65, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv67 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv66, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv67 = conv67 + tf.compat.v1.layers.conv2d(conv65, 32, 1)
conv68 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv67, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv69 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv68, 32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv69 = conv69 + tf.compat.v1.layers.conv2d(conv67, 32, 1)
conv70 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv69, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv71 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv70, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv71 = conv71 + tf.compat.v1.layers.conv2d(conv69, 32, 1)
conv72 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv71, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv73 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv72, 32, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv73 = conv73 + tf.compat.v1.layers.conv2d(conv71, 32, 1)

conv74 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv73, 64, 3, 2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv75 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv74, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv75 = conv75 + tf.compat.v1.layers.conv2d(conv73, 64, 1, 2)
conv76 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv75, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv77 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv76, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv77 = conv77 + tf.compat.v1.layers.conv2d(conv75, 64, 1)
conv78 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv77, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv79 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv78, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv79 = conv79 + tf.compat.v1.layers.conv2d(conv77, 64, 1)
conv80 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv79, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv81 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv80, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv81 = conv81 + tf.compat.v1.layers.conv2d(conv79, 64, 1)
conv82 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv81, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv83 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv82, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv83 = conv83 + tf.compat.v1.layers.conv2d(conv81, 64, 1)
conv84 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv83, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv85 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv84, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv85 = conv85 + tf.compat.v1.layers.conv2d(conv83, 64, 1)
conv86 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv85, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv87 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv86, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv87 = conv87 + tf.compat.v1.layers.conv2d(conv85, 64, 1)
conv88 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv87, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv89 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv88, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv89 = conv89 + tf.compat.v1.layers.conv2d(conv87, 64, 1)
conv90 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv89, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv91 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv90, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv91 = conv91 + tf.compat.v1.layers.conv2d(conv89, 64, 1)
conv92 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv91, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv93 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv92, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv93 = conv93 + tf.compat.v1.layers.conv2d(conv91, 64, 1)
conv94 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv93, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv95 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv94, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv95 = conv95 + tf.compat.v1.layers.conv2d(conv93, 64, 1)
conv96 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv95, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv97 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv96, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv97 = conv97 + tf.compat.v1.layers.conv2d(conv95, 64, 1)
conv98 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv97, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv99 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv98, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv99 = conv99 + tf.compat.v1.layers.conv2d(conv97, 64, 1)
conv100 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv99, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv101 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv100, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv101 = conv101 + tf.compat.v1.layers.conv2d(conv99, 64, 1)
conv102 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv101, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv103 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv102, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv103 = conv103 + tf.compat.v1.layers.conv2d(conv101, 64, 1)
conv104 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv103, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv105 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv104, 64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv105 = conv105 + tf.compat.v1.layers.conv2d(conv103, 64, 1)
conv106 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv105, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv107 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv106, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv107 = conv107 + tf.compat.v1.layers.conv2d(conv105, 64, 1)
conv108 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv107, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv109 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(tf.compat.v1.layers.conv2d(conv108, 64, 3,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))), momentum=.9)
conv109 = conv109 + tf.compat.v1.layers.conv2d(conv107, 64, 1)

conv110 = tf.keras.layers.GlobalAveragePooling2D()(conv109)

full = tf.compat.v1.layers.flatten(conv110)
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
        if e == 2:
            learning_rate *= 50
        elif e == 3:
            learning_rate *= 10
        elif e == 30:
            learning_rate /= 10
        elif e == 60:
            learning_rate /= 10
        elif e == 80:
            learning_rate /= 10
        '''
        elif e == 150:
            learning_rate /= 10
        elif e == 300:
            learning_rate /= 10
        elif e == 450:
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

saver.save(sess, './models/110c_200_l2_batchNorm')
save_data()

print("\nTest confuse:\n", confusion_matrix(y_test, np.argmax(sess.run(pred, feed_dict={X: X_test}), axis=1)))
print("\nLabels: ", words)
