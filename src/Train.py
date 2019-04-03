from __future__ import print_function
import os
import gzip, pickle, argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('epochs_num', metavar='N', type=int, nargs='?', default=20,
                    help='epochs num for train')

args = parser.parse_args()

saved_dir = "/tmp/model"

saved_path = saved_dir + '/model.hdf5'

tensorboard_dir = '/tmp/tensorboard/kaptcha'

batch_size = 32
digit = 4
alphabet = 21
num_classes = digit * alphabet
epochs = args.epochs_num
print('epochs num is {}'.format(epochs))

img_rows, img_cols = 45, 125

f = gzip.open(saved_dir + '/data.pkl.gz', 'rb')
loaded_object = pickle.load(f)
x_train, x_test, y_train, y_test = loaded_object
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train /= 255.
x_test /= 255.
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


def captcha_metric(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, alphabet))
    y_true = K.reshape(y_true, (-1, alphabet))
    y_p = K.argmax(y_pred, axis=1)
    y_t = K.argmax(y_true, axis=1)
    r = K.mean(K.cast(K.equal(y_p, y_t), 'float32'))
    return r


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0,
                                              amsgrad=False),
              metrics=[captcha_metric])

if os.path.isfile(saved_path):
    model = load_model(saved_path, custom_objects={'captcha_metric': captcha_metric})
    print('load model from {}'.format(saved_path))

tensorboard = TensorBoard(log_dir=tensorboard_dir, write_graph=False)

checkpointer = ModelCheckpoint(filepath=saved_path, verbose=1, save_best_only=True, period=10)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard, checkpointer])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
