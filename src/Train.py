from __future__ import print_function
import argparse
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

from train_utils.train_utils import ModelCheckpoint

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('epochs_num', metavar='N', type=int, nargs='?', default=20,
                    help='epochs num for train')

parser.add_argument('data_num', metavar='N', type=int, nargs='?', default=100,
                    help='data num for process')

args = parser.parse_args()

saved_dir = "/tmp/model"

tensorflow_serving_model_path = "/tmp/tensorflow_serving/kaptcha_tensorflow"

tensorboard_dir = '/tmp/tensorboard/kaptcha'

batch_size = 32

digit = 4
alphabet = 21
num_classes = digit * alphabet

epochs = args.epochs_num
N_DATA = args.data_num
N_TRAIN = int(N_DATA * 0.9)
N_TEST = N_DATA - N_TRAIN
print('epochs num is {}'.format(epochs))
print('data num is {}'.format(N_DATA))

steps_per_epoch = int(tf.math.ceil(N_TRAIN / batch_size).numpy())
validation_steps = int(tf.math.ceil(N_TEST / batch_size).numpy())

img_rows, img_cols = 45, 125


def parse_image(x):
    result = tf.io.parse_tensor(x, out_type=tf.uint8)
    result = tf.reshape(result, [45, 125, 3])
    result = tf.cast(result, tf.float32)
    result /= 255.0  # normalize to [0,1] range
    return result


def parse_label(x):
    result = tf.io.parse_tensor(x, out_type=tf.int32)
    result = tf.reshape(result, (84,))
    return result


def ds_process(input_ds, batch_size):
    ds = input_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=100))
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


image_ds_train = tf.data.TFRecordDataset(saved_dir + '/images_train.tfrec').map(parse_image,
                                                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

label_ds_train = tf.data.TFRecordDataset(saved_dir + '/labels_train.tfrec').map(parse_label,
                                                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

image_label_ds_train = tf.data.Dataset.zip((image_ds_train, label_ds_train))

image_ds_test = tf.data.TFRecordDataset(saved_dir + '/images_test.tfrec').map(parse_image,
                                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

label_ds_test = tf.data.TFRecordDataset(saved_dir + '/labels_test.tfrec').map(parse_label,
                                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

image_label_ds_test = tf.data.Dataset.zip((image_ds_test, label_ds_test))

image_label_ds_train = ds_process(image_label_ds_train, batch_size)

image_label_ds_test = ds_process(image_label_ds_test, batch_size)


def captcha_metric(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, alphabet))
    y_true = K.reshape(y_true, (-1, alphabet))
    y_p = K.argmax(y_pred, axis=1)
    y_t = K.argmax(y_true, axis=1)
    r = K.mean(K.cast(K.equal(y_p, y_t), 'float32'))
    return r


input_shape = (img_rows, img_cols, 3)

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

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0,
                                                 amsgrad=False),
              metrics=[captcha_metric])

tensorboard = TensorBoard(log_dir=tensorboard_dir, write_graph=False)

checkpointer = ModelCheckpoint(filepath=tensorflow_serving_model_path, verbose=1, save_best_only=False, period=10)

model.fit(image_label_ds_train,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          verbose=1,
          validation_data=image_label_ds_test,
          validation_steps=validation_steps,
          callbacks=[tensorboard, checkpointer])
