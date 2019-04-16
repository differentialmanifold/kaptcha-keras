from __future__ import print_function
import os, argparse
import tensorflow as tf
from utils import *

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data_num', metavar='N', type=int, nargs='?', default=100,
                    help='data num for process')

args = parser.parse_args()

data_num = args.data_num
print('data number for process is {}'.format(data_num))

data_dir = "/tmp/data"
saved_dir = "/tmp/model"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

N_DATA = data_num
N_TRAIN = int(N_DATA * 0.9)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.io.serialize_tensor(image)
    return image


def preprocess_labels(labels):
    image_labels = []
    for i in range(len(labels)):
        image_labels.append(str2onehot(labels[i]))
    return image_labels


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def create_serialized_image_ds(all_image_paths):
    ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(load_and_preprocess_image,
                                                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


def create_serialized_label_ds(all_image_labels):
    ds = tf.data.Dataset.from_tensor_slices(all_image_labels).map(tf.io.serialize_tensor,
                                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


image_paths = []

for i in range(N_DATA):
    path = data_dir + "/%d.jpg" % (i)
    image_paths.append(path)

with open(data_dir + '/pass.txt') as f:
    image_labels = f.read().split(' ')[:N_DATA]
    image_labels = preprocess_labels(image_labels)

image_paths_tr = image_paths[:N_TRAIN]
image_paths_te = image_paths[N_TRAIN:]
image_labels_tr = image_labels[:N_TRAIN]
image_labels_te = image_labels[N_TRAIN:]

serialized_image_ds_train = create_serialized_image_ds(image_paths_tr)

tfrec = tf.data.experimental.TFRecordWriter(saved_dir + '/images_train.tfrec')
tfrec.write(serialized_image_ds_train)

serialized_label_ds_train = create_serialized_label_ds(image_labels_tr)

tfrec = tf.data.experimental.TFRecordWriter(saved_dir + '/labels_train.tfrec')
tfrec.write(serialized_label_ds_train)

serialized_image_ds_test = create_serialized_image_ds(image_paths_te)

tfrec = tf.data.experimental.TFRecordWriter(saved_dir + '/images_test.tfrec')
tfrec.write(serialized_image_ds_test)

serialized_label_ds_test = create_serialized_label_ds(image_labels_te)

tfrec = tf.data.experimental.TFRecordWriter(saved_dir + '/labels_test.tfrec')
tfrec.write(serialized_label_ds_test)
