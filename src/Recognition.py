from __future__ import print_function
import argparse
from keras import backend as K
from keras.models import load_model
from scipy import misc
from utils import *

digit = 4
alphabet = 21
num_classes = digit * alphabet

img_rows, img_cols = 45, 125

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data_num', metavar='N', type=int, nargs='?', default=100,
                    help='data num for recognize')

args = parser.parse_args()

data_num = args.data_num
print('data number for recognize is {}'.format(data_num))

data_dir = "/tmp/testdata"

saved_dir = "/tmp/model"

saved_path = saved_dir + '/model.hdf5'

N_DATA = data_num
dataX = []

for i in range(N_DATA):
    path = data_dir + "/%d.jpg" % (i)
    img = misc.imread(path).astype(np.float)  # load image
    grayim = np.dot(img[..., :3], [0.299, 0.587, 0.114])  # gray scale
    dataX.append(grayim)

f = open(data_dir + '/pass.txt')
labelY = f.read().split(' ')[:N_DATA]

dataY = []
for y in labelY:
    dataY.append(str2onehot(y))


def captcha_metric(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, alphabet))
    y_true = K.reshape(y_true, (-1, alphabet))
    y_p = K.argmax(y_pred, axis=1)
    y_t = K.argmax(y_true, axis=1)
    r = K.mean(K.cast(K.equal(y_p, y_t), 'float32'))
    return r


x_test = np.asarray(dataX)
y_test = np.asarray(dataY)

x_test_ = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test_scaled = x_test_ / 255.

model = load_model(saved_path, custom_objects={'captcha_metric': captcha_metric})


def getPharse(onehot):
    res = ""
    onehot = onehot.reshape(-1, alphabet)
    onehot = np.argmax(onehot, axis=1)
    for e in onehot:
        res += list_char[e]
    return res


y_pred = model.predict(x_test_scaled[:N_DATA])
y = y_test[:N_DATA]

matched_num = 0
for i in range(N_DATA):
    real = getPharse(y[i])
    pred = getPharse(y_pred[i])
    print(real, pred, real == pred)
    if real == pred:
        matched_num += 1

print('Matched number is {}'.format(matched_num))
print('Total number is {}'.format(N_DATA))
print('Successful rate is {}'.format(matched_num / N_DATA))
