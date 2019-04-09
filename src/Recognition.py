from __future__ import print_function
import argparse
from utils import *
from predict_utils import predict_from_image

alphabet = 21

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data_num', metavar='N', type=int, nargs='?', default=100,
                    help='data num for recognize')

args = parser.parse_args()

data_num = args.data_num
print('data number for recognize is {}'.format(data_num))

data_dir = "/tmp/testdata"

N_DATA = data_num
image_path_list = []

for i in range(N_DATA):
    image_path_list.append(data_dir + "/%d.jpg" % (i))

predict_from_image_class = predict_from_image.PredictNative()

y_pred = predict_from_image_class.predict_from_image(image_path_list)

y_pred = np.asarray(y_pred)

f = open(data_dir + '/pass.txt')
labelY = f.read().split(' ')[:N_DATA]

dataY = []
for y in labelY:
    dataY.append(str2onehot(y))

y_test = np.asarray(dataY)


def getPharse(onehot):
    res = ""
    onehot = onehot.reshape(-1, alphabet)
    onehot = np.argmax(onehot, axis=1)
    for e in onehot:
        res += list_char[e]
    return res


y_pred = y_pred[:N_DATA]
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
