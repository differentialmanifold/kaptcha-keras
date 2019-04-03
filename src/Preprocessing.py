from __future__ import print_function
import pickle, gzip, os, argparse
from scipy import misc
from utils import *

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
dataX = []

for i in range(N_DATA):
    if i % 1000 == 0:
        print("i is {}".format(i))
    path = data_dir + "/%d.jpg" % (i)
    img = misc.imread(path).astype(np.float)  # load image
    grayim = np.dot(img[..., :3], [0.299, 0.587, 0.114])  # gray scale
    dataX.append(grayim)

f = open(data_dir + '/pass.txt')
labelY = f.read().split(' ')[:N_DATA]

dataY = []
for y in labelY:
    dataY.append(str2onehot(y))

trX = dataX[:N_TRAIN]  # from 0 - N_TRAIN
teX = dataX[N_TRAIN:]  # from N_TRAIN to end
trY = dataY[:N_TRAIN]
teY = dataY[N_TRAIN:]

pickle.dump((trX, teX, trY, teY), gzip.open(saved_dir + "/data.pkl.gz", "wb"))
