import numpy as np
from scipy import misc
from keras import backend as K
from keras.models import load_model


class PredictAbstract:
    def __init__(self):
        self.alphabet = 21
        self.img_rows = 45
        self.img_cols = 125
        self.saved_path = "/tmp/model/model.hdf5"

    def data_proprocess(self, image_path_list):
        N_DATA = len(image_path_list)
        dataX = []

        for i in range(N_DATA):
            path = image_path_list[i]
            img = misc.imread(path).astype(np.float)  # load image
            grayim = np.dot(img[..., :3], [0.299, 0.587, 0.114])  # gray scale
            dataX.append(grayim)

        x_test = np.asarray(dataX)

        x_test_ = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        x_test_scaled = x_test_ / 255.

        return x_test_scaled

    def predict_from_image(self, image_path_list):
        raise NotImplementedError("To be implemented")


class PredictNative(PredictAbstract):
    def predict_from_image(self, image_path_list):
        def captcha_metric(y_true, y_pred):
            y_pred = K.reshape(y_pred, (-1, self.alphabet))
            y_true = K.reshape(y_true, (-1, self.alphabet))
            y_p = K.argmax(y_pred, axis=1)
            y_t = K.argmax(y_true, axis=1)
            r = K.mean(K.cast(K.equal(y_p, y_t), 'float32'))
            return r

        model = load_model(self.saved_path, custom_objects={'captcha_metric': captcha_metric})

        x_test_scaled = self.data_proprocess(image_path_list)

        y_pred = model.predict(x_test_scaled[:])

        return y_pred
