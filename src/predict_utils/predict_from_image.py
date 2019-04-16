import os
import tensorflow as tf
import requests
from flask import json

from predict_utils.app import data_proprocess


class PredictAbstract:
    def __init__(self):
        self.alphabet = 21
        self.img_rows = 45
        self.img_cols = 125
        self.saved_path = "/tmp/tensorflow_serving/kaptcha_tensorflow"

    def predict_from_image(self, image_path_list):
        raise NotImplementedError("To be implemented")


class PredictNative(PredictAbstract):
    def predict_from_image(self, image_path_list):
        x_test_scaled = data_proprocess(image_path_list)

        tensorflow_model_path = os.path.join(self.saved_path, max(os.listdir(self.saved_path)))

        model = tf.contrib.saved_model.load_keras_model(tensorflow_model_path)

        y_pred = model.predict(x_test_scaled)

        return y_pred


class PredictTensorflowServing(PredictAbstract):
    def predict_from_image(self, image_path_list):
        flask_query_path = "http://flask_ip:5000/image_pred"
        payload = {
            "image_path_list": image_path_list
        }
        resp = requests.post(flask_query_path, json=payload)

        resp = json.loads(resp.content.decode('utf-8'))

        y_pred = resp['y_pred']['predictions']

        return y_pred
