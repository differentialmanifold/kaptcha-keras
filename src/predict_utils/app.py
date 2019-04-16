import numpy as np
import tensorflow as tf
import requests
from flask import Flask, json, request, Response
from flask_cors import CORS

tf.enable_eager_execution()

app = Flask(__name__)
CORS(app)

alphabet = 21

img_rows = 45
img_cols = 125

tensorflow_serving_predict = 'http://localhost:8501/v1/models/kaptcha_tensorflow:predict'

ALPHABET = 'abcde2345678gfynmnpwx'
list_char = [c for c in ALPHABET]


def data_item_proprocess(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.reshape(image, [img_rows, img_cols, 3])
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image.numpy()


def data_proprocess(image_path_list):
    N_DATA = len(image_path_list)
    dataX = []

    for i in range(N_DATA):
        image_path = image_path_list[i]
        image_arr = data_item_proprocess(image_path)
        dataX.append(image_arr)

    x_test = np.asarray(dataX)

    return x_test


def transfer_image(image_path_list):
    x_test_scaled = data_proprocess(image_path_list)

    payload = {
        "instances": x_test_scaled.tolist()
    }

    r = requests.post(tensorflow_serving_predict, json=payload)

    y_pred = json.loads(r.content.decode('utf-8'))

    return y_pred


def captcha_metric(y_true, y_pred):
    K = tf.keras.backend
    y_pred = K.reshape(y_pred, (-1, alphabet))
    y_true = K.reshape(y_true, (-1, alphabet))
    y_p = K.argmax(y_pred, axis=1)
    y_t = K.argmax(y_true, axis=1)
    r = K.mean(K.cast(K.equal(y_p, y_t), 'float32'))
    return r


def transfer_model_from_keras_to_tensorflow(keras_path, tensorflow_path):
    model = tf.keras.models.load_model(keras_path, custom_objects={'captcha_metric': captcha_metric})
    saved_to_path = tf.contrib.saved_model.save_keras_model(
        model, tensorflow_path)
    print(saved_to_path)


def getPharse(onehot):
    res = ""
    onehot = onehot.reshape(-1, alphabet)
    onehot = np.argmax(onehot, axis=1)
    for e in onehot:
        res += list_char[e]
    return res


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    return transfer_response({'result': 'hello'})


@app.route('/image_pred', methods=['POST'])
def image_pred():
    # make sure tensorflow_serving is running
    print(request.headers)
    data = request.json
    if isinstance(data, str):
        data = json.loads(data)
    print(data)
    image_path_list = data['image_path_list']

    y_pred = transfer_image(image_path_list)

    obj = {'y_pred': y_pred}
    return transfer_response(obj)


@app.route('/image_pred_h', methods=['POST'])
def image_pred_h():
    # make sure tensorflow_serving is running
    print(request.headers)
    data = request.json
    if isinstance(data, str):
        data = json.loads(data)
    print(data)
    image_path_list = data['image_path_list']

    predict = transfer_image(image_path_list)

    predict = np.asarray(predict['predictions'])

    y_pred = []

    for i in range(len(predict)):
        y_pred.append(getPharse(predict[i]))

    obj = {'y_pred': y_pred}
    return transfer_response(obj)


@app.route('/transfer_model', methods=['POST'])
def transfer_model():
    data = request.json
    if isinstance(data, str):
        data = json.loads(data)
    print(data)
    keras_path = data['keras_path']
    tensorflow_path = data['tensorflow_path']

    transfer_model_from_keras_to_tensorflow(keras_path, tensorflow_path)

    obj = {'result': 0}
    return transfer_response(obj)


def transfer_response(obj, status_code=200):
    predict_results = json.dumps(obj, ensure_ascii=False)
    return Response(
        response=predict_results,
        mimetype="application/json; charset=UTF-8",
        status=status_code
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
