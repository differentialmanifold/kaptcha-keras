# Kaptcha with keras

## Generate data from Java

The code is in kaptcha_generator folder.

```bash
nohup sudo docker run --rm -v $PWD:/tmp -w /tmp openjdk:8 java -jar example-1.0-SNAPSHOT-spring-boot.jar 100000 /tmp/data &
```

## Preprocess data

my-tensorflow is a docker image build from tensorflow/tensorflow:1.13.1-py3 with keras, scipy and Pillow.

Read data from kaptcha_generator/data, save preprocessed data to model folder.

```bash
nohup sudo docker run --rm -v $PWD/src:/tmp/src -v $PWD/model:/tmp/model -v ~/Developer/Java/kaptcha/data:/tmp/data  -w /tmp my-tensorflow python -u src/Preprocessing.py 100000 > Preprocessing.out 2>&1 &
```

## Train

Read preprocessed data from model folder, and save model.hdf5 to model folder.

Arguments is the epochs to train.

```bash
nohup sudo docker run --rm -v $PWD/src:/tmp/src -v $PWD/model:/tmp/model  -v ~/tensorboard:/tmp/tensorboard -w /tmp my-tensorflow python -u src/Train.py 1000 > train.out 2>&1 &
```

## Recognition

Generate test data to kaptcha_generator/testdata, and predict from trained model.

```bash
nohup sudo docker run --rm -v $PWD/src:/tmp/src -v $PWD/model:/tmp/model  -v ~/Developer/Java/kaptcha/testdata:/tmp/testdata -w /tmp my-tensorflow python -u src/Recognition.py 100 > recognition.out 2>&1 &
```

## Comment

The project is based on `https://github.com/dukn/Captcha-recognition-Keras`.