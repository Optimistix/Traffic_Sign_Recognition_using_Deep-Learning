#!/usr/bin/env python
# coding: utf-8

# Use first line below for local deployment, the second one for Docker
#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite

import os
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image


MODEL_NAME = os.getenv('MODEL_NAME', 'convnet_from_scratch_with_a_dropout_layer.keras.tflite')


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def prepare_input(x):
    return x / 255.0


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# url = https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00062.png

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }

    return result

link = 'https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00062.png'
event = {'url':link}
print(lambda_handler(event, None))
