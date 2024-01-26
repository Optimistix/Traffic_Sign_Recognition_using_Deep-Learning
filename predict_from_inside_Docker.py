#!/usr/bin/env python
# coding: utf-8

# Use first line below for local deployment, the second one for Docker
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite

import os
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

from tensorflow import keras


MODEL_NAME = os.getenv('MODEL_NAME', 'convnet_from_scratch_with_a_dropout_layer.keras.tflite')

model_file = 'convnet_from_scratch_with_a_dropout_layer.keras.h5'
model = keras.models.load_model(model_file)


#classes = np.array(['Speed limit 20km/h',
classes = ['Speed limit 20km/h',
            'Speed limit 30km/h', 
            'Speed limit 50km/h', 
            'Speed limit 60km/h', 
            'Speed limit 70km/h', 
            'Speed limit 80km/h', 
            'End of speed limit 80km/h', 
            'Speed limit 100km/h', 
            'Speed limit 120km/h', 
            'No passing', 
            'No passing veh over 3.5 tons', 
            'Right-of-way at intersection', 
            'Priority road', 
            'Yield', 
            'Stop', 
            'No vehicles', 
            'Veh > 3.5 tons prohibited', 
            'No entry', 
            'General caution', 
            'Dangerous curve left', 
            'Dangerous curve right', 
            'Double curve', 
            'Bumpy road', 
            'Slippery road', 
            'Road narrows on the right', 
            'Road work', 
            'Traffic signals', 
            'Pedestrians', 
            'Children crossing', 
            'Bicycles crossing', 
            'Beware of ice/snow',
            'Wild animals crossing', 
            'End speed + passing limits', 
            'Turn right ahead', 
            'Turn left ahead', 
            'Ahead only', 
            'Go straight or right', 
            'Go straight or left', 
            'Keep right', 
            'Keep left', 
            'Roundabout mandatory', 
            'End of no passing', 
            'End no passing veh > 3.5 tons' ]
            #'End no passing veh > 3.5 tons' ])


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

# url = https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00051.png

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(32, 32))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    #preds = interpreter.get_tensor(output_index)
 
    #predictions = classes[preds.argmax(axis=0)]
    #predictions = classes[np.argmax(preds)]

    preds = model.predict(X).round(4)

    float_predictions = preds[0].tolist()

    i = np.argmax(float_predictions)

    preds_dict =  dict(zip(classes, float_predictions))
    predicted_class = list(preds_dict.keys())[i]
    probability_of_predicted_class = preds_dict[list(preds_dict.keys())[i]]
    return(predicted_class, probability_of_predicted_class)
    #return classes[i]
    #return preds
    #return predictions


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    
    result = {
        'Predicted Class and its probability': pred
    }

    return result


link = 'https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00051.png'
#link = 'https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00062.png'
#link = 'https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00004_00000_00007.png'

event = {'url':link}
print(lambda_handler(event, None))
