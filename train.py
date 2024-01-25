# import the needed dependencies
import glob
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from keras.models import load_model

# Read in training data & inspect data shape
imgs_path = "./Train"
data = []
labels = []
CLASSES = 43
# using for loop to access each image
for i in range(CLASSES):
    img_path = os.path.join(imgs_path, str(i)) #0-42
    for img in os.listdir(img_path):
        im = Image.open(imgs_path + '/' + str(i) + '/' + img)
        im = im.resize((32,32))
        im = np.array(im)
        data.append(im)
        labels.append(i)
data = np.array(data)
labels = np.array(labels)
print("data[0]: ",data[0])
print("labels[0: ]",labels[0])

# Read in test data  
test = pd.read_csv("./Test.csv")
test_labels = test['ClassId'].values.tolist()

test_img_path = "./"
test_imgs = test['Path'].values
test_data = []

for img in test_imgs:
    im = Image.open(test_img_path + '/' + img)
    im = im.resize((32,32))
    im = np.array(im)
    test_data.append(im)
test_data = np.array(test_data)

# Split training data into training & validation 
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
print("training shape: ",x_train.shape, y_train.shape)
print("testing shape: ",x_val.shape, y_val.shape)
# convert interge label to one-hot data
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

print(y_train[1])



# Initialize Model
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(43, activation="softmax")(x)
cnn_with_dropout = keras.Model(inputs=inputs, outputs=outputs)


# Compile Model
cnn_with_dropout.compile(optimizer="rmsprop",
loss="categorical_crossentropy",
metrics=["accuracy"])

# Train the model
callbacks = [
keras.callbacks.ModelCheckpoint(
filepath="convnet_from_scratch_with_a_dropout_layer.keras",
save_best_only=True,
monitor="val_loss")
]

history = cnn_with_dropout.fit(x_train, y_train, 
                    epochs=30, 
                    validation_data=(x_val, y_val), 
                    callbacks=callbacks)


Loaded_Model = tf.keras.models.load_model('convnet_from_scratch_with_a_dropout_layer.keras')
tf.keras.models.save_model(Loaded_Model, 'convnet_from_scratch_with_a_dropout_layer.keras.h5') # Saving the Model in H5 Format
tf.keras.models.save_model(Loaded_Model, 'convnet_from_scratch_with_a_dropout_layer.keras.tf') # Saving the Model in TF Format
