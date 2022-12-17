#!/usr/bin/env python
# coding: utf-8

# # Converting Tensorflow model to TF-Lite model
# 
# - Tensorflow Lite is a lightweight alternative to Tensorflow that only focuses on inference
# - After conversion, test TF-Lite model for inference
# - Remove Tensorflow dependency

# import numpy as np

import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.applications.xception import preprocess_input


import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

# Convert to TF-Lite model
model = keras.models.load_model('food101-model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('food101-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# Test
interpreter = tflite.Interpreter(model_path='food101-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299, 299))

url = 'https://bit.ly/3PzCqJ2'  # samosa.jpg hosted on Github
X = preprocessor.from_url(url)

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


classes = ['cup_cakes', 'french_fries', 'hamburger', 'pizza', 'ramen', 'onion_rings', 'samosa', 'waffles']

print(dict(zip(classes, preds[0])))

