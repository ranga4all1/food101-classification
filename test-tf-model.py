
# ## 1.4 Using the model
# - Loading the model
# - Evaluating the model
# - Getting predictions

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.xception import preprocess_input

data_dir = './data/food-101/dataset_mini/'
# train_dir = data_dir + 'train'
# val_dir = data_dir + 'val'
test_dir = data_dir + 'test'

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)

model = keras.models.load_model('food101-model.h5')

model.evaluate(test_ds)

# path = './data/food-101/dataset_mini/test/onion_rings/1331610.jpg'
# path = './data/food-101/dataset_mini/test/waffles/1195540.jpg'
path = './data/food-101/dataset_mini/test/samosa/144404.jpg'
img = load_img(path, target_size=(299, 299))

x = np.array(img)
X = np.array([x])
# X.shape

X = preprocess_input(X)

pred = model.predict(X)

classes = ['cup_cakes', 'french_fries', 'hamburger', 'pizza', 'ramen', 'onion_rings', 'samosa', 'waffles']

print(dict(zip(classes, pred[0])))
