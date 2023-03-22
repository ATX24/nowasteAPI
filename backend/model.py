#Install bento requirements
#Install libraries
#!pip install -r https://raw.githubusercontent.com/bentoml/BentoML/main/examples/quickstart/requirements.txt

import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
from tensorflow.python.ops.math_ops import Xlogy
import cv2
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from net import rohanNet
from sklearn.model_selection import train_test_split


#Load Dataset

r = os.listdir('DATASET/R1')

o = os.listdir('DATASET/O1')


labels = []

# 0 = r
# 1 = o
for i in range(0, 100):
  labels.append('0')

for i in range(0, 100):
  labels.append('1')

  
X = []
labels = labels
main_dir = 'DATASET'
i=0
for img in r[0:100]:
  img = cv2.imread(main_dir + '/R1/' + img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_array = Image.fromarray(img, 'RGB')
  # resize image to 227x227 which is the required input size of the Alexnet model
  img_rs = img_array.resize((227,227))
  # convert the image to array
  img_rs = np.array(img_rs)
  X.append(img_rs)
  
for img in o[0:100]:
  img = cv2.imread(main_dir + '/O1/' + img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_array = Image.fromarray(img, 'RGB')
  # resize image to 227x227 which is the required input size of the Alexnet model
  img_rs = img_array.resize((227,227))
  # convert the image to array
  img_rs = np.array(img_rs)
  X.append(img_rs)


#Build model

data = {}
train = {}
test = {}

data['features'] = X
data['labels'] = np.array(labels)
data['features'] = np.array(data['features'])


train['features'], test['features'], train['labels'], test['labels'] = train_test_split(data['features'], data['labels'], test_size=0.2, random_state=0)
X_train, y_train = train['features'], to_categorical(train['labels'])
X_test, y_test = test['features'], to_categorical(test['labels'])

#Make data generators
train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=30)
test_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=30)
print('# of training images:', train['features'].shape[0])
print('# of validation images:', test['features'].shape[0])
print(X_train.shape)

print(train_generator)
print(test_generator)

model = rohanNet()
#Change to 50 epochs later
clf = model.fit_generator(train_generator, epochs=30, validation_data=test_generator, validation_steps=50)
model.save("model.h5")

