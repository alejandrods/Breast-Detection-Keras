# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:09:58 2018

@author: aleja
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import gc
#from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Cropping2D

from random import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

############################################
#GET DATA
############################################

path_dir = './data/'

x_set = []
y_set = []

for path, dirs, files in os.walk('./data/'):
    for d in dirs:
        if '0' in d:
            for f in glob.iglob(os.path.join(path, d, '*.png')):
#                print(f)
                x_set.append(f)
                y_set.append(0)

        else:
            for f in glob.iglob(os.path.join(path, d, '*.png')):
#                print(f)
                x_set.append(f)
                y_set.append(1)
                
x_set_loaded = []
for items in x_set:
    im = mpimg.imread(items)
    x_set_loaded.append(im)
    
gc.collect()

x_set_resize = []
for items in x_set_loaded:
    x_set_resize.append(cv2.resize(items, (50,50)))
    
X_train, X_test, y_train, y_test = train_test_split(x_set_resize, y_set, test_size=0.20)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

y_train_enc= to_categorical(y_train, num_classes = 2)

print('SOME EXAMPLES: /n')
plt.imshow(X_train[0])
print(y_train[0])
plt.show()

plt.imshow(X_train[2000])
print(y_train[2000])

############################################
# MODEL
############################################  
  
input_shape = (X_train.shape[1], X_train.shape[2], 3)
model = Sequential()

model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(y_set)), activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
model.fit(X_train, y_train_enc, validation_split=0.2, shuffle=True, nb_epoch=20)

############################################
# PREDICT
############################################  

def predict_function(images):
    results = []
    for img in images:
        #Reshape
        reshape_input = np.array(img)[np.newaxis, :]
        
        prediction = model.predict(reshape_input)
        result = np.array(y_set)[prediction.argmax(axis=1)][0]
        results.append(result)
        
    return results 


pred = predict_function(X_test)
    
    
    
    
    
    
    