import os
import shutil
import pickle
from PIL import Image
from hashlib import sha256
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras import models
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, image

from keras.applications.xception import Xception, decode_predictions
from keras.applications.xception import preprocess_input as preprocess_input_xce

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_incV3

from keras.applications.efficientnet import EfficientNetB7
from keras.applications.efficientnet import preprocess_input as preprocess_input_eff

### Variables

PIXELS = 300
path = './images/'
trainList = pd.read_csv('train.csv')
testList = pd.read_csv('test.csv')

## Create directories and copy images
### Create directories

if not os.path.exists('./train'):
    os.mkdir('./train')
if not os.path.exists('./train/cup'):
    os.mkdir('./train/cup')     
if not os.path.exists('./train/fork'):
    os.mkdir('./train/fork')     
if not os.path.exists('./train/glass'):
    os.mkdir('./train/glass')     
if not os.path.exists('./train/knife'):
    os.mkdir('./train/knife')     
if not os.path.exists('./train/plate'):
    os.mkdir('./train/plate')     
if not os.path.exists('./train/spoon'):
    os.mkdir('./train/spoon')     
    
if not os.path.exists('./test'):
    os.mkdir('./test')
    
if not os.path.exists('./Models'):
    os.mkdir('./Models')    

### Copy images to directories

for img in os.listdir('./images'):
    
    if (not os.path.exists(f'./train/{img}')) and (not(os.path.exists(f'./test/{img}'))):
        imgName = int(img.split('.')[0])
        
        if imgName in trainList.Id.values:
            
            imgLabel = trainList[trainList.Id == imgName].label.values
            
            if imgLabel == 'cup':
                shutil.copy(f'./images/{img}', f'./train/cup/{img}')
            elif imgLabel == 'fork':
                shutil.copy(f'./images/{img}', f'./train/fork/{img}')
            elif imgLabel == 'glass':
                shutil.copy(f'./images/{img}', f'./train/glass/{img}')
            elif imgLabel == 'knife':
                shutil.copy(f'./images/{img}', f'./train/knife/{img}')
            elif imgLabel == 'plate':
                shutil.copy(f'./images/{img}', f'./train/plate/{img}')
            elif imgLabel == 'spoon':
                shutil.copy(f'./images/{img}', f'./train/spoon/{img}')

        else:            
            
            shutil.copy(f'./images/{img}', f'./test/{img}')
            
totalTrainImgs = len(os.listdir("./train/cup")) + len(os.listdir("./train/fork")) + \
                len(os.listdir("./train/glass")) + len(os.listdir("./train/knife")) + \
                len(os.listdir("./train/plate")) + len(os.listdir("./train/spoon"))

totalTestImgs = len(os.listdir("./test"))

print(f'There is a total of {totalTrainImgs} images in the training set')
print(f'There is a total of {totalTestImgs} images in the test set')            

### Image Generators

def ProcInputs(modelName):
    
    preprocessInputs = {'Xception': preprocess_input_xce,
                    'InceptionV3': preprocess_input_incV3,
                    'efficientnetB7': preprocess_input_eff,
                   }
    return preprocessInputs[modelName]


def DataGenerators (modelName, split=0.2):
   
    preprocess_input = ProcInputs(modelName)
    
    dataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        validation_split=split,
                                      )    

    trainGenerator = dataGenerator.flow_from_directory(directory='./train',
                                                         batch_size=32,
                                                         target_size=(PIXELS, PIXELS), 
                                                         subset="training",
                                                         shuffle=True,
                                                         class_mode='categorical')

    valGenerator = dataGenerator.flow_from_directory(directory='./train',
                                                       batch_size=16,
                                                       target_size=(PIXELS, PIXELS),
                                                       subset="validation",
                                                       shuffle=True,
                                                       class_mode='categorical')
    
    return trainGenerator, valGenerator

# We will be training model EfficientNet Only

def GetModel(modelName):
    
    basemodel_Xce = tf.keras.applications.Xception(weights='imagenet',
                                                   include_top=False,
                                                   input_shape=(PIXELS, PIXELS, 3))

    basemodel_Inc = tf.keras.applications.InceptionV3(weights='imagenet',
                                                      include_top=False,
                                                      input_shape=(PIXELS, PIXELS, 3))
    
    basemodel_EffB7 = tf.keras.applications.EfficientNetB7(weights='imagenet',
                                                       include_top=False,
                                                       input_shape=(PIXELS, PIXELS, 3)) 

   
    basemodels = {'Xception': basemodel_Xce,
                  'InceptionV3': basemodel_Inc,
                  'efficientnetB7': basemodel_EffB7,
                 }
    
    return basemodels[modelName]

### Building the model

def MakeModel(modelName = 'Xception', learning_rate = 0.001, size_inner=16, droprate=0.3):
    
    base_model = GetModel(modelName)
    
    base_model.trainable = False

    inputs = Input(shape=(PIXELS, PIXELS, 3))
    base = base_model(inputs, training=False)
    vectors = layers.GlobalAveragePooling2D()(base)
    
    inner = layers.Dense(size_inner, activation='relu')(vectors)
    drop = layers.Dropout(droprate)(inner)
    
    outputs = layers.Dense(6, activation='softmax')(drop)
    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=learning_rate)
    loss = CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model

### Checkpointing

def GetCheckpoint(modelName):
    checkpoint = keras.callbacks.ModelCheckpoint("./Models/" + modelName + '_{epoch:02d}_{val_accuracy:.3f}.h5',
                                                 save_best_only=True,
                                                 #save_weights_only=True,
                                                 monitor='val_accuracy',
                                                 mode='max')
    return checkpoint

def GetCheckpointFull(modelName):
    checkpoint = keras.callbacks.ModelCheckpoint("./Models/" + modelName + '_{epoch:02d}_{accuracy:.3f}.h5',
                                                 save_best_only=True,
                                                 #save_weights_only=True,
                                                 monitor='accuracy',
                                                 mode='max')
    return checkpoint


### Model preparation

def RunModel(learningRates, modelName, steps_per_epoch=50, epochs=10):
    
    scores = {}
    size = 128
    droprate = 0.3
    trainGenerator, valGenerator = DataGenerators (modelName)
    print()

    for lr in learningRates:
        print(f'Learning rate: {lr}')
        checkpoint = GetCheckpoint(modelName + '_' + f'lr{lr}')
        model = MakeModel(modelName=modelName, learning_rate=lr)
        history = model.fit(trainGenerator,
                            validation_data=valGenerator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            callbacks=[checkpoint])
        scores[lr] = history.history
        print('\n')
    
    return model, history, scores

### Plot history

def PlotHistory(scores, lims=[0.9,0.98], xaxis=10):
        
    for lr , hist in scores.items():
        plt.plot(hist['val_accuracy'], label=lr)

    plt.xticks(np.arange(xaxis))
    plt.ylim(lims[0], lims[1])
    plt.legend()

### Best rate is 0.001

learningRates = [0.001]
modelName = 'efficientnetB7'
model_EN, history_EN, scores_EN = RunModel(learningRates, modelName)

PlotHistory(scores_EN, lims=(0.95,0.99))