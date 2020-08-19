#This program is for loading data and training the U-NET model

from PIL import Image
import matplotlib.pyplot as plt
import numpy  as np
import math
import os
import sys
import random
from skimage.io import imsave,imread

import time
import datetime

import model_unet
import measurment_functions
import data_normalizing
import load_train_test_data


from tensorflow import keras
from keras.models import Model,load_model,save_model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from os.path import join


def mymkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# Learning function of the U-NET model

def Training_Unet(mainPath='',Augmentation=True,nb_Augmentation=3,INPUT_WIDTH = 256,INPUT_HEIGHT = 256, CHANNELS = [0,1,2,3,4,5,6], 
    Normalization ="Normalizing_by_image_by_column",Train_model=True, Load_weights=True,validation_split=0.1, batch_size = 5, epochs=30):
    PathTT= join(mainPath,'data')

    # EVALUATION folder will receive the accuracy graphs
    EVALUATION_PATH=mainPath+"/EVALUATIONS/"+str(INPUT_WIDTH)+'_'+str(len(CHANNELS))+'_'+ str(Augmentation)+'_'+str(nb_Augmentation)+'_'+str(Normalization)+'_'+str(batch_size)+'_'+str(epochs)

    # Creation of the log directory 
    mymkdir(mainPath+"/EVALUATIONS")
    mymkdir(EVALUATION_PATH)

    # Loading data training and test data
    size_image_input=[INPUT_WIDTH ,INPUT_HEIGHT,len(CHANNELS)]
    size_image_output=[INPUT_WIDTH ,INPUT_HEIGHT,1]

    X_Train,Y_Train,X_Test,Y_Test=load_train_test_data.load_training_test_data(mainPath=mainPath, size_image=[INPUT_WIDTH ,INPUT_HEIGHT], input_channels=CHANNELS,
        augment=Augmentation, nb_augment=nb_Augmentation)
    print('Loading of the training and validation datasets completed')
    #Check the max and min of each column of input data,
    print("Initial data")
    print([np.max(X_Train[:,:,cid]) for cid,i in enumerate(CHANNELS)])
    print([np.min(X_Train[:,:,cid]) for cid,i in enumerate(CHANNELS)])

    # Normalizing data
    # Initializing vectors
    X_Train_Normal=np.zeros_like(X_Train)
    X_Test_Normal=np.zeros_like(X_Test)

    #X_Train_Normal=data_normalizing.Normalizing_by_image_by_column(X_Train)
    #X_Test_Normal=data_normalizing.Normalizing_by_image_by_column(X_Test)
    if not Normalization is None:
       normfunc =  getattr(data_normalizing, Normalization)
       X_Train_Normal=normfunc(X_Train)
       X_Test_Normal=normfunc(X_Test)
    else :
       X_Train_Normal=X_Train
       X_Test_Normal=X_Test

    #Check again max and mon of each column of input data, after normalization it must be equal to (1,1,1,1,1,1) and (0 0 0 0 0 0) respectively.
    print("the shape of X_Train is  ",np.shape(X_Train))
    print("the shape of normalized X_Train is  ",np.shape(X_Train_Normal))
    print("the shape of Y_Train is  ",np.shape(Y_Train))
    print([np.max(X_Train_Normal[:,:,cid]) for cid,i in enumerate(CHANNELS)])
    print([np.min(X_Train_Normal[:,:,cid]) for cid,i in enumerate(CHANNELS)])
    print("the shape of X_Test is  ",np.shape(X_Test))
    print("the shape of normalized X_Test is  ",np.shape(X_Test_Normal))
    print("the shape of Y_test is  ",np.shape(Y_Test))
    print([np.max(X_Test_Normal[:,:,cid]) for cid,i in enumerate(CHANNELS)])
    print([np.min(X_Test_Normal[:,:,cid]) for cid,i in enumerate(CHANNELS)])


    # Training model 

    model_unet.Training_Model(Train_model=train_model, Load_weights=Load_weights, validation_split=validation_split, batch_size = batch_size, epochs=epochs,
        X_Train_Normal=X_Train_Normal,Y_Train=Y_Train,X_Test_Normal=X_Test_Normal,Y_Test=Y_Test,INPUT_WIDTH =INPUT_WIDTH, 
        INPUT_HEIGHT = INPUT_HEIGHT,INPUT_CHANNELS = len(CHANNELS),EVALUATION_PATH=EVALUATION_PATH)


# Loading the weights model U-NET if it exists (True or False)
Load_weights = False

#Choose whether or not to learn the model
train_model=True

#Change mainPath to your path
mainPath='/content/gdrive/My Drive/U-NET'

import sys
import ast

if len(sys.argv) >1:
    CHANNELS = ast.literal_eval(sys.argv[1])
else:
    CHANNELS = [0,1,2,3,4,5,6]


Training_Unet(mainPath=mainPath,Augmentation=True,nb_Augmentation=3,INPUT_WIDTH = 96,INPUT_HEIGHT = 96, CHANNELS = CHANNELS,
    Normalization ="Normalizing_by_image_by_column",Train_model=train_model, Load_weights=Load_weights,validation_split=0.1, batch_size = 20, epochs=30)
