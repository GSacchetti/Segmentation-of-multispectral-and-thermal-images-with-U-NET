# -*- coding:Utf_8 -*-


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



def mymkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)

# Learning function of the U-NET model

def Training_Unet(mainPath='',Augmentation=True,nb_Augmentation=3,INPUT_WIDTH = 256,INPUT_HEIGHT = 256, INPUT_CHANNELS = 7, OUTPUT_CHANNELS=1, 
	Normalization ="Normalizing_by_image_by_column",Train_model=True, Load_weights=True,validation_split=0.1, batch_size = 5, epochs=30):
	PathTT=mainPath+'data/'

	EVALUATION_PATH=mainPath+"/EVALUATIONS/"+str(INPUT_WIDTH)+'_'+str(INPUT_CHANNELS)+'_'+ str(Augmentation)+'_'+str(nb_Augmentation)+'_'+str(Normalization)+'_'+str(batch_size)+'_'+str(epochs)

	# Creation of the log directory 
	mymkdir(mainPath+"/EVALUATIONS")
	mymkdir(EVALUATION_PATH)

	# Loading data training and test data
	size_image_input=[INPUT_WIDTH ,INPUT_HEIGHT,INPUT_CHANNELS]
	size_image_output=[INPUT_WIDTH ,INPUT_HEIGHT,OUTPUT_CHANNELS]

	X_Train,Y_Train,X_Test,Y_Test=load_train_test_data.load_training_test_data(mainPath=mainPath, size_image_Input=size_image_input,
	 size_mask_Output=size_image_output,augment=Augmentation, nb_augment=nb_Augmentation)

	#Check the max and min of each column of input data, it must equal to (1,1,1,1,1,1) and (0 0 0 0 0 0) respectively.
	print(np.max(X_Train[:,:,0]),np.max(X_Train[:,:,1]), np.max(X_Train[:,:,2]),np.max(X_Train[:,:,3]),np.max(X_Train[:,:,4]),np.max(X_Train[:,:,5]),np.max(X_Train[:,:,6]))
	print(np.min(X_Train[:,:,0]),np.min(X_Test[:,:,1]), np.max(X_Test[:,:,2]),np.max(X_Test[:,:,3]),np.max(X_Test[:,:,4]),np.max(X_Test[:,:,5]),np.max(X_Test[:,:,6]))

	# Normalizing data
	# Initializing vectors
	X_Train_Normal=np.zeros_like(X_Train)
	X_Test_Normal=np.zeros_like(X_Test)

	X_Train_Normal=data_normalizing.Normalizing_by_image_by_column(X_Train)
	X_Test_Normal=data_normalizing.Normalizing_by_image_by_column(X_Test)
	print("the shape of X_Train is  ",np.shape(X_Train))
	print("the shape of Y_Train is  ",np.shape(Y_Train))
	print("the shape of X_Test is  ",np.shape(X_Test))
	print("the shape of Y_test is  ",np.shape(Y_Test))
	print(np.min(X_Train_Normal), np.max(X_Train_Normal))
	print(np.min(X_Test_Normal), np.max(X_Test_Normal))

	X_Train_Normal=X_Train
	X_Test_Normal=X_Test

	# Training model 

	model_unet.Training_Model(Train_model=train_model, Load_weights=Load_weights, validation_split=validation_split, batch_size = batch_size, epochs=epochs,
		X_Train_Normal=X_Train_Normal,Y_Train=Y_Train,X_Test_Normal=X_Test_Normal,Y_Test=Y_Test,INPUT_WIDTH =INPUT_WIDTH, 
		INPUT_HEIGHT = INPUT_HEIGHT,INPUT_CHANNELS = INPUT_CHANNELS,EVALUATION_PATH=EVALUATION_PATH)


# Loading the weights model U-NET if it exists (True or False)
Load_weights =False

#Choose whether or not to learn the model
train_model=True

#mainPath='/Users/rafikarezki/Desktop/Projet-Master/U-NET/'
mainPath=os.getcwd()

Training_Unet(mainPath=mainPath,Augmentation=True,nb_Augmentation=3,INPUT_WIDTH = 96,INPUT_HEIGHT = 96, INPUT_CHANNELS = 7, OUTPUT_CHANNELS=1, 
	Normalization ="Normalisation_by_image_by_column",Train_model=train_model, Load_weights=Load_weights,validation_split=0.1, batch_size = 20, epochs=30)


