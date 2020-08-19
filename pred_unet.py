#This script is for using the trained U-NET model, To make predictions on other images of our dataset.
# uploading packages 

from PIL import Image
import matplotlib.pyplot as plt
import numpy  as np
import math
import os
import sys
import ast
import random
from skimage.io import imsave,imread
import matplotlib.pyplot as plt

import time
import datetime

import model_unet 
import data_normalizing
import load_train_test_data
import measurment_functions

from os.path import join
from tensorflow import keras
from keras.models import Model,load_model,save_model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# Load_weights = True : if we load the existing model that is already driven.
# validation_split, batch_size, epochs are the parameters of the model.
# INPUT_WIDTH =96(256), INPUT_HEIGHT = 96(256), CHANNELS = [0,1,2,3,4,5,6] the sizes of our data.
# path : is the folder's path of image to predict.
# threshold : it is the discrimination value of the predicted pixels, we say a pixel belongs to the class 'leaves' 
# if its predicted value exceeds such threshold, otherwise it is of the calasse 'other'. To find an optimal threshold,
# you can use the function "optimal_threshold_desc" in the script "measurement_functions".
# Prediction_Images_Mean : 
# Normalization : is the type of normalization
# step : is the displacement step of the prediction patch
# CHANNELS : By default [0,1,2,3,4,5,6], select the channels wanted for the prediction, it must be the same channels as the ones selected for training.

#Functions to get weight values depending on the center of the image
def radius(x,y,input_size):
  return  (((x-(input_size/2))**2)+ ((y-(input_size/2))**2))

def weight(x,y,input_size):
  R = (radius(0,0,input_size))**(1/3)
  r = (radius(x,y,input_size))**(1/3)
  return (1/3)*(R-r)/(3*R)


def Prediction(Path='',INPUT_WIDTH =256, INPUT_HEIGHT = 256,CHANNELS = [0,1,2,3,4,5,6],threshold=0.7,Prediction_Images_Mean=True,
	Normalization='Normalizing_by_image_by_column',step=30):

  # Load model U-NET
  model=model_unet.Model_Unet(INPUT_WIDTH =INPUT_WIDTH , INPUT_HEIGHT = INPUT_HEIGHT,INPUT_CHANNELS = len(CHANNELS))
	# Load of weights's model
  model = load_model('model-U_NET.h5')

  EVALUATION_PATH='/content/gdrive/My Drive/U-NET/EVALUATIONS'
  VERIFICATIONS_PATH='/content/gdrive/My Drive/U-NET/VERIFICATIONS'
    
  print(join(Path,'prediction.log'))
  f=open(join(Path,'prediction.log'),'w')

  f.write('image prediction date: '+str(datetime.datetime.now())+'\n')
  f.write('the size of images input and output:'+'\n')
  f.write('INPUT_WIDTH= '+str(INPUT_WIDTH)+'  '+'CHANNELS='+str(CHANNELS)+'\n')
  f.write('Normalization_data:'+str(Normalization)+' threshold_opt='+str(threshold)+'\n')

  # Image prediction time
  start = time. time()
  
  #Creation of the concentric circle for weight matrix
  indices = [[[(i+0.5),(j+0.5)] for i in range(INPUT_WIDTH)] for j in range(INPUT_HEIGHT)]
  concentric_weights = np.array([[weight(x,y,INPUT_WIDTH) for x,y in ind] for ind in indices])
  #To verify the concentric matrix :
  filename_verif=join(VERIFICATIONS_PATH,'concentric.png')
  imsave(filename_verif,concentric_weights)
  
  if Prediction_Images_Mean is False:
  	# Classical prediction of Images
  	# In the classic prediction, we take the multispectral and thermal images, we do the normalization, then we apply the trained model of U-NET.
  	# Choose between prediction on one image ("npy") or on the full Dalle ("tif")
  	#if len(sys.argv) > 2 : 
  	#    if sys.argv[2] == "tif" :
  	#        list_image = os.listdir(join(Path,'DallesTif'))
  	#    elif sys.argv[2] == "npy" :
  	#        list_image = os.listdir(join(Path,'ImageArray'))
  	#    else : 
  	#        print("Argument 2 not recognized, please enter tif for the Dalles or npy for the Image")
  	#        quit()
  	#else : 
  	#    list_image = os.listdir(join(Path,'ImageArray'))
  	#        
    #for image in list_image :
    #  if len(sys.argv) > 2 :
    #      if sys.argv[2] == "tif" :
    #        image_Path = join(Path,'DallesTif',image)
    #        tif_image = Image.open(image_Path)
    #        image_array = np.array(tif_image)
    #      elif sys.argv[2] == "npy" :
    #        image_Path=join(Path,'ImageArray',image)
    #        image_array=np.load(image_Path)
    for image in os.listdir(join(Path,'ImageArray')):
      image_Path=join(Path,'ImageArray',image)
      image_array=np.load(image_Path)
      print('size Image :',image_array.size)
			
      X_Image=image_array[0:(np.shape(image_array)[0]//INPUT_WIDTH)*INPUT_WIDTH,0:(np.shape(image_array)[1]//INPUT_HEIGHT)*INPUT_HEIGHT]
      Y_Image=image_array[0:(np.shape(image_array)[0]//INPUT_WIDTH)*INPUT_WIDTH,0:(np.shape(image_array)[1]//INPUT_HEIGHT)*INPUT_HEIGHT,0]
      
      print('size X_Image :',X_Image.size ,'size Y_Image :',Y_Image.size)
			
      X_Image_data=np.zeros([1,(np.shape(image_array)[0]//INPUT_WIDTH)*INPUT_WIDTH,(np.shape(image_array)[1]//INPUT_HEIGHT)*INPUT_HEIGHT,len(CHANNELS)], dtype=np.float32)
      X_Image_data[0,:,:,:]=X_Image
      X_Image_data_Normal=(X_Image_data)
      print(np.max(X_Image_data_Normal))
      print(np.min(X_Image_data_Normal))
      for ii in range(0,np.shape(X_Image_data_Normal)[1]-INPUT_WIDTH,INPUT_WIDTH):
        for jj in range(0,np.shape(X_Image_data_Normal)[2]-INPUT_HEIGHT,INPUT_HEIGHT):
          X_pred=np.zeros((INPUT_WIDTH,INPUT_HEIGHT,len(CHANNELS)),dtype=np.float32)
          Mask_pred=np.zeros((INPUT_WIDTH,INPUT_HEIGHT),dtype=np.uint8)
          #Handling of the Normalization function
          if not Normalization is None:
            normfunc =  getattr(data_normalizing, Normalization)
            X_Pred=normfunc(X_Image_data_Normal[:,ii:ii+INPUT_WIDTH,jj:jj+INPUT_HEIGHT,:])
          else :
            X_Pred=X_Image_data_Normal
          #If that part doesn't work, unquote next line to do it manually, and change by the wanted Normalization method
          #X_pred=data_normalizing.Normalizing_by_image_by_column(X_Image_data_Normal[:,ii:ii+INPUT_WIDTH,jj:jj+INPUT_HEIGHT,:])
          results = model.predict(X_pred,verbose=1) 
          
          # Prepare for multiplying by the concentric weight matrix
          results = np.reshape(results,(96,96))
          results = np.multiply(results,concentric_weights)
          
          for i in range(INPUT_WIDTH):
            for j in range(INPUT_HEIGHT):
              if results[i,j] >= threshold: 
                Mask_pred[i,j]=255
              else:
                Mask_pred[i,j]=0
			
          Y_Image[ii:ii+INPUT_WIDTH,jj:jj+INPUT_HEIGHT]=Mask_pred
      Y_Image_pred=Y_Image.astype('uint8')
      filename_image=join(EVALUATION_PATH,str(threshold)+str(image)+'.png')
      print(filename_image)
      imsave(filename_image,Y_Image_pred)

  else:
  	#if len(sys.argv) > 2 : 
  	#    if sys.argv[2] == "tif" :
  	#        list_image = os.listdir(join(Path,'DallesTif'))
  	#    elif sys.argv[2] == "npy" :
  	#        list_image = os.listdir(join(Path,'ImageArray'))
  	#    else : 
  	#        print("Argument 2 not recognized, please enter tif for the Dalles or npy for the Image")
  	#        quit()
  	#else : 
  	#    list_image = os.listdir(join(Path,'ImageArray'))
  	#        
    #for image in list_image :
    #  if len(sys.argv) > 2 :
    #      if sys.argv[2] == "tif" :
    #        image_Path = join(Path,'DallesTif',image)
    #        tif_image = Image.open(image_Path)
    #        image_array = np.array(tif_image)
    #      elif sys.argv[2] == "npy" :
    #        image_Path=join(Path,'ImageArray',image)
    #        image_array=np.load(image_Path)
    
    for image in os.listdir(join(Path,'ImageArray')):
      image_Path=join(Path,'ImageArray',image)
      image_array=np.load(image_Path)
      
      # To improve the prediction, we have introduce the array_mean variable. We make several predictions 
      # on the image by moving the patch square of 256 * 256 pixels (96 * 96, resp) with a step called "step".
      # At the end, we take the average of the values obtained from each pixel on the different predictions.
      # NB_Pred : it is the matrix of prediction numbers of each pixel of the image to predict
      
      NB_Pred=np.zeros([image_array.shape[0],image_array.shape[1]],dtype=np.float32)
      Mask_pred=np.zeros((image_array.shape[0],image_array.shape[1]),dtype=np.float32)
      print(image_array.shape[0],image_array.shape[1])
      
      #Try and get an appropriate step for both witdh and height
      max_step = 10
      step_x = 0
      step_y = 0
      for i in range(1,max_step):
        if (((image_array.shape[0] - INPUT_WIDTH)%(2**i) == 0) and ((2**i) > step_x)) and (((image_array.shape[1] - INPUT_HEIGHT)%(2**i) == 0) and ((2**i) > step_y)):
            step_x = 2**i
            step_y = 2**i
      print(step_x,step_y)
      
      INPUT=np.zeros([1,image_array.shape[0],image_array.shape[1],len(CHANNELS)],dtype=np.float32)
      for nb,l in enumerate(CHANNELS):
          print(INPUT.shape,image_array.shape)
          INPUT[0,:,:,nb]=image_array[:,:,l]
          
      #Handling of the Normalization function
      if not Normalization is None:
        normfunc =  getattr(data_normalizing, Normalization)
        INPUT_Normal=normfunc(INPUT)
      else :
        X_Pred=INPUT
      #If that part doesn't work, unquote next line to do it manually
      #INPUT_Normal=data_normalizing.Normalizing_by_image_by_column(INPUT)
      for x in range(0,(image_array.shape[0]-INPUT_WIDTH)+1,step_x):
        for y in range(0,(image_array.shape[1]-INPUT_HEIGHT)+1,step_y):
          Input_data=np.zeros([1,INPUT_WIDTH,INPUT_HEIGHT,len(CHANNELS)],dtype=np.float32)
          Input_data[0,:,:,:]=INPUT_Normal[:,x:x+INPUT_WIDTH,y:y+INPUT_HEIGHT,:]
          results = model.predict(Input_data,verbose=1)
          
          # Prepare for multiplying by the concentric weith matrix
          results = np.reshape(results,(96,96))
          results = np.multiply(results,concentric_weights)
          
          
          #Mask_pred[x:x+INPUT_WIDTH,y:y+INPUT_HEIGHT]+=results[0,0:INPUT_WIDTH,0:INPUT_HEIGHT]
          NB_Pred[x:x+INPUT_WIDTH,y:y+INPUT_HEIGHT] += concentric_weights
          for i in range(INPUT_WIDTH):
            for j in range(INPUT_HEIGHT):
              Mask_pred[x+i,y+j]+=results[i,j]
            
      # Script to check the number of prediction and their importance on the full image.
      # It saves an image of the same size as the image to predict.
      NB_Pred_tmp = NB_Pred*20
      filename_verif=join(VERIFICATIONS_PATH,'nb_pred.png')
      imsave(filename_verif,NB_Pred_tmp)
      assert(np.count_nonzero(NB_Pred == 0) == 0)
      Mask_pred[:,:] /= NB_Pred[:,:]
      Mask_pred[Mask_pred<=threshold]=0
      Mask_pred[Mask_pred>=threshold]=255
      
      Mask_pred = Mask_pred.astype(np.uint8)
      filename_image=join(EVALUATION_PATH,str(threshold)+str(image)+'.png')
      print(filename_image)
      imsave(filename_image,Mask_pred)
      
      end = time. time()
      f.write('\n The time of execution"s prediction of the Image +'+str(image)+' :'+str(end-start)+'seconds')
  f.close()

output_path=join('/content/gdrive/My Drive/U-NET/data','Image_to_pred')

if len(sys.argv) >1:
    CHANNELS = ast.literal_eval(sys.argv[1])
else:
    CHANNELS = [0,1,2,3,4,5,6]
    
Prediction(Path=output_path,INPUT_WIDTH =96, INPUT_HEIGHT = 96,CHANNELS = CHANNELS, threshold=0.5, Prediction_Images_Mean=True,
	Normalization='Normalizing_by_image_by_column', step=32)
