# -*- coding:Utf_8 -*-

#This script allows to load simply the images and masks of training and the test in in matrices of inputs and vectors of output

from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
import numpy  as np
import math
import os
import random
from skimage.io import imsave,imread
import datetime
import time

start_time = time.time()


# mainPath : the path or the way to our training and testing data set.
# size_image_Input=[96,96,7]([256,256,7]),size_mask_Output=[96,96,1]([256,256,1]: the sizes of images and masks. 
# augment=True, nb_augment=3 : augment: it is a question of whether or not we increase our data set and 
#     "nb_augment = 3" is for the number of rotations of 90 Â°.
# There are packages and functions that can generate them otherwise. However, this increase has greatly 
#      improved our prediction results.


def load_training_test_data(mainPath='', size_image_Input=[96,96,7],
	size_mask_Output=[96,96,1],augment=True, nb_augment=3): 

	#loading of Input and Output data
	#Initialization parameters and vetors training images and maskes set
	PathTT=mainPath+'/data/'
	VERIFICATIONS_PATH='/content/gdrive/My Drive/U-NET/VERIFICATIONS'
	list_Input_train=sorted(os.listdir(PathTT+'TrainNPY/images'))
	list_Output_train=sorted(os.listdir(PathTT+'TrainNPY/images'))
	list_Input_train_Augment=sorted(os.listdir(PathTT+'TrainNPY/images'))
	list_Output_train_Augment=sorted(os.listdir(PathTT+'TrainNPY/images'))

	#Check the size of data training and test

	if (np.size(list_Input_train)==np.size(list_Input_train)) and (np.size(list_Input_train_Augment)==np.size(list_Input_train_Augment)):
		print('the masks and images of training data are consistent')
	else :
		print('Error! the masks and images of training data are not consistent')
		quit()
	list_Input_test=sorted(os.listdir(PathTT+'TestNPY/images'))
	list_Output_test=sorted(os.listdir(PathTT+'TestNPY/images'))
	if (np.size(list_Input_test)==np.size(list_Input_test)):
		print('the masks and images of test data are consistent')
	else :
		print('Error! the masks and images of test data are not consistent')
		quit()


	Train_Batch_size=np.size(list_Input_train)
	if augment==True :
		Train_Batch_size=Train_Batch_size*(nb_augment+1)

	Test_Batch_size=np.size(list_Input_test)

	Size_Input_train=[Train_Batch_size,size_image_Input[0],size_image_Input[1],size_image_Input[2]]
	Size_Output_train=[Train_Batch_size,size_mask_Output[0],size_mask_Output[1],size_mask_Output[2]]

	X_Train=np.zeros(Size_Input_train, dtype=np.float32) 
	Y_Train=np.zeros(Size_Output_train, dtype=np.uint8) 
	nb_Input=0
	nb_Output=0

	# Writing the lists of images and masks set
	# Before teaching the U-net model, it is important to check this list on data folder
	# and to compare the consistency betwen the images and masks, we can even display them in TIFF

	f_images=open(mainPath+'/data/'+'Id_Training_Images.txt','w')
	f_masks=open(mainPath+'/data/'+'Id_Training_Masks.txt','w')
	f_images.write('Date of loading data: '+str(datetime.datetime.now())+'\n')
	f_masks.write('Date of loading data: '+str(datetime.datetime.now())+'\n')
	f_images.write('list of Id_images:\n')
	f_masks.write('list of Id_images:\n')
	if augment==True :
		g_images=open(mainPath+'/data/'+'Id_Training_Augment_images.txt','w')
		g_masks=open(mainPath+'/data/'+'Id_Train_Augment_masks.txt','w')
		g_images.write('Date of loading data: '+str(datetime.datetime.now())+'\n')
		g_masks.write('Date of loading data: '+str(datetime.datetime.now())+'\n')
		g_images.write('list of Id_images:\n')
		g_masks.write('list of Id_images:\n')

	list_Input=sorted(os.listdir(PathTT+'TrainNPY/images'))
	for Input in list_Input:
		InputArray=np.load(PathTT+'TrainNPY/images/'+str(Input))
		X_Train[nb_Input]=InputArray
		nb_Input+=1
		f_images.write(str(Input)+'\n')

	if augment ==True:
		list_Input=sorted(os.listdir(PathTT+'TrainNPY/Augment_images'))
		for Input in list_Input:
			InputArray=np.load(PathTT+'TrainNPY/Augment_images/'+str(Input))
			X_Train[nb_Input]=InputArray
			nb_Input+=1
			g_images.write(str(Input)+'\n')


	list_Output=sorted(os.listdir(PathTT+'TrainNPY/masks/'))
	for Output in list_Output:
		OutputArray=np.load(PathTT+'TrainNPY/masks/'+Output)
		Y_Train[nb_Output]=OutputArray
		nb_Output=+1
		f_masks.write(str(Output)+'\n')
		filename_verif=join(VERIFICATIONS_PATH,'output'+str(nb_Output)+'.png')
		print(OutputArray.size)
		print(type(OutputArray))
		#imsave(filename_verif,OutputArray)


	if augment==True:
		List_Output=sorted(os.listdir(PathTT+'TrainNPY/Augment_masks'))
		#print(list_Output)
		for Output in List_Output:
				OutputArray=np.load(PathTT+'TrainNPY/Augment_masks'+'/'+Output)
				Y_Train[nb_Output]=OutputArray
				nb_Output=nb_Output+1
				g_masks.write(str(Output)+'\n')
				filename_verif=join(VERIFICATIONS_PATH,'output'+str(nb_Output)+'.png')
				print(OutputArray.size)
				print(type(OutputArray))
				#imsave(filename_verif,OutputArray)

	#Initializing parameters of input and output testing vector
	nb_Input=0
	nb_Output=0

	size_Input_test=[Test_Batch_size,size_image_Input[0],size_image_Input[1],size_image_Input[2]]
	size_Output_test=[Test_Batch_size,size_mask_Output[0],size_mask_Output[1],size_mask_Output[2]]
	X_Test=np.zeros(size_Input_test, dtype=np.float32) 
	Y_Test=np.zeros(size_Output_test, dtype=np.uint8) 


	for Input in sorted(os.listdir(PathTT+'TestNPY/images')):
		InputArray=np.load(PathTT+'TestNPY/images/'+str(Input))
		X_Test[nb_Input]=InputArray
		nb_Input+=1

	for Output in sorted(os.listdir(PathTT+'TestNPY/masks')):
		OutputArray=np.load(PathTT+'TestNPY/masks/'+Output)
		Y_Test[nb_Output]=OutputArray
		nb_Output+=1

	f_images.close()
	f_masks.close()
	if augment==True :
		g_images.close()
		g_masks.close()

	return X_Train,Y_Train,X_Test,Y_Test
