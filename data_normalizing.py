# To avoid any anomaly in the training and test data, we normalized the
# images in three different ways and then we compared the three best models 
# of U-net built after learning.


import numpy  as np
import math
import os

# We defined three different standardizations, these standardizations we used them,
# and then we compared them in our study

def Normalizing_by_column(data):

	''' Function for the normalization of data column by column'''
	
	datanorm=np.zeros_like(data)
	for d in range(data.shape[3]):
		col_data=data[:,:,:,d]
		datanorm[:,:,:,d]=(col_data-col_data.min())/(col_data.max()-col_data.min()).astype(np.float32)
	return datanorm

def Normalizing_by_image_by_column(data):

	''' Function for the normalization of data colonne by image on the input vector, then by column'''
	datanorm=np.zeros_like(data)
	for im in range(data.shape[0]):
		for d in range(data.shape[3]):
			col_data=data[im,:,:,d]
			datanorm[im,:,:,d]=(col_data-col_data.min())/(col_data.max()-col_data.min()).astype(np.float32)
	return datanorm
	
def Normalizing_By_image(data):

	''' Function for the normalization of data by image'''
	datanorm=np.zeros_like(data)
	for im in range(data.shape[0]):
		image=data[im,:,:,:]
		datanorm[im,:,:,:]=((image-image.min())/(image.max()-image.min()).astype(np.float32))
	return datanorm
