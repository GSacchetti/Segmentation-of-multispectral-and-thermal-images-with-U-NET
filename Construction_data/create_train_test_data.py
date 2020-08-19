#######  This algorithm  is to create Test and training data sets of U-NET #######
#Loading packages
import numpy  as np
import os,sys
from skimage.io import imsave,imread
from random import sample
import shutil
import random
from os.path import join
random.seed(2019)

#mainPath='/content/drive/My Drive/U-NET'
#mainPath=os.getcwd()
# The path of our data set
mainPath='/content/gdrive/My Drive/U-NET/data/'
PathTIFandNPY=mainPath+'TIFandNPY/'
PathTT=mainPath



#These parameters are to divide and increase our data pool, 
#we have images of large sizes varying between 256 * 256 and 500 * 700 pixels, to have
# them images of same sizes, we chose 96 * 96 pixels, so you can choose another size, 
#it depends on the parameters of your model U-NET

#for our model U-NET, we can choose IMG_WIDTH = 256 IMG_HEIGHT = 256 and Size_image_mask=256
# as parameters for the learning and test data (You can seen the parameters of U-NET on the folder U-NET, file = Model.py)

# We have choseen the folowing parameters : 


IMG_WIDTH = 96
IMG_HEIGHT = 96
Augmentation=True
nb_Augmentation=3

#For learning data of deep learning methods, we often take 70% to 80% data
# for this we look at the number of images or subsets that we have.
# So we take the rest for the test and validation of the model.
size_data=np.size(os.listdir(PathTIFandNPY))
size_train=int(0.7*size_data)
print('size_train = ',size_train)

#Initialization of parameters : 
nb_image=0
nb_mask=0
nb_image_augment=0
nb_mask_augment=0
nb_image_test=0
nb_mask_test=0

# Share  of indices data on indices train and indices test 

liste=set(os.listdir(PathTIFandNPY))
print(liste)
train_ids=set(sample(liste,size_train))
test_ids=liste-train_ids
print(test_ids)
def mymkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)

#CrÃƒÆ’Ã‚Â©ation de dossiers s'ils n'existaient pas 		
mymkdir(PathTT+'TrainNPY')
mymkdir(PathTT+'TrainNPY')
Train_path_data=PathTT+'TrainNPY/'
mymkdir(Train_path_data+'/images')
mymkdir(Train_path_data+'/Augment_images')
Train_path_images=Train_path_data+'/images'
Train_path_image_aug=Train_path_data+'/Augment_images/'
mymkdir(Train_path_data+'/masks')
mymkdir(Train_path_data+'/Augment_masks')
Train_path_masks=Train_path_data+'masks'
Train_path_mask_aug=Train_path_data+'/Augment_masks/'
mymkdir(PathTT+'TestNPY')
Test_path_data=PathTT+'TestNPY/'
mymkdir(Test_path_data+'/images')
Test_path_images=Test_path_data+'/images/'

mymkdir(Test_path_data+'/masks')
Test_path_masks=Test_path_data+'/masks/'



for element in list(os.listdir(PathTIFandNPY)):
	if element in train_ids:
		Train_Path_image=PathTIFandNPY+element+'/ImageArray/'
		Train_Path_mask=PathTIFandNPY+element+'/MaskArray/'

		for element_id in list(os.listdir(Train_Path_image)):
			imageArray=np.load(Train_Path_image+element_id)
			print(np.shape(imageArray))
			for i in range(0,np.shape(imageArray)[0]-IMG_WIDTH,IMG_WIDTH//2):
				for j in range(0,np.shape(imageArray)[1]-IMG_HEIGHT,IMG_HEIGHT//2):
					np.save(join(Train_path_images,'image_'+str(nb_image)+'.npy'),imageArray[i:i+IMG_WIDTH,j:j+IMG_HEIGHT])
					if Augmentation==True:
						for Augment in range(nb_Augmentation):
							np.save(join(Train_path_image_aug,'image_'+str(nb_image_augment)+'.npy'),
								np.rot90(imageArray[i:i+IMG_WIDTH,j:j+IMG_HEIGHT],Augment+1))
							nb_image_augment=nb_image_augment+1
					nb_image+=1


			for i in range(np.shape(imageArray)[0]-IMG_WIDTH,IMG_WIDTH,-IMG_WIDTH//2):
				for j in range(np.shape(imageArray)[1]-IMG_HEIGHT,IMG_HEIGHT,-IMG_HEIGHT//2):
					np.save(join(Train_path_images,'image_'+str(nb_image)+'.npy'),
						imageArray[i-IMG_WIDTH:i,j-IMG_HEIGHT:j])
					if Augmentation==True:
						for Augment in range(nb_Augmentation):
							np.save(join(Train_path_image_aug,'image_'+str(nb_image_augment)+'.npy'),
								np.rot90(imageArray[i-IMG_WIDTH:i,j-IMG_HEIGHT:j],Augment+1))
							nb_image_augment=nb_image_augment+1
					nb_image+=1

		for element_id in list(os.listdir(Train_Path_mask)):
			MaskArray=np.load(Train_Path_mask+element_id)
			print(np.shape(MaskArray))
			print(Train_Path_mask+element_id)
			for i in range(0,np.shape(MaskArray)[0]-IMG_WIDTH,IMG_WIDTH//2):
				for j in range(0,np.shape(MaskArray)[1]-IMG_HEIGHT,IMG_HEIGHT//2):
					np.save(join(Train_path_masks,'mask_'+str(nb_mask)+'.npy'),
						MaskArray[i:i+IMG_WIDTH,j:j+IMG_HEIGHT])
					
					if Augmentation==True:
						for Augment in range(nb_Augmentation):
							np.save(join(Train_path_mask_aug,'mask_'+str(nb_mask_augment)+'.npy'),
								np.rot90(MaskArray[i:i+IMG_WIDTH,j:j+IMG_HEIGHT],Augment+1))
							nb_mask_augment+=1
					nb_mask+=1
			

			for i in range(np.shape(MaskArray)[0]-IMG_WIDTH,IMG_WIDTH,-IMG_WIDTH//2):
				for j in range(np.shape(MaskArray)[1]-IMG_HEIGHT,IMG_HEIGHT,-IMG_HEIGHT//2):
					np.save(join(Train_path_masks,'mask_'+str(nb_mask)+'.npy'),
						MaskArray[i-IMG_WIDTH:i,j-IMG_HEIGHT:j])
					if Augmentation==True:
						for Augment in range(nb_Augmentation):
							np.save(join(Train_path_mask_aug,'mask_'+str(nb_mask_augment)+'.npy'),
								np.rot90(MaskArray[i-IMG_WIDTH:i,j-IMG_HEIGHT:j],Augment+1))
							nb_mask_augment+=1
					nb_mask+=1


	if element in test_ids:
		Test_Path_image=PathTIFandNPY+element+'/ImageArray/'
		Test_Path_mask=PathTIFandNPY+element+'/MaskArray/'
		for element_id in list(os.listdir(Test_Path_image)):


			imageArray=np.load(Test_Path_image+element_id)

			for i in range(0,np.shape(imageArray)[0]-IMG_WIDTH,IMG_WIDTH//2):
				for j in range(0,np.shape(imageArray)[1]-IMG_HEIGHT,IMG_HEIGHT//2):
					np.save(join(Test_path_images,'image_'+str(nb_image_test)+'.npy'),
						imageArray[i:i+IMG_WIDTH,j:j+IMG_HEIGHT]) 
					nb_image_test=nb_image_test+1

			for i in range(np.shape(imageArray)[0]-IMG_WIDTH,IMG_WIDTH,-IMG_WIDTH//2):
				for j in range(np.shape(imageArray)[1]-IMG_HEIGHT,IMG_HEIGHT,-IMG_HEIGHT//2):
					np.save(join(Test_path_images,'image_'+str(nb_image_test)+'.npy'),
						imageArray[i-IMG_WIDTH:i,j-IMG_HEIGHT:j]) 
					nb_image_test=nb_image_test+1


		for element_id in list(os.listdir(Test_Path_mask)):

			maskArray=np.load(Test_Path_mask+element_id)

			for i in range(0,np.shape(maskArray)[0]-IMG_WIDTH,IMG_WIDTH//2):
				for j in range(0,np.shape(maskArray)[1]-IMG_HEIGHT,IMG_HEIGHT//2):
					np.save(join(Test_path_masks,'mask_'+str(nb_mask_test)+'.npy'),
						maskArray[i:i+IMG_WIDTH,j:j+IMG_HEIGHT]) 
					nb_mask_test=nb_mask_test+1

			for i in range(np.shape(maskArray)[0]-IMG_WIDTH,IMG_WIDTH,-IMG_WIDTH//2):
				for j in range(np.shape(maskArray)[1]-IMG_HEIGHT,IMG_HEIGHT,-IMG_HEIGHT//2):
					np.save(join(Test_path_masks,'mask_'+str(nb_mask_test)+'.npy'),
						maskArray[i-IMG_WIDTH:i,j-IMG_HEIGHT:j]) 
					nb_mask_test=nb_mask_test+1


print('nb_mask=',nb_mask)
print('nb_image=',nb_image)
print('nb_mask_augment=',nb_mask_augment)
print('nb_image_augment=',nb_image_augment)
print('nb_image_test=',nb_image_test)
print('nb_mask_test=',nb_mask_test)
