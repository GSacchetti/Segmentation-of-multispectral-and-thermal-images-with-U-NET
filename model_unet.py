from tensorflow import keras
from keras.models import Model,load_model,save_model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt


def Model_Unet(INPUT_WIDTH = 256, INPUT_HEIGHT = 256,INPUT_CHANNELS = 7):


	inputs = Input((INPUT_WIDTH, INPUT_HEIGHT,INPUT_CHANNELS))
	
	#Part 01: DOWN_BLOC

	conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
	conv1 = Dropout(0.1) (conv1)
	conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
	pool1 = MaxPooling2D((2, 2)) (conv1)

	conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool1)
	conv2 = Dropout(0.1) (conv2)
	conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
	pool2 = MaxPooling2D((2, 2)) (conv2)

	conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool2)
	conv3 = Dropout(0.2) (conv3)
	conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
	pool3 = MaxPooling2D((2, 2)) (conv3)

	conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool3)
	conv4 = Dropout(0.2) (conv4)
	conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

	#Part 02: BOTTELNECK
	conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool4)
	conv5 = Dropout(0.3) (conv5)
	conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv5)
	
	#Part 03: UP_BLOC

	up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
	up6 = concatenate([up6, conv4])
	conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (up6)
	conv6 = Dropout(0.2) (conv6)
	conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv6)

	up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
	up7 = concatenate([up7, conv3])
	conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (up7)
	conv7 = Dropout(0.2) (conv7)
	conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv7)

	up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
	up8 = concatenate([up8, conv2])
	conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (up8)
	conv8 = Dropout(0.1) (conv8)
	conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv8)

	up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
	up9 = concatenate([up9, conv1], axis=3)
	conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (up9)
	conv9 = Dropout(0.1) (conv9)
	conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv9)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

	model = Model(inputs=[inputs], outputs=[outputs])
	return model

# Train_model=True : the condition whether one drives the model or not 
# Load_weights=True : if we load the existing model that is already driven
# validation_split, batch_size, epochs are the parameters of the model
# X_Train_Normal, Y_Train, X_Test_Normal, Y_Test are the input and output vectors of training and testing model
# INPUT_WIDTH =96(256), INPUT_HEIGHT = 96(256), INPUT_CHANNELS = 7 the sizes and caracteristics of our data 
# EVALUATION_PATH : it is the path of log folder

def Training_Model(Train_model=True, Load_weights=True, validation_split=0.01, batch_size = int, epochs = int,
	X_Train_Normal = [], Y_Train = [], X_Test_Normal= [], Y_Test=[],
	INPUT_WIDTH =96, INPUT_HEIGHT = 96, INPUT_CHANNELS = 7, EVALUATION_PATH = ''):

	#Training model U-NET
	model=Model_Unet(INPUT_WIDTH =INPUT_WIDTH, INPUT_HEIGHT = INPUT_HEIGHT,INPUT_CHANNELS = INPUT_CHANNELS)

	#load of weights model
	
	# The two following lines introduce a decay for the learning rate
	#lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
    #optimizer = keras.optimizers.Adam(lr=lr_schedule)

    # Unquote the two previous line and quote the next line to get the decay
	optimizer = keras.optimizers.Adam(lr=0.01) #This line allows us to change the learning rate of the optimizer
	model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
	model.summary()
	earlystopper = EarlyStopping(patience=20, verbose=1)
	checkpointer = ModelCheckpoint('model-U_NET.h5', verbose=1, save_best_only=True)

	if Load_weights==True :
		model = load_model('model-U_NET.h5')

	if Train_model==True :
		history=model.fit(X_Train_Normal,Y_Train,validation_split=validation_split, batch_size=batch_size, 
			epochs=epochs,validation_data=(X_Test_Normal, Y_Test),callbacks=[earlystopper, checkpointer])

		# Evaluation of model
		model.evaluate(X_Test_Normal,Y_Test, batch_size=batch_size)

		# Display the graphics of accuracy and loss
		print(history.history.keys())  
		plt.figure(1)  

		# Summarize history for accuracy  
		plt.subplot(211)  
		plt.plot(history.history['accuracy'])  
		plt.plot(history.history['val_accuracy'])  
		plt.title('model accuracy')  
		plt.ylabel('accuracy')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'test'], loc='upper left')  

		# Summarize history for loss  

		plt.subplot(212)  
		plt.plot(history.history['loss'])  
		plt.plot(history.history['val_loss'])  
		plt.title('model loss')  
		plt.ylabel('loss')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'test'], loc='upper left')   
		plt.savefig(EVALUATION_PATH+'/Loss_Accuracy_Graphique.png')
