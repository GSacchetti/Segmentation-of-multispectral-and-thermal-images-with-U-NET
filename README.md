#Segmentation of multispectral and thermal images in vegetation environments with U-NET 

All steps available in the ipynb notebook named "Implementation_U-NET.ipynb"

If done on Google Colab, just follows the steps of the notebook, otherwise :

1° - Packages Installation :
run "pip install -r requirements.txt"

2° - Data Preparation : Building of the training and test data sets : 

  a) Download the EXCELS files from the following drive : https://1drv.ms/f/s!/AgrWrFruN85rbarFzVMaBsOx10I and add them to the data/Excel folder

  b) In the "Construction_data" folder : Run "python excel_to_tif_array.py to convert the excel fies to tif images

  c) Run "python create_train_test_data.py" to split the data into a training and testing sets.

3° - Training U-NET model :

Run "python main.py"

By default the training will be done on the following channels [0,1,2,3,4,5,6], this can be changed by running python main.py "[...]" with within the brackets the wanted channels.

To load a weight file to try and improve the training, change Load_weight to True in main.py, and if you want to train on a specific weight file, also change the name of that weight file in model_unet.py

4° - Prediction and Testing of the U-NET Model :

!! Important !! Make sure that the channels used in pred_unet.py are the same as the one used for the training of the weight file you want to use.
Multiple weight files are provided, each with the channels used to get them.

Run "python pred_unet.py"

By default the prediction will be done on the following channels [0,1,2,3,4,5,6], this can be changed by running python pred_unet.py [...] with wothin the brackets the wanted channels. 

To make prediction on the image contained in image_to_pred/ImageArray, don't put a second argument or add "npy".
To make predictions on the Dalles, add "tif" as such python pred_unet.py "[0,1,2,3,4,5,6]" "tif".


Note: For any instruction, remember to check the correspondance of the parameters in the scripts 
with the data set (the path to the training data, the size of the images and masks, the image channels, etc.).
