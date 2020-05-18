#Segmentation of multispectral and thermal images in vegetation environments with U-NET 

All steps available in the ipynb notebook named "Implementation_U-NET.ipynb"

1° - Packages Installation :
run "pip install -r requirements.txt"

2° - Data Preparation : Building of the training and test data sets : 

  a) Download the EXCELS files from the following drive : https://1drv.ms/f/s!/AgrWrFruN85rbarFzVMaBsOx10I and add them to the data/Excel folder

  b) In the "Construction_data" folder : Run "python excel_to_tif_array.py to convert the excel fies to tif imagis and Numpy tables

  c) Run "python create_train_test_data.py" to split the data into a training and testing sets.

3° - Training U-NET model :

Run "python main.py"

4° - Prediction and Testing of the U-NET Model :

Run "python pred_unet.py"


Note: For any instruction, remember to check the correspondence of the parameters in the scripts 
with the data set (the path to the training data, the size of the images and masks, the number of
image channels, etc.).
