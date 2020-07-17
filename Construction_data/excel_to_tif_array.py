This algorithm is to convert the excel files into tiff images and arrays.   #######
#Our dataset were in the form of excel tables, to convert these tables into a tiff image, I created this Convert function.
#To perform this conversion, we have calculated a displacement step called "step", 
#because this table represents a multispectral and thermal georeferential part (you can see the different 
#columns in the data / Excel folder) extracted from ERDAS software, so the step represents its resolution.
#We can display them all 7 at once, for that we saved them in pictures but each length differently 
#(You can see it in the data / TIFandNPY folder
 
#Loading packages
from PIL import Image
import xlrd
import numpy  as np
import os,sys
from skimage.io import imsave,imread
import math


def mymkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)

def Convert(PathImportExcel,PathExportTif,step=12,factor=1000, Nb_channels=7):
	#Initialization of indices of images for each channel 
	print(os.listdir(PathImportExcel))
	for element in list(os.listdir(PathImportExcel)):
		if element.find('~$')==-1 and element.find('.D')==-1:
			name=element.replace('.xlsx','')
			print(element)
			file= xlrd.open_workbook(PathImportExcel+element)
			#Initilization of indice of subsets
			for k in file.sheet_names():
				tableau = file.sheet_by_name(str(k))
				# Writting the number of lines of each subset 
				print('le nombre de lignes de '+str(k)+' %s ' % tableau.nrows)
				# Writting the number of lines of each subset
				print('le nombre de colonnes  '+str(k)+' '+'%s ' % tableau.ncols)
				minX=sys.maxsize
				maxX=-sys.maxsize
				minY=sys.maxsize
				maxY=-sys.maxsize
				for l in range(1,tableau.nrows):
					x=tableau.cell_value(l,1)*factor
					minX=min(minX,x)
					maxX=max(maxX,x)
					y=tableau.cell_value(l,2)*factor
					minY=min(minY,y)
					maxY=max(maxY,y)
				#Determination's rÃ©solution
				tab=[]
				for i in range(1,4000):
					tab.append(tableau.cell_value(i,1)*factor)
				table=[]
				for i in tab:
					if not i in table:
						table.append(i)
				step=int(table[2]-table[1])
				xSize=1+(maxX-minX)/step
				ySize=1+(maxY-minY)/step
				size =(round(xSize),round(ySize))
				print('the image"s size:',size)
				namesubset=name+'_'+str(k)
				mymkdir(PathExportTif+namesubset)
				mymkdir(PathExportTif+'/'+namesubset+'/'+'ImageTif')
				mymkdir(PathExportTif+'/'+namesubset+'/'+'ImageArray')
				mymkdir(PathExportTif+'/'+namesubset+'/'+'MaskTif')
				mymkdir(PathExportTif+'/'+namesubset+'/'+'MaskArray')
				matrix=np.zeros([size[0],size[1],Nb_channels], dtype=np.float32)
				###liste de channels (faie modif)
				for h in range(3,10):
					image= np.zeros((size[0],size[1]), dtype=np.float32) 
					for l in range(1,tableau.nrows):
						i=math.floor((tableau.cell_value(l,1)*factor-minX+step/2.)/step)
						j=math.floor((tableau.cell_value(l,2)*factor-minY+step/2.)/step)
						image[i,j]=(tableau.cell_value(l,h))
						matrix[i,j,h-3]=tableau.cell_value(l,h)

					imageint=(255*(image-image.min())/(image.max()-image.min())).astype(np.uint8)
					imsave(PathExportTif+'/'+namesubset+'/'+'ImageTif'+'/'+name+'_'+str(k)+'_B'+str(h-1)+'.tif',imageint)
				np.save(PathExportTif+'/'+namesubset+'/'+'ImageArray'+'/'+namesubset+'_image.npy',matrix)

				#SAVE MASK
				image= np.zeros((size[0],size[1],1), dtype=np.uint8) 
				for l in range(1,tableau.nrows):
					i=int((tableau.cell_value(l,1)*factor-minX)/step)
					j=int((tableau.cell_value(l,2)*factor-minY)/step)
					v=tableau.cell_value(l,11) 
					if v=="other":
						image[i,j]=0
					else:
						image[i,j]=255

					#else: 
						#print('UNNKOWN '+v)
						#quit()
				imsave(PathExportTif+'/'+namesubset+'/'+'MaskTif'+'/'+name+'_'+str(k)+'_mask.tif',image)
				np.save(PathExportTif+'/'+namesubset+'/'+'MaskArray'+'/'+namesubset+'_mask.npy',np.float32(image/255.0))
				print(np.shape(image))
				del image


#mainPath=os.getcwd()
mainPath = '/content/gdrive/My Drive/U-NET'
PathImportExcel=mainPath+'/data/Excel/'
mymkdir(mainPath+'/data/TIFandNPY')

PathExportTif=mainPath+'/data/TIFandNPY/'

PathImportExcel='/content/gdrive/My Drive/U-NET/data/Excel/'
mymkdir('/content/gdrive/My Drive/U-NET/data/TIFandNPY')

PathExportTif='/content/gdrive/My Drive/U-NET/data/TIFandNPY/'
#Application of method convert 
Convert(PathImportExcel,PathExportTif)


