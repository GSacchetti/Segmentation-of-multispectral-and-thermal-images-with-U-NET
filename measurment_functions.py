# -*- coding:Utf_8 -*-

#these functions are for measuring the accuracy of the model by comparing 
#the images predicted with the U-NET model and the images of ground truth.

# Precision measurement function

def measure(pred,mask):
	'''this function measures the prediction accuracy of the model U-NET'''
	nbGoodBlack=0
	nbGoodWhite=0
	nbFP=0
	nbFN=0
	INPUT_WIDTH=np.shape(pred)[0]
	INPUT_HEIGHT=np.shape(pred)[1]
	for i in range(INPUT_WIDTH):
		for j in range(INPUT_HEIGHT):
			if pred[i,j]==0 and mask[i,j]==0:
				nbGoodBlack+=1
			elif pred[i,j]==0 and mask[i,j]==1:
				nbFN+=1
			elif pred[i,j]==1 and mask[i,j]==1:
				nbGoodWhite+=1
			elif pred[i,j]==1 and mask[i,j]==0:
				nbFP+=1

			else:
				print('Unknow values ')
				quit()
	print('Results :'+str(100.0*(nbGoodBlack+nbGoodWhite)/(nbGoodBlack+nbGoodWhite+nbFP+nbFN) )+ '%')
	print('nbGoodBlack='+str(nbGoodBlack))
	print('nbGoodWhite='+str(nbGoodWhite))
	print('nbFP='+str(nbFP))
	print('nbFN='+str(nbFN))
	return nbGoodBlack,nbGoodWhite,nbFP,nbFN,100.0*(nbGoodBlack+nbGoodWhite)/(nbGoodBlack+nbGoodWhite+nbFP+nbFN)



# The descrimination level
# To improve the precision of U-NET et its prediction, we have introduced an "optimal" 
# discrimination threshold to be able to better separate the predicted pixels.
# This threshold parameter can be replaced with a predictive pixel discrimination layer in the U-NET model

def optimal_threshold_desc(results):
	list=[]
	for threshold in range(20,80,1):
		globalMeasure=0
		for n in range(results.shape[0]):
			mask_pred=np.zeros((INPUT_WIDTH,INPUT_HEIGHT,3),dtype=np.uint8)
			mask_pred=np.zeros((INPUT_WIDTH,INPUT_HEIGHT,1),dtype=np.uint8)
			threshold=threshold*0.1
			print(threshold)
			for i in range(INPUT_WIDTH):
				for j in range(INPUT_HEIGHT):
					if results[n,i,j] >= threshold: 
						mask_pred[i,j]=1
					else:
						mask_pred[i,j]=0
			mask_test=Y_Test[n,:,:]
			nbGoodBlack,nbGoodWhite,nbFP,nbFN,m=measure(mask_pred,mask_test)
			globalMeasure+=m
		list.append(globalMeasure)
	print(list)	
	#
	threshold=((list.index(max(list)))+1)*0.01+0.2
	return threshold