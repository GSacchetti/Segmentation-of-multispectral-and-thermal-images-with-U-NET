#these functions are for measuring the accuracy of the model by comparing 
#the images predicted with the U-NET model and the images of ground truth.



# Precision measurement function
import numpy as np
def measure(beta,pred,mask):
    '''this function measures the prediction accuracy of the model U-NET'''
    nb_True_Negative=0
    nb_True_Positive=0
    nb_False_Positive=0
    nb_False_Negative=0
    INPUT_WIDTH=np.shape(pred)[0]
    INPUT_HEIGHT=np.shape(pred)[1]
    for i in range(INPUT_WIDTH):
        for j in range(INPUT_HEIGHT):
            if pred[i,j]==0 and mask[i,j]==0:
                nb_True_Negative+=1
            elif pred[i,j]==0 and mask[i,j]==255:
                nb_False_Negative+=1
            elif pred[i,j]==255 and mask[i,j]==255:
                nb_True_Positive+=1
            elif pred[i,j]==255 and mask[i,j]==0:
                nb_False_Positive+=1
            else:
                print(pred[i,j],mask[i,j])
                return ('Unknown values')
    accuracy = 100.0*(nb_True_Negative+nb_True_Positive)/(nb_True_Negative+nb_True_Positive+nb_False_Positive+nb_False_Negative)
    precision = (nb_True_Positive)/(nb_True_Positive+nb_False_Positive)
    recall = (nb_True_Positive)/(nb_True_Positive+nb_False_Negative)
    f_beta_score = (1+beta**2)*(recall*precision)/(recall+((beta**2)*precision))
    
    print('Accuracy : '+str(accuracy)+ '%')
    print('Precision : '+str(precision))
    print('Recall : '+str(recall))
    print('F_beta Score : '+str(f_beta_score)+' for beta = '+str(beta))
    
    print('nb_True_Negative = '+str(nb_True_Negative))
    print('nb_True_Positive = '+str(nb_True_Positive))
    print('nb_False_Positive = '+str(nb_False_Positive))
    print('nb_False_Negative = '+str(nb_False_Negative))
    return nb_True_Negative,nb_True_Positive,nb_False_Positive,nb_False_Negative,accuracy,precision,recall,f_beta_score



# The descrimination level
# To improve the precision of U-NET and its prediction, we have introduced an "optimal" 
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
