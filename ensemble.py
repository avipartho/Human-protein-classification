import os 
import random 
import warnings
import torch 
import numpy as np 
import pandas as pd 

from utils import *
from models.model import*
from data import HumanDataset
from tqdm import tqdm 
from config import config
from datetime import datetime

from torch.utils.data import DataLoader

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_no
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# 4. Ensemble
def ensemble(train_loader,test_loader,models,test_files,metric,fixed_th=None):
	sample_submission_df = test_files
	# confirm the model converted to cuda
	labels_dic,labels,submissions= {},[],[]
	for no,model in enumerate(models):
		model.cuda()
		model.eval()
		submit_results = []
		for i,(input,filepath) in enumerate(tqdm(test_loader)):
			# change everything to cuda and get only basename
			filepath = [os.path.basename(x) for x in filepath]
			with torch.no_grad():
				image_var = input.cuda(non_blocking=True)
				y_pred = model(image_var)
				label = y_pred.sigmoid().cpu().data.numpy()
				labels.append(label)
		labels_dic[no] = np.array(labels)
		labels = []

	# avg prediction probabilities
	total_models = len(models)+0.
	labels = (labels_dic[0]+labels_dic[1]+labels_dic[2]+labels_dic[3]+labels_dic[4])/total_models
	if fixed_th is not None:
		labels = [label > fixed_th for label in labels]
	else:
		th_dic = get_best_thres(train_loader,model)
		labels = [label > np.array(list(th_dic.values())) for label in labels]

	for row in np.concatenate(labels):
		subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
		submissions.append(subrow)
	sample_submission_df['Predicted'] = submissions
	sample_submission_df.to_csv('./submit/ensemble_%s_%s_submission.csv'%(config.model_name,metric), index=None)

# 5. main function
def main(n_folds):
	
	all_files = pd.read_csv(config.data_root+"train.csv")
	test_files = pd.read_csv(config.data_root+"sample_submission.csv")
		
	# load dataset
	train_gen = HumanDataset(all_files,config.train_data,mode="train")
	train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=4)

	test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
	test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)

	models = []
	for fold in range(n_folds):
		model = get_net()
		model.cuda()
		best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		models.append(model)
		del(model)
	ensemble(train_loader,test_loader,models,test_files,"best_loss",0.15)
	
	models = []
	for fold in range(n_folds):
		model = get_net()
		model.cuda()
		best_model = torch.load("%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		models.append(model)
		del(model)
	ensemble(train_loader,test_loader,models,test_files,"best_f1",0.15)

	models = []
	for fold in range(n_folds):
		model = get_net()
		model.cuda()
		best_model = torch.load("%s/%s/%s/checkpoint.pth.tar"%(config.weights,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		models.append(model)
		del(model)
	ensemble(train_loader,test_loader,models,test_files,"last_epoch",0.15)


if __name__ == "__main__":
	n_folds = 5	
	main(n_folds)