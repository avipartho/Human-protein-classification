import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 

from utils import *
from data import HumanDataset
from tqdm import tqdm 
from config import config
from datetime import datetime
from models.model import*
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_no
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

def setup_logger(fold):
	log = Logger()
	log.open("logs/%s_fold_%s_log_train.txt"%(config.model_name,str(fold)),mode="a")
	log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
	log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
	log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
	log.write('-------------------------------------------------------------------------------------------------------------------------------\n')
	return log

def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start,log):
	losses = AverageMeter()
	f1 = AverageMeter()
	model.train()
	for i,(images,target) in enumerate(train_loader):
		images = images.cuda(non_blocking=True)
		target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
		# compute output
		output = model(images)
		loss = criterion(output,target)
		losses.update(loss.item(),images.size(0))
		
		f1_batch = f1_score(target,output.sigmoid().cpu() > 0.15,average='macro')
		f1.update(f1_batch,images.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('\r',end='',flush=True)
		message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
				"train", i/len(train_loader), epoch,
				losses.avg, f1.avg, 
				valid_loss[0], valid_loss[1], 
				str(best_results[0])[:8],str(best_results[1])[:8],
				time_to_str((timer() - start),'min'))
		print(message , end='',flush=True)
	log.write("\n")
	#log.write(message)
	#log.write("\n")
	return [losses.avg,f1.avg]

# 2. evaluate function
def evaluate(val_loader,model,criterion,epoch,train_loss,best_results,start,log):
	# only meter loss and f1 score
	losses = AverageMeter()
	f1 = AverageMeter()
	# switch mode for evaluation
	model.cuda()
	model.eval()
	with torch.no_grad():
		for i, (images,target) in enumerate(val_loader):
			images_var = images.cuda(non_blocking=True)
			target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
			output = model(images_var)
			loss = criterion(output,target)
			losses.update(loss.item(),images_var.size(0))
			f1_batch = f1_score(target,output.sigmoid().cpu().data.numpy() > 0.15,average='macro')
			f1.update(f1_batch,images_var.size(0))
			print('\r',end='',flush=True)
			message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
					"val", i/len(val_loader), epoch,                    
					train_loss[0], train_loss[1], 
					losses.avg, f1.avg,
					str(best_results[0])[:8],str(best_results[1])[:8],
					time_to_str((timer() - start),'min'))

			print(message, end='',flush=True)
		log.write("\n")
		#log.write(message)
		#log.write("\n")

	return [losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,fold,test_files,metric,log,th_dic=None):
	sample_submission_df = test_files
	# confirm the model converted to cuda
	filenames,labels,submissions= [],[],[]
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
			if th_dic is not None: labels.append(label > np.array(list(th_dic.values())))
			else: labels.append(label > 0.15)
			filenames.append(filepath)

	for row in np.concatenate(labels):
		subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
		submissions.append(subrow)
	sample_submission_df['Predicted'] = submissions
	sample_submission_df.to_csv('./submit/%s_%s_fold_%s_submission.csv'%(config.model_name,metric,str(fold)), index=None)

# 4. Ensemble
def ensemble(test_loader,models,fold,test_files,log,th_dic=None):
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
	labels = (labels_dic[0]+labels_dic[1]+labels_dic[2])/total_models
	if th_dic is not None: labels = [label > np.array(list(th_dic.values())) for label in labels] 
	else: labels = [label > 0.15 for label in labels]

	for row in np.concatenate(labels):
		subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
		submissions.append(subrow)
	sample_submission_df['Predicted'] = submissions
	sample_submission_df.to_csv('./submit/ensemble_%s_fold_%s_submission.csv'%(config.model_name,str(fold)), index=None)

# 5. main function
def main(fold, oversample=False):
	log = setup_logger(fold)

	# mkdirs
	if not os.path.exists(config.submit):
		os.makedirs(config.submit)
	if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
		os.makedirs(config.weights + config.model_name + os.sep +str(fold))
	if not os.path.exists(config.best_models):
		os.mkdir(config.best_models)
	if not os.path.exists("./logs/"):
		os.mkdir("./logs/")

	# get model
	model = get_net()
	model.cuda()

	# criterion
	optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
	if config.weighted_loss:
		criterion = nn.BCEWithLogitsLoss(pos_weight=get_class_weight()).cuda()
	else:
		criterion = nn.BCEWithLogitsLoss().cuda()
	# criterion = FocalLoss().cuda()
	# criterion = f1_loss().cuda()
	start_epoch = 0
	best_loss = 999
	best_f1 = 0
	best_results = [np.inf,0]
	val_metrics = [np.inf,0]
	resume = False
	all_files = pd.read_csv(config.data_root+"train.csv")
	test_files = pd.read_csv(config.data_root+"sample_submission.csv")
	# train_data_list,val_data_list = train_test_split(all_files,test_size = 0.13,random_state = 2050)

	# Stratify 
	mlb = MultiLabelBinarizer()
	labels = [[int(i) for i in i.split()] for i in all_files.Target.tolist()]
	labels = mlb.fit_transform(labels)

	X, Y = np.arange(len(labels)), labels
	mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)

	for n_fold, (train_index, test_index) in enumerate(mskf.split(X, Y)):
		print('Fold %s'%str(n_fold))
		train_data_list = all_files.iloc[train_index]
		val_data_list = all_files.iloc[test_index]
		if n_fold==fold: break
	
	# Oversample
	if oversample:
		oversampled_train_data = train_data_list.copy()
		s = Oversampling(oversampled_train_data)
		for ind,idx in enumerate(train_data_list.Id):
			multiplier = s.get(idx)
			if multiplier>1: 
				oversampled_train_data = oversampled_train_data.append([train_data_list.iloc[[ind]]]*(multiplier-1),ignore_index=True)
			train_data_list = oversampled_train_data

	# load dataset
	train_gen = HumanDataset(train_data_list,config.train_data,mode="train")
	train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=4)

	val_gen = HumanDataset(val_data_list,config.train_data,augument=False,mode="train")
	val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=4)

	test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
	test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)

	scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
	start = timer()

	if config.retrain: # retrain
		saved_model = torch.load("%s/%s/%s/checkpoint.pth.tar"%(config.weights,config.model_name,str(fold)))
		model.load_state_dict(saved_model["state_dict"])
		optimizer.load_state_dict(saved_model["optimizer"])
		start_epoch = saved_model["epoch"]
		best_results = [saved_model["best_loss"],saved_model["best_f1"]]
		scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
		del(saved_model)

	if config.train: # train
		for epoch in range(start_epoch,config.epochs):
			scheduler.step(epoch)
			# train
			lr = get_learning_rate(optimizer)
			train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start,log)
			# validate
			val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start,log)
			# check results 
			is_best_loss = val_metrics[0] < best_results[0]
			best_results[0] = min(val_metrics[0],best_results[0])
			is_best_f1 = val_metrics[1] > best_results[1]
			best_results[1] = max(val_metrics[1],best_results[1])   
			# save model
			save_checkpoint({
						"epoch":epoch + 1,
						"model_name":config.model_name,
						"state_dict":model.state_dict(),
						"best_loss":best_results[0],
						"optimizer":optimizer.state_dict(),
						"fold":fold,
						"best_f1":best_results[1],
			},is_best_loss,is_best_f1,fold)
			# print logs
			print('\r',end='',flush=True)
			log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
					"best", epoch, epoch,                    
					train_metrics[0], train_metrics[1], 
					val_metrics[0], val_metrics[1],
					str(best_results[0])[:8],str(best_results[1])[:8],
					time_to_str((timer() - start),'min'))
				)
			log.write("\n")
			time.sleep(0.01)

	if config.test: # test
		best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		test(test_loader,model,fold,test_files,"best_loss",log)

		best_model = torch.load("%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		test(test_loader,model,fold,test_files,"best_f1",log)

		best_model = torch.load("%s/%s/%s/checkpoint.pth.tar"%(config.weights,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		test(test_loader,model,fold,test_files,"last_epoch",log)

	if config.ensemble: # ensemble
		models = []

		del(model)
		model = get_net()
		model.cuda()
		best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		models.append(model)

		del(model)
		model = get_net()
		model.cuda()
		best_model = torch.load("%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		models.append(model)

		del(model)
		model = get_net()
		model.cuda()
		best_model = torch.load("%s/%s/%s/checkpoint.pth.tar"%(config.weights,config.model_name,str(fold)))
		model.load_state_dict(best_model["state_dict"])
		models.append(model)

		ensemble(test_loader,models,fold,test_files,log)

if __name__ == "__main__":
	main(config.curr_fold, config.oversample)