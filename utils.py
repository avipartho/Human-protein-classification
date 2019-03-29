import os
import sys 
import json
import torch
import shutil
import numpy as np 
from config import config
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm

# save best model
def save_checkpoint(state, is_best_loss,is_best_f1,fold):
    filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
    if is_best_f1:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))

# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

class f1_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        
        def get_f1(preds, targs):
            tp = torch.sum(preds * targs, dim=0).float()
            tn = torch.sum((1 - preds) * (1 - targs), dim=0).float()
            fp = torch.sum(preds * (1 - targs), dim=0).float()
            fn = torch.sum((1 - preds) * targs, dim=0).float()

            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)

            f1 = 2*p*r / (p+r+1e-10)
            return torch.mean(f1)

        return 1 - get_f1(torch.sigmoid(input), target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

class Oversampling:
    def __init__(self, df):
        self.train_labels = df.set_index('Id')
        self.train_labels['Target'] = [[int(i) for i in s.split()] for s in self.train_labels['Target']]
        # set the minimum number of duplicates for each class
        self.multi = [1, 1, 1, 1, 1, 1, 1, 1, 8, 8,
                      8, 1, 1, 1, 1, 8, 1, 2, 1, 1,
                      4, 1, 1, 1, 2, 1, 2, 8]
        # TODO : different oversampling? https://www.kaggle.com/wordroid/inceptionresnetv2-resize256-f1loss-lb0-419

    def get(self, image_id):
        labels = self.train_labels.loc[image_id, 'Target'] if image_id in self.train_labels.index else []
        m = 1
        for l in labels:
            if m < self.multi[l]: m = self.multi[l]
        return m
        
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

def get_best_thres(val_loader,model,num_split=100):

    y_true_all, y_pred_all = [], []
    model.cuda()
    model.eval()
    with torch.no_grad():
        for (images,target) in tqdm(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = np.array(target)
            output = model(images_var).sigmoid().cpu().data.numpy()
            y_true_all.extend(target)
            y_pred_all.extend(output)

    cate2th = {}
    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)

    for c in range(28):
        y_true = y_true_all[:,c]
        y_pred = y_pred_all[:,c]
        best_th = 0
        best_f1 = -1

        for th in np.linspace(0,1,num_split,endpoint=False):
            f1 = f1_score(y_true,(y_pred > th).astype(int))
            if best_f1 <= f1:
                best_f1 = f1
                best_th = th

        cate2th[c] = best_th

    return cate2th

def get_class_weight(mu=0.5):
    import math
    labels_dict = {
        40958.0,
        3072.0,
        10871.0,
        3329.0,
        5130.0,
        5938.0,
        3725.0,
        9405.0,
        217.0,
        197.0,
        182.0,
        2194.0,
        2233.0,
        1458.0,
        2692.0,
        63.0,
        1290.0,
        446.0,
        1893.0,
        3672.0,
        438.0,
        13809.0,
        2729.0,
        10345.0,
        428.0,
        37366.0,
        706.0,
        127.0
    }

    total = np.sum(labels_dict)
    class_weight_log = np.zeros(28)

    # for key in range(28):
    #     score_log = math.log(mu * total / float(labels_dict[key]))
    #     class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else 1.0
    # class_weight_log = 28 * class_weight_log / np.sum(class_weight_log)

    labels_dict = {0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 1.0,
        5: 1.0,
        6: 2.0,
        7: 1.0,
        8: 21.0,
        9: 25.0,
        10: 40.0,
        11: 2.0,
        12: 2.0,
        13: 3.0,
        14: 2.0,
        15: 53.0,
        16: 3.0,
        17: 6.0,
        18: 2.0,
        19: 1.0,
        20: 7.0,
        21: 1.0,
        22: 2.0,
        23: 1.0,
        24: 4.0,
        25: 1.0,
        26: 4.0,
        27: 101.0}

    for key in range(28):
        class_weight_log[key] = labels_dict[key]
        
    return torch.from_numpy(np.expand_dims(class_weight_log,0)).float()