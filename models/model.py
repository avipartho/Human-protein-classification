from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
from torchsummary import summary
import torch

def get_net(show_summary=0):
    
    '''show_summary flag is for printing summary of the model
        0 --> prints nothing
        1 --> keras type model summary using torchsummary library
        2 --> prints the model in pyTorch fashion'''

    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    old_weight = model.conv1_7x7_s2.weight
    model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    
    if config.first_layer_pretrained:
        # using pretrained imagenet weight for RGB channels & G channel weight for Y channel
        new_weight = torch.nn.Parameter(torch.cat((old_weight,torch.reshape(old_weight[:,1,:,:],(64,1,7,7))),dim=1))
        model.conv1_7x7_s2.weight = new_weight 

    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, config.num_classes),
            )

    if show_summary == 1: print(summary(model.to("cuda:0"), (config.channels, config.img_width, config.img_height)))
    elif show_summary == 2: print(model)
    # to check weight
    # for param in model.parameters():    
    #     print(param.data[0])
    #     break
    return model
