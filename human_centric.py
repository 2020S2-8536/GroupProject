import torch 
import torchvision
import torch.nn as nn
import numpy as np
import method
from config import config
from target_predicts import target_pre
from PIL import Image
def action_classsify(img):
    '''
    To classcify action

    '''
    return -1 
    # pass
def target_predict(img):
    return -1

def target_location(mu_ah,b_h,b_o,sigma): 
    b_oh = method.b_oh(b_h,b_h)
    g = method.Gaussian_fuc(b_oh,mu_ah,sigma)
    Smooth_L1 = nn.SmoothL1Loss()
    loss = Smooth_L1(mu_ah,b_oh)
    return g,loss
def human_centric (img,b_o,b_h):
    para = config()
    # region = img.crop((x, y, x+w, y+h))
    region = img.crop((b_h[0], b_h[1],b_h[0]+b_h[2],b_h[1]+b_h[3]))
    s_ah = action_classsify(region)
    mu_ah = target_predict(region)
    g,loss = target_location(mu_ah,b_h,b_o,para.sigma)

    return s_ah,g,loss
        

