import os
import torch
import numpy as np
import torch.nn as nn

class target_pre(nn.Module):
    def __init__(self):
        super(target_pre,self).__init__()
        
    def forward(self,x):
        
        return x