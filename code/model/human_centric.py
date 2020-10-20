import torch
import torchvision
import torch.nn as nn
import numpy as np
from model.config import config
from model.target_predicts import target_pre
from PIL import Image
from model.faster_rcnn_vgg16 import decom_vgg16
from utils import array_tool as at
from model.roi_module import RoIPooling2D
import model.method as method

class targetPredict(nn.modules):
    def __init__(self,gt):
        self.pred_label = None # pred
        self.pred_b_box = None # pred

        self.gt_human_action = None  # gt
        self.gt_human_box = None # gt
        self.gt_object_box = None # gt

        self.roi_size = 7
        self.spatial_scale = 1/16
        self.roi = RoIPooling2D(self.roi_size,self.roi_size,self.spatial_scale)
        _, self.classifier = decom_vgg16()

        self.cls_loc = nn.Linear(4096, object_classes * 4)
        self.score = nn.Linear(4096, action_classes)

    def forward(self,x,rois,roi_indices):
        # 在pred_b_box, pred_label找到human box -》 坐标
        pred_human_box = None
        roi_indices = torch.array([0])
        roi_indices = at.totensor().float()
        rois = at.totensor(pred_human_box).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

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
        # s_ah = action_classsify(region)
        # mu_ah = target_predict(region)
        # g,loss = target_location(mu_ah,b_h,b_o,para.sigma)

        # return s_ah,g,loss
        

