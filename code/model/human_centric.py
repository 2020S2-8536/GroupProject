import torch
import torchvision
import torch.nn as nn
import numpy as np
from model.config import config
from torch.nn import functional as F
from PIL import Image
from model.faster_rcnn_vgg16 import decom_vgg16
from utils import array_tool as at
from model.roi_module import RoIPooling2D
import model.method as method
from model.utils.bbox_tools import loc2bbox

VOC_BBOX_LABEL_NAMES =  ('airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining_table',
                            'dog','horse','motorcycle','human','potted_plant','sheep','couch','train','tv')

HICO_ACTIONS = ('board', 'ride', 'sit_on','pet', 'watch', 'feed', 'hold', 'drive', 'board', 'sail', 'stand_on', 'carry', 'drink_with',
              'open', 'hug', 'kiss', 'lie_on', 'herd', 'walk', 'clean', 'eat_at', 'sit_at', 'run', 'train', 'hop_on', 'greet',
              'race')

class targetPredict(nn.modules):
    def __init__(self, loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        self.object_classes = len(list(VOC_BBOX_LABEL_NAMES))
        self.action_classes = len(list(HICO_ACTIONS))
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        # self.pred_label = pred_label # pred
        # self.pred_b_box = pred_b_box # pred

        # self.gt_human_action = gt_human_action  # gt
        # self.gt_human_box = gt_human_box # gt
        # self.gt_object_box = gt_object_box # gt

        self.roi_size = 7
        self.spatial_scale = 1/16
        self.roi = RoIPooling2D(self.roi_size,self.roi_size,self.spatial_scale)
        _, self.classifier = decom_vgg16()

        self.obj_loc = nn.Linear(4096, self.object_classes * 4)
        self.action_score = nn.Linear(4096, self.action_classes)

    def forward(self,x, pred_scores, pred_off, rois, scale):
        bboxes = list()
        labels = list()
        scores = list()

        size = x.shape[1:]
        roi_score = pred_scores.data
        roi_cls_loc = pred_off.data
        roi = at.totensor(rois) / scale

        # Convert predictions to bounding boxes in image coordinates.
        # Bounding boxes are scaled to the scale of the input images.
        mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
            repeat(self.action_classes)[None]
        std = torch.Tensor(self.loc_normalize_std).cuda(). \
            repeat(self.action_classes)[None]

        roi_cls_loc = (roi_cls_loc * std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                            at.tonumpy(roi_cls_loc).reshape((-1, 4)))
        cls_bbox = at.totensor(cls_bbox)
        cls_bbox = cls_bbox.view(-1, self.n_class * 4)
        # clip bounding box
        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
        cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

        prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

        raw_cls_bbox = at.tonumpy(cls_bbox)
        raw_prob = at.tonumpy(prob)
        bbox, label, _ = self._suppress(raw_cls_bbox, raw_prob)


        pred_human_box = self.get_pred_human_box(bbox, label)
        roi_indices = torch.array([0.0])
        rois = at.totensor(pred_human_box).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        pred_object_loc_off = self.obj_loc(fc7)
        action_scores = self.action_score(fc7)

        # pred_object

        return pred_object_loc_off, action_scores

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
    def get_pred_human_box(self, pred_bboxes, pred_labels):
        # get predicted human bbox according to pred labels
        # In case there is no human box
        human_box = torch.zeros(size=(4, 0))
        max = 0
        # ymin xmin ymax xmax
        for label in pred_labels:
            if VOC_BBOX_LABEL_NAMES[label] == 'human':
                if abs((pred_bboxes[label][2] - pred_bboxes[label][0]) * (pred_bboxes[label][3] - pred_bboxes[label][1])) > max:
                    max = abs((pred_bboxes[label][2] - pred_bboxes[label][0]) * (pred_bboxes[label][3] - pred_bboxes[label][1]))
                    human_box = pred_bboxes[label]
        return human_box

