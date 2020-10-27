from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from data.util import flip_bbox, resize_bbox
from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F


def test():
    mean = t.Tensor(self.loc_normalize_mean).cuda(). \
        repeat(self.n_class)[None]
    std = t.Tensor(self.loc_normalize_std).cuda(). \
        repeat(self.n_class)[None]

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

    bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
    print("bbox: ", bbox, "label: ", label, "score:", score)
    bboxes.append(bbox)
    labels.append(label)
    scores.append(score)