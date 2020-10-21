import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import array_tool as at
from model.roi_module import RoIPooling2D
import model.method as method
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression
import cupy as cp
import numpy as np

VOC_BBOX_LABEL_NAMES =  ('airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining_table',
                            'dog','horse','motorcycle','human','potted_plant','sheep','couch','train','tv')

HICO_ACTIONS = ('board', 'ride', 'sit_on','pet', 'watch', 'feed', 'hold', 'drive', 'board', 'sail', 'stand_on', 'carry', 'drink_with',
              'open', 'hug', 'kiss', 'lie_on', 'herd', 'walk', 'clean', 'eat_at', 'sit_at', 'run', 'train', 'hop_on', 'greet',
              'race')

class targetPredict(nn.Module):
    def __init__(self,  classifier, loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2),):
        super(targetPredict, self).__init__()
        self.object_classes = len(list(VOC_BBOX_LABEL_NAMES)) + 1
        self.action_classes = len(list(HICO_ACTIONS))
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.classifier = classifier
        self.roi_size = 7
        self.spatial_scale = 1/16
        self.roi = RoIPooling2D(self.roi_size,self.roi_size,self.spatial_scale)

        self.obj_loc = nn.Linear(4096, 4)
        self.action_score = nn.Linear(4096, self.action_classes)


    def forward(self,x, pred_scores, pred_off, rois, scale, gt_human_box = None, gt_object_box = None, mode = 'train'):
        self.mode = mode
        size = x.shape[2:]
        roi_score = pred_scores.data
        roi_cls_loc = pred_off.data
        roi = at.totensor(rois) / scale

        # Convert predictions to bounding boxes in image coordinates.
        # Bounding boxes are scaled to the scale of the input images.
        mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
            repeat(self.object_classes)[None]
        std = torch.Tensor(self.loc_normalize_std).cuda(). \
            repeat(self.object_classes)[None]


        roi_cls_loc = (roi_cls_loc * std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, self.object_classes, 4)
        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                            at.tonumpy(roi_cls_loc).reshape((-1, 4)))
        cls_bbox = at.totensor(cls_bbox)
        cls_bbox = cls_bbox.view(-1, self.object_classes * 4)
        # clip bounding box
        # print(size[0], size[1])
        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
        cls_bbox[:, 1::3] = (cls_bbox[:, 1::3]).clamp(min=0, max=size[1])

        prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

        raw_cls_bbox = at.tonumpy(cls_bbox)
        raw_prob = at.tonumpy(prob)
        bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)

        pred_human_box_coor, pred_human_score = self.get_pred_human_box(bbox, label, score) # human coordinates
        roi_indices = torch.tensor([0]).cuda().float()
        roiss = at.totensor(pred_human_box_coor).cuda().float()
        indices_and_rois = torch.cat([roi_indices[:, None], roiss], dim=1)
        # NOTE: important: yx->xy xmin, ymin, xmax, ymax
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        pred_object_loc = self.obj_loc(fc7)
        action_scores = self.action_score(fc7)

        if self.mode == 'train':
            # b_oh
            # gt_human: ymin xmin ymax xmax
            x_h = (gt_human_box[0][3] - gt_human_box[0][1]) / 2
            y_h = (gt_human_box[0][2] - gt_human_box[0][0]) / 2
            w_h = gt_human_box[0][3] - gt_human_box[0][1]
            h_h = gt_human_box[0][2] - gt_human_box[0][0]
            gt_human_boh = torch.tensor([x_h, y_h, w_h, h_h])

            # gt_object: ymin xmin ymax xmax
            x_o = (gt_object_box[0][3] - gt_object_box[0][1]) / 2
            y_o = (gt_object_box[0][2] - gt_object_box[0][0]) / 2
            w_o = gt_object_box[0][3] - gt_object_box[0][1]
            h_o = gt_object_box[0][2] - gt_object_box[0][0]
            gt_object_boh = torch.tensor([x_h, y_h, w_h, h_h])

            b_oh = method.b_oh(gt_human_boh, gt_object_boh)

            # Gaussian
            x_mu = (pred_object_loc[0][3] - pred_object_loc[0][1]) / 2
            y_mu = (pred_object_loc[0][2] - pred_object_loc[0][0]) / 2
            w_mu = pred_object_loc[0][3] - pred_object_loc[0][1]
            h_mu = pred_object_loc[0][2] - pred_object_loc[0][0]
            mu_ah = torch.tensor([x_mu, y_mu, w_mu, h_mu])

            Gau = method.Gaussian_fuc(b_oh, mu_ah, sigma=0.3)

            # argmax
            # print(bbox)

            pred_obj_box_coor, pred_object_labels, pred_object_score = self.get_pred_object(Gau, bbox, label, score, pred_human_score)
            # pred_object location coordinates, action score
            return pred_human_box_coor, pred_obj_box_coor, pred_object_labels, pred_object_score, pred_object_loc, action_scores, b_oh

        elif self.mode == 'test':
            # 对每一个object都要求一个gaussian，需要pred_human box, pred_object box
            # b_oh: pred object box, pred human box

            # b_oh
            # gt_human: ymin xmin ymax xmax
            x_h = (pred_human_box_coor[0][3] - pred_human_box_coor[0][1]) / 2
            y_h = (pred_human_box_coor[0][2] - pred_human_box_coor[0][0]) / 2
            w_h = pred_human_box_coor[0][3] - pred_human_box_coor[0][1]
            h_h = pred_human_box_coor[0][2] - pred_human_box_coor[0][0]
            human_boh = torch.tensor([x_h, y_h, w_h, h_h])

            x_mu = (pred_object_loc[0][3] - pred_object_loc[0][1]) / 2
            y_mu = (pred_object_loc[0][2] - pred_object_loc[0][0]) / 2
            w_mu = pred_object_loc[0][3] - pred_object_loc[0][1]
            h_mu = pred_object_loc[0][2] - pred_object_loc[0][0]
            mu_ah = torch.tensor([x_mu, y_mu, w_mu, h_mu])
            best_ids = 0
            max_gau = 0

            for i in range(len(bbox)):
                # gt_object: ymin xmin ymax xmax
                x_o = (bbox[i][3] - bbox[0][1]) / 2
                y_o = (bbox[0][2] - bbox[0][0]) / 2
                w_o = bbox[0][3] - bbox[0][1]
                h_o = bbox[0][2] - bbox[0][0]
                gt_object_boh = torch.tensor([x_h, y_h, w_h, h_h])
                b_oh = method.b_oh(human_boh, gt_object_boh)

                Gau = method.Gaussian_fuc(b_oh, mu_ah, sigma=0.3)
                if Gau > max_gau:
                    max_gau = Gau
                    best_ids = i

            pred_obj_box_coor = bbox[i]
            pred_object_labels = label[i]
            pred_object_score = score[i]

            return pred_human_box_coor, pred_obj_box_coor, pred_object_labels, pred_object_score, action_scores
    # def action_classsify(img):
    #     '''
    #     To classcify action
    #
    #     '''
    #     return -1
    #     # pass
    # def target_predict(img):
    #     return -1

    def target_location(mu_ah,b_h,b_o,sigma):
        b_oh = method.b_oh(b_h,b_h)
        g = method.Gaussian_fuc(b_oh,mu_ah,sigma)
        Smooth_L1 = nn.SmoothL1Loss()
        loss = Smooth_L1(mu_ah,b_oh)
        return g,loss
    # def human_centric (img,b_o,b_h):
    #     para = config()
    #     # region = img.crop((x, y, x+w, y+h))
    #     region = img.crop((b_h[0], b_h[1],b_h[0]+b_h[2],b_h[1]+b_h[3]))
    #     # s_ah = action_classsify(region)
    #     # mu_ah = target_predict(region)
    #     # g,loss = target_location(mu_ah,b_h,b_o,para.sigma)
    #
    #     # return s_ah,g,loss

    def get_pred_human_box(self, pred_bboxes, pred_labels, pred_scores):
        # get predicted human bbox according to pred labels
        # In case there is no human box
        human_box = torch.zeros(size=(1, 4))
        max = 0
        argmax_label = 0
        # ymin xmin ymax xmax
        for label in pred_labels:
            if VOC_BBOX_LABEL_NAMES[label] == 'human':
                if abs((pred_bboxes[label][2] - pred_bboxes[label][0]) * (pred_bboxes[label][3] - pred_bboxes[label][1])) > max:
                    max = abs((pred_bboxes[label][2] - pred_bboxes[label][0]) * (pred_bboxes[label][3] - pred_bboxes[label][1]))
                    human_box = pred_bboxes[label]
                    argmax_label = label
        human_box = torch.tensor(human_box)
        human_box = torch.reshape(human_box, (1,4))
        return human_box, pred_scores[argmax_label]

    def get_pred_object(self, gau, pred_bboxes, pred_labels, pred_obj_scores, pred_human_score):
        max_idx = 0
        max_score = 0
        for i in range(len(pred_labels)):
            if gau * pred_obj_scores[i] * pred_human_score > max_score:
                max_score = gau * pred_obj_scores[i] * pred_human_score
                print(max_score)
                max_idx = i
        print(pred_bboxes[max_idx])
        return pred_bboxes[max_idx], pred_labels[max_idx], pred_obj_scores[max_idx]

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.object_classes):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.object_classes, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > 0
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), 0, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score