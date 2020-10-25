import os
import xml.etree.ElementTree as ET
import random
import numpy as np

from .util import read_image


class HICODataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """
    def __init__(self, data_dir, split='train'): # default split name: HICO train

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        self.ids = self.get_imgNames(data_dir + '/annotation/' + split + '/')[:50]
        self.data_dir = data_dir
        self.split = split
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_imgNames(self,path):
        imgs_list = list()
        imgs = os.listdir(path)
        for i in range(len(imgs)):
            imgs_list.append(int(imgs[i][:-4]))
            # random.shuffle(imgs_list)
        return imgs_list

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes
            10.18 Update, img(W, H, C) -> (C, H, W), bboxes coordinates (xmin, mmax, ymin, ymax) - > (ymin, xmin, ymax, xmax), labels
            gt_human_bbox, gt_object bbox, action
        """
        id_ = self.ids[i]
        # data_dir: /home/dzc/Desktop/ANU/8526Project/GroupProject/code/HICO/images
        anno = self.data_dir + 'annotation/'+ self.split + '/'+ str(id_) + '.txt'
        object = list()
        bbox = list()
        human_box = list()
        object_box = list()
        action = list()
        # for obj in anno.findall('object'):
        #     # when in not using difficult split, and the object is
        #     # difficult, skipt it.
        #     if not self.use_difficult and int(obj.find('difficult').text) == 1:
        #         continue
        #
        #     bndbox_anno = obj.find('bndbox')
        #     # subtract 1 to make pixel indexes 0-based
        #     bbox.append([
        #         int(bndbox_anno.find(tag).text) - 1
        #         for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
        #     name = obj.find('name').text.lower().strip()
        #     label.append(VOC_BBOX_LABEL_NAMES.index(name))
        f = open(anno, "r")
        line = f.readline()
        img_file = line[:-1]

        line = f.readline()
        # width, height, channel = [i for i in line[:-1].split()]

        # human label
        line = f.readline()

        try:
            object.append(VOC_BBOX_LABEL_NAMES.index(line[:-1]))
        except ValueError:
            print(line[:-1])
            print(img_file)
        # human box
        line = f.readline()
        xmin, xmax, ymin, ymax = [i for i in line[:-1].split()]
        bbox.append([int(ymin), int(xmin), int(ymax), int(xmax)])
        human_box.append([int(ymin), int(xmin), int(ymax), int(xmax)])

        # object label
        line = f.readline()
        if line[:-1] == 'person':
            object.append(VOC_BBOX_LABEL_NAMES.index('human'))
        else:
            object.append(VOC_BBOX_LABEL_NAMES.index(line[:-1]))
        # object box
        line = f.readline()
        xmin, xmax, ymin, ymax = [i for i in line[:-1].split()]
        bbox.append([int(ymin), int(xmin), int(ymax), int(xmax)])
        object_box.append([int(ymin), int(xmin), int(ymax), int(xmax)])

        # action
        line = f.readline()
        action.append(HICO_ACTIONS.index(line[:-1]))
        f.close()

        bbox = np.stack(bbox).astype(np.float32)
        human_box = np.stack(human_box).astype(np.float32)
        object_box = np.stack(object_box).astype(np.float32)
        label = np.stack(object).astype(np.int32)
        action = np.stack(action).astype(np.int32)

        # Load a image
        img = read_image(img_file, color=True)
        # print(bbox, label, human_box, object_box, action)
        return img, bbox, label, human_box, object_box, action

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES =  ('airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining_table',
                            'dog','horse','motorcycle','human','potted_plant','sheep','couch','train','tv')

HICO_ACTIONS = ('board', 'ride', 'sit_on','pet', 'watch', 'feed', 'hold', 'drive', 'board', 'sail', 'stand_on', 'carry', 'drink_with',
              'open', 'hug', 'kiss', 'lie_on', 'herd', 'walk', 'clean', 'eat_at', 'sit_at', 'run', 'train', 'hop_on', 'greet',
              'race')