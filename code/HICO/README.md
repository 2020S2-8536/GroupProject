## 2020.10.15 update
HICO annotation preprocess complete.


**object**: {'airplane': 235, 'bicycle': 1569, 'bird': 154, 'boat': 889, 'bottle': 337, 'bus': 810, 'car': 111, 'cat': 161, 'chair': 911, 'cow': 234, 
        'dining_table': 699, 'dog': 286, 'horse': 1363, 'motorcycle': 911, 'person': 639, 'potted_plant': 207, 
        'sheep': 308, 'couch': 291, 'train': 353, 'tv': 3}

**actions**: {'board': 335, 'ride': 539, 'sit_on': 3018, 'pet': 121, 'watch': 5, 'feed': 154, 'hold': 443, 'drive': 99, 'sail': 242, 'stand_on': 1169,
'carry': 685, 'drink_with': 124, 'open': 5, 'hug': 1282, 'kiss': 136, 'lie_on': 79, 'herd': 53, 'walk': 289, 'clean': 87,
'eat_at': 5, 'sit_at': 607, 'run': 645, 'train': 41, 'hop_on': 212, 'greet': 72, 'race': 24}

The annotations are converted from .mat files into text files and are listed in train and test folders respectively. 

There are 10471 sample images in training and validation 2802 images for testing.

For a annotatio file, each line contains one key information, respectively they are:
**Line 1**: HICO/images/test2015/HICO_test2015_00000002.jpg (**image relative path**)
**Line 2**: 640 461 3 (**image size info**)
**Line 3**: human (**Human classification label**)
**Line 4**: 226 340 18 210 (**Human bounding box**, formatted as x1, x2, y1, y2, where (x1, y1) is the top left point and (x2, y2) is thee bottom right)
**Line 5**(: horse (**Interacting Object classification label**)
**Line 6**: 174 393 65 440 (Object Bbox, share the same format with human bbox)
**Line 7**: hug (**Action classification label**)


Demo

1. cd tools

2. Run demo_visualize.m to see how to visualize bbox annotations.


Annotation File

1. anno_bbox.mat contains three variables:
    a. bbox_train: bounding boxes annotation for the train2015 set

        filename: file names
        size:     image width, hieght, depth
        hoi:      HOI annotations
            id:          action index of list_action
            bboxhuman:   human bounding boxes
            bboxobject:  object bounding boxes
            connection:  instances of HOI (human-object pairs); each row is
                         one instance, represented by a pair of human bbox
                         index and object bbox index
            invis:       1: HOI invisible; bboxhuman/bboxobject/connection
                            will be empty
                         0: HOI visible; bboxhuman/bboxobject/connection
                            will not be empty

    b. bbox_test: bounding boxes annotation for the test2015 set; same 
                  stucture as bbox_train

    c. list_action: list of HOIs
