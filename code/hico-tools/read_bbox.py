import scipy.io as sio
import numpy as np
# object: {'airplane': 235, 'bicycle': 1569, 'bird': 154, 'boat': 889, 'bottle': 337, 'bus': 810, 'car': 111, 'cat': 161, 'chair': 911, 'cow': 234, 'dining_table': 699, 'dog': 286, 'horse': 1363, 'motorcycle': 911, 'person': 639, 'potted_plant': 207, 'sheep': 308, 'couch': 291, 'train': 353, 'tv': 3}
# {'board': 335, 'ride': 539, 'sit_on': 3018, 'pet': 121, 'watch': 5, 'feed': 154, 'hold': 443, 'drive': 99, 'sail': 242, 'stand_on': 1169, 'carry': 685, 'drink_with': 124, 'open': 5, 'hug': 1282, 'kiss': 136, 'lie_on': 79, 'herd': 53, 'walk': 289, 'clean': 87, 'eat_at': 5, 'sit_at': 607, 'run': 645, 'train': 41, 'hop_on': 212, 'greet': 72, 'race': 24}

def read_bbox():
    VOC_BBOX_LABEL_NAMES = ('airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining_table',
                            'dog','horse','motorcycle','person','potted_plant','sheep','couch','train','tv')

    a_list = ['board', 'ride', 'sit_on','pet', 'watch', 'feed', 'hold', 'drive', 'board', 'sail', 'stand_on', 'carry', 'drink_with',
              'open', 'hug', 'kiss', 'lie_on', 'herd', 'walk', 'clean', 'eat_at', 'sit_at', 'run', 'train', 'hop_on', 'greet',
              'race']
    num_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num_o = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    dict_object = dict(zip(VOC_BBOX_LABEL_NAMES, num_o))
    dict_action = dict(zip(a_list, num_a))
    voc_objects = list(VOC_BBOX_LABEL_NAMES)


    data = sio.loadmat('/home/dzc/Desktop/ANU/8526Project/GroupProject/code/HICO/anno_bbox.mat')
    f = open('/home/dzc/Desktop/ANU/8526Project/GroupProject/code/HICO/images/label_text.txt', "r")  # 设置文件对象
    HICO_dataset_home_dir = 'HICO/images/'
    action_list = list()

    line = f.readline()
    a, b, c = [i for i in line.split()]
    action_list.append([int(a), b, c])

    while line:  # 直到读取完文件
        line = f.readline()  # 读取一行文件，包括换行符
        if len(line.split()) != 0:
            a, b, c = [i for i in line.split()]
            action_list.append([int(a), b, c])
        line = line[:-1]  # 去掉换行符，也可以不去
    f.close()

    # print(data['bbox_train'][0][13][2][0][0][2][0][0][0][0][0])
    # print(data['bbox_train'][0][0][2][0][1])
    # data['bbox_train'][0][i]是第一个图片的所有信息 data['bbox_train'][0][0][k]是对应的条目
    # data['bbox_train'][0][i][0][0] 图片名称, data['bbox_train'][0][0][1] 是图片大小， data['bbox_train'][0][0][1][0][0][0][0][0]是width
    # data['bbox_train'][0][0][k][0][0]是第一组HOI对应的信息，依次包括动作的id, human bbox, object bbox, 一对多关系[1,N] or [N, 1]，是否被遮挡 0 or 1
    bad_train = list()


    for i in range(len(data['bbox_train'][0])):
        try:
            cur = data['bbox_train'][0][i]
            img_name = cur[0][0]

            w_h_d = [cur[1][0][0][0][0][0], cur[1][0][0][1][0][0], cur[1][0][0][2][0][0]]

            human_box = [cur[2][0][0][1][0][0][0][0][0],
                            cur[2][0][0][1][0][0][1][0][0],
                            cur[2][0][0][1][0][0][2][0][0],
                            cur[2][0][0][1][0][0][3][0][0]]

            action_id = cur[2][0][0][0][0][0]
            object_box = None
            # 遍历所有object，判断是否在voc可以识别的物体里面
            for j in range(len(cur[2][0])):
                action_id = cur[2][0][j][0][0][0]
                object_name, action = action_list[action_id][1], action_list[action_id][2]
                if object_name in voc_objects and action in a_list:
                    object_box = [cur[2][0][j][2][0][0][0][0][0],
                                    cur[2][0][j][2][0][0][1][0][0],
                                    cur[2][0][j][2][0][0][2][0][0],
                                    cur[2][0][j][2][0][0][3][0][0]]
                    break
            if object_box is not None:
                with open("/home/dzc/Desktop/ANU/8526Project/GroupProject/code/HICO/images/annotation/train/%d.txt" % i,"a") as new:
                    new.write(HICO_dataset_home_dir + 'train2015/' + img_name)
                    new.write('\n')
                    new.write(str(w_h_d[0]) + " " + str(w_h_d[1])+ " " +str(w_h_d[2]))
                    new.write('\n')
                    new.write('human')
                    new.write('\n')
                    new.write(str(human_box[0])+ " " +str(human_box[1])+ " " +str(human_box[2])+ " " +str(human_box[3]))
                    new.write('\n')
                    new.write(object_name)
                    new.write('\n')
                    new.write(str(object_box[0])+ " " +str(object_box[1])+ " " +str(object_box[2])+ " " +str(object_box[3]))
                    new.write('\n')
                    new.write(action)
                    new.write('\n')

                    dict_object[object_name] += 1
                    dict_action[action] += 1

        except IndexError:
            bad_train.append(i)
            continue

    for i in range(len(data['bbox_test'][0])):
        try:
            cur = data['bbox_test'][0][i]
            img_name = cur[0][0]

            w_h_d = [cur[1][0][0][0][0][0], cur[1][0][0][1][0][0], cur[1][0][0][2][0][0]]

            human_box = [cur[2][0][0][1][0][0][0][0][0],
                            cur[2][0][0][1][0][0][1][0][0],
                            cur[2][0][0][1][0][0][2][0][0],
                            cur[2][0][0][1][0][0][3][0][0]]

            action_id = cur[2][0][0][0][0][0]
            object_box = None
            # 遍历所有object，判断是否在voc可以识别的物体里面
            for j in range(len(cur[2][0])):
                action_id = cur[2][0][j][0][0][0]
                object_name, action = action_list[action_id][1], action_list[action_id][2]
                if object_name in voc_objects and action in a_list:
                    object_box = [cur[2][0][j][2][0][0][0][0][0],
                                    cur[2][0][j][2][0][0][1][0][0],
                                    cur[2][0][j][2][0][0][2][0][0],
                                    cur[2][0][j][2][0][0][3][0][0]]
                    break
            if object_box is not None:
                with open("/home/dzc/Desktop/ANU/8526Project/GroupProject/code/HICO/images/annotation/test/%d.txt" % i,"a") as new:
                    new.write(HICO_dataset_home_dir + 'test2015/' + img_name)
                    new.write('\n')
                    new.write(str(w_h_d[0]) + " " + str(w_h_d[1])+ " " +str(w_h_d[2]))
                    new.write('\n')
                    new.write('human')
                    new.write('\n')
                    new.write(str(human_box[0])+ " " +str(human_box[1])+ " " +str(human_box[2])+ " " +str(human_box[3]))
                    new.write('\n')
                    new.write(object_name)
                    new.write('\n')
                    new.write(str(object_box[0])+ " " +str(object_box[1])+ " " +str(object_box[2])+ " " +str(object_box[3]))
                    new.write('\n')
                    new.write(action)
                    new.write('\n')
        except IndexError:
            bad_train.append(i)
            continue
    print(dict_object, dict_action)
if __name__ == '__main__':
    read_bbox()