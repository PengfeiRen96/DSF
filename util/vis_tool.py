import numpy as np
import torch
import cv2
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
#改变绘图风格
# import seaborn as sns
import os
import shutil
# sns.set(color_codes=True)


def get_param(dataset):
    if dataset == 'icvl' or dataset == 'nyu':
        return 240.99, 240.96, 160, 120
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120
    elif dataset == 'FHAD' or 'hands' in dataset:
        return 475.065948, 475.065857, 315.944855, 245.287079
    elif dataset == 'itop':
        return 285.71, 285.71, 160.0, 120.0


def get_joint_num(dataset):
    if dataset == 'nyu':
        return 14
    elif dataset == 'icvl':
        return 16
    elif dataset == 'FHAD' or 'hands' in dataset or 'msra' in dataset:
        return 21
    elif dataset == 'itop':
        return 15


def pixel2world(x, dataset):
    fx, fy, ux, uy = get_param(dataset)
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, dataset):
    fx,fy,ux,uy = get_param(dataset)
    x[:, :, 0] = x[:, :, 0] * fx/x[:, :, 2] + ux
    x[:, :, 1] = uy - x[:, :, 1] * fy / x[:, :, 2]
    return x


def jointImgTo3D(uvd, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(uvd, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (uvd[0] - fu) * uvd[2] / fx
        ret[1] = (uvd[1] - fv) * uvd[2] / fy
        ret[2] = uvd[2]
    else:
        ret[:, 0] = (uvd[:,0] - fu) * uvd[:, 2] / fx
        ret[:, 1] = (uvd[:,1] - fv) * uvd[:, 2] / fy
        ret[:, 2] = uvd[:,2]
    return ret


def joint3DToImg(xyz, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(xyz, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (xyz[0] * fx / xyz[2] + fu)
        ret[1] = (xyz[1] * fy / xyz[2] + fv)
        ret[2] = xyz[2]
    else:
        ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
        ret[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
        ret[:, 2] = xyz[:, 2]
    return ret


def save_result_img(index, root_dir,pic_dir, pose):
    img = cv2.imread(root_dir + '/convert/' + '{}.jpg'.format(index), 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    draw_pose(img, pose)
    cv2.imwrite(pic_dir+'/' + str(index) + ".png", img)


def get_hierarchical_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),(0,6),
                (1, 7), (1, 12), (1, 13),(1,14),
                (2, 8), (2, 15), (2, 16), (2, 17),
                (3, 9), (3, 18), (3, 19), (3, 20),
                (4, 10), (4, 21), (4, 22), (4, 23),
                (5, 11), (5, 24), (5, 25), (5, 26),],\
               [(6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
                (7, 12), (12, 13), (13, 14),
                (8, 15), (15, 16), (16, 17),
                (9, 18), (18, 19), (19, 20),
                (10, 21), (21, 22),(22, 23),
                (11, 24), (24, 25), (25, 26)]
    elif dataset == 'nyu':
        # return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),(0,6), # first
        #         (1,7), (1,8), (2,9), (2,10), (3,11), (3, 12), (4, 13), (4, 14),
        #         (5, 15), (5, 16), (5, 17), (6, 18), (6, 18), (6, 20)],\
        #         [(7, 8), (8, 20), (9, 10), (10, 20), (11, 12), (12, 20), (13, 14), (14,20),
        #          (15,16),(16,17),(17,20),(18,20),(19,20)]
        return [(0, 2), (0, 3), (0, 4), (0, 5),(0,6), # first
                (1, 7), (1, 8), (2, 9), (2, 10), (3,11), (3, 12), (4, 13), (4, 14),
                (5, 15), (5, 16), (5, 17), (6, 18), (6, 18), (6, 20)],\
                [(7, 8), (8, 20), (9, 10), (10, 20), (11, 12), (12, 20), (13, 14), (14,20),
                 (15,16),(16,17),(17,20),(18,20),(19,20)]


# return: first represent contain , second represent adj, final ,nodenum
def get_bone_hierarchical_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0,6),
                (1,7),(1,8),(1,9),(1,10),(1,11),
                (2,12), (2, 13), (2, 14),
                (3, 15), (3, 16), (3, 17),
                (4, 18), (4, 19), (4, 20),
                (5, 21), (5, 22), (5, 23),
                (6, 24), (5, 25), (5, 26),],\
               [(7, 12), (12, 13), (13, 14),
                (8, 15), (15, 16), (16, 17),
                (9, 18), (18, 19), (19, 20),
                (10, 21), (21, 22),(22, 23),
                (11, 24), (24, 25), (25, 26)], 27


def get_sketch_group(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [[0,1,2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16],[17,18,19]]


def get_joint_group(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [[0], [1,6,7,8],[2,9,10,11],[3,12,13,14],[4,15,16,17],[5,18,19,20]]
    if dataset == 'nyu':
        return [[0,1],[2,3],[4,5],[6,7],[8,9,10],[11,12,13]]


def get_adj_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11),
                (3, 12), (12, 13),(13, 14),
                (4, 15), (15, 16),(16, 17),
                (5, 18), (18, 19), (19, 20)]
    elif 'nyu' == dataset:
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10),  (11, 13), (12, 13)]


def get_adj_mat(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11),
                (3, 12), (12, 13),(13, 14),
                (4, 15), (15, 16),(16, 17),
                (5, 18), (18, 19), (19, 20)]
    elif 'nyu' == dataset:
        return np.array([[1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                         [1,1,0,0,0,0,0,0,0,0,0,0,0,1],
                         [0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                         [0,0,1,1,0,0,0,0,0,0,0,0,0,1],
                         [0,0,0,0,1,1,0,0,0,0,0,0,0,0],
                         [0,0,0,0,1,1,0,0,0,0,0,0,0,1],
                         [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                         [0,0,0,0,0,0,1,1,0,0,0,0,0,1],
                         [0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],

                         ])


def get_joint_size(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return np.array([0.15,0.15,0.11,0.11,0.11,0.11,
                         0.1,0.1,0.08,
                         0.08,0.08,0.06,
                         0.08,0.08,0.06,
                         0.08,0.08,0.06,
                         0.06,0.06,0.05])
    if dataset == 'nyu':
        return np.array([0.06, 0.08, 0.06, 0.08, 0.06, 0.08, 0.06, 0.08, 0.1, 0.1, 0.12 ,0.1, 0.1, 0.15])


def get_dense_sketch_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11),
                (3, 12), (12, 13),(13, 14),
                (4, 15), (15, 16),(16, 17),
                (5, 18), (18, 19), (19, 20),
                (20,17),(17,14),(14,11),(11,8),
                (19,16),(16,13),(13,10),(10,7),
                (18,15),(15,12),(12,9),(9,6),
                (5,4),(4,3),(3,2),(2,1)]
    elif 'nyu' == dataset:
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]
    elif dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]

def get_multiView_sketch_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11),
                (3, 12), (12, 13),(13, 14),
                (4, 15), (15, 16),(16, 17),
                (5, 18), (18, 19), (19, 20),
                (20,17),(17,14),(14,11),(11,8),
                (19,16),(16,13),(13,10),(10,7),
                (18,15),(15,12),(12,9),(9,6),
                (5,4),(4,3),(3,2),(2,1)]
    elif 'nyu' == dataset:
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]



def get_sketch_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                [1, 6], [6, 7], [7, 8],
                [2, 9], [9, 10], [10, 11],
                [3, 12], [12, 13],[13, 14],
                [4, 15], [15, 16],[16, 17],
                [5, 18], [18, 19], [19, 20]]
    elif 'nyu' == dataset:
        return [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [9, 10], [1, 13],
                [3, 13], [5, 13], [7, 13], [10, 13], [11, 13], [12, 13]]
    elif dataset == 'icvl':
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [0, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15]]
    elif dataset == 'msra':
        return [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20]]
    elif dataset == 'itop':
        return [[0, 1],
                [1, 2], [2, 4], [4, 6],
                [1, 3], [3, 5], [5, 7],
                [1, 8],
                [8, 9], [9, 11], [11, 13],
                [8, 10], [10, 12], [12, 14]]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return [[0, 1],
                [0, 2], [2, 3], [3, 4], [4, 5],
                [0, 6], [6, 7], [7, 8], [8, 9],
                [0, 10], [10, 11], [11, 12], [12, 13],
                [0, 14], [14, 15], [15, 16], [16, 17],
                [0, 18], [18, 19], [19, 20], [20 ,21]]
    else:
        return [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ]


def get_hierarchy_mat(dataset):
    if dataset == 'mano':
        return np.array([
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]])
    elif 'nyu' == dataset:
        return np.array([[1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,1,1,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,1,1,1]])


def get_hierarchy_mapping(dataset):
    if 'mano' in dataset:
        # return [[[0], [1,2,3,16],[4,5,6,17],[10,11,12,19],[7,8,9,18],[13,14,15,20]], ]
        return [[[0], [1, 2], [3, 16], [4, 5], [6,17], [10, 11], [12, 19], [7, 8],[9, 18], [13, 14],[15,20]],\
            [[0], [1, 2], [3, 4], [7, 8], [5, 6], [9, 10]], \
            [[0, 1, 2, 3, 4, 5]],
                ]
    elif 'nyu' == dataset:
        return [[[0, 1], [2,3], [4,5], [6,7], [8,9,10], [11,12,13]], ]


def get_hierarchy_sketch(dataset):
    if 'nyu' == dataset:
        return [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [9, 10], [1, 13],
                [3, 13], [5, 13], [7, 13], [10, 13], [11, 13], [12, 13]], \
               [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
    elif 'mano' == dataset:
        # return [
        #         [0, 13], [13, 14], [14, 15], [15, 20],
        #         [0, 1], [1, 2], [2, 3], [3, 16],
        #         [0, 4], [4, 5], [5, 6], [6, 17],
        #         [0, 10], [10, 11], [11,  12], [12, 19],
        #         [0, 7], [7, 8], [8, 9], [9, 18]
        #         ],\
        #         [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [2, 4], [4, 3], [5, 1]]
        return [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ],\
                [[0, 1], [0, 3], [0, 5], [0, 7], [0, 9], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], \
               [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],\
                [[0, 0]]

                # [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [2, 4], [4, 3], [5, 1]]

############################# start for HGCN 2020/5/16  #############################
def get_bone_num(dataset, dense=True):
    if dataset == 'FHAD' or 'hands' in dataset:
        if dense:
            return 15
        else:
            return 6
    elif 'nyu' == dataset:
        return 6
    elif dataset == 'msra':
        if dense:
            return 15
        else:
            return 6
    elif dataset == 'icvl':
        if dense:
            return 15
        else:
            return 6


def get_bone_edge(dataset, dense=True):
    if dataset == 'FHAD' or 'hands' in dataset:
        if dense:
            return [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11),(12,13),(13,14)]
        else:
            return [(0, 1), (0, 2), (0, 3), (0,4), (0,5),
                    (1, 2), (2, 3), (3, 4), (4, 5)]
    elif 'nyu' == dataset:
        return [(0,5), (1,5), (2,5), (3,5),(4,5)]
    elif dataset =='msra':
        if dense:
            return [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (12, 13), (13, 14)]
        else:
            return [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
    elif dataset =='icvl':
        return [(0, 1), (1, 2),
                (3, 4), (4, 5),
                (6, 7), (7, 8),
                (9, 10), (10, 11),
                (12, 13), (13, 11)]


def get_bone_id_setting(dataset, dense=True):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [(0,1),(1,6),(7,8),
                (0,2),(2,9),(10,11),
                (0,3),(3,12),(13,14),
                (0,4),(4,15),(16,17),
                (0,5),(5,18),(19,20)]
    elif 'nyu' == dataset:
        return [(0,1),(2,3),(4,5),(6,7),(8,10),(12,13)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (3, 4), (0, 5), (5, 6), (7, 8),
                (0, 9), (9, 10), (11, 12), (0, 13), (13, 14), (15, 16),
                (0, 17), (17, 18),  (19, 20)]
    elif dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3),
                (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9),
                (0, 10), (10,11), (11, 12),
                (0, 13), (13, 14), (14, 15)]



############################# end for HGCN 2020/5/16  #############################


def get_HandModel_bone(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [ (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11),
                (3, 12), (12, 13),(13, 14),
                (4, 15), (15, 16),(16, 17),
                (5, 18), (18, 19), (19, 20)]




############################# for hand model #####################################

def get_BoneLen(dataset):
    if 'hands' in dataset or 'FHAD' in dataset:
        return [0.1401, 0.7422, 0.6914, 0.6461, 0.6439, 0.5530, 0.3132, 0.2846, 0.4553,
        0.2343, 0.1887, 0.4910, 0.2821, 0.2172, 0.4489, 0.2558, 0.2135, 0.3522,
        0.1936, 0.1799],\
        [0.0022, 0.0238, 0.0150, 0.0161, 0.0170, 0.0491, 0.0343, 0.0183, 0.0476,
        0.0180, 0.0191, 0.0446, 0.0238, 0.0168, 0.0405, 0.0346, 0.0115, 0.0427,
        0.0295, 0.0215]

def get_FingerGroup(dataset):
    if 'hands' in dataset or 'FHAD' in dataset:
        return [(2,9,10,11),(3,12,13,14),(4,15,16,17),(5,18,19,20)]

def get_PlamGroup(dataset):
    if 'hands' in dataset or 'FHAD' in dataset:
        return [(0,1),(0,2),(0,3),(0,4),(0,5)]

def get_PlamAngleCon(dataset):
    if 'hands' in dataset or 'FHAD' in dataset:
        return np.array([[0.1,0.5],[0.1,0.25],[0.1,0.25],[0.1,0.25]])*np.pi


def get_HandModel_pill(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [ (1, 6), (6, 7), (7, 8),
                (23, 9), (9, 10), (10, 11),
                (24, 12), (12, 13),(13, 14),
                (25, 15), (15, 16),(16, 17),
                (26, 18), (18, 19), (19, 20)]


def get_HandModel_wedge(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [ (26,25,5),(25,5,4),(25,24,4),(24,3,4),(24,2,3),(24,23,2),
                 (5,4,21),(4,0,21),(4,3,0),(3,1,0),(3,2,1),(2,22,1)]


def get_HandModel_size(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return np.array([0.20,0.20,
                         0.15,0.15,0.15,0.15,
                         0.13, 0.10, 0.08,
                         0.08,0.08,0.06,
                         0.08,0.08,0.06,
                         0.08,0.08,0.06,
                         0.08,0.06,0.05,
                         0.20,
                         0.08,
                         0.08, 0.08, 0.08, 0.08])


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (204, 153, 17) #(17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    BROWN = (204, 153, 17)


class Finger_color(Enum):
    THUMB = (0, 0, 255)
    INDEX = (75, 255, 66)
    MIDDLE = (255, 0, 0)
    RING = (17, 240, 244)
    LITTLE = (255, 255, 0)
    WRIST = (255, 0, 255)
    ROOT = (255, 0, 255)


def get_sketch_color(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [Finger_color.THUMB, Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.RING, Finger_color.LITTLE,
                Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX,  Finger_color.INDEX,  Finger_color.INDEX,
              Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
              Finger_color.RING, Finger_color.RING, Finger_color.RING,
              Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
              ]
    elif dataset == 'nyu':
        return (Finger_color.LITTLE,Finger_color.RING,Finger_color.MIDDLE,Finger_color.INDEX,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.LITTLE, Finger_color.RING, Finger_color.MIDDLE, Finger_color.INDEX, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST)
    elif dataset == 'icvl':
        return [Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE, Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE]
    elif dataset == 'msra':
        return [Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                 Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                 Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                 Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,
                 Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB]
    elif dataset == 'itop':
        return [Color.RED,
                Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN,
                Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE,
                ]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return (Finger_color.ROOT,
            Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,)
    else:
        return (Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
               Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
               Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,)

def get_joint_color(dataset):
    if dataset == 'FHAD'or 'hands' in dataset:
        # return [Color.CYAN, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
        #         Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.PURPLE, Color.PURPLE, Color.PURPLE,
        #         Color.RED, Color.RED, Color.RED]
        return [Finger_color.ROOT,
                Finger_color.THUMB, Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.RING, Finger_color.LITTLE,
                Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
                Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                Finger_color.RING, Finger_color.RING, Finger_color.RING,
                Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE]
    elif dataset == 'nyu':
        return [Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.RING,Finger_color.RING,Finger_color.MIDDLE,Finger_color.MIDDLE,
                Finger_color.INDEX, Finger_color.INDEX,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST,Finger_color.WRIST]
    if dataset == 'icvl':
        return [Finger_color.ROOT,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                 Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                 Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                 Finger_color.RING,Finger_color.RING,Finger_color.RING,
                 Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE]
    elif dataset == 'msra':
        return [Finger_color.WRIST,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.MIDDLE,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB]
    elif dataset == 'itop':
        return  [Color.RED,Color.BROWN,
                 Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE,
                 Color.CYAN,
                 Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return [Finger_color.ROOT, Finger_color.ROOT,
            Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,]
    else:
        return  [Finger_color.ROOT,
                 Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
                 Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                 Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
                 Finger_color.RING, Finger_color.RING, Finger_color.RING,
                 Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                 Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.LITTLE, Finger_color.RING, Finger_color.THUMB,
                ]

def draw_point(dataset, img, pose):
    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, colors_joint[0].value, -1)
        idx = idx + 1
    return img


def draw_coll_pose(dataset, img, pose):
    for index, (x, y) in enumerate(get_sketch_setting(dataset)):
        s = pose[x]
        e = pose[y]
        cv2.line(img, (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), (0, 0, 0), 3)
        cv2.circle(img, (int(s[0]), int(s[1])), 6, (255, 0, 0), -1)
        cv2.circle(img, (int(e[0]), int(e[1])), 6, (255, 0, 0), -1)
        cv2.circle(img, (int(s[0]+(e[0]-s[0])/3), int(s[1]+(e[1]-s[1])/3)), 6, (255, 0, 0), -1)
        cv2.circle(img, (int(s[0]+(e[0]-s[0])/3*2), int(s[1]+(e[1]-s[1])/3*2)), 6, (255, 0, 0), -1)
    return img


def draw_pose(dataset, img, pose, scale=1):

    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2*scale, colors_joint[idx].value, -1)
        idx = idx + 1
        if idx>=len(colors_joint):
            break
    colors = get_sketch_color(dataset)
    idx = 0
    for index, (x, y) in enumerate(get_sketch_setting(dataset)):
        if x >= pose.shape[0] or y >= pose.shape[0]:
            break
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 1*scale)
        idx = idx + 1
    return img


def draw_conf(dataset, img, pose, confi):
    colors = get_sketch_color(dataset)
    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 1+int(confi[idx]*5), colors_joint[idx].value, -1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        if x >= pose.shape[0] or y >= pose.shape[0]:
            break
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 2)
        idx = idx + 1
    return img

# def draw_conf(dataset, img, pose, confi):
#     cNorm = colors.Normalize(vmin=0, vmax=1.0)
#     jet = plt.get_cmap('jet')
#     scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
#     joint_color = 255 * scalarMap.to_rgba(1 - confi)
#
#     idx = 0
#     for pt in pose:
#         cv2.circle(img, (int(pt[0]), int(pt[1])), 3, joint_color[idx], -1)
#         idx = idx + 1
#     idx = 0
#     for x, y in get_sketch_setting(dataset):
#         cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
#                  (int(pose[y, 0]), int(pose[y, 1])),Color.BROWN.value , 1)
#         idx = idx + 1
#     return img


def draw_visible(dataset, img, pose, visible):
    idx = 0
    color = [Color.RED, Color.BLUE]
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color[visible[idx]].value, -1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])),Color.BROWN.value , 1)
        idx = idx + 1
    return img


def draw_pcl(pcl, img_size, background_value=1):
    device = pcl.device
    batch_size = pcl.size(0)
    img_pcl = []
    for index in range(batch_size):
        img = torch.ones([img_size, img_size]).to(device) * background_value
        index_x = torch.clamp(torch.floor((pcl[index, :, 0] + 1) / 2 * img_size), 0, img_size - 1).long()
        index_y = torch.clamp(torch.floor((pcl[index, :, 1] + 1) / 2 * img_size), 0, img_size - 1).long()
        # img[index_y, index_x] = pcl[index, :, 2]
        img[index_y, index_x] = -1
        img_pcl.append(img)
    return torch.stack(img_pcl, dim=0).unsqueeze(1)


def draw_depth_heatmap(dataset, pcl, heatmap, joint_id):
    fx, fy, ux, uy = get_param(dataset)
    pcl = pcl.transpose(1, 0)
    # pcl = joint3DToImg(pcl,(fx, fy, ux, uy))
    pcl = (pcl + 1) * 64
    sample_num = pcl.shape[0]
    img = np.ones((128, 128), dtype=np.uint8)*255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors_joint = get_joint_color(dataset)
    for idx in range(sample_num):
        r = int(colors_joint[joint_id].value[0] * heatmap[joint_id,idx])
        b = int(colors_joint[joint_id].value[1] * heatmap[joint_id,idx])
        g = int(colors_joint[joint_id].value[2] * heatmap[joint_id,idx])
        cv2.circle(img, (int(pcl[idx,0]), int(pcl[idx,1])), 1, (r,g,b) , -1)
    return img


def debug_point_heatmap(dataset, data, index, GFM_):
    joint_num = len(get_joint_color(dataset))
    img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    img, joint_world, joint_img, pcl_sample = img.cuda(), joint_world.cuda(), joint_img.cuda(), pcl_sample.cuda()
    pcl_normal, joint_normal, offset, coeff, max_bbx_len = pcl_normal.cuda(), joint_normal.cuda(), offset.cuda(), coeff.cuda(), max_bbx_len.cuda()
    center, cube = center.cuda(),cube.cuda()
    feature = GFM_.joint2heatmap_pcl(joint_normal, pcl_normal, max_bbx_len)
    joint_predict = GFM_.heatmap2joint_pcl(feature, pcl_normal, max_bbx_len)
    joint_predict = torch.matmul((joint_predict + offset.view(-1, 1, 3).repeat(1, joint_num, 1)) * max_bbx_len.view(-1,1,1), coeff.inverse())
    joint_predict = (joint_predict - center.view(-1, 1, 3).repeat(1, joint_num, 1)) / cube.view(-1, 1, 3).repeat(1,joint_num,1) * 2
    print((joint_predict-joint_world).sum())
    for idx in range(pcl_normal.size(0)):
        for idx_joint in range(joint_num):
            img = draw_depth_heatmap(dataset, pcl_normal.cpu().numpy()[idx], feature.cpu().numpy()[idx],idx_joint)
            img_name = './debug/pcl_heatmap_' + str(index) + '_' + str(idx)+'_' + str(idx_joint) + '.png'
            cv2.imwrite(img_name, img)


def debug_mesh(verts, faces, batch_index, data_dir, img_type):
    batch_size = verts.size(0)
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    for index in range(batch_size):
        path = data_dir + '/' + str(batch_index * batch_size + index) + '_' + img_type + '.obj'
        with open(path, 'w') as fp:
            for v in verts[index]:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def debug_only_2d_heatmap(heatmap2d, batch_index, data_dir, img_type='heatmap'):
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    batch_size, head_num, height, width = heatmap2d.size()
    if batch_size==0:
        return 0
    # heatmap2d = heatmap2d.view(batch_size,head_num,-1)
    # heatmap2d = (heatmap2d - heatmap2d.min(dim=-1, keepdim=True)[0])
    # heatmap2d = heatmap2d / (heatmap2d.max(dim=-1, keepdim=True)[0] + 1e-8)
    # heatmap2d = heatmap2d.view(batch_size, head_num, height, width)
    heatmap = heatmap2d.cpu().detach().numpy()
    for index in range(heatmap2d.size(0)):
        for joint_index in range(heatmap2d.size(1)):
                img_dir = data_dir + '/' + img_type + '_' + str(batch_size * batch_index + index) + '_' + str(
                    joint_index) + '.png'
                heatmap_draw = cv2.resize(heatmap[index, joint_index], (128, 128))
                heatmap_color = 255 * scalarMap.to_rgba(1 - heatmap_draw)
                cv2.imwrite(img_dir, heatmap_color.reshape(128, 128, 4)[:, :, 0:3])


def debug_2d_heatmap(heatmap2d, batch_index, data_dir, size, img_type='heatmap', save=False):
    # heatmap2d = GFM_.joint2heatmap2d(joint_img, std = 6.8 ,isFlip=False, heatmap_size=img.size(-1))
    # depth = GFM_.depth2map(joint_img[:, :, 2], heatmap_size=img.size(-1))
    # feature = heatmap2d * (depth + 1) / 2
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    batch_size, head_num, height, width = heatmap2d.size()
    if batch_size==0:
        return 0
    heatmap2d = heatmap2d.view(batch_size, head_num, -1)
    heatmap2d = (heatmap2d - heatmap2d.min(dim=-1, keepdim=True)[0])
    heatmap2d = heatmap2d / (heatmap2d.max(dim=-1, keepdim=True)[0] + 1e-8)
    heatmap2d = heatmap2d.view(batch_size, head_num, height, width) * 1.5
    # heatmap2d = torch.where(heatmap2d.eq(0), torch.ones_like(heatmap2d) * 0.5, heatmap2d)
    # heatmap2d = F.interpolate(heatmap2d, size=[128, 128])
    heatmap_list = []
    heatmap = heatmap2d.cpu().detach().numpy()
    for index in range(heatmap2d.size(0)):
        for joint_index in range(heatmap2d.size(1)):
                img_dir = data_dir + '/' +  str(batch_size * batch_index + index) + '_' + str(
                    joint_index) + '_' +img_type + '.png'
                heatmap_draw = cv2.resize(heatmap[index, joint_index], (size, size))
                heatmap_color = 255 * scalarMap.to_rgba(1 - heatmap_draw)
                if save:
                    heatmap_color = heatmap_color.reshape(size, size, 4)
                    heatmap_color[:, :, 1:3] = heatmap_color[:, :, 1:3]
                    cv2.imwrite(img_dir, heatmap_color)
                heatmap_list.append(heatmap_color.reshape(size, size, 4)[:, :, 0:3])
                # ret, img_show = cv2.threshold(img_draw[index, 0] * 255.0, 245, 255, cv2.THRESH_BINARY)
                # img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB)
                # cv2.imwrite(img_dir, img_show/2 + heatmap_color.reshape(128, 128, 4)[:, :, 0:3])
    return np.stack(heatmap_list, axis=0).squeeze()


def debug_offset(data, batch_index, GFM_):
    img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    img_size = 32
    batch_size,joint_num,_ = joint_world.size()
    offset = GFM_.joint2offset(joint_img, img, feature_size=img_size)
    unit = offset[:, 0:joint_num*3, :, :].numpy()
    for index in range(batch_size):
        fig, ax = plt.subplots()
        unit_plam = unit[index, 0:3, :, :]
        x = np.arange(0,img_size,1)
        y = np.arange(0,img_size,1)

        X, Y = np.meshgrid(x, y)
        Y = img_size - 1 - Y
        ax.quiver(X, Y, unit_plam[0, ...], unit_plam[1, ...])
        ax.axis([0, img_size, 0, img_size])
        plt.savefig('./debug/offset_' + str(batch_index) + '_' + str(index) + '.png')


def debug_offset_heatmap(img, joint, batch_index, GFM_, kernel_size):
    img_size = 128
    batch_size,joint_num,_ = joint.size()
    offset = GFM_.joint2offset(joint, img, kernel_size, feature_size=img_size)
    heatmap = offset[:, joint_num*3:, :, :].numpy()
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    img_draw = img.numpy()
    for index in range(batch_size):
        for joint_index in range(joint_num):
            img_dir = './debug/' + str(batch_size * batch_index + index) + '_' + str(joint_index) + '.png'
            heatmap_color = 255 * scalarMap.to_rgba((kernel_size-heatmap[index, joint_index].reshape(128, 128)) / kernel_size)
            img_show = cv2.cvtColor(img_draw[index, 0] * 255.0/2.0, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(img_dir, img_show + heatmap_color.reshape(128, 128, 4)[:, :, 0:3])


def debug_bone_heatmap(img, joint, batch_index, GFM_, kernel_size):
    feature_size = 32
    up = torch.nn.Upsample(scale_factor=4, mode='nearest')
    # down = torch.nn.MaxPool2d((4,4),stride=4)
    batch_size,joint_num,_ = joint.size()
    offset = GFM_.joint2boneheatmap2d('nyu', joint, img, kernel_size, feature_size=feature_size)
    # heatmap = up(down(offset)).numpy()
    heatmap = up((offset)).numpy()
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    img_draw = img.numpy()
    for index in range(batch_size):
        for joint_index in range(13):
            img_dir = './debug/' + str(batch_size * batch_index + index) + '_' + str(joint_index) + '.png'
            heatmap_color = 255 * scalarMap.to_rgba((kernel_size-heatmap[index, joint_index].reshape(128, 128)) / kernel_size)
            img_show = cv2.cvtColor(img_draw[index, 0] * 255.0/2.0, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(img_dir, img_show + heatmap_color.reshape(128, 128, 4)[:, :, 0:3])


def debug_cluster(img, cluster_index, index, data_dir, name):
    batch_size, num, input_size, input_size = img.size()
    for img_idx in range(img.size(0)):
        for channel_idx in range(img.size(1)):
            cluster_id = cluster_index.detach().cpu().numpy()[img_idx]
            if not os.path.exists(data_dir + '/' + str(cluster_id) + '/'):
                os.makedirs(data_dir + '/' + str(cluster_id) + '/')
            img_draw = (img.detach().cpu().numpy()[img_idx, channel_idx] + 1) / 2 * 255
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(data_dir + '/'+str(cluster_id)+'/' + str(batch_size * index + img_idx) + '_'+str(channel_idx)+"_" + name + '.png', img_draw)


def debug_2d_img(img, index, data_dir, name, batch_size):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_idx in range(img.size(0)):
        for channel_idx in range(img.size(1)):
            img_draw = (img.detach().cpu().numpy()[img_idx,channel_idx] + 1) / 2 * 255
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_'+str(channel_idx)+"_" + name + '.png', img_draw)
            img_list.append(img_draw)
    return np.stack(img_list, axis=0)


def debug_2d_pose(img, joint_img, index, dataset, data_dir, name, batch_size, save=False, alpha=False):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_idx in range(joint_img.size(0)):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_pose(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB),joint_uvd[img_idx], input_size // 128 )
        if alpha:
            r, g, b = cv2.split(img_show)
            a = np.ones([input_size, input_size]) * 255
            for i in range(input_size):
                for j in range(input_size):
                    if (r[i][j] == 255 and g[i][j] == 255 and b[i][j] == 255):
                        a[i][j] = 0
            img_show = np.stack((b, g, r, a), axis=-1)

        if save:
           cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)
        img_list.append(img_show)
    return np.stack(img_list, axis=0)


def debug_confidence_joint(img, joint_img, conf, index, dataset, data_dir, name, batch_size, save=False):
    _, _, input_size, input_size = img.size()
    img_list = []
    conf = conf.detach().cpu().numpy()
    for img_idx in range(img.size(0)):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_conf(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB), joint_uvd[img_idx], conf[img_idx])
        if save:
            cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)
        img_list.append(img_show)
    return np.stack(img_list, axis=0)


def debug_2d_pose_index(img, joint_img, index, dataset, data_dir, name):
    num, input_size, input_size = img.size()
    joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
    img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
    img_show = draw_pose(dataset, cv2.cvtColor(img_draw[0], cv2.COLOR_GRAY2RGB), joint_uvd)
    cv2.imwrite(data_dir + '/' + str(index) + '_' + name + '.png', img_show)


def draw_2d_pose(img, joint_img, dataset):
    num, input_size, input_size = img.size()
    joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
    img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
    img_show = draw_pose(dataset, cv2.cvtColor(img_draw[0], cv2.COLOR_GRAY2RGB), joint_uvd)
    return img_show


def stack_row_pic(batch_img_list):
    out_imgs = []
    list_size = batch_img_list.shape[0]
    batch_size = batch_img_list.shape[1]
    for batch_index in range(batch_size):
        img_list = []
        for list_index in range(list_size):
            img_list.append(batch_img_list[list_index, batch_index])
        imgs = np.hstack(img_list)
        out_imgs.append(imgs)
    return np.stack(out_imgs, axis=0)


def stack_col_pic(batch_img_list):
    out_imgs = []
    list_size = batch_img_list.shape[0]
    batch_size = batch_img_list.shape[1]
    for batch_index in range(batch_size):
        img_list = []
        for list_index in range(list_size):
            img_list.append(batch_img_list[list_index, batch_index])
        imgs = np.concatenate(img_list, axis=0)
        out_imgs.append(imgs)
    return np.stack(out_imgs, axis=0)


def draw_muti_pic(batch_img_list, index, data_dir, name, text=None, save=True, max_col=7):
    batch_size = batch_img_list[0].shape[0]
    for batch_index in range(batch_size):
        img_list = []
        img_list_temp = []
        for img_index, imgs in enumerate(batch_img_list):
            img_list_temp.append(imgs[batch_index].squeeze())
            if (img_index + 1) % max_col == 0:
                img_list.append(np.hstack(img_list_temp))
                img_list_temp = []

        if img_index < max_col:
            imgs = np.hstack(img_list_temp)
        else:
            # img_list.append(np.hstack(img_list_temp))
            imgs = np.concatenate(img_list, axis=0)

        if text:
            cv2.putText(imgs, text[batch_index], (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
        if save:
            cv2.imwrite(data_dir + '/' + name + '_' + str(batch_size * index + batch_index)  + '.png', imgs)
    return imgs


def draw_batch_img(batch_img, index, data_dir, name):
    batch_size = batch_img.shape[0]
    for batch_index in range(batch_size):
        cv2.imwrite(data_dir + '/' + str(batch_size * index + batch_index) + '_' + name + '.png', batch_img[batch_index])


def Matr(axis, theta, batch_size):
    M = torch.eye(4)
    if axis == 0:
        M[1, 1] = torch.cos(theta)
        M[1, 2] = -torch.sin(theta)
        M[2, 1] = torch.sin(theta)
        M[2, 2] = torch.cos(theta)
    elif axis == 1:
        M[0, 0] = torch.cos(theta)
        M[0, 2] = -torch.sin(theta)
        M[2, 0] = torch.sin(theta)
        M[2, 2] = torch.cos(theta)
    elif axis == 2:
        M[0, 0] = torch.cos(theta)
        M[0, 1] = -torch.sin(theta)
        M[1, 0] = torch.sin(theta)
        M[1, 1] = torch.cos(theta)
    else:
        M[axis - 3, 3] = theta
    return M.unsqueeze(0).repeat(batch_size,1,1).permute(0,2,1)


def rotate_pcl(pcl, rot):
    device = pcl.device
    batch_size, point_num, _ = pcl.size()
    pcl_rot = torch.cat((pcl, torch.ones([batch_size, point_num, 1]).to(device)), dim=-1)
    pcl_rot = torch.matmul(pcl_rot, Matr(2, rot[2], batch_size).to(device))
    pcl_rot = torch.matmul(pcl_rot, Matr(0, rot[0], batch_size).to(device))
    pcl_rot = torch.matmul(pcl_rot, Matr(1, rot[1], batch_size).to(device))
    return pcl_rot


def debug_ThreeView_pose(pcl, joint_xyz, index, dataset, data_dir, name):
    """
    :param pcl:
    :param joint_xyz:
    :param index:
    :param dataset:
    :param data_dir:
    :param name:
    :return:
    """
    batch_size = pcl.size(0)
    if batch_size == 0:
        return 0
    device = pcl.device
    for view_index in range(0, 3):
        rot = torch.tensor([0., 0., 0.]).to(device)
        if view_index != 2:
            rot[view_index] = np.pi / 2.0
        pcl_rot = rotate_pcl(pcl, rot)
        joint_rot = rotate_pcl(joint_xyz, rot)
        img = draw_pcl(pcl_rot, 128)
        for img_idx in range(img.size(0)):
            joint_uvd = (joint_rot.detach().cpu().numpy() + 1) / 2 * 128
            img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
            im_color = cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB)
            # mask = (img_draw[img_idx, 0] == 255)[:, :, np.newaxis]
            # img_draw = cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB)
            # im_color = cv2.applyColorMap(img_draw[img_idx, 0].astype(np.uint8), cv2.COLORMAP_HOT)
            # im_color[mask] = 255
            img_show = draw_pose(dataset, im_color, joint_uvd[img_idx])
            cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '-view'+str(view_index) + '-' + name + '.png', img_show)


def debug_visible_joint(img, joint_img, visible, index, dataset, data_dir, name):
    batch_size,_,input_size,input_size = img.size()
    visible = visible.detach().cpu().numpy().astype(np.int)
    for img_idx in range(img.size(0)):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_visible(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB), joint_uvd[img_idx], visible[img_idx])
        cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)
        # cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_ori.png', img_draw[img_idx, 0])


def debug_point(img, so_node, index, dataset, data_dir, name):
    batch_size,_,input_size,input_size = img.size()
    for img_idx in range(img.size(0)):
        joint_uvd = (so_node.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_point(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB), joint_uvd[img_idx])
        cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)


def debug_canny(img,index,data_dir):
    batch = img.size(0)
    img_np = (img.cpu().numpy()+1)/2*255
    for img_idx in range(batch):
        img_draw = cv2.Canny(img_np[img_idx,0].astype(np.uint8), 50, 150)
        cv2.imwrite(data_dir + '/' + str(batch * index + img_idx) + '_canny.png', img_draw)


def debug_pcl_heatmap(pcl_heatmap, batch_index, data_dir='./debug', img_type='pcl'):
    img_size = 128
    batch_size, channel_num, point_num, _ = pcl_heatmap.size()
    if batch_size == 0:
        return 0
    img = draw_pcl(pcl_heatmap.reshape(batch_size*channel_num,point_num,3), img_size)
    img = img.view(batch_size, channel_num, img_size, img_size)
    heatmap = img.cpu().detach().numpy()
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    for index in range(batch_size):
        for joint_index in range(channel_num):
                img_dir = data_dir + '/' + img_type + '_' + str(batch_index*batch_size + index) + '_' + str(joint_index) + '.png'
                heatmap_draw = cv2.resize(heatmap[index, joint_index], (128, 128))
                heatmap_color = 255 * scalarMap.to_rgba(1 - heatmap_draw)
                cv2.imwrite(img_dir, heatmap_color.reshape(128, 128, 4)[:, :, 0:3])

class data_distribute:
    def __init__(self, bin_num=100):
        self.bin_num = bin_num
        self.bin_batch_count, self.bin_batch_joint_count = torch.zeros((self.bin_num)),torch.zeros((self.bin_num))

    def forward(self,joint_pd, label):

        loss_all = torch.sum(torch.pow(joint_pd - label, 2), dim=-1)
        loss_batch = torch.mean(loss_all, -1)
        loss_batch_joint = loss_all.view(-1)
        bin_batch = torch.floor(loss_batch*50 * (self.bin_num - 0.0001)).long()
        bin_batch_joint = torch.floor(loss_batch_joint*50 * (self.bin_num - 0.0001)).long()
        bin_batch = torch.clamp(bin_batch,0,self.bin_num-1)
        bin_batch_joint = torch.clamp(bin_batch_joint, 0, self.bin_num - 1)
        for i in range(self.bin_num):
            self.bin_batch_count[i] += (bin_batch == i).sum().item()
            self.bin_batch_joint_count[i] += (bin_batch_joint == i).sum().item()

        return 0

    def vis(self):
        plt.bar( np.arange(self.bin_num),self.bin_batch_count.numpy())

        plt.legend()

        plt.xlabel('loss value')
        plt.ylabel('number')

        plt.title('loss distribute')

        plt.savefig('./loss_distribute.png')

        plt.bar(np.arange(self.bin_num),self.bin_batch_joint_count.numpy())

        plt.legend()

        plt.xlabel('loss value')
        plt.ylabel('number')

        plt.title('loss distribute')

        plt.savefig('./loss_joint_distribute.png')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_mesh(path, verts, faces, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY
    if transpose:
        verts = verts[:, [0, 2, 1]]

    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = (141 / 255, 184 / 255, 226 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor(edge_color)

    ax.add_collection3d(mesh)

    cam_equal_aspect_3d(ax, verts, transpose=transpose)

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def display_mesh(path, verts, faces, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY
    if transpose:
        verts = verts[:, [0, 2, 1]]

    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = (141 / 255, 184 / 255, 226 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor(edge_color)

    ax.add_collection3d(mesh)

    cam_equal_aspect_3d(ax, verts, transpose=transpose)

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def cam_equal_aspect_3d(ax, verts, flip_x=False, transpose=True):
    '''
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    :param ax:
    :param verts:
    :param flip_x:
    :return:
    '''
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    # min_lim, max_lim = np.min(centers - r), np.max(centers + r)
    if flip_x:
      ax.set_xlim(centers[0] + r, centers[0] - r)
      # ax.set_xlim(max_lim, min_lim)
    else:
      ax.set_xlim(centers[0] - r, centers[0] + r)
      # ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
    # ax.set_ylim(min_lim, max_lim)
    # ax.set_zlim(max_lim, min_lim)
    if transpose:
      ax.set_xlabel('X')
      ax.set_ylabel('Z')
      ax.set_zlabel('Y')
    else:
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
    # ax.view_init(5, -5)
    ax.view_init(5, -85)

########################## segement ##########################
from collections import namedtuple

THUMB = (0, 0, 255)
INDEX = (75, 255, 66)
MIDDLE = (255, 0, 0)
RING = (17, 240, 244)
LITTLE = (255, 255, 0)
WRIST = (255, 0, 255)
ROOT = (255, 0, 255)

def get_segmentFingerColor():
    JointClass = namedtuple('JointClass', ['name', 'id', 'color'])
    classes = [
        JointClass('plam', 0, (255, 255, 255)),
        JointClass('index', 1, INDEX),
        JointClass('middle', 2, MIDDLE),
        JointClass('ring', 4, RING),
        JointClass('little', 3, LITTLE),
        JointClass('thumb', 5, THUMB),
        JointClass('background', 6, (255, 255, 255)),
    ]
    train_id_to_color = [c.color for c in classes if (c.id != -1 and c.id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    return train_id_to_color


def get_segmentJointColor():
    JointClass = namedtuple('JointClass', ['name', 'id', 'color'])
    classes = [
        JointClass('plam', 0, (255, 255, 255)),
        JointClass('index-root', 1, (0, 255, 0)),
        JointClass('index-pip', 2, (0, 205, 0)),
        JointClass('index-dip', 3, (0, 155, 0)),

        JointClass('middle-root', 4, (0, 0, 255)),
        JointClass('middle-pip', 5, (0, 0, 205)),
        JointClass('middle-dip', 6, (0, 0, 155)),

        JointClass('little-root', 7, (0, 104, 139)),
        JointClass('little-pip', 8, (0, 154, 205)),
        JointClass('little-dip', 9, (0, 178, 238)),

        JointClass('ring-root', 10, (255, 255, 0)),
        JointClass('ring-pip', 11, (205, 205, 0)),
        JointClass('ring-dip', 12, (155, 155, 0)),

        JointClass('thumb-root', 13, (105, 0, 0)),
        JointClass('thumb-pip', 14, (155, 0, 0)),
        JointClass('thumb-dip', 15, (205, 0, 0)),

        JointClass('index-tip', 16, (0, 105, 0)),
        JointClass('middle-tip', 17, (0, 0, 105)),
        JointClass('little-tip', 18, (0, 191, 255)),
        JointClass('ring-tip', 19, (105, 105, 0)),
        JointClass('thumb-tip', 20, (255, 0, 0)),
        JointClass('background', 21, (255, 255, 255)),
    ]
    train_id_to_color = [c.color for c in classes if (c.id != -1 and c.id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    return train_id_to_color


def get_nyu_segmentJointColor():
    JointClass = namedtuple('JointClass', ['name', 'id', 'color'])
    classes = [

        JointClass('little-root', 0, (0, 104, 139)),
        JointClass('little-dip', 1, (0, 178, 238)),

        JointClass('ring-root', 2, (255, 255, 0)),
        JointClass('ring-dip', 3, (155, 155, 0)),

        JointClass('middle-root', 4, (0, 0, 255)),
        JointClass('middle-dip', 5, (0, 0, 155)),

        JointClass('index-root', 6, (0, 255, 0)),
        JointClass('index-dip', 7, (0, 155, 0)),

        JointClass('thumb-root', 8, (255, 0, 0)),
        JointClass('thumb-pip', 9, (205, 0, 0)),
        JointClass('thumb-dip', 10, (155, 0, 0)),
        JointClass('plam', 11, (0, 0, 0)),
        JointClass('background', 12, (255, 255, 255)),
    ]
    train_id_to_color = [c.color for c in classes if (c.id != -1 and c.id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    return train_id_to_color


# def get_segmentJointColor():
#     JointClass = namedtuple('JointClass', ['name', 'id', 'color'])
#     classes = [
#         JointClass('plam', 0, (0, 0, 0)),
#         JointClass('index-root', 1, (0, 255, 0)),
#         JointClass('index-pip', 2, (0, 205, 0)),
#         JointClass('index-dip', 3, (0, 155, 0)),
#         JointClass('index-tip', 4, (0, 105, 0)),
#         JointClass('middle-root', 5, (0, 0, 255)),
#         JointClass('middle-pip', 6, (0, 0, 205)),
#         JointClass('middle-dip', 7, (0, 0, 155)),
#         JointClass('middle-tip', 8, (0, 0, 105)),
#         JointClass('ring-root', 9, (255, 255, 0)),
#         JointClass('ring-pip', 10, (205, 205, 0)),
#         JointClass('ring-dip', 11, (155, 155, 0)),
#         JointClass('ring-tip', 12, (105, 105, 0)),
#         JointClass('little-root', 13, (0, 104, 139)),
#         JointClass('little-pip', 14, (0, 154, 205)),
#         JointClass('little-dip', 15, (0, 178, 238)),
#         JointClass('little-tip', 16, (0, 191, 255)),
#         JointClass('thumb-root', 17, (105, 0, 0)),
#         JointClass('thumb-pip', 18, (155, 0, 0)),
#         JointClass('thumb-dip', 19, (205, 0, 0)),
#         JointClass('thumb-tip', 20, (255, 0, 0)),
#         JointClass('background', 21, (255, 255, 255)),
#     ]
#     train_id_to_color = [c.color for c in classes if (c.id != -1 and c.id != 255)]
#     train_id_to_color.append([0, 0, 0])
#     train_id_to_color = np.array(train_id_to_color)
#     id_to_train_id = np.array([c.id for c in classes])
#     return train_id_to_color

def get_segmentBKColor():
    JointClass = namedtuple('JointClass', ['name', 'id', 'color'])
    classes = [
        JointClass('hand', 0, (0, 0, 0)),
        JointClass('background', 1, (255, 255, 255)),
    ]
    train_id_to_color = [c.color for c in classes if (c.id != -1 and c.id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    return train_id_to_color


def vis_segment(segement, type, dataset, batch_index, data_dir='./debug', name='pred'):
    batch_size = segement.size(0)
    if type == 'bk':
        id_to_color = get_segmentBKColor()
    elif type == 'finger':
        id_to_color = get_segmentFingerColor()
    elif type == 'joint':
        id_to_color = get_segmentJointColor()
    segment_img = id_to_color[segement.detach().cpu()]
    for img_index in range(batch_size):
        cv2.imwrite(data_dir+'/segment_%s_%d.png' % (name,batch_size * batch_index + img_index), segment_img[img_index])


