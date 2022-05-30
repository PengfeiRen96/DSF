# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np
import math
import cv2
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
import scipy.io as sio
from util import vis_tool
import random
sys.path.append('..')
import csv
from sklearn.decomposition import PCA
from render_model.mano_layer import Render
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from render_model.transfer import define_Encoder, define_Decoder,define_G


joint_select = np.array([0, 1, 3, 5, #0-3
                         6, 7, 9, 11,#4-7
                         12, 13, 15, 17,#8-11
                         18, 19, 21, 23,#12-15
                         24, 25, 27, 28,#16-19
                         32, 30, 31])#20-22
calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]
calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]

mano_select = np.array([0, 1, 2, 4, #0-3
                         6, 7, 8, 10,#4-7
                         12, 13, 14, 16,#8-11
                         18, 19, 20, 22,#12-15
                         24, 25, 26, 28,#16-19
                         29])#20-22
mano_calculate = [20, 15, 14, 13, 11, 10, 9, 3, 2, 1, 7, 6, 5, 19, 18, 17, 12, 8, 0, 4, 16]

xrange = range



def Matr(axis, theta):
    M = np.eye(3)
    if axis == 0:
        M[1, 1] = np.cos(theta)
        M[1, 2] = -np.sin(theta)
        M[2, 1] = np.sin(theta)
        M[2, 2] = np.cos(theta)
    elif axis == 1:
        M[0, 0] = np.cos(theta)
        M[0, 2] = -np.sin(theta)
        M[2, 0] = np.sin(theta)
        M[2, 2] = np.cos(theta)
    elif axis == 2:
        M[0, 0] = np.cos(theta)
        M[0, 1] = -np.sin(theta)
        M[1, 0] = np.sin(theta)
        M[1, 1] = np.cos(theta)
    else:
        M[axis - 3, 3] = theta
    return M


def pixel2world(x, y, z, paras):
    fx, fy, fu, fv = paras
    worldX = (x - fu) * z / fx
    worldY = (fv - y) * z / fy
    return worldX, worldY


def pixel2world_noflip(x, y, z, paras):
    fx, fy, fu, fv = paras
    worldX = (x - fu) * z / fx
    worldY = (y - fv) * z / fy
    return worldX, worldY


def world2pixel(x, y, z, paras):
    fx, fy, fu, fv = paras
    pixelX = x * fx / z + fu
    pixelY = fv - y * fy / z
    return pixelX, pixelY


def get_center_from_bbx(depth, bbx, upper=807, lower=171):
    centers = np.array([0.0, 0.0, 300.0])
    img = depth[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    centers[0] += bbx[0]
    centers[1] += bbx[1]
    return centers


def get_center_fast(img, upper=650, lower=100):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    return centers


def rotatePoint2D(p1, center, angle):
    """
    Rotate a point in 2D around center
    :param p1: point in 2D (u,v,d)
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated point
    """
    alpha = angle * np.pi / 180.
    pp = p1.copy()
    pp[0:2] -= center[0:2]
    pr = np.zeros_like(pp)
    pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
    pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
    pr[2] = pp[2]
    ps = pr
    ps[0:2] += center[0:2]
    return ps


def rotatePoints2D(pts, center, angle):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i] = rotatePoint2D(pts[i], center, angle)
    return ret


def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
    return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
    return ret


def batchtransformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    new = []
    for index in range(pts.shape[0]):
        new.append(transformPoints2D(pts[index], M[index]))
    return np.stack(new, axis=0)


def hands17_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def nyu_reader(img_path):
    img = cv2.imread(img_path)
    depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256, dtype=np.float32)
    return depth


def icvl_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def msra_reader(image_name, para):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right, bottom = data[:6]
    depth = np.zeros((height, width), dtype=np.float32)
    f.seek(4*6)
    data = np.fromfile(f, dtype=np.float32)
    depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
    depth_pcl = np.reshape(data, (bottom-top, right-left))
    #convert to world
    imgHeight, imgWidth = depth_pcl.shape
    hand_3d = np.zeros([3, imgHeight*imgWidth])
    d2Output_x = np.tile(np.arange(imgWidth), (imgHeight, 1)).reshape(imgHeight, imgWidth).astype('float64') + left
    d2Output_y = np.repeat(np.arange(imgHeight), imgWidth).reshape(imgHeight, imgWidth).astype('float64') + top
    hand_3d[0], hand_3d[1] = pixel2world(d2Output_x.reshape(-1), d2Output_y.reshape(-1), depth_pcl.reshape(-1),para)
    hand_3d[2] = depth_pcl.reshape(-1)
    valid = np.arange(0,imgWidth*imgHeight)
    valid = valid[(hand_3d[0, :] != 0)|(hand_3d[1, :] != 0)|(hand_3d[2, :] != 0)]
    handpoints = hand_3d[:, valid].transpose(1,0)

    return depth,handpoints


def synth_reader(image_name):
    img = Image.open(image_name)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def msra14_reader(image_name, para):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    depth = np.reshape(data, (240, 320))
    return depth


def HO3D_reader(depth_filename):
    """Read the depth image in dataset and decode it"""
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt
    return dpt

import re


def shrec_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


class loader(Dataset):
    def __init__(self, root_dir, phase, img_size, center_type, dataset_name):
        self.rng = np.random.RandomState(23455)
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.phase = phase
        self.img_size = img_size
        self.center_type = center_type
        self.allJoints = False
        # create OBB
        self.pca = PCA(n_components=3)
        self.sample_num = 512

    # numpy
    def jointImgTo3D(self, uvd, paras=None, flip=None):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.paras
        if flip == None:
            flip = self.flip
        ret = np.zeros_like(uvd, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (uvd[0] - fu) * uvd[2] / fx
            ret[1] = flip * (uvd[1] - fv) * uvd[2] / fy
            ret[2] = uvd[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
            ret[:, 1] = flip * (uvd[:, 1] - fv) * uvd[:, 2] / fy
            ret[:, 2] = uvd[:, 2]
        else:
            ret[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
            ret[:, :, 1] = flip * (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
            ret[:, :, 2] = uvd[:, :, 2]

        return ret

    def joint3DToImg(self, xyz, paras=None, flip=None):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.paras
        if flip==None:
            flip = self.flip
        ret = np.zeros_like(xyz, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (xyz[0] * fx / xyz[2] + fu)
            ret[1] = (flip * xyz[1] * fy / xyz[2] + fv)
            ret[2] = xyz[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
            ret[:, 1] = (flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
            ret[:, 2] = xyz[:, 2]
        else:
            ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
            ret[:, :, 1] = (flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
            ret[:, :, 2] = xyz[:, :, 2]
        return ret

    # tensor
    def pointsImgTo3D(self, point_uvd, flip=None):
        if flip == None:
            flip = self.flip
        point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
        point_xyz[:, :, 0] = (point_uvd[:, :, 0] - self.paras[2]) * point_uvd[:, :, 2] / self.paras[0]
        point_xyz[:, :, 1] = flip * (point_uvd[:, :, 1] - self.paras[3]) * point_uvd[:, :, 2] / self.paras[1]
        point_xyz[:, :, 2] = point_uvd[:, :, 2]
        return point_xyz

    def points3DToImg(self, joint_xyz, flip=None):
        fx, fy, fu, fv = self.paras
        if flip == None:
            flip = self.flip
        joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
        joint_uvd[:, :, 0] = (joint_xyz[:, :, 0] * fx / (joint_xyz[:, :, 2]+1e-8) + fu)
        joint_uvd[:, :, 1] = (flip * joint_xyz[:, :, 1] * fy / (joint_xyz[:, :, 2]) + fv)
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    # augment
    def comToBounds(self, com, size, paras):
        fx, fy, fu, fv = paras
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2] * fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2] * fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2] * fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2] * fy + 0.5))
        return xstart, xend, ystart, yend, zstart, zend

    def comToTransform(self, com, size, dsize, paras):
        """
        Calculate affine transform from crop
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: affine transform
        """

        xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size, paras)

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            scale = np.eye(3) * dsize[0] / float(wb)
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            scale = np.eye(3) * dsize[1] / float(hb)
            sz = (wb * dsize[1] / hb, dsize[1])
        scale[2, 2] = 1

        # ori
        # xstart = int(np.floor(dsize[0] / 2. - sz[1] / 2.))
        # ystart = int(np.floor(dsize[1] / 2. - sz[0] / 2.))

        # change by pengfeiren
        xstart = int(np.floor(dsize[0] / 2. - sz[0] / 2.))
        ystart = int(np.floor(dsize[1] / 2. - sz[1] / 2.))
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return np.dot(off, np.dot(scale, trans))

    def recropHand(self, crop, M, Mnew, target_size, paras, background_value=0., nv_val=0., thresh_z=True, com=None,
                   size=(250, 250, 250)):

        flags = cv2.INTER_NEAREST

        warped = cv2.warpPerspective(crop, np.dot(M, Mnew), target_size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(background_value))
        # warped[np.isclose(warped, nv_val)] = background_value # Outliers will appear on the edge
        warped[warped < nv_val] = background_value

        if thresh_z is True:
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, size, paras)
            msk1 = np.logical_and(warped < zstart, warped != 0)
            msk2 = np.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.  # backface is at 0, it is set later

        return warped


    def moveCoM(self, dpt, cube, com, off, joints3D, M, paras, pad_value=0):
        """
        Adjust already cropped image such that a moving CoM normalization is simulated
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param off: offset to center of mass (dx,dy,dz) in 3D coordinates
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if offset is 0, nothing to do
        if np.allclose(off, 0.):
            return dpt, joints3D, com, M

        # add offset to com
        new_com = self.joint3DToImg(self.jointImgTo3D(com) + off)

        # check for 1/0.
        if not (np.allclose(com[2], 0.) or np.allclose(new_com[2], 0.)):
            # scale to original size
            Mnew = self.comToTransform(new_com, cube, dpt.shape, paras)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                      nv_val=np.min(dpt[dpt > 0])-1, thresh_z=True, com=new_com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        # adjust joint positions to new CoM
        new_joints3D = joints3D + self.jointImgTo3D(com) - self.jointImgTo3D(new_com)

        return new_dpt, new_joints3D, new_com, Mnew

    def rotateHand(self, dpt, cube, com, rot, joints3D, pad_value=0):
        """
        Rotate hand virtually in the image plane by a given angle
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param rot: rotation angle in deg
        :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, rotation angle in XXX
        """

        # if rot is 0, nothing to do
        if np.allclose(rot, 0.):
            return dpt, joints3D, rot

        rot = np.mod(rot, 360)

        M = cv2.getRotationMatrix2D((dpt.shape[1] // 2, dpt.shape[0] // 2), -rot, 1)

        flags = cv2.INTER_NEAREST

        new_dpt = cv2.warpAffine(dpt, M, (dpt.shape[1], dpt.shape[0]), flags=flags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)


        if (dpt > 0).sum() > 0:
            dpt_min = np.min(dpt[dpt > 0])-1
            new_dpt[new_dpt < dpt_min] = 0

        com3D = self.jointImgTo3D(com)
        joint_2D = self.joint3DToImg(joints3D + com3D)
        data_2D = np.zeros_like(joint_2D)
        for k in xrange(data_2D.shape[0]):
            data_2D[k] = rotatePoint2D(joint_2D[k], com[0:2], rot)
        new_joints3D = (self.jointImgTo3D(data_2D) - com3D)

        return new_dpt, new_joints3D, rot

    def scaleHand(self, dpt, cube, com, sc, joints3D, M, paras, pad_value=0):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if scale is 1, nothing to do
        if np.allclose(sc, 1.):
            return dpt, joints3D, cube, M

        new_cube = [s * sc for s in cube]

        # check for 1/0.
        if not np.allclose(com[2], 0.):
            # scale to original size
            Mnew = self.comToTransform(com, new_cube, dpt.shape, paras)
            new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                      nv_val=np.min(dpt[dpt>0])-1, thresh_z=True, com=com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        new_joints3D = joints3D

        return new_dpt, new_joints3D, new_cube, Mnew

    def jointmoveCoM(self, dpt, cube, com, off, joints3D, M, paras, pad_value=0):
        """
        Adjust already cropped image such that a moving CoM normalization is simulated
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param off: offset to center of mass (dx,dy,dz) in 3D coordinates
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if offset is 0, nothing to do
        if np.allclose(off, 0.):
            return joints3D,com,M

        # add offset to com
        new_com = self.joint3DToImg(self.jointImgTo3D(com) + off.reshape(1, 3))

        batch_size = joints3D.shape[0]
        Mnew = []
        for index in range(batch_size):
            # check for 1/0.
            if not (np.allclose(com[index,2], 0.) or np.allclose(new_com[index, 2], 0.)):
                # scale to original size
                Mnew.append(self.comToTransform(new_com[index], cube, dpt.shape, paras))
            else:
                Mnew.append(M)

        new_joints3D = []
        for index in range(batch_size):
            # adjust joint positions to new CoM
            new_joints3D.append(joints3D[index] + self.jointImgTo3D(com[index]) - self.jointImgTo3D(new_com[index]))

        return np.stack(new_joints3D, axis=0), new_com, np.stack(Mnew, axis=0)

    def jointrotateHand(self, dpt, cube, com, rot, joints3D, pad_value=0):
        """
        Rotate hand virtually in the image plane by a given angle
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param rot: rotation angle in deg
        :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, rotation angle in XXX
        """

        # if rot is 0, nothing to do
        if np.allclose(rot, 0.):
            return joints3D,rot

        rot = np.mod(rot, 360)
        com3D = self.jointImgTo3D(com)

        batch_size = joints3D.shape[0]
        new_joints3D = []
        for index in range(batch_size):
            joint_2D = self.joint3DToImg(joints3D[index] + com3D[index])
            data_2D = np.zeros_like(joint_2D)
            for k in xrange(data_2D.shape[0]):
                data_2D[k] = rotatePoint2D(joint_2D[k], com[index, 0:2], rot)
            new_joints3D.append(self.jointImgTo3D(data_2D) - com3D[index])

        return np.stack(new_joints3D, axis=0), rot

    def jointscaleHand(self,dpt, cube, com, sc, joints3D, M, paras, pad_value=0):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if scale is 1, nothing to do
        if np.allclose(sc, 1.):
            return joints3D, cube, M

        new_cube = [s * sc for s in cube]

        batch_size = joints3D.shape[0]
        Mnew = []
        for index in range(batch_size):
            # check for 1/0.
            if not np.allclose(com[index, 2], 0.):
                # scale to original size
                Mnew.append(self.comToTransform(com[index], new_cube, dpt.shape, paras))

            else:
                Mnew.append(M)
        new_joints3D = joints3D

        return new_joints3D, new_cube, np.stack(Mnew, axis=0)

    def rand_augment(self, sigma_com=None, sigma_sc=None, rot_range=None):
        if sigma_com is None:
            sigma_com = 35.

        if sigma_sc is None:
            sigma_sc = 0.05

        if rot_range is None:
            rot_range = 180.

        # mode = self.rng.randint(0, len(self.aug_modes))
        # off = self.rng.randn(3) * sigma_com  # +-px/mm
        # rot = self.rng.uniform(-rot_range, rot_range)
        # sc = abs(1. + self.rng.randn() * sigma_sc)
        #
        # mode = np.random.randint(0, len(self.aug_modes))
        # off = np.random.randn(3) * sigma_com  # +-px/mm
        # rot = np.random.uniform(-rot_range, rot_range)
        # sc = abs(1. + np.random.randn() * sigma_sc)

        mode = random.randint(0, len(self.aug_modes)-1)
        off = np.array([random.uniform(-1, 1) for a in range(3)]) * sigma_com# +-px/mm
        rot = random.uniform(-rot_range, rot_range)
        sc = abs(1. + random.uniform(-1,1) * sigma_sc)
        return mode, off, rot, sc

    def augmentCrop(self, img, gt3Dcrop, com, cube, M, mode, off, rot, sc, paras, normZeroOne=False):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :param sigma_com: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: image, 3D annotations(unnormal), com(image coordinates), cube
        """
        assert len(img.shape) == 2
        assert isinstance(self.aug_modes, list)
        premax = img.max()
        if np.max(img) == 0:
            imgD = img
            new_joints3D = gt3Dcrop
        elif self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgD, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, paras, pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            imgD, new_joints3D, rot = self.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgD, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, paras, pad_value=0)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            imgD = img
            new_joints3D = gt3Dcrop
        else:
            raise NotImplementedError()
        imgD = self.normalize_img(premax, imgD, com, cube)
        return imgD, None, new_joints3D, np.asarray(cube), com, M, rot

    def batchaugmentCrop(self, img, batch_gt3Dcrop, com, cube, M, mode, off, rot, sc, paras, normZeroOne=False):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :param sigma_com: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: image, 3D annotations(unnormal), com(image coordinates), cube
        """
        assert len(img.shape) == 2
        assert isinstance(self.aug_modes, list)
        premax = img.max()
        if np.max(img) == 0:
            imgD = img
        elif self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            new_batch_gt3Dcrop, com, M = self.jointmoveCoM(img.astype('float32'), cube, com, off, batch_gt3Dcrop, M,paras, pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            new_batch_gt3Dcrop, rot = self.jointrotateHand(img.astype('float32'), cube, com, rot, batch_gt3Dcrop, pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            new_batch_gt3Dcrop, cube, M = self.jointscaleHand(img.astype('float32'), cube, com, sc, batch_gt3Dcrop, M, paras,pad_value=0)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            new_batch_gt3Dcrop = batch_gt3Dcrop
        else:
            raise NotImplementedError()
        return new_batch_gt3Dcrop, np.asarray(cube), com, M, rot

    def normalize_img(self, premax, imgD, com, cube):
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)
        return imgD

    # use deep-pp's method
    def Crop_Image_deep_pp(self, depth, com, size, dsize, paras):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size, paras)

        # crop patch from source
        cropped = self.getCrop(depth, xstart, xend, ystart, yend, zstart, zend)

        # resize to same size, 等比例缩放
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])

        # if wb != hb:
        #     print('crop!')
        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        if cropped.shape[0] > cropped.shape[1]:
            scale = np.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = np.eye(3) * sz[0] / float(cropped.shape[1])

        scale[2, 2] = 1

        # depth resize
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)


        # 把缩放之后的图片长和宽不一定相同，并且不一定与目标大小相同
        ret = np.ones(dsize, np.float32) * 0  # use background as filler
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape, xstart, ystart
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, np.dot(off, np.dot(scale, trans))

    # use deep-pp's method
    def Crop_Image_deep_pp_nodepth(self, com, size, dsize, paras):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size,paras)
        # resize to same size
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if hb > wb:
            scale = np.eye(3) * sz[1] / float(hb)
        else:
            scale = np.eye(3) * sz[0] / float(wb)
        scale[2, 2] = 1

        off = np.eye(3)
        xstart = int(np.floor(dsize[0] / 2. - sz[1] / 2.))
        ystart = int(np.floor(dsize[1] / 2. - sz[0] / 2.))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return np.dot(off, np.dot(scale, trans))

    # for get trans
    def Batch_Crop_Image_deep_pp(self, depth, com, size, dsize, paras):
        batch_trans=[]
        for index in range(com.shape[0]):
            trans = self.Crop_Image_deep_pp_nodepth(com[index], size, dsize, paras)
            batch_trans.append(trans)
        return np.stack(batch_trans, axis=0)

    def getCrop(self, depth, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
        """
        Crop patch from image
        :param depth: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        """
        if len(depth.shape) == 2:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1]))), mode='constant',
                             constant_values=background)
        elif len(depth.shape) == 3:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1]),
                      :].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1])),
                                       (0, 0)), mode='constant', constant_values=background)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            msk1 = np.logical_and(cropped < zstart, cropped != 0)
            msk2 = np.logical_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later
        return cropped
    # point cloud
    def pca_point(self, pcl, joint):
        point_num = pcl.shape[0]
        if point_num < 10:
            pcl = self.joint2pc(joint)
        self.pca.fit(pcl)
        coeff = self.pca.components_.T
        if coeff[1, 0] < 0:
            coeff[:, 0] = -coeff[:, 0]
        if coeff[2, 2] < 0:
            coeff[:, 2] = -coeff[:, 2]
        coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])
        points_rotation = np.dot(pcl, coeff)
        joint_rotation = np.dot(joint, coeff)

        index = np.arange(points_rotation.shape[0])
        if points_rotation.shape[0] < self.sample_num:
            tmp = math.floor(self.sample_num / points_rotation.shape[0])
            index_temp = index.repeat(tmp)
            index = np.append(index_temp,
                              np.random.choice(index, size=divmod(self.sample_num, points_rotation.shape[0])[1],
                                               replace=False))
        index = np.random.choice(index, size=self.sample_num, replace=False)
        points_rotation_sampled = points_rotation[index]

        # Normalize Point Cloud
        scale = 1.2
        bb3d_x_len = scale * (points_rotation[:, 0].max() - points_rotation[:, 0].min())
        bb3d_y_len = scale * (points_rotation[:, 1].max() - points_rotation[:, 1].min())
        bb3d_z_len = scale * (points_rotation[:, 2].max() - points_rotation[:, 2].min())
        max_bb3d_len = bb3d_x_len / 2.0

        points_rotation_sampled_normalized = points_rotation_sampled / max_bb3d_len
        joint_rotation_normalized = joint_rotation / max_bb3d_len
        if points_rotation.shape[0] < self.sample_num:
            offset = np.mean(points_rotation, 0) / max_bb3d_len
        else:
            offset = np.mean(points_rotation_sampled_normalized, 0)
        points_rotation_sampled_normalized = points_rotation_sampled_normalized - offset
        joint_rotation_normalized = joint_rotation_normalized - offset
        return points_rotation_sampled_normalized, joint_rotation_normalized, offset, coeff, max_bb3d_len

    def joint2pc(self, joint, radius=15):
        joint_num, _ = joint.shape

        radius = np.random.rand(joint_num, 100) * radius
        theta = np.random.rand(joint_num, 100) * np.pi
        phi = np.random.rand(joint_num, 100) * np.pi

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        point = np.tile(joint[:, np.newaxis, :], (1, 100, 1)) + np.concatenate(
            (x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=-1)
        point = point.reshape([100 * joint_num, 3])
        sample = np.random.choice(100 * joint_num, self.sample_num, replace=False)
        return point[sample, :]

    #return normalied pcl
    def getpcl(self, imgD, com3D, cube, M):
        mask = np.where(imgD > 0.99)
        dpt_ori = imgD * cube[2] / 2.0 + com3D[2]
        # change the background value to 1
        dpt_ori[mask] = 0

        pcl = (self.depthToPCL(dpt_ori, M) - com3D)
        pcl_num = pcl.shape[0]
        cube_tile = np.tile(cube / 2.0, pcl_num).reshape([pcl_num, 3])
        pcl = pcl / cube_tile
        return pcl

    def farthest_point_sample(self, xyz, npoint):
        N, C = xyz.shape
        S = npoint
        if N < S:
            centroids = np.arange(N)
            centroids = np.append(centroids, np.random.choice(centroids, size=S - N, replace=False))
        else:
            centroids = np.zeros(S).astype(np.int)
            distance = np.ones(N) * 1e10
            farthest = np.random.randint(0, S)
            for i in range(S):
                centroids[i] = farthest
                centroid = xyz[farthest, :]
                dist = np.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = distance.argmax()
        return np.unique(centroids)

    def depthToPCL(self, dpt, T, background_val=0.):
        fx, fy, fu, fv = self.paras
        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - fu) / fx * depth
        col = self.flip * (pts[:, 1] - fv) / fy * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    def PCLToDepth(self, pcl, img_size, T, background_val=0.):
        pcl_uvd = transformPoints2D(self.joint3DToImg(pcl), T)
        uv = np.clip(pcl_uvd[:, :2], 0, img_size-0.5)
        img = np.ones([img_size * img_size])*background_val
        pcl_index = np.floor(uv[:, 1]) * img_size + uv[:, 0]

        # base version
        # pcl_index = pcl_index.astype('int')
        # img[pcl_index] = pcl_uvd[:, 2]

        # consinder depth
        for index, depth_index in enumerate(pcl_index.astype('int')):
            if img[depth_index] > 0:
                if (img[depth_index] > pcl_uvd[index, 2]):
                    img[depth_index] = pcl_uvd[index, 2]
            else:
                img[depth_index] = pcl_uvd[index, 2]

        return img.reshape(img_size, img_size)


    # tensor
    def unnormal_joint_img(self, joint_img):
        device = joint_img.device
        joint = torch.zeros(joint_img.size()).to(device)
        joint[:, :, 0:2] = (joint_img[:, :, 0:2] + 1) / 2 * self.img_size
        joint[:, :, 2] = (joint_img[:, :, 2] + 1) / 2 * self.cube_size[2]
        return joint

    def uvd_nl2xyz_tensor(self, uvd, center, m, cube):
        batch_size, point_num, _ = uvd.size()
        device = uvd.device
        cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        M_t = m.to(device).view(batch_size, 1, 3, 3)
        M_inverse = torch.inverse(M_t).repeat(1, point_num, 1, 1)

        uv_unnormal = (uvd[:, :, 0:2] + 1) * (self.img_size / 2)
        d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
        uvd_unnormal = torch.cat((uv_unnormal, d_unnormal),dim=-1)
        uvd_world = self.get_trans_points(uvd_unnormal, M_inverse)
        xyz = self.pointsImgTo3D(uvd_world)
        return xyz

    def uvd_nl2xyznl_tensor(self, uvd, center, m, cube):
        batch_size, point_num, _ = uvd.size()
        device = uvd.device
        cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        M_t = m.to(device).view(batch_size, 1, 3, 3)
        M_inverse = torch.inverse(M_t).repeat(1, point_num, 1, 1)

        uv_unnormal= (uvd[:, :, 0:2] + 1) * (self.img_size / 2)
        d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
        uvd_unnormal = torch.cat((uv_unnormal, d_unnormal),dim=-1)
        uvd_world = self.get_trans_points(uvd_unnormal, M_inverse)
        xyz = self.pointsImgTo3D(uvd_world)
        xyz_noraml = (xyz - center_t) / (cube_size_t / 2.0)
        return xyz_noraml

    def xyz_nl2uvdnl_tensor(self, joint_xyz, center, M, cube_size):
        device = joint_xyz.device
        batch_size, joint_num, _ = joint_xyz.size()
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)

        joint_temp = joint_xyz * cube_size_t / 2.0 + center_t
        joint_uvd = self.points3DToImg(joint_temp)
        joint_uvd = self.get_trans_points(joint_uvd, M_t)
        joint_uv = joint_uvd[:, :, 0:2] / self.img_size * 2.0 - 1
        joint_d = (joint_uvd[:, :, 2:] - center_t[:, :, 2:]) / (cube_size_t[:, :, 2:] / 2)
        joint = torch.cat((joint_uv, joint_d), dim=-1)
        return joint

    # get point feature from 2d feature
    def imgFeature2pclFeature(self, pcl_uvd, feature):
        '''
        :param pcl: BxN*3 Tensor
        :param feature: BxCxWxH Tensor
        :param center:  Tensor
        :param M:  FloatTensor
        :param cube_size:  LongTensor
        :return: select_feature: BxCxN
        '''

        batch_size, point_num, _ = pcl_uvd.size()
        feature_size = feature.size(-1)
        feature_dim = feature.size(1)

        pcl_uvd = torch.clamp(pcl_uvd,-1,1)
        pcl_uvd = (pcl_uvd+1)/2.0 * (feature_size)
        uv_idx = torch.floor(pcl_uvd[:,:,1]) * feature_size + torch.floor(pcl_uvd[:,:,0])
        uv_idx = uv_idx.long().view(batch_size,1,point_num).repeat(1,feature_dim,1)
        select_feature = torch.gather(feature.view(batch_size,-1, feature_size*feature_size), dim=-1, index=uv_idx).view(batch_size,feature_dim,point_num)

        return select_feature

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
        joints_trans_xy = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans_z = joints[:, :, 2:]
        return torch.cat((joints_trans_xy,joints_trans_z),dim=-1)

    #return fixed size point cloud
    def Img2pcl(self, img, feature_size, center, M, cube, sample_num=1024):
        batch_size = img.size(0)
        device = img.device
        img_rs = F.interpolate(img, (feature_size, feature_size))
        mask = img_rs.le(0.99)
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        # coords = torch.cat((coords, img_rs), dim=1)
        # return coords
        pcl = torch.cat((coords, img_rs), dim=1).view(batch_size, 3, feature_size*feature_size).permute(0, 2, 1)
        pcl = torch.split(pcl, 1)
        mask = torch.split(mask.view(batch_size, 1, feature_size*feature_size).permute(0, 2, 1), 1)
        pcl_valid = []
        for index in range(batch_size):
            temp = torch.masked_select(pcl[index], mask[index]).view(1, -1, 3)
            temp = self.uvd_nl2xyznl_tensor(temp, center[index], M[index], cube[index])
            temp = temp.squeeze(0)
            point_num = temp.size(0)
            if sample_num == 0:
                pcl_valid.append(temp)
            elif point_num == 0:
                pcl_valid.append(torch.zeros(sample_num, 3).to(device))
            elif sample_num > point_num:
                mult = int(np.floor(sample_num/point_num))
                point_mult = temp.repeat(mult, 1)
                if sample_num-point_num*mult==0:
                    pcl_valid.append(point_mult)
                else:
                    pcl_index = torch.multinomial(torch.ones(point_num).to(device), sample_num-point_num*mult, False)
                    pcl_valid.append(torch.cat((point_mult,torch.index_select(temp,0,pcl_index)),dim=0))
            else:
                pcl_index = torch.multinomial(torch.ones(point_num).to(device), sample_num, False)
                pcl_valid.append(torch.index_select(temp, 0, pcl_index))
        return torch.stack(pcl_valid, dim=0)

    def pcl2Img(self, pcl, feature_size, center, M, cube):
        batch_size = pcl.size(0)
        device = pcl.device
        uvdPoint = self.xyz_nl2uvdnl_tensor(pcl, center, M, cube)
        uv = torch.clamp((uvdPoint[:, :, :2] + 1) / 2, 0, 1) * (feature_size-1)
        img = torch.ones([batch_size, feature_size*feature_size]).to(device)
        pcl_index = torch.floor(uv[:, :, 1]) * feature_size + uv[:, :, 0]
        pcl_index = pcl_index.long()
        # pcl_index = torch.where()
        for index in range(batch_size):
            # select = pcl_index[index].lt(feature_size*feature_size) & pcl_index[index].gt(0)
            # index_select = torch.index_select(pcl_index)
            img[index] = torch.scatter(img[index], -1, pcl_index[index], uvdPoint[index, :, 2])
        return img.view(batch_size, 1, feature_size, feature_size)
        # return torch.gather()
    # return the same size tensor as img, which stand for uvd coord.
    def xyzImg2uvdImg(self, xyz_img, render_size, center, M, cube):
        batch_size = xyz_img.size()[0]
        feature_size = xyz_img.size()[-1]
        device = xyz_img.device
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, xyz_img), dim=1)
        uvdPoint = self.xyz_nl2uvdnl_tensor(coords.view(batch_size, 3, feature_size*feature_size).permute(0, 2, 1), center, M, cube)
        # pcl = torch.cat((coords,img_rs), dim=1).view(batch_size,3,feature_size*feature_size).permute(0,2,1)
        # pcl = self.uvd_nl2xyznl_tensor(pcl, center, m, cube)
        # return pcl.view(batch_size, 3, feature_size, feature_size)
        # uvd_img = self.drawpcl(uvdPoint, render_size)
        return uvdPoint

    def uvdImg2xyzImg(self, uvd_img, center, M, cube):
        batch_size = uvd_img.size()[0]
        feature_size = uvd_img.size()[-1]
        device = uvd_img.device
        mesh_u = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_v = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_v, mesh_u), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, uvd_img), dim=1)
        xyz_img_normal = self.uvd_nl2xyznl_tensor(coords.view(batch_size, 3, feature_size*feature_size).permute(0,2,1), center, M, cube).permute(0,2,1).view(batch_size, 3, feature_size, feature_size)
        xyz_img = self.uvd_nl2xyz_tensor(coords.view(batch_size, 3, feature_size*feature_size).permute(0,2,1), center, M, cube).permute(0,2,1).view(batch_size, 3, feature_size, feature_size)
        return xyz_img, xyz_img_normal
        # pcl = torch.cat((coords,img_rs), dim=1).view(batch_size,3,feature_size*feature_size).permute(0,2,1)
        # pcl = self.uvd_nl2xyznl_tensor(pcl, center, m, cube)
        # return pcl.view(batch_size, 3, feature_size, feature_size)
        # uvd_img = self.drawpcl(uvdPoint, render_size)
        # return uvd_img

    # remove background and arm
    def crop_hand(self, img, joint, center, M, cube, offsetxy=25,  offsetz=20, hand_thickness=20):
        skeleton = joint * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
        minx, maxx = torch.min(skeleton[:, :, 0], dim=-1)[0] - offsetxy, torch.max(skeleton[:, :, 0], dim=-1)[0] + offsetxy
        miny, maxy = torch.min(skeleton[:, :, 1], dim=-1)[0] - offsetxy, torch.max(skeleton[:, :, 1], dim=-1)[0] + offsetxy
        minz, maxz = torch.min(skeleton[:, :, 2], dim=-1)[0] - offsetz, torch.max(skeleton[:, :, 2], dim=-1)[0] + offsetz
        minz -= hand_thickness

        minx, maxx = minx.view(-1, 1, 1, 1), maxx.view(-1, 1, 1, 1)
        miny, maxy = miny.view(-1, 1, 1, 1), maxy.view(-1, 1, 1, 1)
        minz, maxz = minz.view(-1, 1, 1, 1), maxz.view(-1, 1, 1, 1)

        xyz, xyz_normal = self.uvdImg2xyzImg(img, center, M, cube)
        ones = torch.ones_like(img).to(xyz.device)
        mask_x = xyz[:, 0:1, :, :].gt(minx) & xyz[:, 0:1, :, :].lt(maxx)
        mask_y = xyz[:, 1:2, :, :].gt(miny) & xyz[:, 1:2, :, :].lt(maxy)
        mask_z = xyz[:, 2:3, :, :].gt(minz) & xyz[:, 2:3, :, :].lt(maxz)
        mask = mask_x & mask_y & mask_z
        img_hand = torch.where(mask, img, ones)
        return img_hand

    # for DSSF cycleImg
    def augment_cycleImg(self, img, joint_uvd, mesh_xyz, center, M, cube):
        batch_size = img.size(0)
        scale = torch.rand([batch_size, 1, 1])*0.4-0.2 + 1
        # scale = 1
        rot = torch.rand([batch_size, 1])*2*np.pi
        # rot = torch.ones([batch_size, 1]) * np.pi /2
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        theta = torch.stack((torch.cat((cos_rot, -sin_rot), dim=1), torch.cat((sin_rot, cos_rot), dim=1)), dim=1)*scale
        trans = torch.cat((theta, torch.zeros([batch_size, 2, 1])), dim=-1).to(img.device)
        grid = F.affine_grid(trans, img.size())
        img = F.grid_sample(img-1, grid)+1

        scale = 2 - scale
        rot = -rot
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        theta = torch.stack((torch.cat((cos_rot, -sin_rot), dim=1), torch.cat((sin_rot, cos_rot), dim=1)), dim=1) * scale
        theta = theta.to(img.device)

        joint_uv_trans = torch.matmul(theta.unsqueeze(1), joint_uvd[:, :, :2].unsqueeze(-1)).squeeze(-1)
        joint_uvd_trans = torch.cat((joint_uv_trans, joint_uvd[:, :, 2:3]), dim=-1)
        joint_xyz_trans = self.uvd_nl2xyznl_tensor(joint_uvd_trans, center, M, cube)

        mesh_uvd = self.xyz_nl2uvdnl_tensor(mesh_xyz, center, M, cube)
        mesh_uv_trans = torch.matmul(theta.unsqueeze(1), mesh_uvd[:, :, :2].unsqueeze(-1)).squeeze(-1)
        mesh_uvd_trans = torch.cat((mesh_uv_trans, mesh_uvd[:, :, 2:3]), dim=-1)
        mesh_xyz_trans = self.uvd_nl2xyznl_tensor(mesh_uvd_trans, center, M, cube)

        return img, joint_uvd_trans, joint_xyz_trans, mesh_xyz_trans

    def render_depth(self, mesh, center_xyz, cube_size=250, scale=1):
        img_width, img_height = int(scale*640), int(scale*480)
        node_num, _ = mesh.shape

        # # rotate in world coord
        # mesh_xyz = mesh - center_xyz
        # mesh_xyz[:, 2] = - mesh_xyz[:, 2]
        # mesh_xyz = np.concatenate((mesh_xyz, np.ones([node_num,1])), axis=-1)
        # mesh_xyz = np.dot(mesh_xyz, Matr(2, rot[2]).T)
        # mesh_xyz = np.dot(mesh_xyz, Matr(0, rot[0]).T)
        # mesh_xyz = np.dot(mesh_xyz, Matr(1, rot[1]).T)
        # mesh_uvd = self.joint3DToImg(mesh_xyz[:, 0:3] + center_xyz)
        # mesh_xyz = mesh - center_xyz
        # mesh_xyz[:, 2] = - mesh_xyz[:, 2]
        # mesh_xyz = mesh_xyz + center_xyz
        mesh_uvd = self.joint3DToImg(mesh)

        # normal depth
        mesh_uvd[:, 2] = (mesh_uvd[:, 2] - center_xyz[2]) / cube_size * 2
        img = np.ones([img_width * img_height]) * -1

        # mesh_grid_x = torch.clamp(torch.floor((mesh[:, :, 0] + 1) / 2 * img_size), 0, img_size - 1)
        # mesh_grid_y = torch.clamp(torch.floor((-mesh[:, :, 1] + 1) / 2 * img_size), 0, img_size - 1)
        # mesh_grid_z = (mesh[:, :, 2] + 1) / 2
        # value = mesh_grid_x + mesh_grid_y*img_size + mesh_grid_z
        # value_sort, indices = torch.sort(value)
        # value_int = torch.floor(value_sort)
        # # because gpu can't assign value in order
        # img[:, value_int.long()] = (value_sort - value_int).cpu()

        mesh_grid_x = np.ceil(np.clip(mesh_uvd[:, 1] * scale, 0, img_height - 1))
        mesh_grid_y = np.ceil(np.clip(mesh_uvd[:, 0] * scale, 0, img_width - 1))
        mesh_grid_z = np.clip((mesh_uvd[:, 2] + 1) / 2, 0, 1)  # resize to [0,1]
        value = mesh_grid_x * img_width + mesh_grid_y + mesh_grid_z

        value_sort = np.sort(value)
        value_sort = value_sort[::-1]
        value_int = np.floor(value_sort).astype('int64')
        img[value_int] = ((value_sort - value_int) * 2 - 1)# recover to [-1,1]
        mask = np.ones_like(img)
        mask[img < -0.99] = 0
        img = mask * (img * cube_size / 2.0 + center_xyz[2])
        img = img.reshape(img_height, img_width)
        img = cv2.resize(img, (640, 480))
        return img

    # mesh world_coord
    # weight sum to 1
    def weight_pcl2depht(self, mesh, weight):


        cNorm = colors.Normalize(vmin=0, vmax=1.0)
        jet = plt.get_cmap('jet')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
        weight = weight/weight.max()
        img = np.zeros([480*640])
        mesh_uv = self.joint3DToImg(mesh)[:,0:2].astype(np.int)
        index = (mesh_uv[:,1] * 640 + mesh_uv[:,0])
        img[index] = weight
        img = img.reshape((480,640))
        img_color = 255 * scalarMap.to_rgba(1- img)
        return img_color

    def read_modelPara(self, data_rt, view):
        theta = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-pose.txt').reshape(-1, 45)
        quat = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-glb.txt').reshape(-1, 3)
        scale = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-scale.txt').reshape(-1, 1)
        trans = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-trans.txt').reshape(-1, 3)
        shape = np.loadtxt(data_rt+'/posePara_lm_collosion/'+self.dataset_name+'-'+self.phase+'-'+str(view)+'-shape.txt').reshape(-1, 10)

        model_para = np.concatenate([quat, theta, shape, scale, trans], axis=-1)
        return model_para

    def read_modelPara_simple(self, data_rt, file_name='posePara_lm_collosion'):
        theta = np.loadtxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-pose.txt').reshape(-1, 45)
        quat = np.loadtxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-glb.txt').reshape(-1, 3)
        scale = np.loadtxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-scale.txt').reshape(-1, 1)
        trans = np.loadtxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-trans.txt').reshape(-1, 3)
        shape = np.loadtxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-shape.txt').reshape(-1, 10)

        model_para = np.concatenate([quat, theta, shape, scale, trans], axis=-1)
        return model_para

    def save_modelPara_simple(self, data_rt, file_name, model_para):
        np.savetxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-glb.txt', model_para[:, :3], fmt='%.6f')
        np.savetxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-pose.txt', model_para[:, 3:48], fmt='%.6f')
        np.savetxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-shape.txt', model_para[:, 48:58], fmt='%.6f')
        np.savetxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-scale.txt', model_para[:, 58:59], fmt='%.6f')
        np.savetxt(data_rt+'/'+file_name+'/'+self.dataset_name+'-trans.txt', model_para[:, 59:], fmt='%.6f')
        return model_para

    def Joint2BKSeg(self, img, joint, center, M, cube, offset=20, hand_thickness=20):
        skeleton = joint * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
        minx, maxx = torch.min(skeleton[:, :, 0], dim=-1)[0] - offset, torch.max(skeleton[:, :, 0], dim=-1)[0] + offset
        miny, maxy = torch.min(skeleton[:, :, 1], dim=-1)[0] - offset, torch.max(skeleton[:, :, 1], dim=-1)[0] + offset
        minz, maxz = torch.min(skeleton[:, :, 2], dim=-1)[0] - offset, torch.max(skeleton[:, :, 2], dim=-1)[0] + offset
        minz -= hand_thickness

        minx, maxx = minx.view(-1, 1, 1, 1), maxx.view(-1, 1, 1, 1)
        miny, maxy = miny.view(-1, 1, 1, 1), maxy.view(-1, 1, 1, 1)
        minz, maxz = minz.view(-1, 1, 1, 1), maxz.view(-1, 1, 1, 1)

        xyz, xyz_normal = self.uvdImg2xyzImg(img, center, M, cube)
        ones = torch.ones_like(img).to(xyz.device)
        mask_x = xyz[:, 0:1, :, :].gt(minx) & xyz[:, 0:1, :, :].lt(maxx)
        mask_y = xyz[:, 1:2, :, :].gt(miny) & xyz[:, 1:2, :, :].lt(maxy)
        mask_z = xyz[:, 2:3, :, :].gt(minz) & xyz[:, 2:3, :, :].lt(maxz)
        mask = mask_x & mask_y & mask_z & img.lt(0.99)
        img_hand = torch.where(mask, ones, ones*0)
        return img_hand.int().squeeze(1)

    def Joint2FingerSeg(self, img, joint, center, M, cube, interval, mask):
        device = img.device
        joint = self.InterpolationJoint(joint, interval)
        skeleton = joint * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
        xyz, xyz_normal = self.uvdImg2xyzImg(img, center, M, cube)
        b, _, h, w = xyz.size()
        xyz = xyz.view(b, 3, h*w).permute(0, 2, 1)
        offset = xyz.unsqueeze(2) - skeleton.unsqueeze(1)
        offset = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1)+1e-8)
        segement = torch.argmin(offset, dim=-1).view(b, h, w)
        # plam
        segement = torch.where(segement.lt(interval*5+1), torch.zeros_like(segement).to(device), segement)

        for finger_index in range(5):
            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1))&
                                   segement.lt(interval*5+1+(finger_index+1)*3*(interval+1)),
                                   torch.ones_like(segement).to(device)*(finger_index+1), segement)

        segement = torch.where(mask.gt(0), segement, torch.ones_like(mask).long()*6)
        return segement

    def Joint2JointSeg(self, img, joint, center, M, cube, interval, mask):
        device = img.device
        joint = self.InterpolationJoint(joint, interval)
        skeleton = joint * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
        xyz, xyz_normal = self.uvdImg2xyzImg(img, center, M, cube)
        b, _, h, w = xyz.size()
        xyz = xyz.view(b, 3, h*w).permute(0, 2, 1)
        offset = xyz.unsqueeze(2) - skeleton.unsqueeze(1)
        offset = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1)+1e-8)
        segement = torch.argmin(offset, dim=-1).view(b, h, w)

        # plam
        segement = torch.where(segement.lt(interval*5+1), torch.zeros_like(segement).to(device), segement)

        # finger joint
        for finger_index in range(5):
            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1))&
                                   segement.lt(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1),
                                   torch.ones_like(segement).to(device)*(finger_index*3+1), segement)

            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1)&
                                   segement.lt(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval),
                                   torch.ones_like(segement).to(device)*(finger_index*3+2), segement)

            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval)&
                                   segement.lt(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval*2),
                                   torch.ones_like(segement).to(device)*(finger_index*3+3), segement)

            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval*2)&
                                   segement.lt(interval*5+1+(finger_index+1)*3*(interval+1)),
                                   torch.ones_like(segement).to(device)*(finger_index + 16), segement)
        segement = torch.where(mask.gt(0), segement, torch.ones_like(mask).long()*21)
        return segement

    def InterpolationJoint(self, joints, interval):
        batch_size = joints.size(0)
        device = joints.device
        plam_interval_value = torch.linspace(0, 1, interval + 2)[1:-1].reshape(1, 1, -1)
        interval_value = torch.linspace(0, 1, interval + 2)[:-1].reshape(1, 1, -1)
        child = [2, 3, 16, 5, 6, 17, 8, 9, 18, 11, 12, 19, 14, 15, 20]

        # for plam center
        plam_child_c = joints[:, [1, 4, 7, 10, 13], :]
        plam_parent_c = joints[:, 0:1, :]
        plam_bone_c = plam_child_c - plam_parent_c
        plam_shpere_c = plam_bone_c.reshape([batch_size, -1, 1, 3]) * plam_interval_value.to(device).unsqueeze(-1) \
                        + plam_parent_c.reshape([batch_size, -1, 1, 3])
        plam_shpere_c = torch.cat((plam_parent_c, plam_shpere_c.view(batch_size, -1, 3)), dim=1)

        # for finger center
        finger_child_c = joints[:, child, :]
        finger_parent_c = joints[:, 1:16, :]
        finger_shpere_c = finger_child_c - finger_parent_c
        finger_shpere_c = finger_shpere_c.reshape([batch_size, -1, 1, 3]) * interval_value.to(device).unsqueeze(-1) \
                          + finger_parent_c.reshape([batch_size, -1, 1, 3])

        shpere_c = torch.cat((plam_shpere_c.view(batch_size, -1, 3), finger_shpere_c.view(batch_size, -1, 3)), dim=1)
        return shpere_c

    def PCL2JointSeg(self, pcl, joint, interval, mask):
        device = pcl
        joint = self.InterpolationJoint(joint, interval)
        offset = pcl.unsqueeze(2) - joint.unsqueeze(1)
        offset = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1)+1e-8)
        segement = torch.argmin(offset, dim=-1)

        # plam
        segement = torch.where(segement.lt(interval*5+1), torch.zeros_like(segement).to(device), segement)

        # finger joint
        for finger_index in range(5):
            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1))&
                                   segement.lt(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1),
                                   torch.ones_like(segement).to(device)*(finger_index*3+1), segement)

            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1)&
                                   segement.lt(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval),
                                   torch.ones_like(segement).to(device)*(finger_index*3+2), segement)

            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval)&
                                   segement.lt(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval*2),
                                   torch.ones_like(segement).to(device)*(finger_index*3+3), segement)

            segement = torch.where(segement.ge(interval*5+1+finger_index*3*(interval+1) + interval//2 + 1 + interval*2)&
                                   segement.lt(interval*5+1+(finger_index+1)*3*(interval+1)),
                                   torch.ones_like(segement).to(device)*(finger_index + 16), segement)
        segement = torch.where(mask.gt(0), segement, torch.ones_like(mask).long()*21)
        return segement


class nyu_loader_test(loader):
    def __init__(self, root_dir, view=0, aug_para=[10, 0.1, 180],
                 img_size=128, cube_size=[250, 250, 250], center_type='refine', joint_num=14, loader=nyu_reader):
        super(nyu_loader_test, self).__init__(root_dir, 'train', img_size, center_type, 'nyu')
        # np.random.seed(1)
        self.paras = (588.03, 587.07, 320., 240.)
        self.ori_img_size = (640, 480)
        self.train_cube_size = np.ones([72757, 3])*cube_size[0]
        self.test_cubesize = np.ones([8252, 3])*cube_size[0]
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        self.cube_size = np.concatenate((self.train_cube_size, self.test_cubesize), axis=0)
        self.allJoints = True
        self.flip = 1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']#'rot','com','sc','none'
        self.aug_para = aug_para
        self.view = view

        self.all_img_path = []

        # load test data
        data_path = '{}/{}'.format(self.root_dir, 'test')
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_refined.txt'.format(data_path, 'test')
        print('loading test data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path

        self.all_joints_uvd = self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]
        self.refine_center_xyz = np.loadtxt(center_path)

        for index in range(8252):
            self.all_img_path.append(self.data_path + '/depth_'+str(self.view+1)+'_{:07d}.png'.format(index+1))
        print('finish!!')

        self.all_joints_xyz = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz, flip=-1))
        self.refine_center_xyz = self.jointImgTo3D(self.joint3DToImg(self.refine_center_xyz, flip=-1))

        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader

    def __getitem__(self, index):

        img_path = self.all_img_path[index]
        depth = self.loader(img_path)

        joint_xyz = self.all_joints_xyz[index].copy()
        cube_size = self.cube_size[index].copy()
        center_xyz = self.center_xyz[index]
        center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz.reshape(1, 3)
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)
        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
                                                                       trans, mode, off, rot, sc, self.paras)
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        visible = torch.ones([1]).long()
        rotation = torch.zeros([3])
        if self.phase == 'train':
            if self.aug_modes[mode] == 'rot':
                rotation[2] = rot / 180 * np.pi
        return data, joint, joint_img, center, M, cube
        # return data, data, joint, joint_img, joint_img, center, M, cube, visible, rotation

    def __len__(self):
        return len(self.all_img_path)


class nyu_loader_train_test(loader):
    def __init__(self, root_dir, view=0, aug_para=[10, 0.1, 180],
                 img_size=128, cube_size=[250, 250, 250], center_type='refine', joint_num=14, loader=nyu_reader):
        super(nyu_loader_train_test, self).__init__(root_dir, 'train', img_size, center_type, 'nyu')
        # np.random.seed(1)
        self.paras = (588.03, 587.07, 320., 240.)
        self.ori_img_size = (640, 480)
        self.train_cube_size = np.ones([72757, 3])*cube_size[0]
        self.test_cubesize = np.ones([8252, 3])*cube_size[0]
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        self.cube_size = np.concatenate((self.train_cube_size, self.test_cubesize), axis=0)
        self.allJoints = True
        self.flip = 1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']#'rot','com','sc','none'
        self.aug_para = aug_para
        self.view = view

        self.all_img_path = []

        # load train data
        data_path = '{}/{}'.format(self.root_dir, 'train')
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_refined.txt'.format(data_path, 'train')
        print('loading train data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path

        self.all_joints_uvd = self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]
        self.refine_center_xyz = np.loadtxt(center_path)

        for index in range(72757):
            self.all_img_path.append(self.data_path + '/depth_'+str(self.view+1)+'_{:07d}.png'.format(index+1))

        # load test data
        data_path = '{}/{}'.format(self.root_dir, 'test')
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_refined.txt'.format(data_path, 'test')
        print('loading test data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path

        self.all_joints_uvd = np.concatenate((self.all_joints_uvd, self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]), axis=0)
        self.all_joints_xyz = np.concatenate((self.all_joints_xyz, self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]), axis=0)
        self.refine_center_xyz = np.concatenate((self.refine_center_xyz, np.loadtxt(center_path)), axis=0)

        for index in range(8252):
            self.all_img_path.append(self.data_path + '/depth_'+str(self.view+1)+'_{:07d}.png'.format(index+1))
        print('finish!!')

        self.all_joints_xyz = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz, flip=-1))
        self.refine_center_xyz = self.jointImgTo3D(self.joint3DToImg(self.refine_center_xyz, flip=-1))

        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader

    def __getitem__(self, index):

        img_path = self.all_img_path[index]
        depth = self.loader(img_path)

        joint_xyz = self.all_joints_xyz[index].copy()
        cube_size = self.cube_size[index].copy()
        center_xyz = self.center_xyz[index]
        center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz.reshape(1, 3)
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)
        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
                                                                       trans, mode, off, rot, sc, self.paras)
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        visible = torch.ones([1]).long()
        rotation = torch.zeros([3])
        if self.phase == 'train':
            if self.aug_modes[mode] == 'rot':
                rotation[2] = rot / 180 * np.pi
        return data, joint, joint_img, center, M, cube
        # return data, data, joint, joint_img, joint_img, center, M, cube, visible, rotation

    def __len__(self):
        return len(self.all_img_path)


class nyu_CCSSL_loader(loader):
    def __init__(self, root_dir, phase, type='real', view=0, aug_para=[0, 0, 0], pesudo_name='CCSSL',
                 img_size=128, cube_size=[250, 250, 250], center_type='refine', joint_num=14, loader=nyu_reader):
        super(nyu_CCSSL_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        self.paras = (588.03, 587.07, 320., 240.)
        self.ori_img_size = (640, 480)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = 1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']#'rot','com','sc','none'
        self.aug_para = aug_para

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path
        self.view = view
        self.type = type # 'real','synth','render'

        NYU2MANO = [22, 15, 14, 13, 11, 10, 9, 3, 2, 1, 7, 6, 5, 19, 18, 17, 12, 8, 0, 4, 16]
        self.all_joints_uvd = self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz, flip=-1))

        self.all_joints_xyz_MANO = self.labels['joint_xyz'][self.view][:, joint_select, :][:, NYU2MANO, :]
        self.all_joints_xyz_MANO = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz_MANO, flip=-1))

        center_path ='{}/center_{}_{}_refined.txt'.format(data_path, self.phase, view)
        self.refine_center_xyz = np.loadtxt(center_path)

        pseudo_path ='{}/{}/joint_uvd.txt'.format(self.root_dir,pesudo_name)
        self.pseudo_joints_uvd = np.loadtxt(pseudo_path).reshape([-1, 21, 3])
        self.pseudo_joints_xyz = self.jointImgTo3D(self.pseudo_joints_uvd)

        weight_path ='{}/{}/weight.txt'.format(self.root_dir,pesudo_name)
        self.weight = np.loadtxt(weight_path).reshape([-1, 21])

        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True

    def __getitem__(self, index):
        img_path = self.data_path + '/depth_' + str(self.view + 1) + '_{:07d}.png'.format(index + 1)
        depth = self.loader(img_path)

        pseudo_joint_xyz = self.pseudo_joints_xyz[index].copy()
        weight = self.weight[index].copy()
        joint_xyz = self.all_joints_xyz[index].copy()
        cube_size = self.cube_size

        center_xyz = self.center_xyz[index]
        center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz.reshape(1, 3)
        gt3Dcrop_pseudo = pseudo_joint_xyz - center_xyz.reshape(1, 3)
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)

        imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
        curLabel = gt3Dcrop / (cube_size[2] / 2.0)
        curLabel_pseudo = gt3Dcrop_pseudo / (cube_size[2] / 2.0)
        cube = np.array(cube_size)
        com2D = center_uvd
        M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_uvd = torch.from_numpy(joint_img).float()
        joint_xyz = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        joint_xyz_pseudo = torch.from_numpy(curLabel_pseudo).float()
        weight = torch.from_numpy(weight).float()
        return data, joint_xyz, joint_uvd, center, M, cube, joint_xyz_pseudo, weight

    def __len__(self):
        return len(self.all_joints_uvd)


class nyu_loader(loader):
    def __init__(self, root_dir, phase, type='real', mask_para=0.1, view=0, aug_para=[10, 0.1, 180],
                 img_size=128, cube_size=[250, 250, 250], center_type='refine', joint_num=14, loader=nyu_reader):
        super(nyu_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        # np.random.seed(1)
        self.paras = (588.03, 587.07, 320., 240.)
        self.ori_img_size = (640, 480)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = 1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']#'rot','com','sc','none'
        self.aug_para = aug_para

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path
        self.view = view
        self.type = type # 'real','synth','render'

        NYU2MANO = [22, 15, 14, 13, 11, 10, 9, 3, 2, 1, 7, 6, 5, 19, 18, 17, 12, 8, 0, 4, 16]
        self.all_joints_uvd = self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz, flip=-1))
        self.all_model_para = self.read_modelPara(root_dir, view)

        self.all_joints_xyz_MANO = self.labels['joint_xyz'][self.view][:, joint_select, :][:, NYU2MANO, :]
        self.all_joints_xyz_MANO = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz_MANO, flip=-1))

        center_path ='{}/center_{}_{}_refined.txt'.format(data_path, self.phase, view)
        self.refine_center_xyz = np.loadtxt(center_path)
        # self.refine_center_xyz = self.jointImgTo3D(self.joint3DToImg(self.refine_center_xyz, flip=-1))
        self.mask = np.zeros([len(self.all_joints_uvd)])

        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True
        self.mask_para = mask_para

    def __getitem__(self, index):
        # if self.phase == 'train':
        #     img_path = self.data_path + '/depth_'+str(self.view+1)+'_{:07d}.png'.format(index+1)
        #     depth = self.loader(img_path)
        #     if self.type == 'render' or self.type == 'real':
        #         synth = depth
        #     elif self.type == 'synth':
        #         img_path = self.data_path + '/synthdepth_' + str(self.view + 1) + '_{:07d}.png'.format(index + 1)
        #         synth = self.loader(img_path)
        # else:
        #     if self.type == 'synth':
        #         img_path = self.data_path + '/synthdepth_' + str(self.view + 1) + '_{:07d}.png'.format(index + 1)
        #         depth = self.loader(img_path)
        #         synth = depth
        #     elif self.type == 'render':
        #         img_path = self.root_dir + '/render/' + self.phase+'/'+ str(self.view + 1) + '_{:07d}.png'.format(index + 1)
        #         depth = synth_reader(img_path)
        #         synth = depth
        #     else:
        #         img_path = self.data_path + '/depth_' + str(self.view + 1) + '_{:07d}.png'.format(index + 1)
        #         depth = self.loader(img_path)
        #         synth = depth
        # index = 32*12 + index + 20
        img_path = self.data_path + '/depth_' + str(self.view + 1) + '_{:07d}.png'.format(index + 1)
        depth = self.loader(img_path)
        if self.type == 'render' or self.type == 'real':
            synth = depth
        elif self.type == 'synth':
            img_path = self.data_path + '/synthdepth_' + str(self.view + 1) + '_{:07d}.png'.format(index + 1)
            synth = self.loader(img_path)

        joint_xyz_MANO = self.all_joints_xyz_MANO[index].copy()
        joint_xyz = self.all_joints_xyz[index].copy()
        model_para = self.all_model_para[index].copy()
        if self.phase == 'test':
            cube_size = self.test_cubesize[index]
        else:
            cube_size = self.cube_size

        center_xyz = self.center_xyz[index]
        center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz.reshape(1, 3)
        gt3Dcrop_MANO = joint_xyz_MANO - center_xyz.reshape(1, 3)
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)
        synth_crop, trans = self.Crop_Image_deep_pp(synth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)
        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
                                                                       trans, mode, off, rot, sc, self.paras)
            synth_imgD, _, curLabel_MANO, cube, com2D, M, _ = self.augmentCrop(synth_crop, gt3Dcrop_MANO, center_uvd, cube_size,
                                                                       trans, mode, off, rot, sc, self.paras)

            curLabel = curLabel / (cube[2] / 2.0)
            curLabel_MANO = curLabel_MANO / (cube[2] / 2.0)

        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            curLabel_MANO = gt3Dcrop_MANO / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans
            synth_imgD = self.normalize_img(synth_crop.max(), synth_crop, center_xyz, cube_size)

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        synth_data = torch.from_numpy(synth_imgD).float()
        synth_data = synth_data.unsqueeze(0)

        joint_uvd = torch.from_numpy(joint_img).float()
        joint_xyz = torch.from_numpy(curLabel).float()
        joint_MANO = torch.from_numpy(curLabel_MANO).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        model_para = torch.from_numpy(model_para).float()
        return data, joint_xyz, joint_uvd, center, M, cube
        # return data, synth_data, joint_xyz, joint_uvd, center, M, cube, model_para

    def __len__(self):
        return len(self.all_joints_uvd)

class nyu_ST_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180],
                 img_size=128, cube_size=[250, 250, 250], center_type='refine', joint_num=14, loader=nyu_reader):
        super(nyu_ST_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        # np.random.seed(1)
        self.paras = (588.03, 587.07, 320., 240.)
        self.ori_img_size = (640, 480)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = 1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']
        self.aug_para = aug_para

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        center_path ='{}/center_{}_{}_refined.txt'.format(data_path, self.phase, 0)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path
        self.type = type # 'real','synth','render'

        NYU2MANO = [22, 15, 14, 13, 11, 10, 9, 3, 2, 1, 7, 6, 5, 19, 18, 17, 12, 8, 0, 4, 16]
        self.all_joints_uvd = self.labels['joint_uvd'][0][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][0][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz, flip=-1))
        self.all_model_para = self.read_modelPara(root_dir, 0)

        self.all_joints_xyz_MANO = self.labels['joint_xyz'][0][:, joint_select, :][:, NYU2MANO, :]
        self.all_joints_xyz_MANO = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz_MANO, flip=-1))

        self.refine_center_xyz = np.loadtxt(center_path)
        self.mask = np.zeros([len(self.all_joints_uvd)])

        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:,20,:]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True

    def __getitem__(self, index):
        img_path = self.data_path + '/depth_1_{:07d}.png'.format(index + 1)
        depth = self.loader(img_path)

        joint_xyz = self.all_joints_xyz[index].copy()
        model_para = self.all_model_para[index].copy()

        if self.phase == 'test':
            cube_size = self.test_cubesize[index]
        else:
            cube_size = self.cube_size

        center_xyz = self.center_xyz[index]
        center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz.reshape(1, 3)
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size,self.img_size), self.paras)

        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
                                                                       trans, mode, off, rot, sc, self.paras)
            curLabel = curLabel / (cube[2] / 2.0)

        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            curLabel_MANO = gt3Dcrop_MANO / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)
        joint_uvd = torch.from_numpy(joint_img).float()
        joint_xyz = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        model_para = torch.from_numpy(model_para).float()
        return data, joint_xyz, joint_uvd, center, M, cube

    def __len__(self):
        return len(self.all_joints_uvd)


class nyu_modelPara_loader(loader):
    def __init__(self, root_dir, phase, view=0, aug_para=[10, 0.1, 180], img_size=128,
                 cube_size=[250,250,250], center_type='refine', joint_num=14, loader=nyu_reader):
        super(nyu_modelPara_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        # np.random.seed(1)
        self.paras = (588.03, 587.07, 320., 240.)
        self.ori_img_size = (640, 480)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = 1

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc','none']#'rot','com','sc','none'
        self.aug_para = aug_para
        self.view = view

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        print('loading data...')
        self.labels = sio.loadmat(label_path)

        self.all_joints_uvd = self.labels['joint_uvd'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][self.view][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz, flip=-1))
        center_path ='{}/center_{}_refined.txt'.format(data_path, self.phase)
        print('loading data...')
        self.data_path = data_path

        self.refine_center_xyz = np.loadtxt(center_path)
        self.refine_center_xyz = np.loadtxt(center_path)
        self.refine_center_xyz = self.jointImgTo3D(self.joint3DToImg(self.refine_center_xyz, flip=-1))
        self.all_model_para = self.read_modelPara(root_dir, view)

        print('finish!!')
        if center_type =='refine':
            self.center_xyz = self.refine_center_xyz
        elif center_type =='joint':
            self.center_xyz = self.all_joints_xyz[:, 20, :]
        elif center_type =='joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3])*self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        if joint_num == 14:
            self.allJoints = False
        else:
            self.allJoints = True

    def __getitem__(self, index):
        model_para = self.all_model_para[index].copy()
        cube_size = self.cube_size
        center_xyz = self.center_xyz[index].copy()

        cube = torch.from_numpy(np.asarray(cube_size)).float()
        model_para = torch.from_numpy(model_para).float()

        return model_para, cube

    def __len__(self):
        return self.all_model_para.shape[0]


class hands_modelPara_loader(loader):
    def __init__(self, root_dir, phase, cube_size=[250, 250, 250], joint_num=21, file_name='posePara_lm_collosion'):
        super(hands_modelPara_loader, self).__init__(root_dir, phase, 128, 'refine', 'hands')
        self.paras = (475.065948, 475.065857, 315.944855, 245.287079)
        np.random.seed(1)

        # img para
        self.data_path = self.root_dir
        self.flip = 1
        self.cube_size = np.array(cube_size)

        # train para
        self.joint_num = joint_num
        print('loading data...')
        # , file_name = 'PosePara_1K'
        self.all_model_para = self.read_modelPara_simple(root_dir, file_name)
        print('finish!!')
        self.length = len(self.all_model_para)

        # self.sub_list = np.random.choice(np.arange(len(self.all_model_para)), 1000, replace=False)
        # self.save_modelPara_simple(root_dir, 'PosePara_1K', self.all_model_para[self.sub_list])

        # self.sub_list = np.random.choice(np.arange(len(self.all_model_para)), 10000, replace=False)
        # self.save_modelPara_simple(root_dir, 'PosePara_10K', self.all_model_para[self.sub_list])
        #
        # self.sub_list = np.random.choice(np.arange(len(self.all_model_para)), 100000, replace=False)
        # self.save_modelPara_simple(root_dir, 'PosePara_100K', self.all_model_para[self.sub_list])

    def __getitem__(self, index):
        model_para = self.all_model_para[index].copy()
        cube = torch.from_numpy(self.cube_size).float()
        model_para = torch.from_numpy(model_para).float()

        return model_para, cube

    def __len__(self):
        return self.length


class icvl_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], cube_size=[200,200,200],
                 img_size=128, joint_num=16, center_type='refine', loader=icvl_reader, full_img=False):
        super(icvl_loader, self).__init__(root_dir, phase, img_size, center_type, 'icvl')
        self.paras = (240.99, 240.96, 160.0, 120.0)
        self.cube_size = cube_size
        self.flip = 1
        self.joint_num = joint_num
        self.full_img = full_img
        self.ori_img_size = (320, 240)

        self.img_size = img_size
        self.data_path = self.root_dir
        self.loader = loader

        self.all_joints_uvd, self.all_centers_xyz, self.img_dirs = self.read_joints(self.root_dir, self.phase)
        # self.model_paras = self.read_modelPara_simple(self.data_path)
        self.length = len(self.all_joints_uvd)
        self.allJoints = False
        self.aug_modes = ['rot', 'com', 'sc', 'none']#,'com','sc','none'
        self.aug_para = aug_para
        self.centers_xyz = self.all_centers_xyz

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        if not os.path.exists(img_path):
            index = index + 1
            img_path = self.img_dirs[index]
        depth = self.loader(img_path)

        joint_uvd = self.all_joints_uvd[index].copy()
        joint_xyz = self.jointImgTo3D(joint_uvd)

        center_xyz = self.centers_xyz[index].copy()
        center_uvd = self.joint3DToImg(center_xyz)
        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size,(self.img_size,self.img_size),self.paras)

        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc,self.paras)
            curLabel = curLabel / (curCube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, self.cube_size)
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            curCube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans
            sc = 1

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)
        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        # get pcl
        # pcl = self.getpcl(imgD, com3D, curCube, M)
        # pcl_index = np.arange(pcl.shape[0])
        # pcl_num = pcl.shape[0]
        # if pcl_num == 0:
        #     pcl_sample = np.zeros([self.sample_num, 3])
        # else:
        #     if pcl_num < self.sample_num:
        #         tmp = math.floor(self.sample_num / pcl_num)
        #         index_temp = pcl_index.repeat(tmp)
        #         pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
        #     select = np.random.choice(pcl_index, self.sample_num, replace=False)
        #     pcl_sample = pcl[select, :]
        # pcl_sample = torch.from_numpy(pcl_sample).float()


        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()

        model_para = torch.zeros([self.joint_num])
        visible = torch.zeros([self.joint_num])
        outline = torch.zeros([self.joint_num])

        return data, data, joint, joint_img, model_para, center, M, cube, visible, outline

    def read_joints(self, data_rt,phase):
        if phase =='train':
            f = open(data_rt + "/train.txt", "r")
            f_center = open(data_rt+"/center_train_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
        else:
            f1 = open(data_rt+"/test_seq_1.txt", "r")
            f2 = open(data_rt + "/test_seq_2.txt", "r")
            f_center = open(data_rt + "/center_test_refined.txt")
            lines = f1.read().splitlines()+f2.read().splitlines()
            while '' in lines:
                lines.remove('')
            lines_center = f_center.readlines()
            f1.close()
            f2.close()

        centers_xyz = []
        joints_uvd = []

        img_names = []
        subSeq = ['0']
        for index, line in enumerate(lines):
            strs = line.split()
            p = strs[0].split('/')
            if not self.full_img:
                if ('0' in subSeq) and len(p[0]) > 6:
                    pass
                elif not ('0' in subSeq) and len(p[0]) > 6:
                    continue
                elif (p[0] in subSeq) and len(p[0]) <= 6:
                    pass
                elif not (p[0] in subSeq) and len(p[0]) <= 6:
                    continue

            img_path = data_rt + '/Depth/' + strs[0]
            if not os.path.isfile(img_path):
                continue

            joint_uvd = np.array(list(map(float, strs[1:]))).reshape(16, 3)
            strs_center = lines_center[index].split()

            if strs_center[0] == 'invalid':
                continue
            else:
                center_xyz = np.array(list(map(float, strs_center))).reshape(3)

            centers_xyz.append(center_xyz)
            joints_uvd.append(joint_uvd)
            img_names.append(img_path)

        f_center.close()
        return joints_uvd, centers_xyz, img_names

    def __len__(self):
        return self.length


class flip_icvl_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], cube_size=[200,200,200],
                 img_size=128, joint_num=16, center_type='refine', loader=icvl_reader, full_img=False):
        super(flip_icvl_loader, self).__init__(root_dir, phase, img_size, center_type, 'icvl')
        self.paras = (240.99, 240.96, 160.0, 120.0)
        self.cube_size = cube_size
        self.flip = 1
        self.joint_num = joint_num
        self.full_img = full_img
        self.ori_img_size = (320, 240)

        self.img_size = img_size
        self.data_path = self.root_dir
        self.loader = loader

        self.all_joints_uvd, self.all_centers_xyz, self.img_dirs = self.read_joints(self.root_dir, self.phase)
        self.length = len(self.all_joints_uvd)
        self.allJoints = False
        self.aug_modes = ['rot', 'com', 'sc', 'none']#,'com','sc','none'
        self.aug_para = aug_para
        self.centers_xyz = self.all_centers_xyz

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        if not os.path.exists(img_path):
            index = index + 1
            img_path = self.img_dirs[index]
        depth = self.loader(img_path)
        depth = depth[:, ::-1].copy()

        joint_uvd = self.all_joints_uvd[index].copy()
        joint_uvd[:, 0] = 320 - joint_uvd[:, 0]
        joint_xyz = self.jointImgTo3D(joint_uvd)

        center_xyz = self.centers_xyz[index].copy()
        center_uvd = self.joint3DToImg(center_xyz)
        center_uvd[0] = 320 - center_uvd[0]
        center_xyz = self.jointImgTo3D(center_uvd)

        if self.center_type == 'joint_mean':
            center_xyz = joint_xyz.mean(0)
            center_uvd = self.jointImgTo3D(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size,(self.img_size,self.img_size),self.paras)

        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
            imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc,self.paras)
            curLabel = curLabel / (curCube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(),depth_crop, center_xyz, self.cube_size)
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            curCube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans
            sc = 1

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size/2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)

        # get pcl
        pcl = self.getpcl(imgD, com3D, curCube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]
        pcl_sample = torch.from_numpy(pcl_sample).float()

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()

        model_para = torch.zeros([self.joint_num])
        visible = torch.zeros([self.joint_num])
        outline = torch.zeros([self.joint_num])

        return data, joint, joint_img, center, M, cube
        # return data, pcl_sample, joint, joint_img, model_para, center, M, cube, visible, outline

    def read_joints(self, data_rt,phase):
        if phase =='train':
            f = open(data_rt + "/train.txt", "r")
            f_center = open(data_rt+"/center_train_refined.txt")
            lines = f.readlines()
            lines_center = f_center.readlines()
            f.close()
        else:
            f1 = open(data_rt+"/test_seq_1.txt", "r")
            f2 = open(data_rt + "/test_seq_2.txt", "r")
            f_center = open(data_rt + "/center_test_refined.txt")
            lines = f1.read().splitlines()+f2.read().splitlines()
            while '' in lines:
                lines.remove('')
            lines_center = f_center.readlines()
            f1.close()
            f2.close()

        centers_xyz = []
        joints_uvd = []

        img_names = []
        subSeq = ['0']
        for index, line in enumerate(lines):
            strs = line.split()
            p = strs[0].split('/')
            if not self.full_img:
                if ('0' in subSeq) and len(p[0]) > 6:
                    pass
                elif not ('0' in subSeq) and len(p[0]) > 6:
                    continue
                elif (p[0] in subSeq) and len(p[0]) <= 6:
                    pass
                elif not (p[0] in subSeq) and len(p[0]) <= 6:
                    continue

            img_path = data_rt + '/Depth/' + strs[0]
            if not os.path.isfile(img_path):
                continue

            joint_uvd = np.array(list(map(float, strs[1:]))).reshape(16, 3)
            strs_center = lines_center[index].split()

            if strs_center[0] == 'invalid':
                continue
            else:
                center_xyz = np.array(list(map(float, strs_center))).reshape(3)

            centers_xyz.append(center_xyz)
            joints_uvd.append(joint_uvd)
            img_names.append(img_path)

        f_center.close()
        return joints_uvd, centers_xyz, img_names

    def __len__(self):
        return self.length


class msra_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.3, 180], img_size=128, joint_num=21, center_type='refine',
                 test_persons=[0], loader=msra_reader):
        super(msra_loader, self).__init__(root_dir, phase, img_size, center_type, 'msra')
        self.paras = (241.42, 241.42, 160, 120)
        self.cube_size = [200, 200, 200, 180, 180, 180, 170, 160, 150]
        self.centers_type = center_type
        self.aug_para = aug_para
        person_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        train_persons = list(set(person_list).difference(set(test_persons)))
        self.ori_img_size = (320, 240)
        self.flip = 1
        if phase == 'train':
            self.all_joints_xyz, self.all_joints_uvd, self.keys, _ = self.read_joints(root_dir, phase,
                                                                                   persons=train_persons)
            self.length = len(self.all_joints_xyz)
        else:
            self.all_joints_xyz, self.all_joints_uvd, self.keys, _ = self.read_joints(root_dir, phase,
                                                                                   persons=test_persons)
            self.length = len(self.all_joints_xyz)

        self.all_model_para = self.read_modelPara_simple(root_dir)
        self.img_num = np.array([8499, 8499, 8406, 8498, 8500, 8497, 8498, 8498, 8496])
        if self.phase == 'train':
            self.model_para_part1 = self.all_model_para[:self.img_num[:test_persons[0]].sum(), :]
            self.model_para_part2 = self.all_model_para[self.img_num[:(test_persons[0]+1)].sum():, :]
            self.model_para = np.concatenate((self.model_para_part1, self.model_para_part2), axis=0)
            # self.cluster_ids_part1 = self.all_cluster_ids[:self.img_num[:test_persons[0]].sum()]
            # self.cluster_ids_part2 = self.all_cluster_ids[self.img_num[:(test_persons[0]+1)].sum():]
            # self.cluster_ids = np.concatenate((self.cluster_ids_part1, self.cluster_ids_part2), axis=0)
        else:
            self.model_para = self.all_model_para[self.img_num[:(test_persons[0])].sum():self.img_num[:test_persons[0]+1].sum(), :]
            # self.cluster_ids = self.all_cluster_ids[self.img_num[:(test_persons[0])].sum():self.img_num[:test_persons[0]+1].sum()]

        file_uvd = open('./msra_label.txt', 'w')
        for index in range(len(self.all_joints_uvd)):
            np.savetxt(file_uvd, self.all_joints_uvd[index].reshape([1, joint_num * 3]), fmt='%.3f')
        if center_type == 'refine':
            file_name = self.root_dir + '/center_' + phase + '_' + str(test_persons[0]) + '_refined.txt'
            self.centers_xyz = self.jointImgTo3D(self.joint3DToImg(np.loadtxt(file_name), flip=-1))

        self.loader = loader
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']#

    def __getitem__(self, index):
        # index = index + 2008
        person = self.keys[index][0]
        name = self.keys[index][1]
        cube_size = [self.cube_size[person], self.cube_size[person], self.cube_size[person]]
        file = '%06d' % int(self.keys[index][2])

        depth, pcl = msra_reader(self.root_dir + "/P" + str(person) + "/" + str(name) + "/" + str(file) + "_depth.bin",
                                 self.paras)
        # synth_depth = synth_reader(self.root_dir + "/synth/" + str(person) + "/" + str(name) + "/" + str(file) + '.png')
        assert (depth.shape == (240, 320))

        joint_xyz = self.all_joints_xyz[index].copy()
        if self.centers_type == 'joint_mean':
            center_xyz = joint_xyz.mean(0)
        else:
            center_xyz = self.centers_xyz[index].copy()
        center_uvd = self.joint3DToImg(center_xyz)
        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size), self.paras)
        # synth_depth_crop, trans = self.Crop_Image_deep_pp(synth_depth, center_uvd, cube_size, (self.img_size, self.img_size), self.paras)
        #
        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1],
                                                   rot_range=self.aug_para[2])
            imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
                                                                       trans, mode, off, rot, sc, self.paras)
            # synth_imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(synth_depth_crop, gt3Dcrop, center_uvd, cube_size,
            #                                                            trans, mode, off, rot, sc, self.paras)
            curLabel = curLabel / (curCube[2] / 2.0)
        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
            # synth_imgD = self.normalize_img(synth_depth_crop.max(), synth_depth_crop, center_xyz, cube_size)
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            curCube = np.array(cube_size)
            com2D = center_uvd
            M = trans
            sc = 1

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)


        pcl = self.getpcl(imgD, center_xyz, curCube, M)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp,np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1],
                                                               replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)
        # synth_data = torch.from_numpy(synth_imgD).float()
        # synth_data = synth_data.unsqueeze(0)
        pcl_sample = torch.from_numpy(pcl_sample.transpose(1, 0)).float()
        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()

        model_para = self.model_para[index].copy()
        model_para = torch.from_numpy(model_para).float()
        # model_para = torch.ones([1])
        # cluster_id = torch.ones([1]).long()*self.cluster_ids[index]
        rotation = torch.zeros([3])
        if self.phase == 'train':
            if self.aug_modes[mode] == 'rot':
                rotation[2] = rot / 180 * np.pi
        return data, joint, joint_img, center, M, cube
        # return data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation

    # return joint_uvd
    def read_joints(self, data_rt, phase, persons=[0, 1, 2, 3, 4, 5, 6, 7],
                    poses=["1", "2", "3", "4", '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']):
        joints_xyz = []
        joints_uvd = []
        index = 0
        keys = {}
        persons_num = []
        file_record = open('./msra_record_list.txt', "w")
        for person in persons:
            person_num = 0
            for pose in poses:
                with open(data_rt + "/P" + str(person) + "/" + str(pose) + "/joint.txt") as f:
                    num_joints = int(f.readline())
                    for i in range(num_joints):
                        person_num+=1
                        file_record.write('P' + str(person) + "/" + str(pose) + '/' + '%06d' % int(i) + "_depth.bin" + '\r\n')
                        joint = np.fromstring(f.readline(), sep=' ')
                        joint_xyz = joint.reshape(21, 3)
                        # need to chaneg z to -
                        joint_xyz[:, 2] = -joint_xyz[:, 2]
                        # joint_uvd = self.joint3DToImg(joint_xyz)
                        joint_uvd = self.joint3DToImg(joint_xyz, flip=-1)
                        joint_xyz = self.jointImgTo3D(joint_uvd, flip=1)
                        # joint = joint.reshape(63)
                        joints_xyz.append(joint_xyz)
                        joints_uvd.append(joint_uvd)
                        keys[index] = [person, pose, i]
                        index += 1
            persons_num.append(person_num)
        file_record.close()
        return joints_xyz, joints_uvd, keys, np.array(persons_num)

    def __len__(self):
        return self.length


class msra_modelPara_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], img_size=128, joint_num=21, center_type='refine',
                 test_persons=[0]):
        super(msra_modelPara_loader, self).__init__(root_dir, phase, img_size, center_type, 'msra')
        self.paras = (241.42, 241.42, 160, 120)
        self.cube_size = [200, 200, 200, 180, 180, 180, 170, 160, 150]
        self.centers_type = center_type
        self.aug_para = aug_para
        person_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        train_persons = list(set(person_list).difference(set(test_persons)))
        self.ori_img_size = (320, 240)
        self.flip = 1
        if phase == 'train':
            self.all_joints_xyz, self.all_joints_uvd, self.keys, _ = self.read_joints(root_dir, phase,
                                                                                   persons=train_persons)
            self.length = len(self.all_joints_xyz)
        self.flip = 1
        # cluster_path = '{}/{}_globalCluster.txt'.format(self.root_dir, 'msra')
        # self.all_cluster_ids = np.loadtxt(cluster_path).astype(int)
        # self.all_model_para = self.read_modelPara_simple(root_dir)
        # self.img_num = np.array([8499, 8499, 8406, 8498, 8500, 8497, 8498, 8498, 8496])
        # if self.phase == 'train':
        #     self.model_para_part1 = self.all_model_para[:self.img_num[:test_persons[0]].sum(), :]
        #     self.model_para_part2 = self.all_model_para[self.img_num[:(test_persons[0]+1)].sum():, :]
        #     self.model_para = np.concatenate((self.model_para_part1, self.model_para_part2), axis=0)
            # self.cluster_ids_part1 = self.all_cluster_ids[:self.img_num[:test_persons[0]].sum()]
            # self.cluster_ids_part2 = self.all_cluster_ids[self.img_num[:(test_persons[0]+1)].sum():]
            # self.cluster_ids = np.concatenate((self.cluster_ids_part1, self.cluster_ids_part2), axis=0)
        # else:
        #     self.model_para = self.all_model_para[self.img_num[:(test_persons[0])].sum():self.img_num[:test_persons[0]+1].sum(), :]
            # self.cluster_ids = self.all_cluster_ids[self.img_num[:(test_persons[0])].sum():self.img_num[:test_persons[0]+1].sum()]
        if center_type == 'refine':
            file_name = self.root_dir + '/center_' + phase + '_' + str(test_persons[0]) + '_refined.txt'
            self.centers_xyz = self.jointImgTo3D(self.joint3DToImg(np.loadtxt(file_name),flip=-1))

    def __getitem__(self, index):
        person = self.keys[index][0]
        cube_size = [self.cube_size[person], self.cube_size[person], self.cube_size[person]]

        joint_xyz = self.all_joints_xyz[index].copy()
        model_para = self.model_para[index].copy()

        cube = torch.from_numpy(np.asarray(cube_size)).float()
        model_para = torch.from_numpy(model_para).float()

        return model_para, cube

    # return joint_uvd
    def read_joints(self, data_rt, phase, persons=[0, 1, 2, 3, 4, 5, 6, 7],
                    poses=["1", "2", "3", "4", '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']):
        joints_xyz = []
        joints_uvd = []
        index = 0
        keys = {}
        persons_num = []
        file_record = open('./msra_record_list.txt', "w")
        for person in persons:
            person_num = 0
            for pose in poses:
                with open(data_rt + "/P" + str(person) + "/" + str(pose) + "/joint.txt") as f:
                    num_joints = int(f.readline())
                    for i in range(num_joints):
                        person_num+=1
                        file_record.write('P' + str(person) + "/" + str(pose) + '/' + '%06d' % int(i) + "_depth.bin" + '\r\n')
                        joint = np.fromstring(f.readline(), sep=' ')
                        joint_xyz = joint.reshape(21, 3)
                        # need to chaneg z to -
                        joint_xyz[:, 2] = -joint_xyz[:, 2]
                        # joint_uvd = self.joint3DToImg(joint_xyz)
                        joint_uvd = self.joint3DToImg(joint_xyz, flip=-1)
                        joint_xyz = self.jointImgTo3D(joint_uvd, flip=1)
                        # joint = joint.reshape(63)
                        joints_xyz.append(joint_xyz)
                        joints_uvd.append(joint_uvd)
                        keys[index] = [person, pose, i]
                        index += 1
            persons_num.append(person_num)
        file_record.close()
        return joints_xyz, joints_uvd, keys, np.array(persons_num)

    def __len__(self):
        return self.length

# class msra_loader(loader):
#     def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], img_size=128, joint_num=21, center_type='refine',
#                  test_persons=[0], loader=msra_reader):
#         super(msra_loader, self).__init__(root_dir, phase, img_size, center_type, 'msra')
#         self.paras = (241.42, 241.42, 160, 120)
#         self.cube_size = [200, 200, 200, 180, 180, 180, 170, 160, 150]
#         self.centers_type = center_type
#         self.aug_para = aug_para
#         person_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#         train_persons = list(set(person_list).difference(set(test_persons)))
#         self.flip = -1
#         if phase == 'train':
#             self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir, phase,
#                                                                                    persons=train_persons)
#             self.length = len(self.all_joints_xyz)
#         else:
#             self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir, phase,
#                                                                                    persons=test_persons)
#             self.length = len(self.all_joints_xyz)
#         file_uvd = open('./msra_label.txt', 'w')
#         for index in range(len(self.all_joints_uvd)):
#             np.savetxt(file_uvd, self.all_joints_uvd[index].reshape([1, joint_num * 3]), fmt='%.3f')
#         if center_type == 'refine':
#             file_name = self.root_dir + '/center_' + phase + '_' + str(test_persons[0]) + '_refined.txt'
#             self.centers_xyz = np.loadtxt(file_name)
#
#         self.loader = loader
#         self.joint_num = joint_num
#         self.aug_modes = ['rot', 'com', 'sc', 'none']
#
#     def __getitem__(self, index):
#         person = self.keys[index][0]
#         name = self.keys[index][1]
#         cube_size = [self.cube_size[person], self.cube_size[person], self.cube_size[person]]
#         file = '%06d' % int(self.keys[index][2])
#
#         depth, pcl = msra_reader(self.root_dir + "/P" + str(person) + "/" + str(name) + "/" + str(file) + "_depth.bin",
#                                  self.paras)
#         assert (depth.shape == (240, 320))
#
#         joint_xyz = self.all_joints_xyz[index].copy()
#         center_xyz = self.centers_xyz[index].copy()
#         center_uvd = self.joint3DToImg(center_xyz)
#         gt3Dcrop = joint_xyz - center_xyz
#
#         depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size),self.paras)
#
#         if self.phase == 'train':
#             mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1],
#                                                    rot_range=self.aug_para[2])
#             imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
#                                                                        trans, mode, off, rot, sc,self.paras)
#             curLabel = curLabel / (curCube[2] / 2.0)
#         else:
#             imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
#             curLabel = gt3Dcrop / (cube_size[2] / 2.0)
#             curCube = np.array(cube_size)
#             com2D = center_uvd
#             M = trans
#             sc = 1
#
#         com3D = self.jointImgTo3D(com2D)
#         joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
#         joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
#         joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)
#         pcl = self.getpcl(imgD, center_xyz, curCube, M)
#         pcl_index = np.arange(pcl.shape[0])
#         pcl_num = pcl.shape[0]
#         if pcl_num == 0:
#             pcl_sample = np.zeros([self.sample_num, 3])
#         else:
#             if pcl_num < self.sample_num:
#                 tmp = math.floor(self.sample_num / pcl_num)
#                 index_temp = pcl_index.repeat(tmp)
#                 pcl_index = np.append(index_temp,
#                                       np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1],
#                                                        replace=False))
#             select = np.random.choice(pcl_index, self.sample_num, replace=False)
#             pcl_sample = pcl[select, :]
#
#         data = torch.from_numpy(imgD).float()
#         data = data.unsqueeze(0)
#
#         pcl_sample = torch.from_numpy(pcl_sample.transpose(1, 0)).float()
#         joint_img = torch.from_numpy(joint_img).float()
#         joint = torch.from_numpy(curLabel).float()
#         center = torch.from_numpy(com3D).float()
#         M = torch.from_numpy(M).float()
#         cube = torch.from_numpy(curCube).float()
#
#         so_node = torch.ones([self.joint_num,3])
#         visible = torch.zeros([self.joint_num])
#         outline = torch.zeros([self.joint_num])
#
#         return data, pcl_sample, joint, joint_img, so_node, center, M, cube,visible,outline
#
#     # return joint_uvd
#     def read_joints(self, data_rt, phase, persons=[0, 1, 2, 3, 4, 5, 6, 7],
#                     poses=["1", "2", "3", "4", '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']):
#         joints_xyz = []
#         joints_uvd = []
#         index = 0
#         keys = {}
#         file_record = open('./msra_record_list.txt', "w")
#         for person in persons:
#             for pose in poses:
#                 with open(data_rt + "/P" + str(person) + "/" + str(pose) + "/joint.txt") as f:
#                     num_joints = int(f.readline())
#                     for i in range(num_joints):
#                         file_record.write(
#                             'P' + str(person) + "/" + str(pose) + '/' + '%06d' % int(i) + "_depth.bin" + '\r\n')
#                         joint = np.fromstring(f.readline(), sep=' ')
#                         joint_xyz = joint.reshape(21, 3)
#                         # need to chaneg z to -
#                         joint_xyz[:, 2] = -joint_xyz[:, 2]
#                         joint_uvd = self.joint3DToImg(joint_xyz)
#                         # joint = joint.reshape(63)
#                         joints_xyz.append(joint_xyz)
#                         joints_uvd.append(joint_uvd)
#                         keys[index] = [person, pose, i]
#                         index += 1
#         file_record.close()
#         return joints_xyz, joints_uvd, keys
#
#     def __len__(self):
#         return self.length


# class msra_loader(loader):
#     def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], img_size=128, joint_num=21, center_type='refine',
#                  test_persons=[0], loader=msra_reader):
#         super(msra_loader, self).__init__(root_dir, phase, img_size, center_type, 'msra')
#         self.paras = (241.42, 241.42, 160, 120)
#         self.cube_size = [200, 200, 200, 180, 180, 180, 170, 160, 150]
#         self.centers_type = center_type
#         self.aug_para = aug_para
#         person_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#         train_persons = list(set(person_list).difference(set(test_persons)))
#         self.flip = -1
#         if phase == 'train':
#             self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir, phase, persons=train_persons)
#             self.length = len(self.all_joints_xyz)
#         else:
#             self.all_joints_xyz, self.all_joints_uvd, self.keys = self.read_joints(root_dir, phase, persons=test_persons)
#             self.length = len(self.all_joints_xyz)
#         file_uvd = open('./msra_label.txt', 'w')
#         for index in range(len(self.all_joints_uvd)):
#             np.savetxt(file_uvd, self.all_joints_uvd[index].reshape([1, joint_num * 3]), fmt='%.3f')
#         if center_type == 'refine':
#             file_name = self.root_dir + '/center_' + phase + '_' + str(test_persons[0]) + '_refined.txt'
#             self.centers_xyz = np.loadtxt(file_name)
#
#         self.loader = loader
#         self.joint_num = joint_num
#         self.aug_modes = ['rot', 'com', 'sc', 'none']
#
#     def __getitem__(self, index):
#         person = self.keys[index][0]
#         name = self.keys[index][1]
#         cube_size = [self.cube_size[person], self.cube_size[person], self.cube_size[person]]
#         file = '%06d' % int(self.keys[index][2])
#
#         depth, pcl = msra_reader(self.root_dir + "/P" + str(person) + "/" + str(name) + "/" + str(file) + "_depth.bin",
#                                  self.paras)
#         assert (depth.shape == (240, 320))
#
#         joint_xyz = self.all_joints_xyz[index].copy()
#
#         if self.center_type == 'refine':
#             center_xyz = self.centers_xyz[index].copy()
#             center_uvd = self.joint3DToImg(center_xyz)
#         elif self.center_type == 'joint_mean':
#             center_xyz = joint_xyz.mean(0)
#             center_uvd = self.joint3DToImg(center_xyz)
#
#         gt3Dcrop = joint_xyz - center_xyz
#
#         depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size),self.paras)
#
#         if self.phase == 'train':
#             mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1],
#                                                    rot_range=self.aug_para[2])
#             imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size,
#                                                                        trans, mode, off, rot, sc,self.paras)
#             curLabel = curLabel / (curCube[2] / 2.0)
#         else:
#             imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
#             curLabel = gt3Dcrop / (cube_size[2] / 2.0)
#             curCube = np.array(cube_size)
#             com2D = center_uvd
#             M = trans
#             sc = 1
#
#         com3D = self.jointImgTo3D(com2D)
#         joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
#         joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
#         joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)
#         data = torch.from_numpy(imgD).float()
#         data = data.unsqueeze(0)
#
#         # get pcl
#         pcl = self.getpcl(imgD, com3D, curCube, M)
#         pcl_index = np.arange(pcl.shape[0])
#         pcl_num = pcl.shape[0]
#         if pcl_num == 0:
#             pcl_sample = np.zeros([self.sample_num, 3])
#         else:
#             if pcl_num < self.sample_num:
#                 tmp = math.floor(self.sample_num / pcl_num)
#                 index_temp = pcl_index.repeat(tmp)
#                 pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1], replace=False))
#             select = np.random.choice(pcl_index, self.sample_num, replace=False)
#             pcl_sample = pcl[select, :]
#
#         pcl_sample = torch.from_numpy(pcl_sample).float()
#         joint_img = torch.from_numpy(joint_img).float()
#         joint = torch.from_numpy(curLabel).float()
#         center = torch.from_numpy(com3D).float()
#         M = torch.from_numpy(M).float()
#         cube = torch.from_numpy(curCube).float()
#         so_node = torch.ones([self.joint_num,3])
#         visible = torch.zeros([self.joint_num])
#         outline = torch.zeros([self.joint_num])
#
#         return data, pcl_sample, joint, joint_img, so_node, center, M, cube,visible,outline
#
#
#     # return joint_uvd
#     def read_joints(self, data_rt, phase, persons=[0, 1, 2, 3, 4, 5, 6, 7],
#                     poses=["1", "2", "3", "4", '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']):
#         joints_xyz = []
#         joints_uvd = []
#         index = 0
#         keys = {}
#         file_record = open('./msra_record_list.txt', "w")
#         for person in persons:
#             for pose in poses:
#                 with open(data_rt + "/P" + str(person) + "/" + str(pose) + "/joint.txt") as f:
#                     num_joints = int(f.readline())
#                     for i in range(num_joints):
#                         file_record.write(
#                             'P' + str(person) + "/" + str(pose) + '/' + '%06d' % int(i) + "_depth.bin" + '\r\n')
#                         joint = np.fromstring(f.readline(), sep=' ')
#                         joint_xyz = joint.reshape(21, 3)
#                         # need to chaneg z to -
#                         joint_xyz[:, 2] = -joint_xyz[:, 2]
#                         joint_uvd = self.joint3DToImg(joint_xyz)
#                         # joint = joint.reshape(63)
#                         joints_xyz.append(joint_xyz)
#                         joints_uvd.append(joint_uvd)
#                         keys[index] = [person, pose, i]
#                         index += 1
#         file_record.close()
#         return joints_xyz, joints_uvd, keys
#
#     def __len__(self):
#         return self.length


class DHG_loader(loader):
    def __init__(self, root_dir, aug_para=[10, 0.1, 180], img_size=128, cube_size=[250, 250, 250]):
        super(DHG_loader, self).__init__(root_dir, 'train', img_size, 'joint_mean', 'shrec')
        self.paras = (463.889, 463.889, 320.00000, 240.00000)
        self.ori_img_size = (640, 480)

        # img para
        self.root = root_dir
        self.cube_size = cube_size
        self.img_size = img_size
        self.aug_para = aug_para
        self.flip = 1

        print('loading data...')
        self.img, self.all_joints_xyz, self.all_joints_uvd, self.all_center_uvd = self.read_joints(self.root)
        self.all_center_uvd[:, 0] = 640 - self.all_center_uvd[:, 0]
        self.length = len(self.all_joints_xyz)
        print('finish!!')
        print(self.length)
        self.aug_modes = ['none', 'com', 'sc',  'rot']  # ,'com','sc','none''rot','com',

    def __getitem__(self, index):
        depth = shrec_reader(self.img[index])
        depth = depth[:, ::-1].copy()
        assert (depth.shape == (480, 640))

        center_uvd = self.all_center_uvd[index].copy()
        # center_uvd[0] = 640 - center_uvd[0]
        center_xyz = self.jointImgTo3D(center_uvd)
        cube_size = self.cube_size

        joint_xyz = self.all_joints_xyz[index].copy().reshape([-1, 3])
        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size), self.paras)

        mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
        imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size, trans, mode, off, rot, sc, self.paras)
        curLabel = curLabel / (curCube[2] / 2.0)
        com3D = self.jointImgTo3D(com2D)

        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint = torch.from_numpy(curLabel).float()
        joint_img = torch.from_numpy(joint_img).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()
        visible = torch.ones([1]).long()
        rotation = torch.zeros([3])

        return data, data, joint, joint_img, joint, center, M, cube, visible, rotation

    # return joint_uvd
    def read_joints(self, data_rt):
        img_list = []
        joint_xyz_list = []
        joint_uvd_list = []
        center_uvd_list = []
        r = re.compile('[ \t\n\r]+')
        prefix = data_rt + "/gesture_{}/finger_{}/subject_{}/essai_{}/"
        input_list = open(data_rt + "/informations_troncage_sequences.txt").readlines()
        for idx, line in enumerate(input_list):
            # Loading dataset
            splitLine = r.split(line)
            dir_path = prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3])
            joint_xyz_gt = np.loadtxt(dir_path + 'skeletons_world_pred_normal_DSSF.txt')
            joint_uvd_gt = np.loadtxt(dir_path + 'skeleton_image.txt')
            start_id = int(splitLine[4])
            end_id = int(splitLine[5])
            for frame_id in range(start_id, end_id+1):
            # for frame_id in range(joint_xyz_gt.shape[0]):
                img_list.append(dir_path+'depth_%d.png'%(frame_id+ 1))
                joint_xyz_list.append(joint_xyz_gt[frame_id-start_id])
                # joint_xyz_list.append(joint_xyz_gt[frame_id])
                joint_uvd_list.append(joint_uvd_gt[frame_id])
                center_uv = joint_uvd_gt[frame_id].reshape(-1, 2).mean(0)
                # center_d = joint_xyz_gt[frame_id].reshape(-1, 3).mean(0)[2:3]*1000
                center_d = joint_xyz_gt[frame_id-start_id].reshape(-1, 3).mean(0)[2:3]*1000
                center_uvd = np.concatenate((center_uv, center_d))
                center_uvd_list.append(center_uvd)
        return img_list, np.array(joint_xyz_list), np.array(joint_uvd_list), np.array(center_uvd_list)

    # return joint_uvd
    def write_joints(self, data_rt, predict_xyz):
        r = re.compile('[ \t\n\r]+')
        prefix = data_rt + "/gesture_{}/finger_{}/subject_{}/essai_{}/"
        input_list = open(data_rt + "/informations_troncage_sequences.txt").readlines()

        all_frame_idx = 0

        for idx, line in enumerate(input_list):
            # Loading dataset
            splitLine = r.split(line)
            dir_path = prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3])
            joint_xyz_gt = np.loadtxt(dir_path + 'skeleton_world.txt')
            start_id =  splitLine[4]
            end_id = splitLine[5]
            frame_len = int(end_id) - int(start_id) + 1
            np.savetxt(dir_path + 'skeletons_world_pred_normal_AWR_50.txt', predict_xyz[all_frame_idx:all_frame_idx+frame_len], fmt='%.8f')
            all_frame_idx = all_frame_idx + frame_len
        print(all_frame_idx)
        return 0


    def __len__(self):
        return self.length


class shrec_loader(loader):
    def __init__(self, root_dir, aug_para=[10, 0.1, 180], img_size=128, cube_size=[250, 250, 250]):
        super(shrec_loader, self).__init__(root_dir, 'train', img_size, 'joint_mean', 'shrec')
        self.paras = (463.889, 463.889, 320.00000, 240.00000)
        self.ori_img_size = (640, 480)

        # img para
        self.root = root_dir
        self.cube_size = cube_size
        self.img_size = img_size
        self.aug_para = aug_para
        self.flip = 1

        print('loading data...')
        self.img, self.all_joints_xyz, self.all_joints_uvd, self.all_center_uvd = self.read_joints(self.root)
        self.all_center_uvd[:, 0] = 640 - self.all_center_uvd[:, 0]
        self.length = len(self.all_joints_xyz)
        print('finish!!')
        print(self.length)
        self.aug_modes = ['none', 'com', 'sc',  'rot']  # ,'com','sc','none''rot','com',

    def __getitem__(self, index):
        depth = shrec_reader(self.img[index])
        depth = depth[:, ::-1].copy()
        assert (depth.shape == (480, 640))

        center_uvd = self.all_center_uvd[index].copy()
        # center_uvd[0] = 640 - center_uvd[0]
        center_xyz = self.jointImgTo3D(center_uvd)
        cube_size = self.cube_size

        joint_xyz = self.all_joints_xyz[index].copy().reshape([-1, 3])
        gt3Dcrop = joint_xyz - center_xyz

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size), self.paras)

        mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1], rot_range=self.aug_para[2])
        imgD, _, curLabel, curCube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, cube_size, trans, mode, off, rot, sc, self.paras)
        curLabel = curLabel / (curCube[2] / 2.0)
        com3D = self.jointImgTo3D(com2D)

        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (curCube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (curCube[0] / 2.0)


        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        joint = torch.from_numpy(curLabel).float()
        joint_img = torch.from_numpy(joint_img).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()
        visible = torch.ones([1]).long()
        rotation = torch.zeros([3])

        return data, data, joint, joint_img, joint, center, M, cube, visible, rotation

    # return joint_uvd
    def read_joints(self, data_rt):
        img_list = []
        joint_xyz_list = []
        joint_uvd_list = []
        center_uvd_list = []
        r = re.compile('[ \t\n\r]+')
        prefix = data_rt + "/gesture_{}/finger_{}/subject_{}/essai_{}/"
        train_list = open(data_rt + "/train_gestures.txt").readlines()
        test_list = open(data_rt + "/test_gestures.txt").readlines()
        input_list = train_list + test_list
        for idx, line in enumerate(input_list):
            # Loading dataset
            splitLine = r.split(line)
            dir_path = prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3])
            joint_xyz_gt = np.loadtxt(dir_path + 'skeletons_world_pred.txt')
            joint_uvd_gt = np.loadtxt(dir_path + 'skeletons_image.txt')
            for frame_id in range(int(splitLine[-2])):
                img_list.append(dir_path+str(frame_id)+'_depth.png')
                joint_xyz_list.append(joint_xyz_gt[frame_id])
                joint_uvd_list.append(joint_uvd_gt[frame_id])
                center_uv = joint_uvd_gt[frame_id].reshape(-1, 2).mean(0)
                # center_d = joint_xyz_gt[frame_id].reshape(-1, 3).mean(0)[2:3]*1000
                center_d = joint_xyz_gt[frame_id].reshape(-1, 3).mean(0)[2:3]
                center_uvd = np.concatenate((center_uv, center_d))
                center_uvd_list.append(center_uvd)
        return img_list, np.array(joint_xyz_list), np.array(joint_uvd_list), np.array(center_uvd_list)

    # return joint_uvd
    def write_joints(self, data_rt, predict_xyz):
        r = re.compile('[ \t\n\r]+')
        prefix = data_rt + "/gesture_{}/finger_{}/subject_{}/essai_{}/"
        train_list = open(data_rt + "/train_gestures.txt").readlines()
        test_list = open(data_rt + "/test_gestures.txt").readlines()
        input_list = train_list + test_list
        all_frame_idx = 0

        for idx, line in enumerate(input_list):
            # Loading dataset
            splitLine = r.split(line)
            dir_path = prefix.format(splitLine[0], splitLine[1], splitLine[2], splitLine[3])
            frame_len = int(splitLine[-2])
            np.savetxt(dir_path + 'skeletons_world_pred_normal_SSR.txt', predict_xyz[all_frame_idx:all_frame_idx+frame_len], fmt='%.8f')
            all_frame_idx = all_frame_idx + frame_len
        return 0


    def __len__(self):
        return self.length


def xyz2error(output, joint, center, cube_size):
    output = output.detach().cpu().numpy()
    joint = joint.detach().cpu().numpy()
    center = center.detach().cpu().numpy()
    cube_size = cube_size.detach().cpu().numpy()
    batchsize, joint_num, _ = output.shape
    center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
    cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

    joint_xyz = output * cube_size / 2 + center
    joint_world_select = joint * cube_size / 2 + center

    errors = (joint_xyz - joint_world_select) * (joint_xyz - joint_world_select)
    errors = np.sqrt(np.sum(errors, axis=2))
    return errors


def render_msra():
    root = '/home/pfren/dataset/msra'
    batch_size = 128
    mano_model_path = '/home/pfren/pycharm/hwt-mini-IK/model_files'
    ImgRender = Render(mano_model_path, 'msra', (241.42, 241.42, 160, 120), (320, 240)).cuda()
    if not os.path.exists(root + '/synth/'):
        os.makedirs(root + '/synth/')

    for person_id in range(1, 9):
        dataset = msra_loader(root, 'test', center_type='refine', test_persons=[person_id])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        if not os.path.exists(root + '/synth/' + str(person_id) + '/'):
            os.makedirs(root + '/synth/' + str(person_id) + '/')
        for index, data in enumerate(dataloader):
            img, pcl, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, outline = data
            with torch.no_grad():
                ori_imgs = ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), M.cuda(),)
            ori_imgs = ori_imgs.cpu().numpy().astype(np.uint16)
            for batch_index in range(img.size(0)):
                img_index = batch_index + index * batch_size
                name = dataset.keys[img_index][1]
                img_dir = root + '/synth/' + str(person_id) + '/'+name+'/'
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                file = '%06d' % int(dataset.keys[img_index][2])
                cv2.imwrite(img_dir + str(file)+'.png', ori_imgs[batch_index, 0])
        print('finish:'+str(person_id))


def render_nyu():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 128
    view = 2
    phase = 'test'
    mano_model_path = '/home/pfren/pycharm/hand_mixed/MANO/'
    ImgRender = Render(mano_model_path, 'nyu', (588.03, 587.07, 320., 240.), (640, 480)).cuda()
    if not os.path.exists(root + '/render/'+phase+'/'):
        os.makedirs(root + '/render/'+phase+'/')
    img_dir = root + '/render/'+phase+'/'
    dataset = nyu_loader(root, 'test', type='real',percent=1, view=view, center_type='refine', aug_para=[0, 0, 0])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    for index, data in enumerate(dataloader):
        img, pcl, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, outline = data
        with torch.no_grad():
            ori_imgs = ImgRender(model_para.cuda(), center.cuda(), cube.cuda())
        ori_imgs = ori_imgs.cpu().numpy().astype(np.uint16)
        for batch_index in range(img.size(0)):
            img_index = batch_index + index * batch_size
            cv2.imwrite(img_dir + str(view+1) +'_'+str(img_index).zfill(7) + '.png', ori_imgs[batch_index, 0])
        print(index)


def display_point(path, pcl, keypoints=None, spheres_c=None, spheres_r=None, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY
    if transpose:
        pcl = pcl[:, [0, 2, 1]]

    for i in range(pcl.shape[0]):
        ax.scatter(pcl[i, 0], pcl[i, 1], pcl[i, 2], color='gray', alpha=0.5, marker=".")

    if keypoints is not None:
        display_keypoints('', keypoints=keypoints, ax=ax, transpose=transpose)
    if spheres_c is not None:
        display_sphere(spheres_c, spheres_r, ax=ax)

    cam_equal_aspect_3d(ax, pcl, transpose=transpose)

    plt.xticks([])
    plt.yticks([])
    plt.axis('on')
    # plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close()


def display_mesh(path, verts, faces,keypoints=None, spheres_c=None, spheres_r=None, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY
    if transpose:
        verts = verts[:, [0, 2, 1]]

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    # face_color = (141 / 255, 184 / 255, 226 / 255)
    # edge_color = (50 / 255, 50 / 255, 50 / 255)
    # face_color = 'slategrey'
    # edge_color = 'slategrey'
    face_color = 'grey'
    edge_color = 'grey'
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor(edge_color)
    # skin_color = (238/255, 175/255, 147/255)
    # mesh.set_facecolor(skin_color)
    # mesh.set_edgecolor(skin_color)

    ax.add_collection3d(mesh)

    if keypoints is not None:
        display_keypoints('', keypoints=keypoints, ax=ax, transpose=transpose)
    if spheres_c is not None:
        display_sphere(spheres_c, spheres_r, ax=ax)
    cam_equal_aspect_3d(ax, verts, transpose=transpose)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=900)
    plt.close()


def display_keypoints(path, keypoints=None, ax=None, transpose=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    COLOR = ['#330000', '#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
    if transpose:
        keypoints = keypoints[:, [0, 2, 1]]
    for i in range(keypoints.shape[0]):
        ax.scatter(keypoints[i, 0], keypoints[i, 1], keypoints[i, 2], color='red')

    cam_equal_aspect_3d(ax, keypoints, transpose=transpose)

    if path:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


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
    # ax.view_init(5, -85)

sphere_color = ['orangered']*21 + ['mediumblue']*9 + ['lawngreen']*9 + ['gold']*9+['teal']*9 +['blueviolet']*9
def display_sphere(centers, radiuss, ax=None, transpose=True):
    num = centers.shape[0]
    if transpose:
        centers = centers[:, [0, 2, 1]]
    for index in range(num):
        center = centers[index]
        radius = radiuss[index]
        t = np.linspace(0, np.pi * 2, 20)
        s = np.linspace(0, np.pi, 20)

        t, s = np.meshgrid(t, s)
        x = np.cos(t) * np.sin(s)
        y = np.sin(t) * np.sin(s)
        z = np.cos(s)
        ax.plot_surface(x*radius + center[0], y*radius + center[1], z*radius + center[2], rstride=1, cstride=1, color=sphere_color[index])


pcl_color = ['royalblue', 'mediumblue', 'navy', 'springgreen','green','darkgreen', 'khaki','gold','orange','lightcoral','indianred','brown','lightgrey','darkgrey','k']
pcl_color =np.array([
[0, 0, 0],
[0, 255, 0],[0, 205, 0],[0, 155, 0],
[0, 0, 255],[0, 0, 205],[0, 0, 155],
[0, 104, 139],[0, 154, 205],[0, 178, 238],
[255, 255, 0],[205, 205, 0], [155, 155, 0],
[105, 0, 0],[155, 0, 0],[205, 0, 0],
])/255


def display_pcl_segment(path, verts, pcl_segment, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY
    if transpose:
        verts = verts[:, [0, 2, 1]]

    # for i in range(verts.shape[0]):
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=pcl_color[pcl_segment],s=0.8)

    cam_equal_aspect_3d(ax, verts, transpose=transpose)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


from mpl_toolkits.mplot3d.art3d import Line3DCollection


def display_pcl_corr(path, pcl, shpere_c, shpere_r,min_index, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pcl_corr = (pcl - shpere_c[min_index]) / np.sqrt(np.sum((pcl - shpere_c[min_index]) ** 2, axis=-1) + 1e-8)[:, np.newaxis] \
               * shpere_r[min_index][:, np.newaxis] +  shpere_c[min_index]

    # XYZ -> XZY
    if transpose:
        pcl = pcl[:, [0, 2, 1]]
        pcl_corr = pcl_corr[:, [0, 2, 1]]

    ax.plot(pcl[:, 0], pcl[:, 1], pcl[:, 2], '.r', markersize=1)
    # ax.plot(pcl_corr[:, 0], pcl_corr[:, 1], pcl_corr[:, 2], '.y', markersize=1)
    ls = np.hstack([pcl, pcl_corr]).copy()
    ls = ls.reshape((-1, 2, 3))
    lc = Line3DCollection(ls, linewidths=0.5, colors='b')
    ax.add_collection(lc)

    cam_equal_aspect_3d(ax, pcl, transpose=transpose)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


def save_label():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    dataset = nyu_loader(root, 'test', type='synth', percent=1, view=2, center_type='joint_mean', aug_para=[0, 0, 0])
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    # synthetic_dataset = icvl_loader(root, 'train', center_type='joint_mean', aug_para=[0, 0, 0])
    # dataset = msra_loader(root, 'test', test_persons=[0], img_size=128, center_type='joint_mean')
    # dataset = hands_modelPara_loader(root, 'train')
    print(dataset.__len__())
    label_file = open('1.txt', 'w')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    for index, data in enumerate(dataloader):
        # model_para, cube = data
        img, synth_img, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, rot = data
        noraml_joint, noraml_joint_xyz, normal_verts_xyz, center3d, cube_size, M = ImgRender.getJoint(model_para.cuda(), center.cuda(), cube.cuda())
        joint_xyz = (noraml_joint_xyz)/2*cube_size.unsqueeze(1) + center3d.unsqueeze(1)
        joint_uvd = dataset.joint3DToImg(joint_xyz.cpu().numpy())
        np.savetxt(label_file, joint_uvd.reshape([-1, 12 * 3]), fmt='%.3f')

    print('done')




def vis_multiView_dataset():
    from render_model.mano_layer_0 import RotationNormalPoints
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    dataset = nyu_multiView_loader(root, 'train', aug_para=[0, 0, 0])
    print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    for index, data in enumerate(dataloader):
        # model_para, cube = data
        img_0, j3d_xyz_0, j3d_uvd_0, center_0, M_0, cube_0, \
        img_1, j3d_xyz_1, j3d_uvd_1, center_1, M_1, cube_1,\
        img_2, j3d_xyz_2, j3d_uvd_2, center_2, M_2, cube_2 = data
        crop_hand = dataset.crop_hand(img_0, j3d_xyz_0, center_0, M_0, cube_0 )
        # rot1 = torch.Tensor([[-0.125, 0, 0]]).repeat([batch_size, 1]) * np.pi * 2
        # j3d_xyz_1 = RotationNormalPoints(j3d_xyz_1, rot1)
        # j3d_uvd_1 = dataset.xyz_nl2uvdnl_tensor(j3d_xyz_1, center_1, M_1, cube_1)
        #
        # rot2 = torch.Tensor([[0.125, 0, 0]]).repeat([batch_size, 1]) * np.pi * 2
        # j3d_xyz_2 = RotationNormalPoints(j3d_xyz_2, rot2)
        # j3d_uvd_2 = dataset.xyz_nl2uvdnl_tensor(j3d_xyz_2, center_2, M_2, cube_2)
        # vis_tool.debug_2d_img(noraml_img, index,'./debug', 'renderImg')
        # vis_tool.debug_2d_img(outer_img.float(), index, './debug', 'outer_img')
        # vis_tool.debug_2d_img(inner_img.float(), index, './debug', 'inner_img')
        # vis_tool.debug_2d_img(img_select.float(), index, './debug', 'heatmap_img')
        # vis_tool.debug_2d_img(transferImg, index, './debug', 'transferImg-ori')
        # vis_tool.debug_2d_img(transferImg_my, index, './debug', 'transferImg-consis')
        # vis_tool.debug_2d_img(img, index, './debug', 'oirImg')
        vis_tool.debug_2d_pose(img_0, j3d_uvd_0, index, 'nyu', './debug', 'joint0', batch_size)
        vis_tool.debug_2d_pose(crop_hand, j3d_uvd_0, index, 'nyu', './debug', 'joint_crop', batch_size)
        # vis_tool.debug_2d_pose(img_1, j3d_uvd_1, index, 'nyu', './debug', 'joint1', batch_size)
        # vis_tool.debug_2d_pose(img_2, j3d_uvd_2, index, 'nyu', './debug', 'joint2', batch_size)
        # vis_tool.debug_2d_pose(img, j3d_uvd, index, 'msra', './debug', 'render_img', batch_size)

        # vis_tool.debug_2d_heatmap(img, torch.abs(img - transferImg_my), index, './debug', 'heatmap-consis')
        # vis_tool.debug_2d_heatmap(img, torch.abs(img - transferImg), index, './debug', 'heatmap-ori')
        print(index)
        if index == 2:
            break

    print('done')


def heatmap2mask(heatmap, threshold=0.4):
    with torch.no_grad():
        batch_size = heatmap.size(0)
        mask = torch.sum(heatmap, dim=1, keepdim=True)
        mask = F.interpolate(mask, size=[128, 128], mode='bilinear')
        mask = mask.view(batch_size, -1)
        mask = mask - mask.min(dim=1, keepdim=True)[0]
        mask = mask / (mask.max(dim=1, keepdim=True)[0] + 1e-8)
        mask = mask.gt(threshold).float().view(batch_size, 1, 128, 128)
    return mask


class outer_mask(torch.nn.Module):
    def __init__(self):
        super(outer_mask, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=False)
        ones = torch.ones_like(self.conv.weight)
        self.conv.weight = torch.nn.Parameter(ones)
        self.conv.weight.requires_grad = False

    def forward(self, real):
        mask = real.lt(0.99).float()
        with torch.no_grad():
            mask_dilate = ~(self.conv(mask).gt(0))
        return mask_dilate

def calculate_coll():
    root = '/home/pfren/dataset/hand/nyu'
    dataset = nyu_loader(root, 'test', type='synth', percent=1, view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    print(dataset.__len__())
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    mano_para_coll = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune_MANO_18_finetune-allLoss(10coll)/MANO_result_1_0.txt')
    mano_para_nocoll = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune_MANO_18_withoutColl/MANO_result_1_0.txt')
    mano_para_coll = torch.from_numpy(mano_para_coll).cuda().float()
    mano_para_nocoll = torch.from_numpy(mano_para_nocoll).cuda().float()
    batch_size = mano_para_coll.size(0)
    noraml_joint, noraml_joint_xyz, normal_verts_xyz, center3d, cube_size, M = \
        ImgRender.getJoint(mano_para_coll, None, torch.ones([batch_size,3]).cuda()*250, all_joints=True)
    noraml_joint_nocoll, noraml_joint_xyz_nocoll, normal_verts_xyz_nocoll, center3d_nocoll, cube_size_nocoll, M_nocoll = \
        ImgRender.getJoint(mano_para_nocoll.cuda(), None, torch.ones([batch_size,3]).cuda()*250, all_joints=True)
    coll_nocoll = ImgRender.mano_layer.calculate_coll(noraml_joint_xyz_nocoll, normal_verts_xyz_nocoll)
    coll = ImgRender.mano_layer.calculate_coll(noraml_joint_xyz, normal_verts_xyz)
    diff = coll - coll_nocoll
    v, i = torch.topk(diff, k=100, dim=-1,largest=False)
    print('finish')

def plot_pointcloud(points, title=""):
    # Sample points uniformly from the surface of the mesh.
    x, y, z = points[0].clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    # plt.show()
    plt.savefig(title+'.png')

MSRA_MANO = [
    0,
    1,2,3,
    5,6,7,
    13,14,15,
    9,10,11,
    17,18,19,
    4,8,16,12,20

]


def opt_depth():
    from torch.optim import Adam, SGD
    root = '/home/pfren/dataset/msra'
    batch_size = 32
    synthetic_dataset = msra_loader(root, 'test', test_persons=[0],
                                           img_size=128, center_type='refine')
    print(synthetic_dataset.__len__())
    mano_model_path = '/home/pfren/pycharm/hwt-mini-IK/model_files'
    # mano_model_path = '/data/users/pfren/pycharm/mano_v1_2/models'
    ImgRender = Render(mano_model_path, 'msra', (241.42, 241.42, 160, 120), (320, 240)).cuda()
    dataloader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    lr = 0.01
    iter_num=1000
    L1Loss = torch.nn.SmoothL1Loss().cuda()
    for index, data in enumerate(dataloader):
        img, synth_img, pcl, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, rot = data

        quat, cam = torch.zeros(batch_size, 3, requires_grad=True, device="cuda"), \
                    torch.zeros(batch_size, 4, requires_grad=True, device="cuda")
        theta, beta = torch.zeros(batch_size, 45, requires_grad=True, device="cuda"), \
                      torch.zeros(batch_size, 10, requires_grad=True, device="cuda")
        # quat.data.fill_(model_para[:, :3].data)
        model_para = model_para
        # quat = torch.tensor(model_para[:, :3], requires_grad=True).cuda()
        # theta = torch.tensor(model_para[:, 3:3+45], requires_grad=True).cuda()
        # cam = torch.tensor(model_para[:, 48:], requires_grad=True).cuda()
        with torch.no_grad():
            quat = model_para[:, :3].detach().clone()
            theta = model_para[:, 3:3+45].detach().clone()
            cam = model_para[:, 48:].detach().clone()
        optimList = [{"params": theta, "initial_lr": lr}, {"params": beta, "initial_lr": lr},
                     {"params": quat, "initial_lr": lr}, {"params": cam, "initial_lr": lr}]
        optimizer = Adam(optimList, lr=lr)
        j3d_xyz = j3d_xyz * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
        j3d_xyz = j3d_xyz.cuda()
        pcl = pcl.permute(0, 2, 1) * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
        pcl = pcl.cuda()
        for iter_index in range(iter_num):
            optimizer.zero_grad()
            # hand_verts, hand_joints = mano(theta)
            hand_verts, hand_joints = ImgRender.mano_layer.get_mano_vertices(quat, theta, beta, cam)
            pcl_offset = torch.min(torch.sum(torch.pow(pcl.unsqueeze(2) - hand_verts.unsqueeze(1),2), dim=-1),dim=-1)[0]

            # print(hand_joints)
            # 100,0.05
            # + lapla(hand_verts)*0.0001
            loss = L1Loss(hand_joints, j3d_xyz) + torch.pow(beta, 2).mean() + torch.pow(theta, 2).mean()+pcl_offset.mean()
            # loss = mse(hand_joints, j3d_xyz)
            if iter_index % 100 == 0:
                error = torch.pow(hand_joints-j3d_xyz, 2).sum(-1).sqrt().mean()
                print(error)
            loss.backward()
            optimizer.step()


        print(index)
        if index == 1:
            break

    print('done')


def mix_dataset():
    from render_model.mano_layer import RotationPoints
    from CPT.util import WeightedFeatureExtract
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    dataset = nyu_loader(root, 'test', view=0, center_type='joint_mean', aug_para=[10, 0.2, 180])
    # synthetic_dataset = icvl_loader(root, 'train', center_type='joint_mean', aug_para=[0, 0, 0])
    # synthetic_dataset = msra_loader(root, 'test', test_persons=[0], img_size=128, center_type='refine')
    print(dataset.__len__())
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    for index, data in enumerate(dataloader):
        # ind_num, img = data
        img, synth_img, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, rot = data
        # img, center, M, cube = img.cuda(), center.cuda(), M.cuda(), cube.cuda()
        # j3d_uvd = j3d_uvd.cuda()
        # pcl = dataset.Img2pcl(img.cuda(), 128, center.cuda(), M.cuda(), cube.cuda(), sample_num=1024)
        # pcl_uvd = dataset.xyz_nl2uvdnl_tensor(pcl, center.cuda(), M.cuda(), cube.cuda())
        # pcl_feature = WeightedFeatureExtract(img, img, j3d_uvd)
        # synth_img = ImgRender.renderPointCloud(pcl, pcl_depth)
        # noraml_img, noraml_joint, center3d, cube_size, M = ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), M.cuda(), rot.cuda(),
        #           useInputM=True)
        # print(noraml_img[0,0])
        # joint_temp = torch.zeros_like(j3d_xyz)
        # joint_temp[:, :12, :] = noraml_joint
        # vis_tool.debug_2d_pose(img, j3d_uvd, index, 'nyu', './debug', 'img', batch_size)
        # vis_tool.debug_2d_pose(noraml_img, joint_temp, index, 'nyu', './debug', 'synth_img', batch_size)
        vis_tool.debug_2d_pose(img, j3d_uvd, index, 'nyu', './debug', 'render_img', batch_size)
        # vis_tool.debug_2d_heatmap(img, img-synth_img, index, './debug')
        print(index)
        if index == 3:
            break

    print('done')


def save_nyu_cam(view=2):
    root = '/home/pfren/dataset/hand/nyu'
    dataset = nyu_loader(root, 'test', percent=1, view=view, center_type='joint_mean')
    print(dataset.__len__())
    np.savetxt('./view'+str(view+1)+'.txt', dataset.all_joints_uvd.reshape([-1, 14 * 3]), fmt='%.3f' )
    print('done')

# vis sphere model
def vis_sphere():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 2
    dataset = nyu_loader(root, 'test', type='synth', percent=1, view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    # synthetic_dataset = icvl_loader(root, 'train', center_type='joint_mean', aug_para=[0, 0, 0])
    # dataset = msra_loader(root, 'test', test_persons=[0], img_size=128, center_type='joint_mean')
    # dataset = hands_modelPara_loader(root, 'train')
    print(dataset.__len__())
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    for index, data in enumerate(dataloader):
        # model_para, cube = data
        img, synth_img, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, rot = data
        noraml_img, noraml_joint, noraml_joint_xyz, normal_verts_xyz, center3d, cube_size, M = ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), all_joints=True)

        # debug sphere
        spheres_c, spheres_r = ImgRender.mano_layer.get_sphere_radius(noraml_joint_xyz, normal_verts_xyz)
        for mesh_index in range(normal_verts_xyz.size(0)):
            verts = normal_verts_xyz.cpu().numpy()[mesh_index]
            sphere_c = spheres_c.cpu().numpy()[mesh_index]
            sphere_r = spheres_r.cpu().numpy()[mesh_index]
            faces = ImgRender.mano_layer.faces.cpu().numpy().astype('int')
            # display_mesh(path, verts, faces, keypoints=sphere_c)
            for seg_index in range(16):
                path = './debug/' + str(index * normal_verts_xyz.size(0) + mesh_index) + 'test%d.png'%(seg_index)
                verts_select = torch.index_select(torch.from_numpy(verts).unsqueeze(0), 1, ImgRender.mano_layer.vertex_joint_index_list[seg_index]).squeeze().numpy()
                display_point(path, verts_select, spheres_c=sphere_c, spheres_r=sphere_r)
        break

    print('done')


GLOBAL_SEED = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GLOBAL_WORKER_ID = None

seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


# vis rendered image from hands20
def vis_Render():
    dataset = 'nyu'
    root = '/home/pfren/dataset/hand/'
    batch_size = 32
    mano_model_path = '../MANO/'
    dataset_loader = nyu_loader(root + dataset, 'train', center_type='joint_mean', aug_para=[10, 0.1, 180])
    # dataset_loader = flip_icvl_loader(root + dataset, 'test', center_type='joint_mean', aug_para=[10, 0.1, 180])
    dataloader = DataLoader(dataset_loader, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    # modelParaDataset = hands_modelPara_loader(root + '/hands20/', 'train', cube_size=[250, 250, 250])
    # modelParaLoader = DataLoader(modelParaDataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    ImgRender = Render(mano_model_path, dataset, dataset_loader.paras, dataset_loader.ori_img_size).cuda()

    # transferNet = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
    # model_dict = torch.load('/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/hands19_consis_cyclegan-40epoch/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
    # transferNet.load_state_dict(model_dict)
    # transferNet.eval()

    for index, data in enumerate(dataloader):
        img, j3d_xyz, j3d_uvd, center, M, cube, model_para = data
        # noraml_img, joint_uvd_gt, verts_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = \
        #     ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), mask=False)
        # vis_tool.debug_2d_pose(img, j3d_uvd, index, 'nyu', './debug', 'real', batch_size, True)
        # vis_tool.debug_2d_pose(noraml_img, joint_uvd_gt, index, 'mano', './debug', 'render', batch_size, True)
        # print(index)
        # break
        M_pred = ImgRender.test_M(center, cube)
        print((M-M_pred).sum())
        # img = img.cuda()
        # xyz = dataset_loader.uvd_nl2xyznl_tensor(j3d_uvd, center, M, cube)
        # print((xyz-j3d_xyz).sum())
        # model_para, cube = data
        # model_para, cube = data
        # model_para = np.loadtxt('/home/pfren/pycharm/hand_mixed/model/manoPara.txt')
        # model_para = model_para[0:1].repeat(batch_size, 1)

        # augmentShape = torch.randn([batch_size, 10]).cuda() * 3
        # augmentCenter = (torch.rand([batch_size, 3]).cuda() - 0.5) * 40
        # augmentSize = (1 + (torch.rand([batch_size, 1]).cuda() - 0.5) * 0.4)
        # augmentView = torch.rand([model_para.size(0), 3]).cuda() * np.pi * 2
        # noraml_img, joint_uvd_gt, verts_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), cube.cuda(),
        #                                                    augmentView=augmentView, augmentShape=augmentShape,
        #                                                    augmentCenter=augmentCenter, augmentSize=augmentSize, mask=False)
        # vis_tool.debug_2d_pose(noraml_img, joint_uvd_gt, index, 'mano', './debug', 'pose', batch_size, True)

        # model_para = torch.from_numpy(model_para).float().cuda().unsqueeze(0).repeat([32,1])
        # augmentShape = torch.randn([batch_size, 10]).cuda() * 3 * 0
        # augmentCenter = (torch.rand([batch_size, 3]).cuda() - 0.5) * 40 * 0
        # augmentSize = (1 + (torch.rand([batch_size, 1]).cuda() - 0.5) * 0.4)
        # augmentView = torch.rand([model_para.size(0), 3]).cuda() * np.pi * 2 * 0
        # noraml_img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), None, cube.cuda(),
        #               augmentView=augmentView, augmentShape=augmentShape, augmentCenter=augmentCenter, augmentSize=augmentSize, mask=False)
        # vis_tool.debug_2d_img(noraml_img, index, './debug', 'renderImg0', 32)
        #
        # augmentView = torch.Tensor([[0, 0, np.pi/2]]).cuda().repeat(model_para.size(0),1)
        # noraml_img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), None, cube.cuda(),
        #               augmentView=augmentView, augmentShape=augmentShape, augmentCenter=augmentCenter, augmentSize=augmentSize, mask=False)
        # vis_tool.debug_2d_img(noraml_img, index, './debug', 'renderImg1', 32)
        # transferImg = transferNet(noraml_img)
        # add_noise = ImgRender.synth2real(transferImg, noise=0.02, noise_patch=4, sigma=0.5)

        # vis_tool.debug_2d_img(noraml_img, index, './debug', 'transferImg-ori')
        # vis_tool.debug_2d_img(add_noise, index, './debug', 'transferImg-noise')
        # vis_tool.debug_2d_pose(img, j3d_uvd, index, 'icvl', './debug', 'pose', batch_size)
        # print(index)
        # if index == 0:
        #     break

    print('done')


def campare_icploss():
    from metric.meshLoss import FingerICPLoss, ICPLoss, JointICPLoss
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 1
    mano_model_path = '../MANO/'
    dataset = nyu_loader(root, 'test', view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, dataset.ori_img_size).cuda()

    index = 500
    img, _, joint_xyz, joint_uvd, _, center, M, cube, _, _ = dataset.__getitem__(index)
    img, center, M, cube = img.cuda().unsqueeze(0), center.cuda().unsqueeze(0), M.cuda().unsqueeze(0), cube.cuda().unsqueeze(0)
    joint_xyz, joint_uvd = joint_xyz.cuda(), joint_uvd.cuda()

    label_index = 500
    model_para_label = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-alljoint-P2M-Depth-ICP-ICPJoint/MANO_result_1_0.txt')[label_index:label_index+1,:]
    model_para_label = torch.from_numpy(model_para_label).cuda().float()
    model_img_label, model_joint_uvd_label, model_joint_xyz_label, _, model_mesh_label = ImgRender.noraml_render(model_para_label, center, cube, all_joints=True)

    index = 1710
    model_para = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-alljoint-P2M-Depth-ICP-ICPJoint/MANO_result_1_0.txt')[index:index+1,:]
    model_para = torch.from_numpy(model_para).cuda().float()
    model_img, model_joint_uvd, model_joint_xyz, _, model_mesh = ImgRender.noraml_render(model_para, center, cube, all_joints=True)

    img = model_img_label
    _, pcl = dataset.uvdImg2xyzImg(img, center, M, cube)
    pcl = pcl.view(batch_size, 3, -1).permute(0, 2, 1)
    segment = ImgRender.mano_layer.seg_pcl(model_joint_xyz_label, model_joint_xyz_label, model_mesh_label, pcl)
    segment = segment.view(batch_size, 1, 128, 128)
    segment = torch.where(img.lt(0.99), segment, torch.zeros_like(segment))
    segment = torch.where(segment.eq(1) | segment.eq(4) | segment.eq(7) | segment.eq(10) | segment.eq(13)| segment.eq(14),
                          torch.zeros_like(segment), segment)

    img_select = torch.where(segment.gt(0), img, torch.ones_like(img))
    pcl = dataset.Img2pcl(img_select, 128, center, M, cube, 1024)

    id_to_color = vis_tool.get_segmentJointColor()
    segment_img = id_to_color[segment.squeeze(1).long().detach().cpu()]
    cv2.imwrite('./debug/segment.png', segment_img[0])

    segment = ImgRender.mano_layer.seg_pcl(model_joint_xyz_label, model_joint_xyz_label, model_mesh_label, pcl)
    display_pcl_segment('./debug/pcl_segment.png', pcl.detach().cpu().numpy()[0], segment.detach().cpu().numpy()[0])

    print("%.6f" % ICPLoss(model_mesh, pcl, ImgRender.mano_layer.faces).mean())
    print("%.6f" % ICPLoss(model_mesh_label, pcl, ImgRender.mano_layer.faces).mean())
    # print("%.6f"%JointICPLoss(model_mesh, pcl, ImgRender.mano_layer.joint_faces, segment).view(5,3).mean(-1)[0])
    # print("%.6f"%JointICPLoss(model_mesh, pcl, ImgRender.mano_layer.joint_faces, segment).mean(-1))
    # print("%.6f"%JointICPLoss(model_mesh_label, pcl, ImgRender.mano_layer.joint_faces, segment).mean(-1))

    # print("%.6f"%ImgRender.mano_layer.JointICP(model_joint_xyz, model_joint_xyz_label, model_mesh_label, pcl).mean())
    # print("%.6f"%ImgRender.mano_layer.JointICP(model_joint_xyz_label, model_joint_xyz_label, model_mesh_label, pcl).mean())
    loss, dis, shpere_idx, shpere, shpere_r = ImgRender.mano_layer.JointICP(model_joint_xyz_label, model_joint_xyz_label, model_mesh_label, pcl)
    print("%.6f" % loss.mean())
    joint = shpere.squeeze()[shpere_idx.squeeze(), :]
    line = joint - pcl.squeeze()
    direct = line / torch.sqrt(torch.pow(line, 2).sum(-1)).unsqueeze(-1)
    end_points = pcl.squeeze() + direct * dis.squeeze().unsqueeze(-1)

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    start_points = pcl.squeeze().cpu()
    end_points = end_points.cpu()
    ls = np.hstack([start_points, end_points]).copy()
    ls = ls.reshape((-1, 2, 3))
    lc = Line3DCollection(ls, linewidths=0.5, colors='b')
    ax.add_collection(lc)
    display_sphere(shpere.squeeze().cpu().numpy(), shpere_r.squeeze().cpu().numpy(), ax=ax, transpose=False)

    mesh = model_mesh_label.cpu().numpy().squeeze()

    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], color='red', s=1)

    cam_equal_aspect_3d(ax, start_points.numpy(), transpose=False)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.savefig('hh.png', bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()
    # verts = model_mesh.cpu().numpy()[0]
    # faces = ImgRender.mano_layer.faces
    # path = './%d.obj'%(index)
    # with open(path, 'w') as fp:
    #     for v in verts:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #     for f in faces + 1:
    #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    # print("%.6f"%ICPLoss(model_mesh_label, pcl, ImgRender.mano_layer.faces).mean())
    # print("%.6f"%JointICPLoss(model_mesh_label, pcl, ImgRender.mano_layer.joint_faces, segment).view(5,3).mean(-1)[0])
    # print("%.6f"%JointICPLoss(model_mesh_label, pcl, ImgRender.mano_layer.joint_faces, segment).mean(-1))

    # vis_tool.debug_2d_img(img, 0, './debug', 'img',1)
    # vis_tool.debug_2d_pose(model_img, model_joint_uvd, 0, 'MANO', './debug', 'pred1', 1, True)
    # vis_tool.debug_2d_pose(model_img_label, model_joint_uvd_label, 0, 'MANO', './debug', 'pred2', 1, True)


# compare joint for different dataset
def vis_nyu_Joint():
    dataset_name = 'nyu'
    root = '/home/pfren/dataset/hand/'
    batch_size = 32
    mano_model_path = '../MANO/'
    dataset = nyu_modelPara_loader(root + '/nyu/', 'train', cube_size=[250, 250, 250])
    Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ImgRender = Render(mano_model_path, dataset_name, dataset.paras, dataset.ori_img_size).cuda()

    for index, data in enumerate(Loader):
        # model_para, cube = data
        model_para, cube = data
        # noraml_img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), None, cube.cuda(), all_joints=True, mask=False)
        # vis_tool.debug_2d_pose(noraml_img, joint_uvd_gt, index, 'MANO', './debug', 'pose', batch_size)
        noraml_img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), None, cube.cuda(), mask=False)

        # vis_tool.debug_2d_pose(noraml_img, joint_uvd_gt[:,:-1,:], index, 'nyu', './debug', 'pose', batch_size)
        # print(index)
        # if index == 0:
        #     break
    print('done')


# compare joint for different dataset
def vis_msra_Joint():
    dataset_name = 'msra'
    root = '/home/pfren/dataset/hand/'
    batch_size = 32
    mano_model_path = '../MANO/'
    dataset = msra_loader(root + dataset_name, 'test')
    Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ImgRender = Render(mano_model_path, dataset_name, dataset.paras, dataset.ori_img_size).cuda()

    for index, data in enumerate(Loader):
        data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
        # model_para, cube = data
        # model_para, cube = data
        noraml_img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), all_joints=True, mask=False)
        vis_tool.debug_2d_pose(noraml_img, joint_uvd_gt, index, 'MANO', './debug', 'pose', batch_size)
        # noraml_img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), None, cube.cuda(), mask=False)
        # vis_tool.debug_2d_pose(noraml_img, joint_uvd_gt[:, :, :], index, 'MANO', './debug', 'pose-select', batch_size)
        # vis_tool.debug_2d_pose(noraml_img, joint_uvd_gt, index, 'MANO', './debug', 'pose-synth', batch_size)
        vis_tool.debug_2d_pose(data, joint_img, index, 'msra', './debug', 'pose-ori', batch_size)
        # vis_tool.debug_2d_img(data, index, './debug', 'ori')
        print(index)
        if index == 0:
            break
    print('done')


def joint2offset(joint, img, kernel_size, feature_size):
    device = joint.device
    batch_size, _, img_height, img_width = img.size()
    img = F.interpolate(img, size=[feature_size, feature_size])
    _, joint_num, _ = joint.view(batch_size, -1, 3).size()
    joint_feature = joint.reshape(joint.size(0), -1, 1, 1).repeat(1, 1, feature_size, feature_size)
    # mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
    # mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1)
    offset = joint_feature - coords
    offset = offset.view(batch_size, joint_num, 3, feature_size, feature_size)
    dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2) + 1e-8)
    offset_norm = (offset / (dist.unsqueeze(2)))
    heatmap = (kernel_size - dist) / kernel_size
    mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size, 1, feature_size, feature_size)
    offset_norm_mask = (offset_norm * mask.unsqueeze(2)).view(batch_size, -1, feature_size, feature_size)
    heatmap_mask = heatmap * mask.float()
    return torch.cat((offset_norm_mask, heatmap_mask), dim=1)


def vis_heatmap():
    # root = '/home/pfren/dataset/hand/nyu'
    # batch_size = 32
    # dataset = nyu_loader(root, 'test', view=0, center_type='joint_mean')
    # mano_model_path = '../MANO/'
    # ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    #
    # for index, data in enumerate(dataloader):
    #     img, synth_img, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, rot = data
    #     noraml_img, noraml_joint_uvd, noraml_joint_xyz, mesh, center3d, cube_size, M = \
    #         ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), all_joints=True, M=M.cuda(), rot=rot.cuda(), useInputM=True,mask=False)
    #     heatmap = joint2offset(j3d_uvd, img, 0.8, 128)
    #     vis_tool.debug_2d_heatmap(img, heatmap, index, './debug', 128, save=True)
    #     vis_tool.debug_2d_img(img, index, './debug', 'img', batch_size)
    #     vis_tool.debug_2d_img(noraml_img, index, './debug', 'synth', batch_size)
    #     vis_tool.debug_2d_pose(torch.ones([batch_size, 1, 256, 256]), noraml_joint_uvd, index,'MANO','./debug','pose',batch_size,True)
    #     for batch_index in range(batch_size):
    #         verts = mesh.cpu().numpy()[batch_index]
    #         faces = ImgRender.mano_layer.faces
    #         path = './debug/%d.obj' % (index*batch_size+batch_index)
    #         with open(path, 'w') as fp:
    #             for v in verts:
    #                 fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #             for f in faces + 1:
    #                 fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    #     if index == 3:
    #         break
    # print('done')

    # root = '/home/pfren/dataset/hand/nyu'
    # batch_size = 32
    # dataset = nyu_loader(root, 'test', view=0, center_type='joint_mean')
    # mano_model_path = '../MANO/'
    # ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    #
    # transferNet_consis = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
    # model_dict = torch.load('/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/nyu_background_consis_cyclegan-40epoch/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
    # transferNet_consis.load_state_dict(model_dict)
    # transferNet_consis.eval()
    #
    # for index, data in enumerate(dataloader):
    #     img, synth_img, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, rot = data
    #     noraml_img, noraml_joint_uvd, noraml_joint_xyz, mesh, center3d, cube_size, M = \
    #         ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), all_joints=True, M=M.cuda(), rot=rot.cuda(), useInputM=True,mask=False)
    #     vis_tool.debug_2d_img(noraml_img, index, './debug', 'synth', batch_size)
    #     noraml_img, noraml_joint_uvd, noraml_joint_xyz, mesh, center3d, cube_size, M = \
    #         ImgRender(model_para.cuda(), center.cuda(), cube.cuda(), all_joints=True, M=M.cuda(), rot=rot.cuda(), useInputM=True,mask=True)
    #     trans_img = transferNet_consis(noraml_img)
    #     heatmap = joint2offset(noraml_joint_uvd, noraml_img, 0.8, 128)
    #     vis_tool.debug_2d_heatmap(img, heatmap, index, './debug', 128, save=True)
    #     vis_tool.debug_2d_img(noraml_img, index, './debug', 'img', batch_size)
    #     vis_tool.debug_2d_img(trans_img, index, './debug', 'img_trans', batch_size)
    #     vis_tool.debug_2d_pose(torch.ones([batch_size, 1, 256, 256]), noraml_joint_uvd, index,'MANO','./debug','pose',batch_size,True)
    #     for batch_index in range(batch_size):
    #         verts = mesh.cpu().numpy()[batch_index]
    #         faces = ImgRender.mano_layer.faces
    #         path = './debug/%d.obj' % (index*batch_size+batch_index)
    #         with open(path, 'w') as fp:
    #             for v in verts:
    #                 fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #             for f in faces + 1:
    #                 fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    #     if index == 3:
    #         break
    # print('done')
    from model.multiView_fusion import MultiView_UNet
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    view = 0
    dataset = nyu_loader(root, 'test', view=view, center_type='joint_mean')
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    net = MultiView_UNet('ResNet_multiView_18', 21, 3).cuda()
    model_dict = torch.load('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/Finetune-UNet-CycleGAN-UPGCN-GroupConfi-MultiView-TestData/best.pth', map_location=lambda storage, loc: storage)
    net.load_state_dict(model_dict['model'])
    net.eval()
    for index, data in enumerate(dataloader):
        img, synth_img, j3d_xyz, j3d_uvd, model_para, center, M, cube, visible, rot = data
        offset, model_para, _ = net.forward_single([img.cuda()])
        # heatmap = offset[-1][:, 21 * 3:21 * 4, :]
        # weight = offset[-1][:, 21 * 4:, :]
        # weight = torch.softmax(weight.view(batch_size, 21, -1), dim=-1).view(batch_size, 21, 32, 32)
        # vis_tool.debug_2d_img(img, index, './debug', 'img_{}'.format(view), 32)
        # vis_tool.debug_2d_heatmap(heatmap, index, './debug', 128, save=True, img_type='heatmap_{}'.format(view))
        # vis_tool.debug_2d_heatmap(weight, index, './debug', 128, save=True, img_type='weight_{}'.format(view))
        model_img, model_joint_uvd, model_joint_xyz, _, model_mesh = ImgRender.noraml_render(model_para, center.cuda(), cube.cuda(), all_joints=True)
        # vis_tool.debug_2d_img(model_img, index, './debug', 'render_{}'.format(view), 32)
        # print(xyz2error(model_joint_xyz[:,ImgRender.mano_layer.transfer,:],model_joint_xyz))
        vis_tool.debug_2d_pose(torch.ones([batch_size,1,256,256]), model_joint_uvd, index, 'mano', './debug', 'pose_{}'.format(view), batch_size, save=True)
        # faces = ImgRender.mano_layer.faces
        # for batch_index in range(batch_size):
        #     path = './debug/mesh{}.obj'.format(batch_index)
        #     verts = normal_verts_xyz.cpu().detach().numpy()[batch_index]
        #     with open(path, 'w') as fp:
        #         for v in verts:
        #             fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        #         for f in faces + 1:
        #             fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        if index == 0:
            break
    print('done')


def shrec_data():
    # dataset_name = 'shrec'
    # root = '/home/pfren/dataset/hand/'
    # dataset = shrec_loader(root + dataset_name, aug_para=[0, 0, 0])
    # joint_uvds = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/shrec/srrnet_18_attention/result_1.txt').reshape([-1,21,3])
    # joint_xyzs = dataset.jointImgTo3D(joint_uvds)
    # # joint_xyzs_normal = (joint_xyzs - dataset.jointImgTo3D(dataset.all_center_uvd.reshape([-1, 1, 3]))) / 125.0
    # joint_xyzs_normal = joint_xyzs / 1000
    # joint_xyzs_normal_plam = np.mean(joint_xyzs_normal, axis=1, keepdims=True)
    # MANO2SHREC = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
    # joint_xyzs_normal = joint_xyzs_normal[:, MANO2SHREC, :]
    # joint_write = np.concatenate((joint_xyzs_normal[:, 0:1, :], joint_xyzs_normal_plam, joint_xyzs_normal[:, 1:, :]), axis=1)
    # dataset.write_joints(root + dataset_name, joint_write.reshape([-1, 22*3]))

    dataset_name = 'DHG2016'
    root = '/home/pfren/dataset/hand/'
    dataset = DHG_loader(root + dataset_name, aug_para=[0, 0, 0])
    Loader = DataLoader(dataset, batch_size=32, shuffle=False)

    for index, data in enumerate(Loader):
        data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
        # vis_tool.debug_2d_pose(data, joint_img, index, 'MANO', './debug', 'ori', 32)
        vis_tool.debug_2d_img(data, index,  './debug', 'ori')
        if index == 10:
            break
    print('done')


def DHG_data():
    dataset_name = 'DHG2016'
    root = '/home/pfren/dataset/hand/'

    # dataset = DHG_loader(root + dataset_name, aug_para=[0, 0, 0])
    # joint_uvds = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/shrec/DHG-AWR-50/result_0.txt').reshape([-1,21,3])
    # joint_xyzs = dataset.jointImgTo3D(joint_uvds)
    # # joint_xyzs_normal = (joint_xyzs - dataset.jointImgTo3D(dataset.all_center_uvd.reshape([-1, 1, 3]))) / 125.0
    # joint_xyzs_normal = joint_xyzs / 1000
    # joint_xyzs_normal_plam = np.mean(joint_xyzs_normal, axis=1, keepdims=True)
    # HAND2MANO = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 11, 14, 20, 17, 8]
    # MANO2SHREC = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
    # joint_xyzs_normal = joint_xyzs_normal[:, HAND2MANO, :][:, MANO2SHREC, :]
    # joint_write = np.concatenate((joint_xyzs_normal[:, 0:1, :], joint_xyzs_normal_plam, joint_xyzs_normal[:, 1:, :]), axis=1)
    # dataset.write_joints(root + dataset_name, joint_write.reshape([-1, 22*3]))
    #
    dataset_name = 'DHG2016'
    root = '/home/pfren/dataset/hand/'
    dataset = DHG_loader(root + dataset_name, aug_para=[0, 0, 0])
    Loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for index, data in enumerate(Loader):
        data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
        vis_tool.debug_2d_pose(data, joint_img, index, 'DHG', './debug', 'ori', 32, True)
        # vis_tool.debug_2d_img(data, index,  './debug', 'ori')
        if index == 10:
            break
    print('done')


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=0, keepdim=True)
    w, x, y, z = norm_quat[0], norm_quat[1], norm_quat[2], norm_quat[3]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=0).view(3, 3)

    return rotMat


def rodrigues(theta):
    l1norm = torch.norm(theta + 1e-8, p=2, dim=0)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = quat2mat(torch.cat([v_cos, v_sin * normalized], dim=0))
    return quat


def RotationPoints(points, rot):
    rotMat = quat2mat(rot).unsqueeze(1)
    rot_points = torch.matmul(rotMat, points.unsqueeze(-1)).squeeze()
    return rot_points


def joint_augment(joint):
    device = joint.device
    batch_size = joint.size(0)
    augment_types = np.random.randint(2, 3, batch_size)

    thumb = np.array([13, 14, 15, 20])
    index = np.array([1, 2, 3, 16])
    middle = np.array([4, 5, 6, 17])
    ring = np.array([10, 11, 12, 19])
    little = np.array([7, 8, 9, 18])
    all_finger = np.stack([thumb, index, middle, ring, little], axis=0)

    for index, augment_type in enumerate(augment_types):
        # 相邻手指之间的插值
        if augment_type == 0:
            R1 = np.random.uniform(0, 0.2)
            R2 = np.random.uniform(0, 0.4)
            R3 = np.random.uniform(0.2, 0.6)
            R4 = np.random.uniform(0.4, 1)
            Disturb = torch.from_numpy(np.array([R1, R2, R3, R4])).to(device).unsqueeze(-1).float()

            fingerID = np.random.randint(0, 4)
            nerberID = np.random.randint(0, 1)
            nerberID = nerberID if nerberID > 0 else -1

            finger1 = all_finger[fingerID]
            finger2 = all_finger[(fingerID + nerberID) % 5]

            joint[index, finger1, :] = joint[index, finger2, :] * Disturb + joint[index, finger1, :] * (1 - Disturb)

        # 关节点扰动
        if augment_type == 1:
            fingerID = np.random.randint(0, 4)
            noise = torch.rand([3]).to(device).float() * 0.1

            finger = all_finger[fingerID]

            joint[index, finger, :] = joint[index, finger, :] + noise

        if augment_type == 2:
            rot = torch.rand([3]).to(device).float() * 0.1 * np.pi
            joint[index, :, :] = torch.matmul(rodrigues(rot), joint[index, :, :].unsqueeze(-1)).squeeze(-1)
    return joint


def display_hand(verts, mano_faces=None, ax=None, alpha=0.2):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts.numpy())
    plt.axis('off')
    plt.savefig('./mesh.png', bbox_inches='tight', pad_inches=0, dpi=150)


# from open3d import *


def multiView_calibration():
    root = '/home/pfren/dataset/hand/icvl'
    # root = 'D:\\dataset\\nyu'
    batch_size = 32
    sample_num = 1024
    dataset = icvl_multiAug_loader(root, aug_para=[10, 0.2, 180], center_type='refine')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    # mano_model_path = '../MANO/'
    # ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()
    # pred1s = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-traintest/result_0_1.txt').reshape([-1, 21, 3])
    # pred2s = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-traintest/result_0_2.txt').reshape([-1, 21, 3])
    # joint = np.loadtxt('D:\\pycharm\\hand_mixed\\result_0_0.txt').reshape([-1, 21, 3])
    # mesh = np.loadtxt('D:\\pycharm\\hand_mixed\\mesh_result_1_0.txt').reshape([8252, -1, 3])
    # verts = mesh[448].reshape([-1, 3])
    # faces =  ImgRender.mano_layer.faces.cpu().numpy().astype(np.int16)
    # path = './test.obj'
    # with open(path, 'w') as fp:
    #     for v in verts:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #     for f in faces + 1:
    #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    # display_hand(mesh_vis, ImgRender.mano_layer.faces.cpu().numpy().astype(np.int16))
    for index, data in enumerate(dataloader):
        data_0, joint_0, joint_img_0, center_0, M_0, cube_0, \
        data_1, joint_1, joint_img_1, center_1, M_1, cube_1, \
        data_2, joint_2, joint_img_2, center_2, M_2, cube_2 = data
        # vis_tool.debug_2d_img(data_0, index, './debug', 'img0', batch_size)
        # vis_tool.debug_2d_img(data_1, index, './debug', 'img1', batch_size)
        # vis_tool.debug_2d_img(data_2, index, './debug', 'img2', batch_size)

        # vis_tool.debug_2d_pose(img_0, joint_img_0, index, 'msra', './debug/', 'pose0', batch_size, True)
        # img_trans, joint_trans, M = aug_img(img_0, joint_img_0, theta=90, trans=0, scale=0.3)
        # img_trans, joint_trans, M = aug_img(img_trans, joint_trans, theta=0, trans=40, scale=0)
        # vis_tool.debug_2d_pose(img_trans, joint_trans, index, 'msra', './debug/', 'pose1', batch_size, True)

        # img_tran_0, M = rot_img(img_0, rot)
        # joint_trans_0 = joint_img_0.clone()
        # joint_trans_0[:, :, :2] = torch.matmul(joint_trans_0[:, :, :2].unsqueeze(-2), M.unsqueeze(1)).squeeze()
        # vis_tool.debug_2d_pose(img_tran_0, joint_trans_0, index, 'msra', './debug/', 'pose0', batch_size, True)
        # rot = torch.rand([batch_size])*360
        # img_tran_0, M = rot_img(img_0, rot)
        # joint_trans_0 = joint_img_0.clone()
        # joint_trans_0[:, :, :2] = torch.matmul(joint_trans_0[:, :, :2].unsqueeze(-2), M.unsqueeze(1)).squeeze()
        # vis_tool.debug_2d_pose(img_tran_0, joint_trans_0, index, 'msra', './debug/', 'pose1', batch_size, True)
        # rot = torch.rand([batch_size])*360
        # data_0 = rot_img(img_0, rot)
        # vis_tool.debug_2d_pose(data_0, joint_img_0, index, 'msra', './debug/', 'pose1', batch_size, True)
        # vis_tool.debug_2d_pose(data_1, joint_img_1, index, 'msra', './debug/', 'pose1', batch_size, True)
        # vis_tool.debug_2d_pose(data_2, joint_img_2, index, 'msra', './debug/', 'pose2', batch_size, True)
        # return 0
        # vis_tool.debug_2d_img(data_0, index, './debug', 'img0', batch_size)
        # vis_tool.debug_2d_img(data_1, index, './debug', 'img1', batch_size)
        # vis_tool.debug_2d_img(data_2, index, './debug', 'img2', batch_size)
        # break
        ############### laod prediected joint and mesh  #####################################
        # pred0 = dataset.pointsImgTo3D(torch.from_numpy(joint[index * batch_size:(index + 1) * batch_size]).float())
        # pred1 = dataset.pointsImgTo3D(torch.from_numpy(joint[index * batch_size:(index + 1) * batch_size]).float())
        # pred2 = dataset.pointsImgTo3D(torch.from_numpy(joint[index * batch_size:(index + 1) * batch_size]).float())
        # pred0 = (pred0 - center_0.unsqueeze(1)) / cube_0.unsqueeze(1) * 2
        # pred1 = (pred1 - center_1.unsqueeze(1)) / cube_1.unsqueeze(1) * 2
        # pred2 = (pred2 - center_2.unsqueeze(1)) / cube_2.unsqueeze(1) * 2

        # pred0 = dataset.xyz_nl2uvdnl_tensor(pred0,center_0,M_0, cube_0)
        # pred1 = dataset.xyz_nl2uvdnl_tensor(pred1,center_1,M_1, cube_1)
        # pred2 = dataset.xyz_nl2uvdnl_tensor(pred2,center_2,M_2, cube_2)
        # vis_tool.debug_2d_pose(torch.ones([batch_size, 1, 512, 512]), pred0, index, 'mano', './debug/', 'pose0', batch_size, True)
        # vis_tool.debug_2d_pose(torch.ones([batch_size, 1, 512, 512]), pred1, index, 'mano', './debug/', 'pose1', batch_size, True)
        # vis_tool.debug_2d_pose(torch.ones([batch_size, 1, 512, 512]), pred2, index, 'mano', './debug/', 'pose2', batch_size, True)
        # if index > 15:
        #     break

        # pcl_0 = dataset.Img2pcl(data_0, 128,  center_0, M_0, cube_0, sample_num)
        # pcl_0 = pcl_0 / 2 * cube_0.unsqueeze(1) + center_0.unsqueeze(1)
        # pcl_1 = dataset.Img2pcl(data_1, 128,  center_1, M_1, cube_1, sample_num)
        # pcl_1 = pcl_1 / 2 * cube_1.unsqueeze(1) + center_1.unsqueeze(1)
        # pcl_2 = dataset.Img2pcl(data_2, 128,  center_2, M_2, cube_2, sample_num)
        # pcl_2 = pcl_2 / 2 * cube_2.unsqueeze(1) + center_2.unsqueeze(1)
        #
        # joint_num = joint_0.size(1)
        # ones_pad = torch.ones([batch_size, joint_num, 1])
        # joint_world_0 = joint_0 * cube_0.unsqueeze(1) / 2 + center_0.unsqueeze(1)
        # joint_world_1 = joint_1 * cube_1.unsqueeze(1) / 2 + center_1.unsqueeze(1)
        # joint_world_2 = joint_2 * cube_2.unsqueeze(1) / 2 + center_2.unsqueeze(1)
        # temp0 = torch.cat((joint_world_0, ones_pad), dim=-1)[:, :4]
        # temp1 = torch.cat((joint_world_1, ones_pad), dim=-1)[:, :4]
        # temp2 = torch.cat((joint_world_2, ones_pad), dim=-1)[:, :4]

        # TE = torch.eye(4).unsqueeze(0).repeat(batch_size,1,1)
        # T01 = torch.linalg.lstsq(temp0, temp1).solution
        # T02 = torch.linalg.lstsq(temp0, temp2).solution
        # T10 = torch.linalg.lstsq(temp1, temp0).solution
        # T20 = torch.linalg.lstsq(temp2, temp0).solution

        # temp0 = temp0.cpu().numpy()
        # temp1 = temp1.cpu().numpy()
        # temp2 = temp2.cpu().numpy()
        # T10, T20= [], []
        # for T_index in range(batch_size):
        #     T10.append(np.linalg.lstsq(temp1[T_index], temp0[T_index])[0])
        #     T20.append(np.linalg.lstsq(temp2[T_index], temp0[T_index])[0])
        # T10 = torch.from_numpy(np.stack(T10, axis=0)).float()
        # T20 = torch.from_numpy(np.stack(T20, axis=0)).float()
        #
        # pcl_one_pad = torch.ones([batch_size, sample_num, 1])
        # pcl_1 = dataset.Img2pcl(data_1, 128, center_1, M_1, cube_1, sample_num)
        # pcl_2 = dataset.Img2pcl(data_2, 128, center_2, M_2, cube_2, sample_num)
        # pcl_1 = pcl_1 * cube_1.unsqueeze(1) / 2 + center_1.unsqueeze(1)
        # pcl_2 = pcl_2 * cube_2.unsqueeze(1) / 2 + center_2.unsqueeze(1)
        # pcl_10 = torch.matmul(torch.cat((pcl_1, pcl_one_pad), dim=-1), T10)[:, :, :3]
        # pcl_20 = torch.matmul(torch.cat((pcl_2, pcl_one_pad), dim=-1), T20)[:, :, :3]
        # pcl_10 = (pcl_10 - center_0.unsqueeze(1)) * 2 / cube_0.unsqueeze(1)
        # pcl_20 = (pcl_20 - center_0.unsqueeze(1)) * 2 / cube_0.unsqueeze(1)
        # pcl_0 = (pcl_0 - center_0.unsqueeze(1)) * 2 / cube_0.unsqueeze(1)
        #
        # pcl_all = torch.cat((pcl_0, pcl_10, pcl_20), dim=1)
        # from pointCloud.util.pointnet_util import farthest_point_sample,index_points
        # pcl_sample_index = farthest_point_sample(pcl_all, 1024)
        # pcl_sample = index_points(pcl_all, pcl_sample_index)
        # pcl_sample = pcl_0.cpu().numpy()[0]
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(pcl_sample)
        #
        # def create_open3d_sphere(centers, radiuses):
        #     sphere_list = []
        #     for index, center in enumerate(centers):
        #         mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radiuses[index])
        #         mesh_sphere.compute_vertex_normals()
        #         mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        #         mesh_sphere.translate((center[0], center[1], center[2]))
        #         sphere_list.append(mesh_sphere)
        #     return sphere_list
        #
        # from render_model.mano_layer_0 import MANO_SMPL
        # mano_layer = MANO_SMPL('D:\\pycharm\\hand_mixed\\MANO\\MANO_RIGHT.pkl', 'mano')
        #
        # pred0 = dataset.pointsImgTo3D(torch.from_numpy(joint[index * batch_size:(index + 1) * batch_size]).float())
        # pred0 = (pred0 - center_0.unsqueeze(1)) / cube_0.unsqueeze(1) * 2
        # mesh = (torch.from_numpy(mesh[index * batch_size:(index + 1) * batch_size]).float() - center_0.unsqueeze(1)) / cube_0.unsqueeze(1) * 2

        # hand_mesh = mesh.cpu().numpy()[0]
        # face = mano_layer.faces.cpu().numpy()
        # o3d_mesh = open3d.geometry.TriangleMesh()
        # o3d_mesh.triangles = open3d.utility.Vector3iVector(face)
        # o3d_mesh.vertices = open3d.utility.Vector3dVector(hand_mesh)
        # o3d_mesh.compute_vertex_normals()

        # loss, dis, shpere_idx, shpere, shpere_r = \
        #     mano_layer.JointICP(pred0.cuda(), pred0.cuda(), mesh.cuda(), pcl_0.cuda())
        #
        # open3d.visualization.draw_geometries([point_cloud] + create_open3d_sphere(shpere.cpu().numpy()[0],shpere_r.cpu().numpy()[0]))
        # break

        ########################## for render img ##########################
        # augmentShape = torch.randn([batch_size, 10]).cuda() * 3
        # augmentShape = augmentShape.unsqueeze(0).repeat(3,1,1).view(3*batch_size, 10)
        # augmentCenter = (torch.rand([3*batch_size, 3]).cuda() - 0.5) * 40
        # augmentSize = (1 + (torch.rand([3*batch_size, 1]).cuda() - 0.5) * 0.4)
        # augmentView = torch.rand([3*batch_size, 3]).cuda() * np.pi * 2
        # model_para = model_para.unsqueeze(0).repeat(3,1,1).view(3*batch_size, -1)
        # cube = cube_0.unsqueeze(0).repeat(3, 1, 1).view(3 * batch_size, -1)
        # img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), None, cube.cuda(),
        #                        augmentView=augmentView, augmentShape=augmentShape, augmentCenter=augmentCenter,
        #                        augmentSize=augmentSize, mask=False, all_joints=True)
        # data_0, data_1, data_2 = torch.split(img, batch_size, dim=0)
        # joint_0, joint_1, joint_2 = torch.split(joint_xyz_gt, batch_size, dim=0)
        # center_0, center_1, center_2 = torch.split(center_s, batch_size, dim=0)
        # cube_0, cube_1, cube_2 = torch.split(cube_s, batch_size, dim=0)
        # M_0, M_1, M_2 = torch.split(M_s, batch_size, dim=0)
        # joint_num = joint_0.size(1)
        # ones_pad = torch.ones([batch_size, joint_num, 1]).cuda()

        # # for index in range(batch_size):
        # vis_tool.debug_2d_pose(img[:batch_size], joint_uvd_gt[:batch_size], index, 'mano', './debug/', 'pseudo0', batch_size, True)
        # vis_tool.debug_2d_pose(img[batch_size:2*batch_size], joint_uvd_gt[batch_size:2*batch_size], index, 'mano', './debug/', 'pseudo1', batch_size, True)
        # vis_tool.debug_2d_pose(img[2*batch_size:3*batch_size], joint_uvd_gt[2*batch_size:3*batch_size], index, 'mano', './debug/', 'pseudo2', batch_size, True)
        #

        # print(out[0])
        # latent_xyz_0 = joint_0 * cube_0.unsqueeze(1) / 2 + center_0.unsqueeze(1)
        # latent_xyz_1 = joint_1 * cube_1.unsqueeze(1) / 2 + center_1.unsqueeze(1)
        # latent_xyz_2 = joint_2 * cube_2.unsqueeze(1) / 2 + center_2.unsqueeze(1)
        #
        # latent_xyz_1 = torch.matmul(torch.cat((latent_xyz_1, torch.ones([batch_size, 14, 1])), dim=-1), T10)[:, :, :3]
        # latent_xyz_2 = torch.matmul(torch.cat((latent_xyz_2, torch.ones([batch_size, 14, 1])), dim=-1), T20)[:, :, :3]
        # pseudoLabel0 = torch.median(torch.stack((latent_xyz_0, latent_xyz_1, latent_xyz_2), dim=-1), dim=-1)[0]
        # pseudoLabel1 = torch.matmul(torch.cat((pseudoLabel0, torch.ones([batch_size, 14, 1])), dim=-1), T01)[:, :, :3]
        # pseudoLabel2 = torch.matmul(torch.cat((pseudoLabel0, torch.ones([batch_size, 14, 1])), dim=-1), T02)[:, :, :3]
        #
        # pseudoxyz0 = (pseudoLabel0 - center_0.unsqueeze(1)) / cube_0.unsqueeze(1) * 2
        # pseudoxyz1 = (pseudoLabel1 - center_1.unsqueeze(1)) / cube_1.unsqueeze(1) * 2
        # pseudoxyz2 = (pseudoLabel2 - center_2.unsqueeze(1)) / cube_2.unsqueeze(1) * 2
        #
        # pseudouvd0 = dataset.xyz_nl2uvdnl_tensor(pseudoxyz0,center_0,M_0,cube_0)
        # pseudouvd1 = dataset.xyz_nl2uvdnl_tensor(pseudoxyz1,center_1,M_1,cube_1)
        # pseudouvd2 = dataset.xyz_nl2uvdnl_tensor(pseudoxyz2,center_2,M_2,cube_2)

        # latent_xyz_0 = joint_0 * cube_0.unsqueeze(1) / 2 + center_0.unsqueeze(1)
        # latent_xyz_1 = joint_1 * cube_1.unsqueeze(1) / 2 + center_1.unsqueeze(1)
        # latent_xyz_2 = joint_2 * cube_2.unsqueeze(1) / 2 + center_2.unsqueeze(1)
        #
        # latent_xyz_1 = torch.matmul(torch.cat((latent_xyz_1, ones_pad), dim=-1), T10)[:, :, :3]
        # latent_xyz_2 = torch.matmul(torch.cat((latent_xyz_2, ones_pad), dim=-1), T20)[:, :, :3]
        # pseudoLabel0 = torch.median(torch.stack((latent_xyz_0, latent_xyz_1, latent_xyz_2), dim=-1), dim=-1)[0]
        # pseudoLabel1 = torch.matmul(torch.cat((pseudoLabel0, ones_pad), dim=-1), T01)[:, :, :3]
        # pseudoLabel2 = torch.matmul(torch.cat((pseudoLabel0, ones_pad), dim=-1), T02)[:, :, :3]
        #
        # pseudoxyz0 = (pseudoLabel0 - center_0.unsqueeze(1)) / cube_0.unsqueeze(1) * 2
        # pseudoxyz1 = (pseudoLabel1 - center_1.unsqueeze(1)) / cube_1.unsqueeze(1) * 2
        # pseudoxyz2 = (pseudoLabel2 - center_2.unsqueeze(1)) / cube_2.unsqueeze(1) * 2
        #
        # pseudouvd0 = dataset.xyz_nl2uvdnl_tensor(pseudoxyz0, center_0, M_0, cube_0)
        # pseudouvd1 = dataset.xyz_nl2uvdnl_tensor(pseudoxyz1, center_1, M_1, cube_1)
        # pseudouvd2 = dataset.xyz_nl2uvdnl_tensor(pseudoxyz2, center_2, M_2, cube_2)
        #
        # vis_tool.debug_2d_pose(data_0, pseudouvd0, index, 'mano', './debug/', 'pseudo0', batch_size, True)
        # vis_tool.debug_2d_pose(data_1, pseudouvd1, index, 'mano', './debug/', 'pseudo1', batch_size, True)
        # vis_tool.debug_2d_pose(data_2, pseudouvd2, index, 'mano', './debug/', 'pseudo2', batch_size, True)


        # img = dataset.pcl2Img(pcl_10, 128, center_0, M_0, cube_0)
        # vis_tool.debug_2d_img(img, index, './debug', 'trans_10', batch_size)
        # img = dataset.pcl2Img(pcl_20, 128, center_0, M_0, cube_0)
        # vis_tool.debug_2d_img(img, index, './debug', 'trans_20', batch_size)
        # break
        # print(T01.cpu().numpy())
        # render_imgs, mano_joints,mano_meshs = ImgRender.multiView_render(model_para.cuda(),
        #                                                                  [TE.cuda(), T01.cuda(), T02.cuda()],
        #                                                                  [center_0.cuda(), center_1.cuda(), center_2.cuda()],
        #                                                                  [cube_0.cuda(), cube_1.cuda(), cube_2.cuda()])
        # render_imgs, mano_joints, mano_meshs = ImgRender.multiView_noraml_render(model_para.cuda(),center_0.cuda(),cube_0.cuda(),
        #                                                                  [TE.cuda(), T01.cuda(), T02.cuda()],
        #                                                                  [center_0.cuda(), center_1.cuda(), center_2.cuda()],
        #                                                                  [cube_0.cuda(), cube_1.cuda(), cube_2.cuda()])
        # for view_index in range(3):
        #     vis_tool.debug_2d_img(render_imgs[view_index], index, './debug', 'render_%d'%view_index, batch_size)

        # vis_tool.debug_2d_img(data_0, index, './debug', 'real_0', batch_size)
        # vis_tool.debug_2d_img(data_1, index, './debug', 'real_1', batch_size)
        # vis_tool.debug_2d_img(data_2, index, './debug', 'real_2', batch_size)

        # display_point('pcl.png', pcl_1.numpy()[0], joint_world_1.numpy()[0])
        # joint_uvd = dataset.xyz_nl2uvdnl_tensor(joint_world_1, center_1, M_1, cube_1)
        # vis_tool.debug_2d_pose(data_1, joint_uvd, index, 'nyu', './debug/', 'trans', batch_size, True)
        #
        # trans_0 = torch.matmul(torch.cat((joint_world_0, torch.ones([batch_size, 14, 1])), dim=-1), T)[:,:,:3]
        # joint_xyz = (trans_0 - center_0.unsqueeze(1)) * 2 / cube_0.unsqueeze(1)
        # joint_uvd = dataset.xyz_nl2uvdnl_tensor(joint_xyz, center_0, M_0, cube_0)
        # vis_tool.debug_2d_pose(data_0, joint_uvd, index, 'nyu', './debug/', 'trans', batch_size, True)
        # display_point('pcl_1to0.png', pcl_0.numpy()[0], trans_0.numpy()[0])
        #
        # pcl_1 = torch.matmul(torch.cat((pcl_1, torch.ones([batch_size, sample_num, 1])), dim=-1), T)[:, :, :3]
        # display_point('pcl_trans.png', pcl_1.numpy()[0], trans_0.numpy()[0])
        # display_point('pcl.png', pcl_0.numpy()[0], trans_0.numpy()[0])

        # pcl_01 = torch.matmul(torch.cat((pcl_0, torch.ones([batch_size, sample_num, 1])), dim=-1), T01)[:, :, :3]
        # pcl_01 = (pcl_01 - center_1.unsqueeze(1)) * 2 / cube_1.unsqueeze(1)
        # img = dataset.pcl2Img(pcl_01, 128, center_1, M_1, cube_1)
        # vis_tool.debug_2d_img(img, index, './debug', 'trans_01', batch_size)
        # vis_tool.debug_2d_pose(img, joint_img_1, index, 'nyu', './debug/', 'trans_pose_01', batch_size, True)
        #
        # pcl_02 = torch.matmul(torch.cat((pcl_0, torch.ones([batch_size, sample_num, 1])), dim=-1), T02)[:, :, :3]
        # pcl_02 = (pcl_02 - center_2.unsqueeze(1)) * 2 / cube_2.unsqueeze(1)
        # img = dataset.pcl2Img(pcl_02, 128, center_2, M_2, cube_2)
        # vis_tool.debug_2d_img(img, index, './debug', 'trans_02', batch_size)
        # vis_tool.debug_2d_pose(img, joint_img_2, index, 'nyu', './debug/', 'trans_pose_02', batch_size, True)

        # debug joint error after trans
        # joint0 = dataset.pointsImgTo3D(torch.from_numpy(pred0[index * batch_size:(index + 1) * batch_size]).float())
        # joint1 = dataset.pointsImgTo3D(torch.from_numpy(pred1[index * batch_size:(index + 1) * batch_size]).float())
        # joint2 = dataset.pointsImgTo3D(torch.from_numpy(pred2[index * batch_size:(index + 1) * batch_size]).float())
        # joint10 = torch.matmul(torch.cat((joint1, torch.ones([batch_size, 21, 1])), dim=-1), T10)[:, :, :3]
        # joint20 = torch.matmul(torch.cat((joint2, torch.ones([batch_size, 21, 1])), dim=-1), T20)[:, :, :3]
        # joint = torch.median(torch.stack((joint0, joint10, joint20), dim=-1), dim=-1)[0]
        # error+=np.sqrt(np.sum((joint.numpy()[:,ImgRender.mano_layer.transfer,:][:,:11,:] -
        #                        joint_world_0.numpy()[:,:11,:])**2, axis=-1)).mean()

        # print(np.sqrt(np.sum((joint01.numpy()[:,ImgRender.mano_layer.transfer,:] - joint_world_0.numpy()[:,:12,:])**2, axis=-1)).mean())
        # joint01 = torch.matmul(torch.cat((joint01, torch.ones([batch_size, 21, 1])), dim=-1), T01)[:, :, :3]
        # print(np.sqrt(np.sum((joint01.numpy()[:,ImgRender.mano_layer.transfer,:] - joint_world_1.numpy()[:,:12,:])**2, axis=-1)).mean())

        # trans_0 = (trans_0 - center_0.unsqueeze(1)) * 2 / cube_0.unsqueeze(1)
        # pcl = (pcl_0 - center_2.unsqueeze(1)) * 2 / cube_2.unsqueeze(1)
        # img = dataset.pcl2Img(pcl, 128, center_2, M_2, cube_2)
        # pcl = dataset.Img2pcl(data_0, 128,  center_0, M_0, cube_0, 2048)
        # vis_tool.debug_2d_pose(data_2, joint_img_2, index, 'nyu', './debug/', 'view2', batch_size, True)

        # joint_uvd = dataset.xyz_nl2uvdnl_tensor(trans_0, center_0, M_0, cube_0)
        # vis_tool.debug_2d_pose(data_0, joint_uvd, index, 'nyu', './debug/', 'view0', batch_size, True)

        # noraml_img, noraml_joint, noraml_joint_xyz, normal_verts_xyz, center3d, cube_size, M= \
        #     ImgRender(model_para.cuda(), center_0.cuda(), cube_0.cuda(), mask=False)
        # vis_tool.debug_2d_img(noraml_img, index, './debug', 'render_ori', batch_size)
        #
        # noraml_img, noraml_joint, noraml_joint_xyz, normal_verts_xyz = \
        #     ImgRender.trans_render(model_para.cuda(), T.cuda(), center_2.cuda(), cube_2.cuda())
        # vis_tool.debug_2d_img(noraml_img, index, './debug', 'render_trans', batch_size)
        #

        # pcl, pcl_normal = dataset.uvdImg2xyzImg(data_1, center_1, M_1, cube_1)
        # pcl_1to0 = torch.matmul(torch.cat((pcl.reshape(batch_size, 3, -1).permute(0, 2, 1), torch.ones([batch_size, 128*128, 1])), dim=-1), T)[:, :, :3]
        # pcl_1to0 = (pcl_1to0 - center_0.unsqueeze(1)) / cube_0.unsqueeze(1) * 2
        # img_1to0 = dataset.xyz_nl2uvdnl_tensor(pcl_1to0, center_0, M_0, cube_0).permute(0,2,1).view(batch_size,3,128,128)

        # img_1tp0 = dataset.xyzImg2uvdImg(pcl_1to0.permute(0,2,1).view(batch_size,3,128,128), 128, center_0, M_0, cube_0)
        # vis_tool.debug_2d_img(img_1to0, index, './debug', 'img', batch_size)

        # print(T)
        # joint_world_0, joint_world_1, joint_world_2 = joint_world_0[0].numpy(),joint_world_1[0].numpy(),joint_world_2[0].numpy()
        #         # joint_world_0 = np.concatenate((joint_world_0, np.ones([14, 1])), axis=1)[:4]
        #         # joint_world_1 = np.concatenate((joint_world_1, np.ones([14, 1])), axis=1)[:4]
        #         # print(solve(joint_world_0, joint_world_1))
        # break
        print(index)
    print('done')


def multiView_center_generate():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 512
    # joint_num = 14
    sample_num = 1024
    phase = 'test'
    dataset = nyu_multiView_loader(root, phase, aug_para=[0, 0, 0], center_type='refine')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)

    centers_xyz_0 = []
    centers_xyz_1 = []
    centers_xyz_2 = []
    T10s = []
    T20s = []
    for index, data in enumerate(dataloader):
        data_0, joint_0, joint_img_0, center_0, M_0, cube_0, \
        data_1, joint_1, joint_img_1, center_1, M_1, cube_1, \
        data_2, joint_2, joint_img_2, center_2, M_2, cube_2 = data
        joint_num = joint_0.size(1)
        current_batch_size = data_0.size(0)
        ones_pad = torch.ones([current_batch_size, joint_num, 1])
        joint_world_0 = joint_0 * cube_0.unsqueeze(1) / 2 + center_0.unsqueeze(1)
        joint_world_1 = joint_1 * cube_1.unsqueeze(1) / 2 + center_1.unsqueeze(1)
        joint_world_2 = joint_2 * cube_2.unsqueeze(1) / 2 + center_2.unsqueeze(1)
        temp0 = torch.cat((joint_world_0, ones_pad), dim=-1)[:, :4]
        temp1 = torch.cat((joint_world_1, ones_pad), dim=-1)[:, :4]
        temp2 = torch.cat((joint_world_2, ones_pad), dim=-1)[:, :4]
        # T01 = torch.linalg.lstsq(temp0, temp1).solution
        # T02 = torch.linalg.lstsq(temp0, temp2).solution
        # torch.dist(T01, temp0.pinverse() @ temp1)
        temp0 = temp0.numpy()
        temp1 = temp1.numpy()
        temp2 = temp2.numpy()
        T01 = []
        T02 = []
        T10 = []
        T20 = []
        for T_index in range(current_batch_size):
            T01.append(np.linalg.lstsq(temp0[T_index], temp1[T_index])[0])
            T02.append(np.linalg.lstsq(temp0[T_index], temp2[T_index])[0])
            T10.append(np.linalg.lstsq(temp1[T_index], temp0[T_index])[0])
            T20.append(np.linalg.lstsq(temp2[T_index], temp0[T_index])[0])
        T01 = torch.from_numpy(np.stack(T01, axis=0)).float()
        T02 = torch.from_numpy(np.stack(T02, axis=0)).float()
        T10 = torch.from_numpy(np.stack(T10, axis=0)).float()
        T20 = torch.from_numpy(np.stack(T20, axis=0)).float()

        center_xyz_0 = center_0.unsqueeze(1)
        center_xyz_1 = torch.matmul(torch.cat((center_xyz_0, torch.ones([current_batch_size, 1, 1])), dim=-1), T01)[:, :, :3]
        center_xyz_2 = torch.matmul(torch.cat((center_xyz_0, torch.ones([current_batch_size, 1, 1])), dim=-1), T02)[:, :, :3]

        centers_xyz_0.append(center_xyz_0)
        centers_xyz_1.append(center_xyz_1)
        centers_xyz_2.append(center_xyz_2)
        T10s.append(T10)
        T20s.append(T20)
        print(index)
    np.savetxt('./T_{}_10_refined.txt'.format(phase), torch.cat(T10s, dim=0).numpy().reshape([-1, 16]),  fmt='%.6f')
    np.savetxt('./T_{}_20_refined.txt'.format(phase), torch.cat(T20s, dim=0).numpy().reshape([-1, 16]),  fmt='%.6f')

    # np.savetxt('./center_{}_0_refined.txt'.format(phase), torch.cat(centers_xyz_0, dim=0).numpy().reshape([-1, 3]),  fmt='%.3f')
    # np.savetxt('./center_{}_1_refined.txt'.format(phase), torch.cat(centers_xyz_1, dim=0).numpy().reshape([-1, 3]),  fmt='%.3f')
    # np.savetxt('./center_{}_2_refined.txt'.format(phase), torch.cat(centers_xyz_2, dim=0).numpy().reshape([-1, 3]), fmt='%.3f')
    print('done')


def draw_GCNBone():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    sample_num = 1024
    dataset = nyu_multiView_loader(root, 'test', aug_para=[10, 0.2, 180], center_type='refine')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()

    for index, data in enumerate(dataloader):
        data_0, joint_0, joint_img_0, center_0, M_0, cube_0, \
        data_1, joint_1, joint_img_1, center_1, M_1, cube_1, \
        data_2, joint_2, joint_img_2, center_2, M_2, cube_2 = data

        # for render img
        model_para = model_para.unsqueeze(0).repeat(3,1,1).view(3*batch_size, -1)
        cube = cube_0.unsqueeze(0).repeat(3, 1, 1).view(3 * batch_size, -1)
        img, joint_uvd_gt, joint_xyz_gt, verts_xyz_gt, center_s, cube_s, M_s = ImgRender(model_para.cuda(), None, cube.cuda(), mask=False, all_joints=True)

        # vis_tool.debug_2d_pose(torch.ones([batch_size*3,1,512,512]), joint_uvd_gt, index, 'mano', './debug/', 'pseudo0', batch_size, True)


def pytorch3d_renderSphere():
    root = '/home/pfren/dataset/hand/msra'
    batch_size = 1
    mano_model_path = '../MANO/'
    # dataset = nyu_loader(root, 'test', percent=1, view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    dataset = msra_loader(root, 'test', center_type='joint_mean', aug_para=[0, 0, 0])

    ImgRender = Render(mano_model_path, 'msra', dataset.paras, dataset.ori_img_size).cuda()

    label_index = 0
    model_para_label = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-alljoint-P2M-Depth-ICP-ICPJoint/MANO_result_1_0.txt')[label_index:label_index+1,:]
    model_para_label = torch.from_numpy(model_para_label).cuda().float()
    joint_uvd, mesh_uvd = ImgRender.get_mesh_uvd(model_para_label)
    cam = model_para_label[:,-4:]
    joint_uvd = (joint_uvd - cam[:,1:4])/ cam[:,0:1]
    vis_tool.debug_2d_pose(torch.ones([1, 1, 128, 128]), joint_uvd, 0, 'mano', './debug/', 'pseudo0', batch_size, True)

    # index = 500
    # img, _, joint_xyz, joint_uvd, _, center, M, cube, _, _ = dataset.__getitem__(index)
    # img, center, M, cube = img.cuda().unsqueeze(0), center.cuda().unsqueeze(0), M.cuda().unsqueeze(0), cube.cuda().unsqueeze(0)
    # joint_xyz, joint_uvd = joint_xyz.cuda(), joint_uvd.cuda()
    #
    # label_index = 500
    # model_para_label = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-alljoint-P2M-Depth-ICP-ICPJoint/MANO_result_1_0.txt')[label_index:label_index+1,:]
    # model_para_label = torch.from_numpy(model_para_label).cuda().float()
    # noraml_img, noraml_joint, noraml_joint_xyz, normal_verts_xyz, _, _, _ = ImgRender(model_para_label, center, cube, all_joints=True)
    # vis_tool.debug_2d_pose(noraml_img, noraml_joint, index, 'mano', './debug/', 'pseudo0', batch_size, True)
    #
    # view = torch.Tensor([[np.pi/2, 0, 0]]).cuda()
    # noraml_img, noraml_joint, noraml_joint_xyz, normal_verts_xyz, _, _, _ = ImgRender(model_para_label, center, cube, augmentView=view)
    # vis_tool.debug_2d_pose(noraml_img, noraml_joint, index, 'mano', './debug/', 'pseudo1', batch_size, True)
    #

    # img = model_img_label
    # _, pcl = dataset.uvdImg2xyzImg(img, center, M, cube)
    # pcl = pcl.view(batch_size, 3, -1).permute(0, 2, 1)
    # segment = ImgRender.mano_layer.seg_pcl(model_joint_xyz_label, model_joint_xyz_label, model_mesh_label, pcl)
    # segment = segment.view(batch_size, 1, 128, 128)
    # segment = torch.where(img.lt(0.99), segment, torch.zeros_like(segment))
    # segment = torch.where(segment.eq(1) | segment.eq(4) | segment.eq(7) | segment.eq(10) | segment.eq(13)| segment.eq(14),
    #                       torch.zeros_like(segment), segment)
    #
    # img_select = torch.where(segment.gt(0), img, torch.ones_like(img))
    # pcl = dataset.Img2pcl(img_select, 128, center, M, cube, 1024)
    #
    # id_to_color = vis_tool.get_segmentJointColor()
    # segment_img = id_to_color[segment.squeeze(1).long().detach().cpu()]
    # cv2.imwrite('./debug/segment.png', segment_img[0])
    #
    # segment = ImgRender.mano_layer.seg_pcl(model_joint_xyz_label, model_joint_xyz_label, model_mesh_label, pcl)
    # display_pcl_segment('./debug/pcl_segment.png', pcl.detach().cpu().numpy()[0], segment.detach().cpu().numpy()[0])
    #
    # loss, dis, shpere_idx, sphere, sphere_r = ImgRender.mano_layer.JointICP(model_joint_xyz_label, model_joint_xyz_label, model_mesh_label, pcl)
    #
    # from pytorch3d.structures import Pointclouds
    # from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
    # from pytorch3d.renderer import (
    #     look_at_view_transform,
    #     PerspectiveCameras,
    #     PointsRasterizationSettings,
    #     PointsRenderer,
    #     PulsarPointsRenderer,
    #     PointsRasterizer,
    #     AlphaCompositor,
    #     NormWeightedCompositor
    # )
    # sphere_world = sphere * cube / 2 + center
    # sphere_radii = sphere_r * cube / 2
    # point_cloud = Pointclouds(points=sphere_world)
    #
    # fx, fy, px, py = 588.03, 587.07, 320., 240.
    # R = torch.eye(3).unsqueeze(0)
    # R[:, 0, 0] = -1
    # R[:, 1, 1] = -1
    # T = torch.zeros(3).unsqueeze(0)
    # cameras = PerspectiveCameras(
    #     focal_length=((fx, fy),),  # (fx, fy)
    #     principal_point=((px, py),),  # (px, py)
    #     image_size=((640, 480),),  # (imwidth, imheight)
    #     device="cuda",
    #     R=R.cuda(), T=T.cuda(),
    # )
    # raster_settings = PointsRasterizationSettings(
    #     image_size=640,
    #     radius=0.003,
    #     points_per_pixel=10
    # )
    #
    # renderer = PulsarPointsRenderer(
    #     rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
    #     n_channels=4
    # ).cuda()
    #
    # images = renderer(point_cloud, gamma=(1e-4,),
    #                   bg_col=torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32))


def draw_sdf():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 1
    mano_model_path = '../MANO/'

    dataset = nyu_loader(root, 'test', center_type='joint_mean', aug_para=[0, 0, 0])
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, dataset.ori_img_size).cuda()

    index = 0
    img,joint_xyz, joint_uvd, center, M, cube = dataset.__getitem__(index)
    img, center, M, cube = img.cuda().unsqueeze(0), center.cuda().unsqueeze(0), M.cuda().unsqueeze(0), cube.cuda().unsqueeze(0)

    label_index = 0
    model_para_label = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-alljoint-P2M-Depth-ICP-ICPJoint/MANO_result_1_0.txt')[
                       label_index:label_index + 1, :]
    model_para_label = torch.from_numpy(model_para_label).cuda().float()
    _, _, _, model_mesh = ImgRender.noraml_render(model_para_label, center, cube)

    from sdf import SDF
    sdf = SDF()
    faces = ImgRender.mano_layer.joint_faces[0].int()
    vertex = model_mesh[:, ImgRender.mano_layer.vertex_joint_index_list[0], :]
    phi = sdf(faces, vertex, grid_size=40)
    phi = phi[0].cpu().numpy()
    normal_phi = phi / phi.max()
    mycolormap = plt.get_cmap('plasma')
    colorsvalues = mycolormap(normal_phi)
    colorsvalues[:, :, :, 3] = 0.2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(phi,  facecolors=colorsvalues, edgecolor=None, shade=False,)
    plt.savefig('./volex.png', bbox_inches='tight', pad_inches=0, dpi=150)


def debug_intersect():
    from mesh_intersection.mano_intersection_loss import mano_intersection_loss
    import  trimesh
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', (588.03, 587.07, 320., 240.), (640, 480)).cuda()
    loss_term = mano_intersection_loss(ImgRender.mano_layer.faces, ImgRender.mano_layer.face_segment,
                                       ImgRender.mano_layer.face_parent).cuda()

    input_mesh = trimesh.load('/home/pfren/pycharm/TIP2021/mesh_intersection/hand.obj')
    vertices = torch.tensor(input_mesh.vertices, dtype=torch.float32).cuda()
    vertices = vertices.unsqueeze(0).repeat(1, 1, 1)
    print(loss_term(vertices))


def save_part():
    # mano_model_path = '../MANO/'
    # ImgRender = Render(mano_model_path, 'nyu', (588.03, 587.07, 320., 240.), (640, 480)).cuda()
    # model_para = np.loadtxt('/home/pfren/pycharm/hand_mixed/checkpoint/nyu/finetune-alljoint-P2M-Depth-ICP-ICPJoint/MANO_result_1_0.txt')
    # model_para = torch.from_numpy(model_para[0:1, :]).cuda().float()
    # _, mesh = ImgRender.get_mesh_xyz(model_para)
    # faces = ImgRender.mano_layer.faces.cpu().numpy()
    # mesh = mesh.cpu().numpy().squeeze()

    # path = './test.obj'
    # with open(path, 'w') as fp:
    #     for v in mesh:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #     for f in faces + 1:
    #         fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    # for part_index in range(16):
    #     path = './obj/part%d.obj'%(part_index)
    #     with open(path, 'w') as fp:
    #         for v in mesh:
    #             fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #         for face_index, f in enumerate(faces+1):
    #             if part_index == ImgRender.mano_layer.face_segment[face_index]:
    #                 fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    # 创建水密性hand part
    # edge_vertex_id = [[274,260,258,172,169,168,138,137,263,186],
    #                [225, 167, 166, 48, 49, 156, 58, 59, 87, 213],
    #                [297, 296, 295, 294, 299, 300, 301, 341, 340, 298],
    #                [75, 246, 277, 269, 270, 187, 185, 262, 228, 288,],
    #                 [389,365,363,362,373,359,358,376,377,394,],
    #                   [412, 411, 406, 407, 408, 409, 410, 452, 453, 413],
    #                   [606,607,596,597,612,611,627,593,592,608],
    #                   [602,603,622,617,589,587,586,599,583,582],
    #                   [641,640,639,634,635,636,637,638,680,682],
    #                   [76,141,197,198,162,163,290,276,247],
    #                   [499,504,487,486,470,471,483,474,475,477],
    #                   [523, 522,517,518,519,520,521, 551, 565, 567],
    #                   [105, 286, 253, 252, 248, 123, 126, 266, 10, 30, 29],
    #                   [707, 708, 709, 710, 711, 753, 755, 714, 713, 712]]
    #
    # face_edge_id = [[0, 3, 6, 9, 12],
    #                 [0, 1], [1, 2], [2],
    #                 [3, 4], [4, 5], [5],
    #                 [6, 7], [7, 8], [8],
    #                 [9, 10], [10, 11], [11],
    #                 [12, 13], [13]]
    # normal_direct = [[-1,-1,-1,-1,-1],
    #                  [1, 1], [-1, 1], [-1],
    #                  [1, -1], [1, -1], [1],
    #                  [1, -1], [1, -1], [1],
    #                  [1, 1], [-1, -1], [1],
    #                  [1, -1], [1]]
    # for part_index in range(15):
    #     path = './obj/part%d.obj' % (part_index)
    #     points = []
    #     face = []
    #     with open(path, 'r') as fp:
    #         while 1:
    #             line = fp.readline()
    #             if not line:
    #                 break
    #             strs = line.split(" ")
    #             if strs[0] == "v":
    #                 points.append(np.array([float(strs[1]), float(strs[2]), float(strs[3])]))
    #             if strs[0] == "f":
    #                 face.append([int(strs[1]), int(strs[2]), int(strs[3])])
    #     points_num = len(points)
    #     points_np = np.array(points)
    #     for vertex in edge_vertex_id:
    #         points.append(points_np[vertex, :].mean(0))
    #
    #     for i, face_id in enumerate(face_edge_id[part_index]):
    #         vertex_id_list = edge_vertex_id[face_id]
    #         vertex_len = len(vertex_id_list)
    #         if normal_direct[part_index][i] == 1:
    #             for index, vertex_id in enumerate(vertex_id_list):
    #                 face.append([vertex_id + 1, vertex_id_list[(index + 1) % vertex_len] + 1, face_id + points_num + 1])
    #         else:
    #             for index, vertex_id in enumerate(vertex_id_list):
    #                 face.append([vertex_id + 1, vertex_id_list[(index - 1) % vertex_len] + 1, face_id + points_num + 1])
    #
    #     save_path = './waterObj/part%d.obj'%(part_index)
    #     with open(save_path, 'w') as fp:
    #         for v in points:
    #             fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #         for face_index, f in enumerate(face):
    #             fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    # 转化为PKL格式
    import pickle
    datas = {}
    for part_index in range(15):
        path = './waterObj/part%d.obj' % (part_index)
        points = []
        faces = []
        with open(path, 'r') as fp:
            while 1:
                line = fp.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    points.append(np.array([float(strs[1]), float(strs[2]), float(strs[3])]))
                if strs[0] == "f":
                    faces.append(np.array([int(strs[1]), int(strs[2]), int(strs[3])]))

        points = np.stack(points, axis=0)
        faces = np.stack(faces, axis=0)-1
        vertex_id = np.unique(faces.reshape(-1))
        face_map = []
        for face in faces:
            face_map.append(np.array([np.where(vertex_id == face[0])[0][0],
                                      np.where(vertex_id == face[1])[0][0],
                                      np.where(vertex_id == face[2])[0][0]]))
        vertex_map = points[vertex_id, :]
        save_path = './waterObjMap/part%d.obj' % (part_index)
        with open(save_path, 'w') as fp:
            for v in vertex_map:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for face_index, f in enumerate(face_map):
                fp.write('f %d %d %d\n' % (f[0]+1, f[1]+1, f[2]+1))

        datas['v-%d' % part_index] = vertex_id
        datas['f-%d' % part_index] = np.stack(face_map, axis=0)

    with open("./MANO_PART.pkl", "wb") as f:
        pickle.dump(datas, f)
    f.close()


def debug_trimesh_intersection():
    import trimesh
    # import os
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # from mesh_to_sdf import sample_sdf_near_surface, mesh_to_voxels
    part_mesh = []
    for part_index in range(15):
        path = './waterObjMap/part%d.obj' % (part_index)
        part_mesh.append(trimesh.load_mesh(path))
    index_tip = part_mesh[3]
    index_mip = part_mesh[2]
    inter = index_mip.intersection(index_tip, engine="scad")
    print(inter.volume)

    # index_tip_vox = index_tip.voxelized(pitch=0.01)
    # obj_points = index_tip_vox.points
    # save_path = './points.obj'
    # with open(save_path, 'w') as fp:
    #     for v in obj_points:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    # inside = index_mip.contains(obj_points)
    # volume = inside.sum() * np.power(0.01, 3)
    # print(volume)
    # voxels = mesh_to_voxels(index_tip, 64, pad=True)


def get_trans_M(theta, scale, trans):
    a11 = scale * torch.cos(theta)
    a12 = -torch.sin(theta)
    a21 = torch.sin(theta)
    a22 = scale * torch.cos(theta)
    a1 = torch.stack((a11, a21), dim=-1)
    a2 = torch.stack((a12, a22), dim=-1)
    return torch.stack((a1, a2, trans), dim=-1)


def debug_CCSSL():
    from CCSSL.consistency import prediction_check_pytorch
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    dataset = nyu_loader(root, 'test', type='synth', view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    for index, data in enumerate(dataloader):
        img, j3d_xyz, j3d_uvd, center, M, cube = data
        offset = joint2offset(j3d_uvd, img, 0.8, 128)
        joint_pd = offset2joint_softmax(offset, img, 0.8)
        vis_tool.debug_2d_pose(img, joint_pd, 0, 'nyu', './debug', 'ori', batch_size, True)

        rot = torch.rand([batch_size]) * 360
        scale = (torch.rand([batch_size]) - 0.5) * 2 * 0.2
        # scale = 1.2 * torch.ones([batch_size])
        trans = (torch.rand([batch_size, 2]) - 0.5) * 2 * 0.2

        M = get_trans_M(rot, 1 + scale, trans)
        gird = F.affine_grid(M, img.size())
        img_trans = F.grid_sample(img - 1, gird, mode='nearest', padding_mode='zeros') + 1


        # gird = F.affine_grid(M, offset.size())
        # offset_trans = F.grid_sample(offset, gird, mode='bilinear', padding_mode='zeros')
        # joint_tran = offset2joint_softmax(offset_trans, img_trans, 0.8)

        # M = get_trans_M(rot, 1 - scale, trans)
        # joint_uv = j3d_uvd[:, :, :2]
        # joint_uv = (j3d_uvd[:, :, :2] + 1) * 64
        # joint_tran = torch.matmul(joint_uv, M)
        # joint_tran = joint_tran / 64 - 1
        vis_tool.debug_2d_pose(img_trans, joint_tran, 0, 'nyu', './debug', 'trans', batch_size, True)


        # img_list = prediction_check_pytorch(img[0:1], None, 10)
        # imgs = torch.cat(img_list, dim=0)
        # vis_tool.debug_2d_img(imgs, 0,'./debug', 'rot', 10)
        break
    print('done')


def joint2offset(joint, img, kernel_size, feature_size):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        _,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint.reshape(joint.size(0),-1,1,1).repeat(1,1,feature_size,feature_size)
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1)
        offset = joint_feature - coords
        offset = offset.view(batch_size, joint_num, 3, feature_size, feature_size)
        dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=2)+1e-8)
        offset_norm = (offset / (dist.unsqueeze(2)))
        heatmap = (kernel_size - dist)/kernel_size
        mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size).float()
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1)


def offset2joint_softmax(offset, depth, kernel_size, scale=100):
    device = offset.device
    batch_size, joint_num, feature_size, feature_size = offset.size()
    joint_num = int(joint_num / 4)
    if depth.size(-1) != feature_size:
        depth = F.interpolate(depth, size=[feature_size, feature_size])
    offset_unit = offset[:, :joint_num * 3, :, :].contiguous()
    heatmap = offset[:, joint_num * 3:, :, :].contiguous()
    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
    mask = depth.lt(0.99).float().view(batch_size, 1, feature_size, feature_size)
    offset_mask = (offset_unit * mask).view(batch_size, joint_num, 3, -1)
    heatmap_mask = (heatmap * mask).view(batch_size, joint_num, -1)
    normal_heatmap = F.softmax(heatmap_mask * scale, dim=-1)

    dist = kernel_size - heatmap_mask * kernel_size
    joint = torch.sum(
        (offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * normal_heatmap.unsqueeze(2).repeat(1, 1, 3, 1),
        dim=-1)
    return joint

def get_rot_mat(theta):
    a00 = torch.cos(theta)
    a01 = -torch.sin(theta)
    a10 = torch.sin(theta)
    a11 = torch.cos(theta)
    a = torch.stack((torch.stack([a00, a01], dim=-1), torch.stack([a10,a11], dim=-1)), dim=1)
    a_z = torch.zeros([a00.size(0), 2, 1])
    return torch.cat((a,a_z), dim=-1)

def get_AT_mat(theta, trans, scale):
    batch_size = theta.size(0)
    device = theta.device
    a00 = torch.cos(theta)
    a01 = -torch.sin(theta)
    a10 = torch.sin(theta)
    a11 = torch.cos(theta)
    a = torch.stack((torch.stack([a00, a01], dim=-1), torch.stack([a10, a11], dim=-1)), dim=1)
    a_z = torch.zeros([a00.size(0), 2, 1]).to(device)
    rot = torch.cat((a, a_z), dim=-1)
    rot_s = rot * scale
    rot_pad = torch.Tensor([0,0,1]).to(device).type_as(theta).reshape([1,1,3]).repeat([batch_size,1,1])
    rot_s = torch.cat((rot_s,rot_pad), dim=1)
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    eye[:, 0:2, 2:3] = trans
    return torch.matmul(eye, rot_s)

def aug_img(img, joint, theta=180, trans=0, scale=0.1):
    batch_size, _, img_size, img_size = img.size()
    joint_num = joint.size(1)
    device = img.device
    batch_rot = torch.rand([batch_size]) * theta / 180 * np.pi
    batch_trans = (torch.ones([batch_size, 2, 1])*2 - 1) * trans / img_size
    batch_scale = (torch.rand([batch_size, 1, 1])*2 - 1) * scale
    M = get_AT_mat(-batch_rot, -batch_trans, 1 - batch_scale)
    grid = F.affine_grid(M[:, 0:2, :], img.size())
    img_trans = F.grid_sample(img, grid, padding_mode='border', mode='nearest')

    M = get_AT_mat(batch_rot, batch_trans, 1 + batch_scale)
    joint_trans = joint.clone()
    joint_trans_temp = torch.cat((joint_trans[:,:,:2], torch.ones([batch_size,joint_num,1]).to(device)), dim=-1).unsqueeze(-1)
    joint_trans[:, :, :2] = torch.matmul(M.unsqueeze(1), joint_trans_temp)[:,:,0:2,0]
    return img_trans, joint_trans, M

def debug_trans():
    from data.transform import get_affine_transform, affine_transform

    from CCSSL.consistency import prediction_check_pytorch
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    dataset = nyu_loader(root, 'test', type='synth', view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    for index, data in enumerate(dataloader):
        img, j3d_xyz, j3d_uvd, center, M, cube = data
        vis_tool.debug_2d_pose(img, j3d_uvd, 0, 'nyu', './debug', 'ori', batch_size, True)

        img_trans, M = rand_aug(img)
        joint_trans = trans_joint(j3d_uvd, M)

        vis_tool.debug_2d_pose(img_trans, joint_trans, 0, 'nyu', './debug', 'trans', batch_size, True)

        M_INV = Inverse_M(M)
        joint_trans_back = trans_joint(j3d_uvd, M_INV)
        vis_tool.debug_2d_pose(img, joint_trans_back, 0, 'nyu', './debug', 'trans_back', batch_size, True)

        break

from data.transform import get_affine_transform, affine_transform


def rand_aug(img, r=180, s=0.2, t=0.2):
    B, _, H, W = img.size()
    rot = np.random.rand(B) * r
    scale = 1 + (np.random.rand(B) - 0.5) * 2 * s
    trans = (np.random.rand(B, 2) - 0.5) * 2 * t
    img_np = img.cpu().numpy()
    center = np.array([H//2, W//2])
    img_list = []
    M_list = []
    for j in range(B):
        s = scale[j]
        r = rot[j]
        t = trans[j]
        M = get_affine_transform(center, s, r, (H, W), t)
        img_trans = cv2.warpAffine(img_np[j, 0], M, (H, W), flags=cv2.INTER_NEAREST, borderValue=1)
        img_list.append(img_trans)
        M_list.append(M)
    img_np = np.stack(img_list, 0)[:, np.newaxis, :, :]
    M_np = np.stack(M_list, 0)
    return torch.from_numpy(img_np).to(img.device), M_np


def trans_joint(joint, M):
    B, J, _ = joint.size()
    joint_np = joint.cpu().detach().numpy()
    joint_np[:, :, :2] = (joint_np[:, :, :2] + 1) * 64
    for i in range(B):
        for j in range(J):
            joint_np[i, j, 0:2] = affine_transform(joint_np[i, j, 0:2], M[i])
    joint_np[:, :, :2] = joint_np[:, :, :2] / 64 - 1
    return torch.from_numpy(joint_np).to(joint.device)


def inverse_M(M):
    a = np.array([0, 0, 1]).reshape(1, 1, 3)
    a = np.tile(a, (M.shape[0], 1, 1))
    M_cat = np.concatenate((M, a), axis=1)
    return torch.from_numpy(M_cat).inverse().numpy()[:, :2, :]


def debug_coll():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 1
    mano_model_path = '../MANO/'
    # dataset = nyu_loader(root, 'test', view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    ImgRender = Render(mano_model_path, 'nyu', (463.889, 463.889, 320.00000, 240.00000), (640, 480)).cuda()
    model_paras = np.loadtxt('/home/pfren/pycharm/TIP2021/checkpoint/nyu/finetune_ResNet_stage_18_adamw_centerTyperefine_coord_weight_100_deconv_weight_1_step_size_10_CubeSize_250_offset_0.8test/MANO_result_1_0.txt')
    joint_xyz, mesh_xyz = ImgRender.get_mesh_xyz(torch.from_numpy(model_paras[2321]).unsqueeze(0).float().cuda())
    coll = ImgRender.mano_layer.calculate_coll(joint_xyz, mesh_xyz).sum()
    print(coll)
    shpere_c, shpere_r = ImgRender.mano_layer.get_sphere_radius(joint_xyz, mesh_xyz)
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    display_sphere(shpere_c.squeeze().cpu().numpy(), shpere_r.squeeze().cpu().numpy(), ax=ax, transpose=False)
    mesh = mesh_xyz.cpu().numpy().squeeze()

    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], color='red', s=1)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.savefig('hh.png', bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()

    # coll_list = []
    # for index in range(len(model_paras)):
    #     joint_xyz, mesh_xyz = ImgRender.get_mesh_xyz(torch.from_numpy(model_paras[index]).unsqueeze(0).float().cuda())
    #     coll = ImgRender.mano_layer.calculate_coll(joint_xyz, mesh_xyz).sum()
    #     coll_list.append(coll.item())
    #     print(index)
    # np.savetxt('./coll.txt', np.array([coll_list]), fmt='%.8f')


def compare_transferNet_nyu():
    root = '/home/pfren/dataset/hand/nyu'
    batch_size = 32
    dataset = nyu_loader(root, 'train', type='synth', view=0, center_type='joint_mean', aug_para=[0, 0, 0])
    print(dataset.__len__())
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', dataset.paras, (640, 480)).cuda()

    transferNet = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
    model_dict = torch.load('../../pytorch-CycleGAN-and-pix2pix/checkpoints/nyu_background_consis_cyclegan-40epoch/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
    transferNet.load_state_dict(model_dict)
    transferNet.eval()

    transferNet_cycle = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
    model_dict = torch.load('../../pytorch-CycleGAN-and-pix2pix/checkpoints/cyclegan/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
    transferNet_cycle.load_state_dict(model_dict)
    transferNet_cycle.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    for index, data in enumerate(dataloader):
        # model_para, cube = data
        img, synth_img, j3d_xyz, j3d_uvd, center, M, cube, model_para = data
        # synth_img = synth_img.cuda()
        synth_img = ImgRender.M_render(model_para.cuda(), center.cuda(), cube.cuda(), M.cuda(), mask=False)

        img_transfer = transferNet(synth_img)
        # img_cycle = transferNet_cycle(synth_img)
        vis_tool.debug_2d_img(synth_img, index, './debug', 'synth', batch_size)
        vis_tool.debug_2d_img(img_transfer, index, './debug', 'transfer', batch_size)
        # vis_tool.debug_2d_img(img_cycle, index, './debug', 'cycle', batch_size)
        vis_tool.debug_2d_img(img, index, './debug', 'real', batch_size)
        # vis_tool.debug_2d_heatmap(torch.abs(synth_img-img_transfer), index, './debug', 128, 'trans_diff', True)
        # vis_tool.debug_2d_heatmap(torch.abs(synth_img-img_cycle), index, './debug', 128, 'cycle_diff', True)
        vis_tool.debug_2d_heatmap(torch.abs(img.cuda()-img_transfer), index, './debug', 128, 'trans_diff_r', True)
        # vis_tool.debug_2d_heatmap(torch.abs(img.cuda()-img_cycle), index, './debug', 128, 'cycle_diff_r', True)

        print(index)
        if index == 2:
            break

    print('done')

@ torch.no_grad()
def compare_transferNet_mano():
    root = '/home/pfren/dataset/hand/hands20'
    batch_size = 32
    dataset = hands_modelPara_loader(root, 'train')
    print(dataset.__len__())
    mano_model_path = '../MANO/'
    ImgRender = Render(mano_model_path, 'nyu', (588.03, 587.07, 320., 240.), (640, 480)).cuda()

    transferNet = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
    model_dict = torch.load('../../pytorch-CycleGAN-and-pix2pix/checkpoints/nyu_background_consis_cyclegan-40epoch/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
    transferNet.load_state_dict(model_dict)
    transferNet.eval()

    transferNet_cycle = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
    model_dict = torch.load('../../pytorch-CycleGAN-and-pix2pix/checkpoints/nyu_ori_cyclegan-40epoch/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
    transferNet_cycle.load_state_dict(model_dict)
    transferNet_cycle.eval()

    transferNet_task = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
    model_dict = torch.load('../../pytorch-CycleGAN-and-pix2pix/checkpoints/ada-10/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
    transferNet_task.load_state_dict(model_dict)
    transferNet_task.eval()

    diff_consis = 0
    diff_task = 0
    diff_cycle = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    for index, data in enumerate(dataloader):
        model_para, cube = data
        model_para, cube = model_para.cuda(), cube.cuda()
        augmentShape = torch.randn([batch_size, 10]).cuda() * 3
        augmentCenter = (torch.rand([batch_size, 3]).cuda() - 0.5) * 40
        augmentSize = (1 + (torch.rand([batch_size, 1]).cuda() - 0.5) * 0.1) # 250 mm
        augmentView = torch.rand([model_para.size(0), 3]).cuda() * np.pi * 2

        img, joint_uvd_gt, mesh_uvd_gt, joint_xyz_gt, mesh_xyz_gt, center, cube, M = \
            ImgRender(model_para, None, cube, augmentView=augmentView, augmentShape=augmentShape,
                           augmentCenter=augmentCenter, augmentSize=augmentSize, mask=False)

        img_transfer = transferNet(img)
        img_cycle = transferNet_cycle(img)
        img_task = transferNet_task(img)

        # vis_tool.debug_2d_img(img, index, './debug', 'synth', batch_size)
        # vis_tool.debug_2d_img(img_transfer, index, './debug', 'consis', batch_size)
        # vis_tool.debug_2d_img(img_cycle, index, './debug', 'cycle', batch_size)
        # vis_tool.debug_2d_img(img_task, index, './debug', 'task', batch_size)
        # vis_tool.debug_2d_heatmap(torch.abs(img-img_transfer), index, './debug', 128, 'trans_transfer', True)
        # vis_tool.debug_2d_heatmap(torch.abs(img-img_cycle), index, './debug', 128, 'trans_cycle', True)
        # vis_tool.debug_2d_heatmap(torch.abs(img-img_task), index, './debug', 128, 'trans_task', True)

        diff_consis += inter_diff(img, img_transfer)
        diff_cycle += inter_diff(img, img_cycle)
        diff_task += inter_diff(img, img_task)
        print(index)
        if index == 5:
            break
    print(diff_consis)
    print(diff_cycle)
    print(diff_task)
    print('done')

def inter_diff(img, img_trans):
    B = img.size(0)
    img = img.view(B, -1)
    img_trans = img_trans.view(B, -1)
    mask1 = img.lt(0.99)
    mask2 = img_trans.lt(0.99)
    inter = mask1 & mask2
    return (torch.abs(img_trans-img).sum(-1) / inter.float().sum(-1)).mean()

def result2video():
    # NYU
    batch_size = 32
    img_size = 256
    dataset_name = 'nyu'
    root = '/home/pfren/dataset/hand/'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWrite_l = cv2.VideoWriter('./label2.avi', apiPreference=0, fourcc=fourcc, fps=30, frameSize=(img_size, img_size))
    videoWrite_p = cv2.VideoWriter('./pred2.avi', apiPreference=0, fourcc=fourcc, fps=30, frameSize=(img_size, img_size))

    dataset = nyu_loader(root + dataset_name, 'test', view=0, img_size=img_size, center_type='refine')
    Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    joint_preds = np.loadtxt('/home/pfren/pycharm/hand_mixed/data/result_0_0.txt').reshape([-1, 21, 3])
    for index, data in enumerate(Loader):
        data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
        joint_pred = joint_preds[index*batch_size:(index+1)*batch_size]
        joint_pred = batchtransformPoints2D(joint_pred[:, :, 0:2], M.numpy()) / 128 - 1
        joint_pred = torch.from_numpy(joint_pred)
        # pose_label = vis_tool.debug_2d_pose(data, joint_img, index, dataset_name, './debug', 'label', batch_size, True)
        pose_pred = vis_tool.debug_2d_pose(data, joint_pred, index, 'nyu', './debug', 'pred', batch_size, True)
        # pose_pred = vis_tool.debug_2d_pose(data, joint_pred, index, 'mano', './debug', 'pred', batch_size, True)
        # pose_img = vis_tool.debug_2d_img(data, index, './debug', 'img', batch_size)
        # for frame in pose_label:
        #     videoWrite_l.write(frame.astype('uint8'))
        # for frame in pose_pred:
        #     videoWrite_p.write(frame.astype('uint8'))
        print(index)
    print('done')

    # batch_size = 32
    # img_size = 256
    # dataset_name = 'icvl'
    # root = '/home/pfren/dataset/hand/'
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # videoWrite = cv2.VideoWriter('./pred.avi', apiPreference=0, fourcc=fourcc, fps=10, frameSize=(img_size, img_size))
    #
    # dataset = flip_icvl_loader(root + dataset_name, 'test', img_size=img_size)
    # Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # joint_preds = np.loadtxt('/home/pfren/pycharm/hand_mixed/data/result_pixel.txt').reshape([-1,21,3])
    # for index, data in enumerate(Loader):
    #     # data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
    #     data, joint, joint_img, center, M, cube = data
    #     joint_pred = joint_preds[index*batch_size:(index+1)*batch_size]
    #     joint_pred = batchtransformPoints2D(joint_pred[:, :, 0:2], M.numpy()) / 128 - 1
    #     joint_pred = torch.from_numpy(joint_pred)
    #     # pose_img = vis_tool.debug_2d_pose(data, joint_img, index, dataset_name, './debug', 'label', batch_size, True)
    #     pose_img = vis_tool.debug_2d_pose(data, joint_pred, index, 'MANO', './debug', 'pred', batch_size, True)
    #     # for frame in pose_img:
    #     #     videoWrite.write(frame.astype('uint8'))
    #     print(index)
    # print('done')

    # batch_size = 32
    # img_size = 256
    # dataset_name = 'msra'
    # root = '/home/pfren/dataset/hand/'
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # videoWrite = cv2.VideoWriter('./pred.avi', apiPreference=0, fourcc=fourcc, fps=30, frameSize=(img_size, img_size))
    #
    # dataset = msra_loader(root + dataset_name, 'test', img_size=img_size)
    # Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # joint_preds = np.loadtxt('/home/pfren/pycharm/hand_mixed/data/result_pixel.txt').reshape([-1, 21, 3])
    # for index, data in enumerate(Loader):
    #     # data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
    #     data, joint, joint_img, center, M, cube = data
    #     joint_pred = joint_preds[index*batch_size:(index+1)*batch_size]
    #     joint_pred = batchtransformPoints2D(joint_pred[:, :, 0:2], M.numpy()) / 128 - 1
    #     joint_pred = torch.from_numpy(joint_pred)
    #     # pose_img = vis_tool.debug_2d_pose(data, joint_img, index, dataset_name, './debug', 'label', batch_size, True)
    #     # pose_img = vis_tool.debug_2d_pose(data, joint_pred, index, dataset_name, './debug', 'pred', batch_size, True)
    #     pose_img = vis_tool.debug_2d_pose(data, joint_pred, index, 'MANO', './debug', 'pred', batch_size, True)
    #     # pose_img = vis_tool.debug_2d_img(data, index, './debug', 'img', batch_size)
    #     for frame in pose_img:
    #         videoWrite.write(frame.astype('uint8'))
    #     print(index)
    # print('done')

def result2Img():
    # NYU
    batch_size = 32
    img_size = 256
    dataset_name = 'nyu'
    root = '/home/pfren/dataset/hand/'

    dataset = nyu_loader(root + dataset_name, 'test', view=0, img_size=img_size, center_type='refine')
    Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    joint_preds = np.loadtxt('/home/pfren/pycharm/TIP2021/checkpoint/nyu/Finetune-PM-V3/result_1_0.txt').reshape([-1, 21, 3])
    for index, data in enumerate(Loader):
        data, joint, joint_img, center, M, cube = data
        joint_pred = joint_preds[index*batch_size:(index+1)*batch_size]
        joint_pred = batchtransformPoints2D(joint_pred[:, :, 0:2], M.numpy()) / 128 - 1
        joint_pred = torch.from_numpy(joint_pred)
        pose_pred = vis_tool.debug_2d_pose(data, joint_pred, index, 'mano', './debug', '', batch_size, True)
        print(index)
    print('done')

    # batch_size = 32
    # img_size = 256
    # dataset_name = 'icvl'
    # root = '/home/pfren/dataset/hand/'
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # videoWrite = cv2.VideoWriter('./pred.avi', apiPreference=0, fourcc=fourcc, fps=10, frameSize=(img_size, img_size))
    #
    # dataset = flip_icvl_loader(root + dataset_name, 'test', img_size=img_size)
    # Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # joint_preds = np.loadtxt('/home/pfren/pycharm/hand_mixed/data/result_pixel.txt').reshape([-1,21,3])
    # for index, data in enumerate(Loader):
    #     # data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
    #     data, joint, joint_img, center, M, cube = data
    #     joint_pred = joint_preds[index*batch_size:(index+1)*batch_size]
    #     joint_pred = batchtransformPoints2D(joint_pred[:, :, 0:2], M.numpy()) / 128 - 1
    #     joint_pred = torch.from_numpy(joint_pred)
    #     # pose_img = vis_tool.debug_2d_pose(data, joint_img, index, dataset_name, './debug', 'label', batch_size, True)
    #     pose_img = vis_tool.debug_2d_pose(data, joint_pred, index, 'MANO', './debug', 'pred', batch_size, True)
    #     # for frame in pose_img:
    #     #     videoWrite.write(frame.astype('uint8'))
    #     print(index)
    # print('done')

    # batch_size = 32
    # img_size = 256
    # dataset_name = 'msra'
    # root = '/home/pfren/dataset/hand/'
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # videoWrite = cv2.VideoWriter('./pred.avi', apiPreference=0, fourcc=fourcc, fps=30, frameSize=(img_size, img_size))
    #
    # dataset = msra_loader(root + dataset_name, 'test', img_size=img_size)
    # Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # joint_preds = np.loadtxt('/home/pfren/pycharm/hand_mixed/data/result_pixel.txt').reshape([-1, 21, 3])
    # for index, data in enumerate(Loader):
    #     # data, pcl_sample, joint, joint_img, model_para, center, M, cube, rotation, rotation = data
    #     data, joint, joint_img, center, M, cube = data
    #     joint_pred = joint_preds[index*batch_size:(index+1)*batch_size]
    #     joint_pred = batchtransformPoints2D(joint_pred[:, :, 0:2], M.numpy()) / 128 - 1
    #     joint_pred = torch.from_numpy(joint_pred)
    #     # pose_img = vis_tool.debug_2d_pose(data, joint_img, index, dataset_name, './debug', 'label', batch_size, True)
    #     # pose_img = vis_tool.debug_2d_pose(data, joint_pred, index, dataset_name, './debug', 'pred', batch_size, True)
    #     pose_img = vis_tool.debug_2d_pose(data, joint_pred, index, 'MANO', './debug', 'pred', batch_size, True)
    #     # pose_img = vis_tool.debug_2d_img(data, index, './debug', 'img', batch_size)
    #     for frame in pose_img:
    #         videoWrite.write(frame.astype('uint8'))
    #     print(index)
    # print('done')

def debug_CCSSL():
    batch_size = 32
    dataset_name = 'nyu'
    root = '/home/pfren/dataset/hand/'

    dataset = nyu_CCSSL_loader(root + dataset_name, 'test', view=1, center_type='refine')
    center = dataset.jointImgTo3D(dataset.joint3DToImg(dataset.refine_center_xyz,flip=-1)).reshape([-1, 3])
    np.savetxt('/home/pfren/dataset/hand/nyu/center_test_1_refine_xyz.txt',center, fmt='%.3f')
    # Loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # for index, data in enumerate(Loader):
    #     img, joint, joint_img, center, M, cube, joint_p, weight = data
    #     vis_tool.debug_visible_joint(img, joint_p, weight, index, 'mano', './debug', 'pesudo')
    #     vis_tool.debug_2d_pose(img, joint, index, 'mano', './debug', 'gt',batch_size, True)
    #     if index>5:
    #         break


if __name__ == "__main__":
    # save_label()
    # save_nyu_cam()
    # vis_dataset()
    # render_msra()
    # vis_multiView_dataset()
    # calculate_coll()
    # mix_dataset()
    # opt_depth()
    # vis_sphere()
    # vis_nyu_Joint()
    # vis_Joint()[0]
    # vis_msra_Joint()
    # vis_Render()
    # shrec_data()
    # DHG_data()
    # campare_icploss()
    # result2video()
    # vis_heatmap()
    # multiView_calibration()
    # multiView_center_generate()
    # draw_GCNBone()
    # pytorch3d_renderSphere()
    # debug_GCN_pool()
    # draw_coll_bone()
    # vis_Render()
    # draw_sdf()
    # debug_noise_depthImage()
    # debug_intersect()
    # save_part()
    # debug_trimesh_intersection()
    # debug_CCSSL()
    # debug_trans()
    # debug_coll()
    compare_transferNet_nyu()
    # compare_transferNet_mano()
    # result2Img()
    # debug_CCSSL()