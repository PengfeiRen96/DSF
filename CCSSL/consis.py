import cv2
import numpy as np
import torch

import torch.nn.functional as F
from CCSSL.transforms import *
from data.transform import get_affine_transform, affine_transform

def prediction_check(inp, model, num_transform=5, num_kpts=21):
    """
    Input:
        Image: 1x128x128
    Output:
        generated_kpts: 21x3
    """
    device = inp.device
    # equivariant consistency
    _,H,W = inp.size()
    s0 = 1
    sf = 0.25
    rf = 30
    joint = 0
    confidence = 0
    for i in range(num_transform):
        img = inp.clone().squeeze()
        if i == 0:
            s = s0
            rot = 0
        else:
            s = s0*np.random.randn()*sf + 1
            s = np.clip(s, 1-sf, 1+sf)
            rot = np.random.randn()*rf
            rot = np.clip(rot, -2*rf, 2*rf)

        M = get_affine_transform(np.array([H//2, W//2]), s, rot, (H, W))
        img_trans = cv2.warpAffine(img.cpu().numpy(), M, (H, W), flags=cv2.INTER_NEAREST, borderValue=1)
        img_trans = torch.from_numpy(img_trans).view(1, 1, H, W).to(device)

        outputs = model(img_trans)
        pixel_pd, _ = outputs[-1]
        joint_uvd = offset2joint_softmax(pixel_pd, img_trans, 0.8)
        confidence += torch.max(torch.softmax(30*pixel_pd[:, num_kpts * 3:].view(num_kpts, -1), dim=-1), dim=-1)[0]

        INV_M = inverse_M(M[np.newaxis, :, :])
        joint_uvd = trans_joint(joint_uvd, INV_M)
        joint += joint_uvd

    joint = joint / num_transform
    confidence = confidence / num_transform
    return joint, confidence


def offset2joint_softmax(offset, depth, kernel_size, scale=30):
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


def rand_aug(img, r=180, s=0.2, t=0.2):
    B, _, H, W = img.size()
    rot = np.random.rand(B) * r
    scale = 1 + (np.random.rand(B) - 0.5) * 2 * s
    trans = (np.random.rand(B, 2) - 0.5) * 2 * t
    img_np = img.cpu().numpy()
    center = np.array([H // 2, W // 2])
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


if __name__ == '__main__':
    img = torch.rand([1, 128, 128])
    prediction_check(img, img)

