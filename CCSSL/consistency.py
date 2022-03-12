import cv2
import numpy as np
import torch

import torch.nn.functional as F
from CCSSL.imutils import *
from CCSSL.transforms import *


# def prediction_check(inp, model, num_transform=5, num_kpts=21):
#     """
#     Input:
#         Image: 1x128x128
#     Output:
#         generated_kpts: 21x3
#     """
#
#     # equivariant consistency
#     s0 = 128/100.0
#     sf = 0.25
#     rf = 30
#     c = torch.Tensor((64, 64))
#     score_map_avg = np.zeros((1,num_kpts*4,64,64))
#     for i in range(num_transform):
#         img = inp.clone()
#         if i == 0:
#             s = s0
#             rot = 0
#         else:
#             s = s0*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf).item()
#             rot = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf).item()
#
#         img = crop(img, c, s, [128, 128], rot)
#
#         outputs = model(img.unsqueeze(0).to(inp.device))
#         output, _ = outputs[-1]
#         score_map = output[-1].cpu() if type(output) == list else output.cpu()
#
#         # rotate and scale score_map back
#         feat_map = score_map.squeeze(0).detach().cpu().numpy()
#         for j in range(feat_map.shape[0]):
#             feat_map_j = feat_map[j]
#             M = cv2.getRotationMatrix2D((32, 32), -rot, 1)
#             feat_map_j = cv2.warpAffine(feat_map_j, M, (64, 64))
#             feat_map_j = cv2.resize(feat_map_j, None, fx=s*100.0/128.0,fy=s*100.0/128.0, interpolation=cv2.INTER_LINEAR)
#
#             if feat_map_j.shape[0]<64:
#                 start = 32-feat_map_j.shape[0]//2
#                 end = start+feat_map_j.shape[0]
#                 score_map_avg[0][j][start:end, start:end] += feat_map_j
#             else:
#                 start = feat_map_j.shape[0]//2-32
#                 end = feat_map_j.shape[0]//2+32
#                 score_map_avg[0][j] += feat_map_j[start:end, start:end]
#
#     score_map_avg = score_map_avg/num_transform
#     confidence_score = np.max(score_map_avg[:, num_kpts*3:], axis=(0,2,3))
#
#     confidence = confidence_score.astype(np.float32)
#     score_map_avg = torch.Tensor(score_map_avg).to(inp.device)
#
#     preds = offset2joint_softmax(score_map_avg, inp.unsqueeze(0), 0.8)
#     preds = preds.squeeze(0)
#     pts = preds.clone().cpu().numpy()
#
#     generated_kpts = np.zeros((num_kpts, 3)).astype(np.float32)
#     generated_kpts[:, :3] = pts
#     # generated_kpts[:, 3] = confidence
#     generated_kpts = torch.from_numpy(generated_kpts)
#     return generated_kpts

def prediction_check_pytorch(inp, model, num_transform=5, num_kpts=21):
    """
    Input:
        Image: 1x128x128
    Output:
        generated_kpts: 21x3
    """
    device = inp.device
    # equivariant consistency
    B,_,H,W = inp.size()
    s0 = 1
    sf = 0.25
    rf = 30
    score_map_avg = torch.zeros((B, num_kpts*4, H // 2, W//2)).to(device)
    img_list = []
    for i in range(num_transform):
        img = inp.clone()
        if i == 0:
            s = s0*torch.ones([B]).to(device)
            rot = torch.zeros([B]).to(device)
        else:
            s = s0*torch.randn([B]).mul_(sf).add_(1).clamp(1-sf, 1+sf).to(device)
            rot = torch.randn([B]).mul_(rf).clamp(-2*rf, 2*rf).to(device)

        M = get_trans_M(rot, s)
        gird = F.affine_grid(M, inp.size())
        img = F.grid_sample(img-1, gird, mode='nearest', padding_mode='zeros')+1
    #     img_list.append(img)
    # return img_list
    #
        outputs = model(img)
        output, _ = outputs[-1]
        # rotate and scale score_map back
        M = get_Inverse_M(M)
        gird = F.affine_grid(M, output.size())
        score_map_avg += F.grid_sample(output, gird, mode='nearest', padding_mode='zeros')

    score_map_avg = score_map_avg/num_transform
    confidence_score = torch.max(score_map_avg[:, num_kpts*3:].view(B,num_kpts,-1), dim=-1)

    preds = offset2joint_softmax(score_map_avg, inp, 0.8)
    return preds


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

def get_trans_M(theta, scale):
    B = theta.size(0)
    a11 = scale * torch.cos(theta)
    a12 = -torch.sin(theta)
    a21 = torch.sin(theta)
    a22 = scale * torch.cos(theta)
    a1 = torch.stack((a11, a21), dim=-1)
    a2 = torch.stack((a12, a22), dim=-1)
    a3 = torch.zeros([B, 2]).to(theta.device)
    return torch.stack((a1, a2, a3), dim=-1)


def get_Inverse_M(M):
    B = M.size(0)
    pad_val = torch.Tensor([0, 0, 1]).view(1, 1, 3).repeat(B, 1, 1).to(M.device)
    return torch.inverse(torch.cat((M,pad_val), dim=1))[:, :2, :]


if __name__ == '__main__':
    img = torch.rand([1, 128, 128])
    prediction_check(img, img)

