import numpy as np
import torch

import torch.nn.functional as F
from util.vis_tool import *

#generate feature module
class GFM:
    def __init__(self):
        self.sigmod = torch.nn.Sigmoid()
        self.softmax2d = torch.nn.Softmax2d()
        self.softmax = torch.nn.Softmax(dim=-1)

    def joint2offset(self, joint, img, kernel_size, feature_size):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        _,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint.reshape(joint.size(0),-1,1,1).repeat(1,1,feature_size,feature_size)
        # mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        # mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
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
        # offset 和 heatmap 用作refine是否需要进行mask，这样是否会损失信息
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1)
        # return torch.cat((offset_norm.view(batch_size,-1,feature_size,feature_size), heatmap),dim=1).float()

    def offset2joint_softmax(self, offset, depth, kernel_size, scale=30):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 4)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num*3:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        normal_heatmap = F.softmax(heatmap_mask*scale, dim=-1)

        dist = kernel_size - heatmap_mask * kernel_size
        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1,1,3,1) + coords) * normal_heatmap.unsqueeze(2).repeat(1,1,3,1), dim=-1)
        return joint

    def feature2joint(self, img, pixel_pd, feature_types, feature_paras):
        for feature_index, feature_type in enumerate(feature_types):
            if feature_type == 'offset':
                joint = self.offset2joint_softmax(pixel_pd, img, feature_paras[feature_index])

        return joint

    def joint2feature(self, joint, img, feature_paras, feature_size, feature_types):
        device = img.device
        batch_size, joint_num, _ = joint.size()
        for feature_index, feature_type in enumerate(feature_types):
            if 'offset' == feature_type:
                feature = self.joint2offset(joint, img, feature_paras[feature_index], feature_size)
        return feature
