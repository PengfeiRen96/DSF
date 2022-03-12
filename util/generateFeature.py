import numpy as np
import torch
# import edt
# import skfmm
from scipy.ndimage.morphology import distance_transform_edt
import torch.nn.functional as F
from util.vis_tool import *

def Matr(axis, theta):
    device = theta.device
    batchsize = theta.size()[0]
    M = torch.eye(4, requires_grad=False).repeat(batchsize, 1, 1)
    if axis == 0:
        M[:, 1, 1] = torch.cos(theta)
        M[:, 1, 2] = -torch.sin(theta)
        M[:, 2, 1] = torch.sin(theta)
        M[:, 2, 2] = torch.cos(theta)
    elif axis == 1:
        M[:, 0, 0] = torch.cos(theta)
        M[:, 0, 2] = -torch.sin(theta)
        M[:, 2, 0] = torch.sin(theta)
        M[:, 2, 2] = torch.cos(theta)
    elif axis == 2:
        M[:, 0, 0] = torch.cos(theta)
        M[:, 0, 1] = -torch.sin(theta)
        M[:, 1, 0] = torch.sin(theta)
        M[:, 1, 1] = torch.cos(theta)
    else:
        M[:, axis - 3, 3] = theta
    return M.to(device)

#generate feature module
class GFM:
    def __init__(self):
        self.sigmod = torch.nn.Sigmoid()
        self.softmax2d = torch.nn.Softmax2d()
        self.softmax = torch.nn.Softmax(dim=-1)
    def rotation_points(self, points, rot):
        device = points.device
        points_rot = points.view(points.size(0), -1, 3)
        points_rot = torch.cat((points_rot, torch.ones(points_rot.size(0), points_rot.size(1), 1).to(device)),dim=-1)
        theta_x = torch.tensor(rot[:, 0]).float().to(device)
        theta_y = torch.tensor(rot[:, 1]).float().to(device)
        theta_z = torch.tensor(rot[:, 2]).float().to(device)

        points_rot = torch.matmul(points_rot, Matr(2, theta_z))
        points_rot = torch.matmul(points_rot, Matr(0, theta_x))
        points_rot = torch.matmul(points_rot, Matr(1, theta_y))

        return points_rot[:, :, 0:3]

    def pcl2img(self, pcl, img_size):
        divce = pcl.device
        pcl = pcl.permute(0,2,1)
        img_pcl = torch.ones([pcl.size(0), img_size*img_size]).to(divce) * -1
        index_x = torch.clamp(torch.floor((pcl[:, :, 0] + 1) / 2 * img_size), 0, img_size - 1).long().to(divce)
        index_y = torch.clamp(torch.floor((1 - pcl[:, :, 1]) / 2 * img_size), 0, img_size - 1).long().to(divce)*img_size
        index = index_y + index_x
        batch_index = torch.arange(pcl.size(0)).unsqueeze(-1).expand(-1, index.size(1))
        img_pcl[batch_index, index] = (pcl[:, :, 2] + 1) / 2
        return img_pcl.view(pcl.size(0), 1, img_size, img_size)

    def joint2heatmap2d(self, joint, img, std, heatmap_size):
        # joint depth is norm[-1,1]
        divce = joint.device
        img_down = F.interpolate(img, size=[heatmap_size, heatmap_size])
        batch_size, joint_num, _ = joint.size()
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        mesh_x = torch.from_numpy(xx).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)+0.5
        mesh_y = torch.from_numpy(yy).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)+0.5
        joint_ht = torch.zeros_like(joint).to(divce)
        joint_ht[:, :, 0] = (joint[:, :, 0] + 1) / 2 * heatmap_size
        joint_ht[:, :, 1] = (joint[:, :, 1] + 1) / 2 * heatmap_size
        joint_x = joint_ht[:, :, 0].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        joint_y = joint_ht[:, :, 1].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        heatmap = torch.exp(-(torch.pow((mesh_x - joint_x) / std, 2) + torch.pow((mesh_y.to(divce) - joint_y) / std, 2)))
        mask = heatmap.ge(0.01).float() * img_down.lt(0.99).float().view(batch_size, 1, heatmap_size, heatmap_size)
        return heatmap, mask

    def joint2plainoffset(self, joint, img, kernel_size, feature_size):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        batch_size,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint[:,:,:2].contiguous().view(batch_size,joint_num*2,1,1).repeat(1,1,feature_size,feature_size)
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).repeat(1, joint_num, 1, 1).to(device)
        offset = joint_feature - coords
        offset = offset.view(batch_size,joint_num,2,feature_size,feature_size)
        dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=2)+1e-8)
        offset_norm = (offset / (dist.unsqueeze(2)))
        heatmap = (kernel_size - dist)/kernel_size
        mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1), mask

    def joint2depthoffset(self, joint, img, mask, feature_size):
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        batch_size,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint[:,:,2].contiguous().view(batch_size,joint_num,1,1).repeat(1,1,feature_size,feature_size)
        offset = joint_feature - img.view(batch_size,1,feature_size,feature_size)
        offset = offset.view(batch_size,joint_num,1,feature_size,feature_size)
        offset_mask = (offset*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        return offset_mask

    def joint2iso(self, joint, img, mask, feature_size):
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        batch_size,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint[:,:,2].contiguous().view(batch_size,joint_num,1,1).repeat(1,1,feature_size,feature_size)
        offset = joint_feature - img.view(batch_size,1,feature_size,feature_size)
        offset = offset.view(batch_size,joint_num,1,feature_size,feature_size)
        offset_mask = (offset*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        return offset_mask

    def joint2distance(self, joint, pcl, feature_size):
        device = joint.device
        batch_size, joint_num, _ = joint.size()
        offset = joint.view(batch_size,joint_num,1,3) - pcl.view(batch_size,1,-1,3)
        offset = offset.view(batch_size,joint_num,-1,3)
        dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=-1)+1e-8)

        dist_topmin = torch.topk(dist.view(batch_size,joint_num,-1),20,largest=False)[0]
        dist_mean = torch.mean(dist_topmin,dim=-1)
        return dist_mean

    def joint2visible(self, joint,joint_uvd, pcl, feature_size):
        device = joint.device
        batch_size, joint_num, _ = joint.size()
        offset = joint.view(batch_size,joint_num,1,3) - pcl.view(batch_size,1,-1,3)
        offset = offset.view(batch_size,joint_num,-1,3)
        dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=-1)+1e-8)

        dist_topmin = torch.topk(dist.view(batch_size,joint_num,-1),20,largest=False)[0]
        # dist_topmin = dist_topmin[:,1:,:] #remove wirst
        dist_mean = torch.mean(dist_topmin,dim=-1)
        joint_mean = torch.FloatTensor([0.2287, 0.1506, 0.1166, 0.1131, 0.1054, 0.1061, 0.1403, 0.1118, 0.0944,0.1087, 0.0966, 0.0875, 0.1108, 0.0992, 0.0889, 0.1066, 0.0948, 0.0862, 0.0921, 0.0881, 0.0948])
        visible = (dist_mean - joint_mean*2.0) > 0

        joint_uv = (joint_uvd[:,:,0:2] + 1) / 2 * feature_size
        joint_d = joint_uvd[:,:,2:]
        joint_uv_offset = joint_uv.view(batch_size,joint_num,1,2) - joint_uv.view(batch_size,1,joint_num,2)
        joint_d_offset = joint_d.view(batch_size,joint_num,1) - joint_d.view(batch_size,1,joint_num)
        joint_uv_dis = torch.sqrt(torch.sum(joint_uv_offset * joint_uv_offset,dim=-1))
        unvisible = torch.sum((joint_uv_dis < 3) * (joint_d_offset<-0.05),dim=-1) > 0
        # unvisible =
        # visible = torch.cat((torch.zeros([batch_size,1]).byte(),visible),dim=-1)
        # return (unvisible + visible)>0
        return visible

    def joint2edt(self, img, joint, feature_size, kernel_size):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        _,joint_num,_ = joint.size()
        img = (1 - F.interpolate(img,size=[feature_size,feature_size]).cpu().numpy())/2
        joint_np = np.floor((joint.cpu().numpy() + 1)/2*feature_size).astype(int)
        batch_list = []
        for img_index in range(batch_size):
            joint_list = []
            image_hmap = img[img_index,0]
            mask = (1e-4 > image_hmap)
            # contour = cv2.Canny((image_hmap * 255.0).astype(np.uint8), 50, 150) / 255
            # mask = mask | contour.astype(bool)
            masked_edt = np.ma.masked_array(img[img_index,0], mask)
            edt_out = distance_transform_edt(mask) * 2 / feature_size
            edt = image_hmap - edt_out
            for joint_index in range(joint_num):
                pose = joint_np[img_index,joint_index]
                l,t = np.clip(pose[1]-1,0,feature_size-3), np.clip(pose[0]-1,0,feature_size-3)
                r,d = l+2,t+2
                val = edt[l:r, t:d].max()
                ring = np.ones_like(image_hmap)

                ring[l:r, t:d] = 0
                ring = - distance_transform_edt(ring)
                ring = np.ma.masked_array(ring, np.logical_and((val > ring), mask))
                ring = np.max(ring) - ring
                ring[~mask] = 0
                phi = image_hmap + ring + 1

                phi[l:r, t:d] = 0.
                df = skfmm.distance(phi, dx=1e-1)
                df_max = np.max(df)
                df = (df_max - df) / df_max
                df[mask] = 0.
                df[1. < df] = 0.
                joint_list.append(df) # edt_l.append(edt)
            # for joint_index in range(joint_num):
            #     pose = np.clip(joint_np[img_index,joint_index], 0, feature_size-1)
            #     val = edt[pose[1], pose[0]]
            #     if 0 > val:
            #         ring = np.ones_like(image_hmap)
            #         ring[pose[1], pose[0]] = 0
            #         ring = - distance_transform_edt(ring)
            #         ring = np.ma.masked_array(ring, np.logical_and((val > ring), mask))
            #         ring = np.max(ring) - ring
            #         ring[~mask] = 0
            #         phi = image_hmap + ring + 1
            #     else:
            #         phi = masked_edt.copy()
            #     phi[pose[1], pose[0]] = 0.
            #     df = skfmm.distance(phi, dx=1e-1)
            #     df_max = np.max(df)
            #     df = (df_max - df) / df_max
            #     df[mask] = 0.
            #     df[1. < df] = 0.
            #     joint_list.append(df) # edt_l.append(edt)
            batch_list.append(np.stack(joint_list, axis=0))
        edts = torch.from_numpy(np.stack(batch_list,axis=0)).to(device)
        # edts  = self.softmax(30*edts.view(batch_size,joint_num,-1)).view(batch_size,joint_num,feature_size,feature_size)
        return edts.float()

    def JointConf2offset(self, joint, img, conf, kernel_size, feature_size):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        _,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint.reshape(joint.size(0),-1,1,1).repeat(1,1,feature_size,feature_size)
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords,img),dim=1).repeat(1, joint_num, 1, 1)
        offset = joint_feature - coords
        offset = offset.view(batch_size,joint_num,3,feature_size,feature_size)
        dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=2)+1e-8)
        offset_norm = (offset / (dist.unsqueeze(2)))
        conf_kernel = conf.unsqueeze(-1) * kernel_size + kernel_size
        # conf_kernel = conf.unsqueeze(-1)
        heatmap = (conf_kernel - dist) / conf_kernel
        mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1)

    def joint2offset_noNormal(self, joint, img, kernel_size, feature_size):
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
        heatmap = kernel_size - dist
        mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1)

    def joint2Confoffset_noNormal(self, joint, img, conf, kernel_size, feature_size):
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
        heatmap = (kernel_size*conf + conf).unsqueeze(-1) - dist
        mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1)

    # normal heatmap
    def Heatmap2Confusing(self, heatmap):
        finger_num = 5
        batch_size, _, height, width = heatmap.size()
        device = heatmap.device
        bone_list = [[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20]]
        bone_heatmaps = torch.Tensor([]).to(device)
        for index in range(finger_num):
            bone_heatmap = torch.mean(heatmap[:,bone_list[index],:,:],dim=1,keepdim=True)
            bone_heatmaps = torch.cat((bone_heatmaps,bone_heatmap),dim=1)

        mask = torch.eye(finger_num).view(1, finger_num, finger_num, 1).repeat(batch_size, 1, 1, height * width).to(device)
        feats_intersection = bone_heatmaps.unsqueeze(1) * bone_heatmaps.unsqueeze(2)
        feats_intersection_mask = feats_intersection.view(batch_size, finger_num, finger_num, height * width) * (1 - mask)
        confusing_heatmap = feats_intersection_mask.view(batch_size,finger_num*finger_num, height, width)
        confusing_heatmap = torch.sum(confusing_heatmap,dim=1,keepdim=True)/2.0
        return bone_heatmaps,confusing_heatmap

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


    def joint2jointoffset(self, joint, jointPoint, kernel_size):
        batch_size, joint_num, _ = joint.size()
        node_num = jointPoint.size(1)
        joint_feature = joint.reshape(batch_size, 1, joint_num, 3)
        coords = jointPoint.reshape(batch_size, node_num, 1, 3)
        offset = joint_feature - coords
        dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1)+1e-8)
        offset_norm = (offset / (dist.unsqueeze(-1)))
        heatmap = (kernel_size - dist)/kernel_size
        mask = heatmap.ge(0).float()
        offset_norm = offset_norm.view(batch_size, node_num, joint_num*3)
        offset_norm_mask = offset_norm * mask.view(batch_size, node_num, joint_num, 1).repeat(1,1,1,3).view(batch_size, node_num, joint_num*3)
        heatmap_mask = heatmap*mask
        return (torch.cat((offset_norm_mask, heatmap_mask), dim=-1)).permute(0,2,1)
        # return torch.cat((offset_norm.view(batch_size,-1,feature_size,feature_size), heatmap),dim=1).float()

    def offset2joint(self, offset, depth, kernel_size, topk=30):
        device = offset.device
        batch_size, joint_num, feature_size,feature_size = offset.size()
        joint_num = int(joint_num / 4)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous().view(batch_size,joint_num,3,-1)
        heatmap = offset[:, joint_num*3:, :, :].contiguous().view(batch_size,joint_num,-1)
        # mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        # mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        value, index = torch.topk(heatmap, topk, dim=-1)
        index = index.unsqueeze(2).repeat(1,1,3,1)
        value = value.unsqueeze(2).repeat(1,1,3,1)
        offset_unit_select = torch.gather(offset_unit, -1, index)
        coords_select = torch.gather(coords, -1, index)
        dist = kernel_size - value * kernel_size
        joint = torch.sum((offset_unit_select*dist + coords_select)*value, dim=-1)
        joint = joint / (torch.sum(value, -1)+1e-8) # avoid 0
        return joint

    def jointoffset2joint(self, offset, jointPoint, kernel_size, scale=30):
        batch_size, offset_num, node_num = offset.size()
        joint_num = offset_num // 4
        offset_unit = offset[:, :joint_num*3, :].contiguous().reshape([batch_size, joint_num, 3, node_num])
        heatmap = offset[:, joint_num * 3:joint_num * 4, :].contiguous().reshape([batch_size, joint_num, 1, node_num])

        normal_weigth = F.softmax(scale*heatmap, dim=-1)
        dist = kernel_size - heatmap * kernel_size
        joint = torch.sum((offset_unit * dist.repeat(1, 1, 3, 1) + jointPoint.permute(0,2,1).unsqueeze(1)) * normal_weigth.repeat(1, 1, 3, 1), dim=-1)
        return joint

    def jointoffset2joint_weight(self, offset, jointPoint, kernel_size, scale=1):
        batch_size, offset_num, node_num = offset.size()
        joint_num = offset_num // 5
        offset_unit = offset[:, :joint_num*3, :].contiguous().reshape([batch_size, joint_num, 3, node_num])
        heatmap = offset[:, joint_num * 3:joint_num * 4, :].contiguous().reshape([batch_size, joint_num, 1, node_num])
        weight = offset[:, joint_num * 4:, :].contiguous().reshape([batch_size, joint_num, 1, node_num])

        normal_weigth = F.softmax(scale*weight, dim=-1)
        dist = kernel_size - heatmap * kernel_size
        joint = torch.sum((offset_unit * dist.repeat(1, 1, 3, 1) + jointPoint.permute(0,2,1).unsqueeze(1)) * normal_weigth.repeat(1, 1, 3, 1), dim=-1)
        return joint



    # change to with mask 2019/9/3
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

    # change to with mask 2019/9/3
    def offset2joint_selectsoftmax(self, offset, depth, kernel_size, scale=30, sample_num=1024):
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
        select_id = torch.multinomial(mask.view(batch_size,-1), sample_num, replacement=True).view(batch_size, 1, sample_num)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        offset_select = torch.gather(offset_mask, -1, select_id.unsqueeze(1).repeat(1,joint_num,3,1))
        heatmap_select = torch.gather(heatmap_mask, -1, select_id.repeat(1,joint_num,1))
        coords_select = torch.gather(coords, -1, select_id.unsqueeze(1).repeat(1,joint_num,3,1))
        normal_heatmap = F.softmax(heatmap_select*scale, dim=-1)

        dist = kernel_size - heatmap_select * kernel_size
        joint = torch.sum((offset_select * dist.unsqueeze(2).repeat(1,1,3,1) + coords_select) * normal_heatmap.unsqueeze(2).repeat(1,1,3,1), dim=-1)
        return joint

    # change to with weight 2020/4/14
    def directoffset2joint_weight(self, offset, depth):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 4)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous().reshape(batch_size, joint_num, 3, -1)
        weight = offset[:, joint_num * 3:, :, :].contiguous().reshape(batch_size,joint_num,1,-1)
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)

        joint = torch.sum((offset_unit + coords) * weight, dim=-1)
        return joint

    # change to with weight 2020/4/14
    def offset2joint_weight(self, offset, depth, kernel_size, scale=1):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 5)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
        weight = offset[:, joint_num * 4:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size, 1, feature_size,feature_size)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        weight_mask = (weight*mask).view(batch_size, joint_num, -1)
        normal_weigth = F.softmax(weight_mask*scale, dim=-1)

        dist = kernel_size - heatmap_mask * kernel_size
        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1,1,3,1) + coords) * normal_weigth.unsqueeze(2).repeat(1,1,3,1), dim=-1)
        return joint

    # change to with weight 2020/4/14
    def offset2joint_weight_sample(self, offset, depth, kernel_size, scale=10):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 5)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
        weight = offset[:, joint_num * 4:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        weight_mask = (weight*mask).view(batch_size, joint_num, -1)
        normal_weigth = F.softmax(weight_mask*scale, dim=-1)

        sample_weight = torch.ones([batch_size, 64*64]).to(device)
        sample_weight = torch.where(depth.lt(0.99).view(batch_size, -1), sample_weight, torch.zeros_like(sample_weight).to(device))
        sample_index = torch.multinomial(sample_weight.view(batch_size, -1), 1024,replacement=True).unsqueeze(1)

        heatmap_mask = torch.gather(heatmap_mask, dim=-1, index=sample_index.repeat(1,21,1))
        offset_mask = torch.gather(offset_mask, dim=-1, index=sample_index.unsqueeze(1).repeat(1, 21, 3, 1))
        normal_weigth = torch.gather(normal_weigth, dim=-1, index=sample_index.repeat(1, 21, 1))
        coords = torch.gather(coords, dim=-1, index=sample_index.unsqueeze(1).repeat(1, 21, 3, 1))

        dist = kernel_size - heatmap_mask * kernel_size
        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1,1,3,1) + coords) * normal_weigth.unsqueeze(2).repeat(1,1,3,1), dim=-1) / normal_weigth.unsqueeze(2).repeat(1,1,3,1).sum(-1)
        return joint


    # change to with weight 2020/4/14
    def offset2joint_weight_nosoftmax(self, offset, depth, kernel_size):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 5)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
        weight = offset[:, joint_num * 4:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        weight_mask = (weight*mask).view(batch_size, joint_num, -1)

        dist = kernel_size - heatmap_mask * kernel_size
        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * weight_mask.unsqueeze(2).repeat(1,1,3,1), dim=-1) / torch.sum(weight_mask, -1,keepdim=True)
        return joint


    def heatmap2joint_softmax(self, heatmap):
        device = heatmap.device
        batch_size, joint_num, feature_size, _= heatmap.size()

        # mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float()+0.5) / feature_size - 1.0
        # mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float()+0.5) / feature_size - 1.0
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = coords.view(1, 1, 2, -1).repeat(batch_size, joint_num, 1, 1).float().to(device)

        normal_heatmap = F.softmax(heatmap.view(batch_size, joint_num, -1)*30, dim=-1).unsqueeze(2).repeat(1,1,2,1).float()
        out = normal_heatmap * coords.view(batch_size, joint_num, 2, -1)
        joint = torch.sum(out, -1)
        return joint
        # heatmap_temp = heatmap.view(batch_size, joint_num, -1).unsqueeze(2).repeat(1,1,2,1)
        # out =  heatmap_temp * coords.view(batch_size, joint_num, 2, -1)
        # joint = torch.sum(out, -1) / heatmap_temp.sum(-1)
        # return joint

    def plainoffset2joint_softmax(self, offset, weight, kernel_size):
        device = offset.device
        batch_size, joint_num, feature_size,feature_size = offset.size()
        joint_num = int(joint_num / 2)
        # mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        # mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 2, -1).to(device)
        dist = kernel_size - weight * kernel_size
        dist = dist.view(batch_size, joint_num, -1).unsqueeze(2).repeat(1, 1, 2, 1)
        normal_weight = F.softmax(30*weight.view(batch_size, joint_num, -1), dim=-1)
        joint = torch.sum((offset.view(batch_size, joint_num, 2, -1) * dist + coords) * normal_weight.unsqueeze(2).repeat(1,1,2,1), dim=-1)
        return joint

    def weight_pos2joint(self, weight_pos):
        batch_size,joint_num,feature_size,feature_size = weight_pos.size()
        joint_num = int(joint_num / 4)
        weight = F.softmax(weight_pos[:, :joint_num, :, :].contiguous().view(batch_size, joint_num, 1, -1),dim=-1).repeat(1,1,3,1)
        pos = weight_pos[:, joint_num:, :, :].view(batch_size,joint_num,3,-1).contiguous()
        joint = torch.sum(weight * pos, -1)
        return joint.view(-1, joint_num, 3)

    def heatmap_depth2joint(self, pixel_pd, img):
        batch_size,joint_num, feature_size, _ = pixel_pd.size()
        batch_size,joint_num, feature_size, _ = pixel_pd.size()
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        joint_num = joint_num//2
        heatmap, depth = pixel_pd[:, joint_num:, :, :].contiguous(), pixel_pd[:, :joint_num,:,:].contiguous()
        joint_uv = self.heatmap2joint_softmax(heatmap)

        mask = heatmap.ge(0.01).float() * img_down.lt(0.99).float().view(batch_size, 1, feature_size, feature_size)
        heatmap_mask = heatmap * mask
        heatmap_mask = heatmap_mask.view(batch_size, joint_num, -1)

        normal_heatmap = F.softmax(10*heatmap_mask.view(batch_size,joint_num,-1), dim=-1)
        joint_depth = torch.sum(depth.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)
        return joint
        # joint_depth = torch.sum(depth.view(batch_size, joint_num, -1) * heatmap_mask, dim=-1)/ torch.sum(heatmap_mask, -1)
        # joint = torch.cat((joint_uv, joint_depth.unsqueeze(-1)), dim=-1)
        # return joint

    def heatmap_depthoffset2joint(self, pixel_pd, img):
        batch_size, joint_num, feature_size, _ = pixel_pd.size()
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        joint_num = joint_num//2
        heatmap, depth_offset = pixel_pd[:, joint_num:,:,:].contiguous(), pixel_pd[:, :joint_num,:,:].contiguous()
        joint_uv = self.heatmap2joint_softmax(heatmap)
        depth = img_down + depth_offset
        mask = heatmap.ge(0).float() * img_down.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        heatmap_temp = heatmap.view(batch_size, joint_num, -1) * mask.view(batch_size, joint_num, -1) + 1e-12
        depth = depth * mask
        # depth = heatmap_temp * depth.view(batch_size, joint_num, -1)
        # joint_depth = (torch.sum(depth, -1) / heatmap_temp.sum(-1)).unsqueeze(-1)
        normal_heatmap = F.softmax(heatmap_temp.view(batch_size, joint_num, -1) * 30, dim=-1).float()
        joint_depth = torch.sum(depth.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)

        return joint

    # change to with mask 2019/9/3
    def plainoffset_depth2joint(self, img, pixel_pd, kernel_size):
        batch_size, joint_num, feature_size, _ = pixel_pd.size()
        joint_num = joint_num // 4
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        offset, weight, depth = pixel_pd[:,:2*joint_num,:,:].contiguous(),pixel_pd[:,2*joint_num:3*joint_num,:,:].contiguous(),\
                              pixel_pd[:,3*joint_num:,:,:].contiguous()
        mask = img_down.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask, weight_mask, depth_mask = offset * mask, weight * mask, depth * mask
        joint_uv = self.plainoffset2joint_softmax(offset_mask, weight_mask, kernel_size)
        normal_heatmap = F.softmax(30 * weight_mask.view(batch_size, joint_num, -1), dim=-1)
        joint_depth = torch.sum(depth_mask.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)
        return joint

    # change to with mask 2019/9/3
    def plainoffset_depthoffset2joint(self, img, pixel_pd, kernel_size):
        batch_size, joint_num,feature_size,_ = pixel_pd.size()
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        joint_num = joint_num // 4
        offset,weight,depth_offset = pixel_pd[:,:2*joint_num,:,:].contiguous(),pixel_pd[:,2*joint_num:3*joint_num,:,:].contiguous(),\
                              pixel_pd[:,3*joint_num:,:,:].contiguous()
        depth = depth_offset + img_down

        mask = img_down.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask, weight_mask, depth_mask = offset * mask, weight * mask, depth * mask

        joint_uv = self.plainoffset2joint_softmax(offset_mask, weight_mask, kernel_size)
        normal_heatmap = F.softmax(30*weight_mask.view(batch_size,joint_num,-1), dim=-1)
        joint_depth = torch.sum(depth_mask.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)
        return joint

    # output size : B*4(xyz+point_type)*N
    def joint2pc(self, joint, seed = 12345, sample_point=1024, radius=0.08):
        device = joint.device
        batch_size, joint_num, _ = joint.size()

        radius = torch.rand([batch_size, joint_num, 100]).to(device)*radius
        theta = torch.rand([batch_size, joint_num, 100]).to(device)*np.pi
        phi = torch.rand([batch_size, joint_num, 100]).to(device)*np.pi

        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        type = torch.arange(1, joint_num+1).float().to(device).view(1, joint_num, 1).repeat(batch_size, 1, 100)

        point = joint.unsqueeze(-2).repeat(1,1,100,1) + torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim = -1)
        point = torch.cat((point, type.unsqueeze(-1)), dim=-1)
        point = point.view(batch_size,-1,4)
        sample = np.random.choice(point.size(1), sample_point, replace=False)
        return point[:, sample, :].permute(0, 2, 1)

    # depth[-1,1]
    def depth2map(self, depth, heatmap_size=32):
        batchsize, jointnum = depth.size()
        depthmap = ((depth + 1) / 2).contiguous().view(batchsize, jointnum, 1, 1).expand(batchsize, jointnum, heatmap_size, heatmap_size)
        return depthmap

    # select feature
    def joint2feature(self, joint, img, feature_paras, feature_size, feature_types):
        device = img.device
        all_feature = torch.Tensor().to(device)
        batch_size, joint_num, _ = joint.size()
        for feature_index, feature_type in enumerate(feature_types):
            if 'heatmap' == feature_type:
                heatmap, mask = self.joint2heatmap2d(joint, img, feature_paras[feature_index], feature_size)
                feature = heatmap
            elif 'heatmap_depth' == feature_type:
                heatmap, mask = self.joint2heatmap2d(joint, img, feature_paras[feature_index], feature_size)
                depth = torch.ones_like(heatmap).to(device) * joint[:, :, 2].view(joint.size(0), -1, 1, 1)
                depth[mask == 0] = -1
                feature = torch.cat((heatmap, depth), dim=1)
            elif 'heatmap_depthoffset' == feature_type:
                heatmap, mask = self.joint2heatmap2d(joint, img, feature_paras[feature_index], feature_size)
                depth_offset = self.joint2depthoffset(joint, img, mask, feature_size)
                feature = torch.cat((heatmap, depth_offset), dim=1)
            elif 'plainoffset_depth' == feature_type:
                plain_offset, mask = self.joint2plainoffset(joint, img, feature_paras[feature_index], feature_size)
                depth = torch.ones([joint.size(0),joint.size(1),feature_size,feature_size]).to(device) * joint[:, :, 2].view(joint.size(0), -1, 1, 1)
                feature = torch.cat((plain_offset, depth), dim=1)
            elif 'plainoffset_depthoffset' == feature_type:
                plain_offset, mask = self.joint2plainoffset(joint, img, feature_paras[feature_index], feature_size)
                depth_offset = self.joint2depthoffset(joint, img, mask, feature_size)
                feature = torch.cat((plain_offset, depth_offset), dim=1)
            elif 'offset' == feature_type or 'weight_offset' == feature_type or 'weight_offset_nosoftmax' == feature_type:
                feature = self.joint2offset(joint, img, feature_paras[feature_index], feature_size)
            elif feature_type == 'weight_pos':
                feature = joint.view(batch_size,joint_num,3,1,1).repeat(1,1,1,feature_size,feature_size)
                feature = feature.view(batch_size,-1,feature_size,feature_size)
            all_feature = torch.cat((all_feature, feature), dim=1)
        return all_feature

    # select feature
    def joint2hardCls(self, img, joint, feature_size, feature_paras):
        finger_index = [[1,6,7,8],[2,9,10,11],[3,12,13,14],[4,15,16,17],[5,18,19,20]]
        batchSize, jointNum,_ = joint.size()
        offset = self.joint2offset(joint, img, feature_paras[0], feature_size)
        distance = offset[:,jointNum*3:,:,:]
        # edt = self.joint2edt(img,joint,feature_size,feature_paras)
        heatmap = distance
        joint_cls = torch.argmax(heatmap, dim=1, keepdim=True)
        finger_cls = []
        for index in range(len(finger_index)):
            finger = heatmap[:,finger_index[index],:,:].sum(1,keepdim=True)
            finger_cls.append(finger)
        finger_cls = torch.cat(finger_cls, dim=1)
        finger_cls = torch.argmax(finger_cls, dim=1, keepdim=True)
        finger_label = torch.zeros([batchSize,len(finger_index),feature_size,feature_size]).long()
        for index in range(len(finger_index)):
            finger_label[:,index:index+1,:,:] = (finger_cls == index).long()

        joint_label = torch.zeros([batchSize,jointNum,feature_size,feature_size]).long()
        for index in range(jointNum):
            joint_label[:,index:index+1,:,:] = (joint_cls == index).long()
        all_label = (heatmap.sum(1, keepdim=True) > 0).long()
        final = torch.cat((all_label,finger_label*all_label,joint_label*all_label),dim=1)
        return final

    # select feature
    def joint2softCls(self, img, joint, feature_size, feature_paras):
        img_mask = img.lt(0.99).float()
        finger_index = [[1,6,7,8],[2,9,10,11],[3,12,13,14],[4,15,16,17],[5,18,19,20]]
        batchSize, jointNum,_ = joint.size()
        offset = self.joint2offset(joint, img, feature_paras[0], feature_size)
        distance = offset[:, jointNum * 3:, :, :]
        # edt = self.joint2edt(img, joint, feature_size, feature_paras)
        # heatmap = edt * distance
        heatmap = distance
        # normal_heatmap = self.softmax(10*heatmap.view(batchSize,jointNum,-1)).view(batchSize,jointNum,feature_size,feature_size)
        per_joint_normal = (heatmap - heatmap.view(batchSize,jointNum,-1).min(-1, keepdim=True)[0].unsqueeze(-1)) / \
                           (heatmap.view(batchSize,jointNum,-1).max(-1, keepdim=True)[0].unsqueeze(-1)  - heatmap.view(batchSize,jointNum,-1).min(-1, keepdim=True)[0].unsqueeze(-1)  + 1e-8)
        joint_cls = per_joint_normal
        finger_cls = []
        for index in range(len(finger_index)):
            finger_cls.append(per_joint_normal[:, finger_index[index], :, :].max(1, keepdim=True)[0])
        finger_cls = torch.cat(finger_cls, dim=1)

        all_label = torch.clamp(heatmap.sum(1, keepdim=True), 0, 1)
        DRM = torch.cat((finger_cls,joint_cls),dim=1)
        CRM = torch.cat((all_label, finger_cls),dim=1)
        KRM = joint_cls
        return DRM,CRM,KRM

    # select feature
    def joint2thCls(self, img, joint, feature_size, feature_paras):
        finger_index = [[1,6,7,8],[2,9,10,11],[3,12,13,14],[4,15,16,17],[5,18,19,20]]
        batchSize, jointNum,_ = joint.size()
        offset = self.joint2offset(joint, img, feature_paras[0], feature_size)
        edt = self.joint2edt(img,joint,feature_size,feature_paras[0])
        heatmap = offset[:,jointNum*3:,:,:] * edt
        normal_heatmap = self.softmax(30*heatmap.view(batchSize,jointNum,-1)).view(batchSize,jointNum,feature_size,feature_size)
        joint_cls = []
        for index in range(jointNum):
            joint = normal_heatmap[:,index:index+1,:,:]
            th = joint[joint>0].mean()*1.2
            joint_cls.append((joint > th).long())
        joint_cls = torch.cat(joint_cls, dim=1)
        finger_cls = []
        for index in range(len(finger_index)):
            finger = joint_cls[:,finger_index[index],:,:].sum(1,keepdim=True)
            finger_cls.append(torch.clamp(finger, 0, 1))
        finger_cls = torch.cat(finger_cls, dim=1)

        all_label = (torch.clamp(heatmap.sum(1, keepdim=True), 0, 1)>0).long()
        final = torch.cat((all_label,finger_cls,joint_cls),dim=1)
        return final

    def joint2boneDist(self,img,joint, feature_size,kernel_size,dataset):
        bone_list = get_sketch_setting(dataset)
        bone = []
        root = []
        child = []
        for pair in bone_list:
            bone.append(joint[:, pair[1], :] - joint[:,pair[0],:])
            root.append(joint[:, pair[0], :])
            child.append(joint[:, pair[1], :])
        bone = torch.stack(bone, dim=1)
        bone_len = torch.sqrt(torch.sum(bone*bone, dim=-1, keepdim=True)+1e-8)
        root = torch.stack(root, dim=1)
        child = torch.stack(child,dim=1)
        bone_num = bone.size(1)

        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords,img),dim=1).view(batch_size,3,-1).permute(0,2,1)

        AP = coords.view(batch_size,1,-1,3) - root.view(batch_size,bone_num,1,3)
        BP = coords.view(batch_size,1,-1,3) - child.view(batch_size,bone_num,1,3)
        AC = torch.sum(AP*bone.view(batch_size,bone_num,1,3), dim=-1, keepdim=True) / bone_len.view(batch_size,bone_num,1,1)
        R = AC / bone_len.view(batch_size,bone_num,1,1)

        dist = torch.sqrt(torch.sum(AP*AP,dim=-1,keepdim=True) - torch.sum(AC*AC,dim=-1,keepdim=True) + 1e-8)
        dist[R<0] = torch.sqrt(torch.sum(AP*AP,dim=-1,keepdim=True))[R<0]
        dist[R>1] = torch.sqrt(torch.sum(BP*BP,dim=-1,keepdim=True))[R>1]
        # dist[R<0] = kernel_size
        # dist[R>1] = kernel_size
        kernel_size = torch.ones_like(dist).to(device) * kernel_size
        kernel_size[:,0:5,:,:] = kernel_size[:,0:5,:,:]*1.5
        dist = (kernel_size - dist) / dist
        dist = dist.view(batch_size,bone_num,feature_size,feature_size)
        mask = dist.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        return dist*mask

    def joint2boneHeatmap_2d(self,img,joint, feature_size,dataset,kernel_size=0):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        bone_list = get_sketch_setting(dataset)
        if kernel_size == 0:
            kernel_size = get_joint_size(dataset)
            kernel_size = torch.from_numpy(kernel_size).to(device).view(1,joint.size(1),1).repeat(batch_size,1,1)
        bone = []
        root = []
        child = []
        root_depth = []
        child_depth = []
        root_size = []
        child_size = []
        for pair in bone_list:
            bone.append(joint[:, pair[1], :2] - joint[:,pair[0],:2])
            root.append(joint[:, pair[0], :2])
            child.append(joint[:, pair[1], :2])
            root_depth.append(joint[:, pair[0], 2:])
            child_depth.append(joint[:, pair[1], 2:])
            root_size.append(kernel_size[:,pair[0],:])
            child_size.append(kernel_size[:,pair[1],:])
        bone = torch.stack(bone, dim=1)
        root = torch.stack(root, dim=1)
        child = torch.stack(child,dim=1)
        child_depth = torch.stack(child_depth, dim=1)
        root_depth = torch.stack(root_depth, dim=1)
        root_size = torch.stack(root_size, dim=1)
        child_size = torch.stack(child_size,dim=1)
        bone_num = bone.size(1)
        bone_len = torch.sqrt(torch.sum(bone*bone, dim=-1, keepdim=True)+1e-8).view(batch_size,bone_num,1)

        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device).view(batch_size,2,-1).permute(0,2,1)

        AP = coords.view(batch_size,1,-1,2) - root.view(batch_size,bone_num,1,2)
        BP = coords.view(batch_size,1,-1,2) - child.view(batch_size,bone_num,1,2)
        AP_len = torch.sqrt(torch.sum(AP * AP, dim=-1) + 1e-8)
        AC_len = torch.sum(AP*bone.view(batch_size,bone_num,1,2), dim=-1) / bone_len
        R = AC_len / bone_len

        PC_len = torch.sqrt(AP_len*AP_len - AC_len*AC_len+ 1e-8)
        PC_len[R<0] = torch.sqrt(torch.sum(AP*AP,dim=-1))[R<0]
        PC_len[R>1] = torch.sqrt(torch.sum(BP*BP,dim=-1))[R>1]

        P_depth = (child_depth- root_depth).view(batch_size,bone_num,1) * R + root_depth.view(batch_size,bone_num,1)
        P_size = (child_size- root_size).view(batch_size,bone_num,1) * R + root_size.view(batch_size,bone_num,1)
        P_depth[R < 0] = root_depth.repeat(1,1,feature_size**2)[R<0]
        P_depth[R > 1] = child_depth.repeat(1,1,feature_size**2)[R>1]
        P_size[R < 0] = root_size.repeat(1,1,feature_size**2)[R<0]
        P_size[R > 1] = child_size.repeat(1, 1, feature_size ** 2)[R > 1]
        bone_max_size = torch.cat((root_size,child_size),dim=1).max(dim=1,keepdim=True)[0].repeat(1,1,feature_size*feature_size)
        P_size[P_size>bone_max_size] = 0
        P_depth = torch.where(PC_len>P_size, torch.full_like(P_depth,1).to(device), P_depth)

        return P_depth.view(batch_size,bone_num,feature_size,feature_size)

    def joint2boneHeatmap_direct(self,img,joint, feature_size,dataset,kernel_size=0):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        bone_list = get_sketch_setting(dataset)
        if kernel_size == 0:
            kernel_size = get_joint_size(dataset)
            kernel_size = torch.from_numpy(kernel_size).to(device).view(1,joint.size(1),1).repeat(batch_size,1,1)
        bone = []
        root = []
        child = []
        root_depth = []
        child_depth = []
        root_size = []
        child_size = []
        for pair in bone_list:
            bone.append(joint[:, pair[1], :2] - joint[:,pair[0],:2])
            root.append(joint[:, pair[0], :2])
            child.append(joint[:, pair[1], :2])
            root_depth.append(joint[:, pair[0], 2:])
            child_depth.append(joint[:, pair[1], 2:])
            root_size.append(kernel_size[:,pair[0],:])
            child_size.append(kernel_size[:,pair[1],:])
        bone = torch.stack(bone, dim=1)
        root = torch.stack(root, dim=1)
        child = torch.stack(child, dim=1)
        child_depth = torch.stack(child_depth, dim=1)
        root_depth = torch.stack(root_depth, dim=1)
        root_size = torch.stack(root_size, dim=1)
        child_size = torch.stack(child_size,dim=1)
        bone_num = bone.size(1)
        bone_len = torch.sqrt(torch.sum(bone*bone, dim=-1, keepdim=True)+1e-8).view(batch_size,bone_num,1)

        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device).view(batch_size,2,-1).permute(0,2,1)

        AP = coords.view(batch_size,1,-1,2) - root.view(batch_size,bone_num,1,2)
        BP = coords.view(batch_size,1,-1,2) - child.view(batch_size,bone_num,1,2)
        AP_len = torch.sqrt(torch.sum(AP * AP, dim=-1) + 1e-8)
        AC_len = torch.sum(AP*bone.view(batch_size,bone_num,1,2), dim=-1) / bone_len
        R = AC_len / bone_len

        PC_len = torch.sqrt(AP_len*AP_len - AC_len*AC_len + 1e-8)
        PC_len[R<0] = torch.sqrt(torch.sum(AP*AP, dim=-1))[R<0]
        PC_len[R>1] = torch.sqrt(torch.sum(BP*BP, dim=-1))[R>1]

        P_depth = (child_depth - root_depth).view(batch_size,bone_num,1) * R + root_depth.view(batch_size,bone_num,1)
        P_size = (child_size - root_size).view(batch_size,bone_num,1) * R + root_size.view(batch_size,bone_num,1)
        P_depth[R < 0] = root_depth.repeat(1,1,feature_size**2)[R<0]
        P_depth[R > 1] = child_depth.repeat(1,1,feature_size**2)[R>1]
        P_size[R < 0] = root_size.repeat(1,1,feature_size**2)[R<0]
        P_size[R > 1] = child_size.repeat(1, 1, feature_size ** 2)[R > 1]
        bone_max_size = torch.cat((root_size,child_size),dim=1).max(dim=1,keepdim=True)[0].repeat(1,1,feature_size*feature_size)
        P_size[P_size>bone_max_size] = 0
        P_depth = torch.where(PC_len>P_size, torch.full_like(P_depth,1).to(device), P_depth).view(batch_size,bone_num,feature_size,feature_size)
        P_depth_all = torch.min(P_depth,dim=1)[0].unsqueeze(1)
        return torch.cat((P_depth_all, P_depth), dim=1).float()

    def HandJoint2ModelJoint(self, joint):
        wirst_right =  1.2*(joint[:,0:1,:] - joint[:,1:2,:]) + joint[:,0:1,:]
        thumb_add  = (joint[:, 1:2, :] + joint[:, 6:7,:])/2.
        index_add =  (joint[:, 2:3, :] + joint[:, 9:10, :]) / 2.
        middle_add = (joint[:, 3:4, :] + joint[:, 12:13, :]) / 2.
        ring_add =   (joint[:, 4:5, :] + joint[:, 15:16, :]) / 2.
        little_add = (joint[:, 5:6, :] + joint[:, 18:19, :]) / 2.
        return torch.cat((joint,wirst_right,thumb_add,index_add,middle_add,ring_add,little_add), dim=1)

    def randMaskOffset(self, feature, p=0.9, value=0.0):
        b, n, _, _ = feature.size()
        j = n//4
        device = feature.device
        # 随机mask掉一部分图
        mask = torch.masked_fill(torch.rand(b, j).gt(p).to(device).view(b, n, 1, 1), value)
        return feature*mask

    def randExchange(self, joint, p=0.9):
        b, j, _ = joint.size()
        device = joint.device
        exchangeId = torch.randint(j, [b, 2])
        batchId = torch.rand(b).gt(p).to(device).view(b,1,1)
        joint_ex = joint.clone()
        joint_ex[:, exchangeId[:, 0], :] = joint[:, exchangeId[:, 1], :]
        joint_ex[:, exchangeId[:, 1], :] = joint[:, exchangeId[:, 0], :]
        return torch.where(batchId, joint_ex, joint)

    def randShift(self, joint, scale=0.5):
        b, j, _ = joint.size()
        device = joint.device
        joint_var = np.array([0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.15, 0.15, 0.15, 0.05])
        noise = (torch.rand([b, j, 3]) - 0.5) * torch.from_numpy(joint_var).view(1,-1,1)*scale
        return joint + noise.to(device)


    # render hand model to depth map
    def Model2Img(self, joint, dataset, kernel_size=0,sr=128):
        device = joint.device
        batch_size = joint.size(0)
        model_joint =  self.HandJoint2ModelJoint(joint)
        mesh_x = 2.0 * torch.arange(sr).unsqueeze(1).expand(sr, sr).float() / (sr - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(sr).unsqueeze(0).expand(sr, sr).float() / (sr - 1.0) - 1.0
        P = torch.stack((mesh_y,mesh_x), dim=-1).view(1,1,sr*sr,2).repeat(batch_size,1,1,1).to(device)

        if kernel_size ==0:
            kernel_size = get_HandModel_size(dataset)
            kernel_size = torch.from_numpy(kernel_size).float().to(device).view(1,model_joint.size(1),1).repeat(batch_size,1,1)

        pill_list = get_HandModel_pill(dataset)
        A = []
        B = []
        A_size = []
        B_size = []
        for pair in pill_list:
            A.append(model_joint[:, pair[0], :])
            B.append(model_joint[:, pair[1], :])
            A_size.append(kernel_size[:,pair[0],:])
            B_size.append(kernel_size[:,pair[1],:])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        A_size = torch.stack(A_size, dim=1)
        B_size = torch.stack(B_size, dim=1)

        pill_depth = self.Pill2Depth(P, A, B, A_size, B_size)

        wedge_list = get_HandModel_wedge(dataset)
        C = []
        B = []
        A = []
        A_size = []
        B_size = []
        C_size = []
        for pair in wedge_list:
            A.append(model_joint[:, pair[0], :])
            B.append(model_joint[:, pair[1], :])
            C.append(model_joint[:, pair[2], :])
            A_size.append(kernel_size[:,pair[0],:])
            B_size.append(kernel_size[:,pair[1],:])
            C_size.append(kernel_size[:, pair[2], :])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        C = torch.stack(C, dim=1)
        A_size = torch.stack(A_size, dim=1)
        B_size = torch.stack(B_size, dim=1)
        C_size = torch.stack(C_size, dim=1)
        wedge_depth = self.Wedge2Depth(P, A,B,C,A_size,B_size,C_size)

        dist = torch.cat((wedge_depth,pill_depth), dim=1)


        return dist.view(batch_size,-1,sr,sr)

    def Pill2Depth(self,P, A, B, A_size, B_size):
        device = P.device

        batch_size, pill_num= A.size()[:2]
        A = A.view(batch_size,pill_num,1,3)
        B = B.view(batch_size, pill_num, 1, 3)
        A_size = A_size.view(batch_size, pill_num, 1, 1)
        B_size = B_size.view(batch_size, pill_num, 1, 1)
        P = P.view(batch_size, 1, -1, 2).repeat(1, pill_num, 1, 1)
        point_num = P.size(2)

        #only consider 2d
        residual = A_size - B_size
        H = torch.where(residual >= 0, A[:,:,:,:2], B[:,:,:,:2])
        H_depth = torch.where(residual >= 0, A[:, :, :, 2:], B[:, :, :, 2:])
        I = torch.where(residual < 0, A[:,:,:,:2], B[:,:,:,:2])
        I_depth = torch.where(residual < 0, A[:, :, :, 2:], B[:, :, :, 2:])
        H_size = torch.where(residual >= 0, A_size, B_size)
        I_size = torch.where(residual < 0, A_size, B_size)
        residual_radius_abs = torch.abs(residual)

        n, n_sy, incircle = self.CircleTangent(P, H, I, residual_radius_abs)
        N = H + n*H_size
        N_s = H + n_sy*H_size
        O = I + n*I_size
        O_s = I + n_sy * I_size
        Q = n*H_size + (N_s - N)/2.0 +H
        R = n*I_size + (O_s - O)/2.0 +I
        lower = self.vec_len(Q - H) /self.vec_len(I - H)
        upper = self.vec_len(R - H) / self.vec_len(I - H)
        HI = I - H
        HP =  P - H
        HF_len = torch.sum(HI*HP, dim=-1,keepdim=True) / self.vec_len(HI)
        # 存在负值的情况，当HP和HI平行的时候
        PF_len = torch.sqrt(torch.abs(self.vec_len(HP)*self.vec_len(HP) - HF_len*HF_len)+1e-8)
        ratio = HF_len / self.vec_len(HI)
        radius = H_size*(1-ratio) + I_size*ratio
        F_depth = H_depth*(1-ratio) +I_depth*ratio
        Pill_valid = (radius - PF_len).ge(0)
        P_depth_abs = torch.sqrt(radius*radius - PF_len*PF_len*Pill_valid.float() + 1e-8)
        pill_depth = F_depth - P_depth_abs
        pill_depth = torch.where(Pill_valid,pill_depth,torch.ones_like(pill_depth).to(device))


        PH_len = self.vec_len(P-H)
        H_valid = (H_size - PH_len).gt(0)
        H_depth_abs = torch.sqrt(H_size*H_size - PH_len*PH_len*H_valid.float()+ 1e-8)
        H_circle_depth = H_depth - H_depth_abs
        H_circle_depth = torch.where(H_valid,H_circle_depth, torch.ones_like(H_circle_depth).to(device))

        PI_len = self.vec_len(P-I)
        I_valid = (I_size - PI_len).gt(0)
        I_depth_abs = torch.sqrt(I_size*I_size - PI_len*PI_len*I_valid.float()+ 1e-8)
        I_circle_depth = I_depth - I_depth_abs
        I_circle_depth = torch.where(I_valid,I_circle_depth, torch.ones_like(I_circle_depth).to(device))

        pill_depth = torch.where(ratio < lower, H_circle_depth, pill_depth)
        pill_depth = torch.where(ratio > upper, I_circle_depth, pill_depth)
        # 小圆内切于大圆
        pill_depth = torch.where(incircle, H_circle_depth,pill_depth)

        # pill_depth = torch.where((incircle&H_valid)|(~incircle&(H_valid|I_valid|Pill_valid)) , pill_depth, torch.ones_like(pill_depth).to(device))

        return pill_depth.view(batch_size,pill_num,point_num)


    def Wedge2Depth(self,P, A_3d, B_3d, C_3d, A_size, B_size, C_size):
        device = P.device
        batch_size, wedge_num = A_3d.size()[0:2]

        A = A_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,0:2]
        B = B_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,0:2]
        C = C_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,0:2]

        A_size = A_size.view(batch_size, wedge_num, 1, 1)
        B_size = B_size.view(batch_size, wedge_num, 1, 1)
        C_size = C_size.view(batch_size, wedge_num, 1, 1)

        A_depth = A_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,2:]
        B_depth = B_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,2:]
        C_depth = C_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,2:]

        P = P.view(batch_size,1,-1,2)
        point_num = P.size(2)

        if_in_triangle = self.ifInTriangle(P,A,B,C)


        H_depth,I_depth,G_depth = A_depth - A_size, B_depth - B_size, C_depth - C_size
        circle_center = torch.cat((A.unsqueeze(-2),B.unsqueeze(-2),C.unsqueeze(-2)),dim=-2)
        circle_depth = torch.cat((H_depth,I_depth,G_depth),dim=-1)
        sort_depth, sort_index = torch.sort(circle_depth,dim=-1)
        sort_index = sort_index.unsqueeze(-1).repeat(1,1,1,1,2)
        circle_sort = torch.gather(circle_center, -2, sort_index)
        H_depth = sort_depth[:, :, :, 0:1]
        I_depth = sort_depth[:, :, :, 1:2]
        G_depth = sort_depth[:, :, :, 2:3]
        H = circle_sort[:, :, :, 0,:]
        I = circle_sort[:, :, :, 1,:]
        G = circle_sort[:, :, :, 2,:]

        k_HP = (P[:, :, :, 1] - H[:, :, :, 1]) / (P[:, :, :, 0] - H[:, :, :, 0] + 1e-8)
        b_HP = H[:, :, :, 1] - H[:, :, :, 0] * k_HP

        k_IG = (G[:, :, :, 1] - I[:, :, :, 1]) / (G[:, :, :, 0] - I[:, :, :, 0] + 1e-8)
        b_IG = I[:, :, :, 1] - I[:, :, :, 0] * k_IG

        F_x = (b_IG - b_HP) / (k_HP - k_IG + 1e-8)
        F_y = k_HP * F_x + b_HP
        F = torch.stack((F_x, F_y), dim=-1)

        R_IG = self.vec_len(F-I) / self.vec_len(I - G)
        F_depth = (1 - R_IG) * I_depth + R_IG * G_depth
        P_depth_abs = self.vec_len(F - P) / self.vec_len(F - H) * (F_depth - H_depth)
        P_depth = F_depth - P_depth_abs


        depth_pill = self.Pill2Depth(P, torch.cat((A_3d,B_3d,C_3d),dim=2).view(batch_size,wedge_num*3,3),
                                        torch.cat((C_3d,A_3d,B_3d),dim=2).view(batch_size,wedge_num*3,3),
                                    torch.cat((A_size,B_size,C_size),dim=2).view(batch_size,wedge_num*3,1),
                                    torch.cat((C_size,A_size,B_size),dim=2).view(batch_size,wedge_num*3,1),
                                    )
        depth_pill_min = depth_pill.view(batch_size,wedge_num,3,point_num).min(dim=-2)[0].view(batch_size,wedge_num,point_num,1)

        P_depth_all = torch.where(if_in_triangle, P_depth, depth_pill_min)
        return P_depth_all.view(batch_size,wedge_num,point_num)


    # render hand model to depth map
    def Model2Img_sample(self, joint, dataset, kernel_size=0,sr=16,img_size=128):
        device = joint.device
        batch_size = joint.size(0)
        model_joint =  self.HandJoint2ModelJoint(joint)
        if kernel_size ==0:
            kernel_size = get_HandModel_size(dataset)
            kernel_size = torch.from_numpy(kernel_size).float().to(device).view(1,model_joint.size(1),1).repeat(batch_size,1,1)


        wedge_list = get_HandModel_wedge(dataset)
        C = []
        B = []
        A = []
        A_size = []
        B_size = []
        C_size = []
        for pair in wedge_list:
            A.append(model_joint[:, pair[0], :])
            B.append(model_joint[:, pair[1], :])
            C.append(model_joint[:, pair[2], :])
            A_size.append(kernel_size[:,pair[0],:])
            B_size.append(kernel_size[:,pair[1],:])
            C_size.append(kernel_size[:, pair[2], :])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        C = torch.stack(C, dim=1)
        A_size = torch.stack(A_size, dim=1)
        B_size = torch.stack(B_size, dim=1)
        C_size = torch.stack(C_size, dim=1)
        wedge_sample = self.Wedge2Depth_sample(A,B,C,A_size,B_size,C_size,sr)
        wedge_sample = wedge_sample.view(batch_size,-1,sr*sr,3)

        pill_list = get_HandModel_pill(dataset)
        A = []
        B = []
        A_size = []
        B_size = []
        for pair in pill_list:
            A.append(model_joint[:, pair[0], :])
            B.append(model_joint[:, pair[1], :])
            A_size.append(kernel_size[:,pair[0],:])
            B_size.append(kernel_size[:,pair[1],:])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        A_size = torch.stack(A_size, dim=1)
        B_size = torch.stack(B_size, dim=1)

        pill_smaple = self.Pill2Depth_sample(A, B, A_size, B_size,sr)
        pill_smaple = pill_smaple.view(batch_size,-1,sr*sr,3)

        return torch.cat((pill_smaple,wedge_sample),dim=1)

    def Pill2Depth_sample(self,A, B, A_size, B_size, sr=16):
        device = A.device
        mesh_x = (torch.arange(sr).unsqueeze(1).expand(sr, sr).float() + 0.5) / (sr - 1.0)
        mesh_y = (torch.arange(sr).unsqueeze(0).expand(sr, sr).float() + 0.5) / (sr - 1.0)
        point_num = sr * sr

        l = torch.min(torch.cat((A[:, :, 0:1] - A_size, B[:, :, 0:1] - B_size), dim=-1), dim=-1, keepdim=True)[0]
        r = torch.max(torch.cat((A[:, :, 0:1] + A_size, B[:, :, 0:1] + B_size), dim=-1), dim=-1, keepdim=True)[0]
        t = torch.min(torch.cat((A[:, :, 1:2] - A_size, B[:, :, 1:2] - B_size), dim=-1), dim=-1, keepdim=True)[0]
        d = torch.max(torch.cat((A[:, :, 1:2] + A_size, B[:, :, 1:2] + B_size), dim=-1), dim=-1, keepdim=True)[0]
        mesh_x = mesh_x.view(1, 1, point_num) * (d - t) + t
        mesh_y = mesh_y.view(1, 1, point_num) * (r - l) + l
        P = torch.stack((mesh_y, mesh_x), dim=-1).to(device)

        batch_size, pill_num= A.size()[:2]
        A = A.view(batch_size,pill_num,1,3)
        B = B.view(batch_size, pill_num, 1, 3)
        A_size = A_size.view(batch_size, pill_num, 1, 1)
        B_size = B_size.view(batch_size, pill_num, 1, 1)
        P = P.view(batch_size, pill_num, sr*sr, 2)
        point_num = P.size(2)

        #only consider 2d
        residual = A_size - B_size
        H = torch.where(residual >= 0, A[:,:,:,:2], B[:,:,:,:2])
        H_depth = torch.where(residual >= 0, A[:, :, :, 2:], B[:, :, :, 2:])
        I = torch.where(residual < 0, A[:,:,:,:2], B[:,:,:,:2])
        I_depth = torch.where(residual < 0, A[:, :, :, 2:], B[:, :, :, 2:])
        H_size = torch.where(residual >= 0, A_size, B_size)
        I_size = torch.where(residual < 0, A_size, B_size)

        HI = I - H
        HP =  P - H
        HF_len = torch.sum(HI*HP, dim=-1,keepdim=True) / self.vec_len(HI)
        PF_len = torch.sqrt(self.vec_len(HP)*self.vec_len(HP) - HF_len*HF_len +1e-6)
        F = HI / (self.vec_len(HI) * HF_len + 1e-8)
        R = HF_len / self.vec_len(HI)
        radius = H_size*(1-R) + I_size*R
        line_depth = H_depth*(1-R) +I_depth*R
        Pill_valid = (radius - PF_len).ge(0) & R.ge(0) & R.le(1)
        P_depth_abs = torch.sqrt(radius*radius - PF_len*PF_len*Pill_valid.float() + 1e-8)
        pill_depth = line_depth - P_depth_abs

        PH_len = self.vec_len(P-H)
        H_valid = (H_size - PH_len).gt(0)
        H_depth_abs = torch.sqrt(H_size*H_size - PH_len*PH_len*H_valid.float()+ 1e-8)
        H_circle_depth = H_depth - H_depth_abs

        PI_len = self.vec_len(P-I)
        I_valid = (I_size - PI_len).gt(0)
        I_depth_abs = torch.sqrt(I_size*I_size - PI_len*PI_len*I_valid.float()+ 1e-8)
        I_circle_depth = I_depth - I_depth_abs

        pill_depth = torch.where(R < 0, H_circle_depth, pill_depth)
        pill_depth = torch.where(R > 1, I_circle_depth, pill_depth)
        pill_depth = torch.where(H_valid|I_valid|Pill_valid, pill_depth, torch.ones_like(pill_depth).to(device))

        # return P,pill_depth.view(batch_size,pill_num,point_num,1)
        return torch.cat((P,pill_depth.view(batch_size,pill_num,point_num,1)),dim=-1)

    def Wedge2Depth_sample(self,A_3d, B_3d, C_3d, A_size, B_size, C_size,sr=16):
        batch_size, wedge_num = A_3d.size()[0:2]
        device = A_3d.device

        A = A_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,0:2]
        B = B_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,0:2]
        C = C_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,0:2]

        A_size = A_size.view(batch_size, wedge_num, 1, 1)
        B_size = B_size.view(batch_size, wedge_num, 1, 1)
        C_size = C_size.view(batch_size, wedge_num, 1, 1)


        A_depth = A_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,2:]
        B_depth = B_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,2:]
        C_depth = C_3d.view(batch_size, wedge_num, 1, 3)[:,:,:,2:]

        point_num = sr * sr

        mesh_x =  (torch.arange(sr).unsqueeze(1).expand(sr, sr).float() + 0.5) / (sr - 1.0)
        mesh_y =  (torch.arange(sr).unsqueeze(0).expand(sr, sr).float() + 0.5) / (sr - 1.0)
        l = torch.min(torch.cat((A[:,:,0:1]-A_size,B[:,:,0:1]-B_size,C[:,:,0:1]-C_size),dim=-1),dim=-1,keepdim=True)[0]
        r = torch.max(torch.cat((A[:, :, 0:1] + A_size, B[:, :, 0:1] + B_size,C[:,:,0:1]+C_size), dim=-1), dim=-1, keepdim=True)[0]
        t = torch.min(torch.cat((A[:,:, 1:2]-A_size,B[:,:,1:2]-B_size,C[:,:,1:2]-C_size),dim=-1),dim=-1,keepdim=True)[0]
        d = torch.max(torch.cat((A[:, :, 1:2] + A_size, B[:, :, 1:2] + B_size,C[:,:,1:2]+C_size), dim=-1), dim=-1, keepdim=True)[0]
        mesh_x = mesh_x.view(1, 1,point_num)  * (d - t) + t
        mesh_y = mesh_y.view(1, 1, point_num) * (r - l) + l
        P = torch.stack((mesh_y, mesh_x), dim=-1).to(device)


        P = P.view(batch_size,wedge_num,point_num,2)

        if_in_triangle = self.ifInTriangle(P,A,B,C)


        H_depth,I_depth,G_depth = A_depth - A_size, B_depth - B_size, C_depth - C_size
        circle_center = torch.cat((A.unsqueeze(-2),B.unsqueeze(-2),C.unsqueeze(-2)),dim=-2)
        circle_depth = torch.cat((H_depth,I_depth,G_depth),dim=-1)
        sort_depth, sort_index = torch.sort(circle_depth,dim=-1)
        sort_index = sort_index.unsqueeze(-1).repeat(1,1,1,1,2)
        circle_sort = torch.gather(circle_center, -2, sort_index)
        H_depth = sort_depth[:, :, :, 0:1]
        I_depth = sort_depth[:, :, :, 1:2]
        G_depth = sort_depth[:, :, :, 2:3]
        H = circle_sort[:, :, :, 0,:]
        I = circle_sort[:, :, :, 1,:]
        G = circle_sort[:, :, :, 2,:]

        k_HP = (P[:, :, :, 1] - H[:, :, :, 1]) / (P[:, :, :, 0] - H[:, :, :, 0] + 1e-8)
        b_HP = H[:, :, :, 1] - H[:, :, :, 0] * k_HP

        k_IG = (G[:, :, :, 1] - I[:, :, :, 1]) / (G[:, :, :, 0] - I[:, :, :, 0] + 1e-8)
        b_IG = I[:, :, :, 1] - I[:, :, :, 0] * k_IG

        F_x = (b_IG - b_HP) / (k_HP - k_IG + 1e-8)
        F_y = k_HP * F_x + b_HP
        F = torch.stack((F_x, F_y), dim=-1)

        R_IG = self.vec_len(F-I) / self.vec_len(I - G)
        F_depth = (1 - R_IG) * I_depth + R_IG * G_depth
        P_depth_abs = self.vec_len(F - P) / self.vec_len(F - H) * (F_depth - H_depth)
        P_depth = F_depth - P_depth_abs


        depth_pill = self.Pill2Depth(P, torch.cat((A_3d,B_3d,C_3d),dim=2).view(batch_size,wedge_num*3,3),
                                        torch.cat((C_3d,A_3d,B_3d),dim=2).view(batch_size,wedge_num*3,3),
                                    torch.cat((A_size,B_size,C_size),dim=2).view(batch_size,wedge_num*3,1),
                                    torch.cat((C_size,A_size,B_size),dim=2).view(batch_size,wedge_num*3,1),
                                    )
        depth_pill_min = depth_pill.view(batch_size,wedge_num,3,point_num).min(dim=-2)[0].view(batch_size,wedge_num,point_num,1)

        P_depth_all = torch.where(if_in_triangle, P_depth, depth_pill_min)
        # return P,P_depth_all.view(batch_size,wedge_num,point_num,1)
        return torch.cat((P,P_depth_all.view(batch_size,wedge_num,point_num,1)), dim=-1)

    """
    para: P batch*1*point_num*2/3
    """
    def ifInTriangle(self, P, A,B,C):
        batch_size, _, point_num, coord_dim = P.size()
        device = P.device
        wedge_num = A.size(1)
        # whether the point is D inside ABC
        AB = B - A
        BC = C - B
        AC = C - A
        AP = P - A
        BP = P - B
        CP = P - C
        if coord_dim == 2:
            # pytorch only support 3D cross product
            AB_AP_normal = torch.cross(torch.cat((AB, torch.zeros(batch_size, wedge_num, 1, 1).to(device)), dim=-1).repeat(1, 1, point_num, 1)
                , torch.cat((AP, torch.zeros(batch_size, wedge_num, point_num, 1).to(device)), dim=-1), dim=-1)
            BC_BP_normal = torch.cross(torch.cat((BC, torch.zeros(batch_size, wedge_num, 1, 1).to(device)), dim=-1).repeat(1, 1, point_num, 1)
                , torch.cat((BP, torch.zeros(batch_size, wedge_num, point_num, 1).to(device)), dim=-1), dim=-1)
            CA_CP_normal = torch.cross(torch.cat((-AC, torch.zeros(batch_size, wedge_num, 1, 1).to(device)), dim=-1).repeat(1, 1, point_num, 1)
                , torch.cat((CP, torch.zeros(batch_size, wedge_num, point_num, 1).to(device)), dim=-1), dim=-1)
        elif coord_dim == 2:
            AB_AP_normal = torch.cross(AB.repeat(1, 1, point_num, 1), AP)
            BC_BP_normal = torch.cross(BC.repeat(1, 1, point_num, 1), BP)
            CA_CP_normal = torch.cross(-AC.repeat(1, 1, point_num, 1), CP)

        if_in_triangle = AB_AP_normal / self.vec_len(AB_AP_normal) + \
                             BC_BP_normal / self.vec_len(BC_BP_normal) + \
                             CA_CP_normal / self.vec_len(CA_CP_normal)
        if_in_triangle = torch.sqrt(torch.sum(if_in_triangle * if_in_triangle, dim=-1, keepdim=True) + 1e-8)

        return if_in_triangle.ge(2.99)

    """
    use the hand model in Tkach'17
    pcl: B*N*3
    """
    def Img2Model(self, pcl, joint, dataset, kernel_size=0):
        device = pcl.device
        batch_size = pcl.size(0)
        model_joint =  self.HandJoint2ModelJoint(joint)
        if kernel_size ==0:
            kernel_size = get_HandModel_size(dataset)
            kernel_size = torch.from_numpy(kernel_size).float().to(device).view(1,model_joint.size(1),1).repeat(batch_size,1,1)

        wedge_list = get_HandModel_wedge(dataset)
        C = []
        B = []
        A = []
        A_size = []
        B_size = []
        C_size = []
        for pair in wedge_list:
            A.append(model_joint[:, pair[0], :])
            B.append(model_joint[:, pair[1], :])
            C.append(model_joint[:, pair[2], :])
            A_size.append(kernel_size[:,pair[0],:])
            B_size.append(kernel_size[:,pair[1],:])
            C_size.append(kernel_size[:, pair[2], :])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        C = torch.stack(C, dim=1)
        A_size = torch.stack(A_size, dim=1)
        B_size = torch.stack(B_size, dim=1)
        C_size = torch.stack(C_size, dim=1)
        dist_wedge = self.Point2WedgeDis(pcl,A,B,C,A_size,B_size,C_size)

        pill_list = get_HandModel_pill(dataset)
        A = []
        B = []
        A_size = []
        B_size = []
        for pair in pill_list:
            A.append(model_joint[:, pair[0], :])
            B.append(model_joint[:, pair[1], :])
            A_size.append(kernel_size[:,pair[0],:])
            B_size.append(kernel_size[:,pair[1],:])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        A_size = torch.stack(A_size, dim=1)
        B_size = torch.stack(B_size, dim=1)

        dist_pill = self.Point2PillDis(pcl, A, B, A_size, B_size)

        dist = torch.cat((dist_pill,dist_wedge), dim=1)
        return dist.squeeze(-1)

    # in the world coord, the more a point range the camera, the lager a point
    def Point2WedgeDis(self, P, A, B, C,A_size, B_size, C_size):
        device = P.device
        batch_size, point_num, _ = P.size()
        wedge_num = A.size(1)
        camera_unit_normal = torch.tensor([0, 0, -1.]).to(device).view(1,1,1,3)

        A = A.view(batch_size, wedge_num, 1, 3)
        B = B.view(batch_size, wedge_num, 1, 3)
        C = C.view(batch_size, wedge_num, 1, 3)
        A_size = A_size.view(batch_size, wedge_num, 1, 1)
        B_size = B_size.view(batch_size, wedge_num, 1, 1)
        C_size = C_size.view(batch_size, wedge_num, 1, 1)
        P = P.view(batch_size,1,-1,3)
        AB = B - A
        AC = C - A
        BC = C - B

        plane_unit_noraml = torch.cross(AB,AC,dim=-1)
        plane_unit_noraml = plane_unit_noraml / self.vec_len(plane_unit_noraml)
        direction  = torch.sum(plane_unit_noraml * camera_unit_normal, dim=-1, keepdim=True)
        direction = torch.where(direction>0, torch.ones_like(direction).to(device), -torch.ones_like(direction).to(device))
        # 表面法向量朝外,相机方向向量朝外，方向不同时的应变为反方向
        plane_unit_noraml = (direction) * plane_unit_noraml

        AU = A + plane_unit_noraml * A_size
        BU = B + plane_unit_noraml * B_size
        CU = C + plane_unit_noraml * C_size
        AB_U = BU - AU
        AC_U = CU - AU
        BC_U = CU - BU
        plane_U_normal = torch.cross(AB_U,AC_U,dim=-1)
        plane_U_unit_noraml = plane_U_normal / torch.sqrt(torch.sum(plane_U_normal*plane_U_normal,dim=-1, keepdim=True)+1e-8)
        direction  = torch.sum(plane_U_unit_noraml * camera_unit_normal, dim=-1, keepdim=True)
        direction = torch.where(direction>0, torch.ones_like(direction).to(device), -torch.ones_like(direction).to(device))
        plane_U_unit_noraml = (direction) * plane_U_unit_noraml

        AP_U = P - AU
        PD_U_len = torch.sum(AP_U*plane_U_unit_noraml, dim=-1, keepdim=True)
        PD_U =  -plane_U_unit_noraml * PD_U_len
        DU = P + PD_U

        # whether the point is D inside ABC
        AD_U = DU - AU
        BD_U = DU - BU
        CD_U = DU - CU
        AB_AD_U_normal = torch.cross(AB_U.repeat(1,1,point_num,1), AD_U)
        BC_BD_U_normal = torch.cross(BC_U.repeat(1,1,point_num,1), BD_U)
        CA_CD_U_normal = torch.cross(-AC_U.repeat(1,1,point_num,1), CD_U)
        if_in_triangle  = AB_AD_U_normal / self.vec_len(AB_AD_U_normal)+ \
                          BC_BD_U_normal / self.vec_len(BC_BD_U_normal)+\
                          CA_CD_U_normal / self.vec_len(CA_CD_U_normal)
        if_in_triangle = torch.sqrt(torch.sum(if_in_triangle*if_in_triangle, dim=-1, keepdim=True)+1e-8)


        sihouette_sample = self.Wedge2Sihouette_Sample(A,B,C,A_size,B_size,C_size)
        dist_pill = self.Point2PillDis(P.view(batch_size,point_num,3), torch.cat((A,B,C),dim=2).view(batch_size,wedge_num*3,3),torch.cat((C,A,B),dim=2).view(batch_size,wedge_num*3,3),
                                    torch.cat((A_size,B_size,C_size),dim=2).view(batch_size,wedge_num*3,1),
                                    torch.cat((C_size,A_size,B_size),dim=2).view(batch_size,wedge_num*3,1),
                                    sihouette_sample)
        dist_pill_min = dist_pill.view(batch_size,wedge_num,3,point_num).min(dim=-2)[0].view(batch_size,wedge_num,point_num,1)
        #
        dist = torch.where(if_in_triangle>2.99, PD_U_len, dist_pill_min)
        return torch.abs(dist)

    def Point2PillDis(self, P, A, B, A_size, B_size, wedge_sihouette_sample=0):
        camera_unit_normal = torch.tensor([0, 0, 1.]).to(P.device).view(1, 1, 1, 3)
        batch_size, point_num, _ = P.size()
        pill_num = A.size(1)
        P = P.view(batch_size,1,point_num,3)
        A = A.view(batch_size,pill_num,1,3)
        B = B.view(batch_size, pill_num, 1, 3)
        A_size = A_size.view(batch_size, pill_num, 1, 1)
        B_size = B_size.view(batch_size, pill_num, 1, 1)

        residual = A_size - B_size
        H_size = torch.where(residual >= 0, A_size, B_size)
        I_size = torch.where(residual < 0, A_size, B_size)
        H = torch.where(residual >= 0, A, B)
        I = torch.where(residual < 0, A, B)
        residual_radius_abs = residual.abs()

        n, n_symmetry, incircle = self.CircleTangent(P ,H, I,residual_radius_abs)

        # N,0 为H,I切点
        N = H + n * H_size
        O = I + n * I_size
        NO = O - N
        NP = P - N
        # F为P垂直于NO的垂足
        FP_len = torch.sum(NP*n,dim=-1,keepdim=True)
        NF = NP + FP_len * (-n)
        R = self.vec_len(NF) / self.vec_len(NO) * torch.sum((NF/self.vec_len(NF))*(NO/self.vec_len(NO)),dim=-1,keepdim=True)
        HP = P - H
        IP = P - I
        HP_len = self.vec_len(HP) - H_size
        IP_len = self.vec_len(IP) - I_size

        if isinstance(wedge_sihouette_sample,int):
            # consider front
            direction_pill = torch.sum(n * (-camera_unit_normal), dim=-1, keepdim=True)
            direction_HP = torch.sum(HP * (-camera_unit_normal), dim=-1, keepdim=True) / self.vec_len(HP)
            direction_IP = torch.sum(IP * (-camera_unit_normal), dim=-1, keepdim=True) / self.vec_len(IP)

            # 位于轮廓或者对称的垂足上
            # 计算垂足
            N = H + n_symmetry * H_size
            FP_symmetry_len = torch.abs(torch.sum((P - N) * n_symmetry, dim=-1, keepdim=True))
            # 计算轮廓上最近的点
            sihouette_sample = self.Pill2Sihouette_Sample(H, I, H_size,I_size)  # batch_size*pill_num*(sample_point*4)*3
            sihouette_dist = self.vec_len(P.view(batch_size, 1, point_num, 1, 3) - sihouette_sample.view(batch_size, pill_num, 1, -1, 3))
            sihouette_dist_min = torch.min(sihouette_dist.squeeze(-1), dim=-1, keepdim=True)[0]  # bactch_size*pill_num*point_num

            back_dist = torch.where(FP_symmetry_len <= sihouette_dist_min, FP_symmetry_len, sihouette_dist_min)
            FP_len = torch.where(direction_pill < 0, back_dist, FP_len)
            HP_len = torch.where(direction_HP < 0, back_dist, HP_len)
            IP_len = torch.where(direction_IP < 0, back_dist, IP_len)
        else:
            wedge_num = pill_num // 3
            sample_num = wedge_sihouette_sample.size(-2)//6
            direction_pill = torch.sum(n * (-camera_unit_normal), dim=-1, keepdim=True)
            direction_HP = torch.sum(HP * (-camera_unit_normal), dim=-1, keepdim=True) / self.vec_len(HP)
            direction_IP = torch.sum(IP * (-camera_unit_normal), dim=-1, keepdim=True) / self.vec_len(IP)

            wedge_sihouette_sample = wedge_sihouette_sample.view(batch_size,wedge_num,6,sample_num,3) # AC,AB,BC,A,B,C
            pill_ss = wedge_sihouette_sample[:,:,:3,:,:].clone()
            ABC = wedge_sihouette_sample[:,:,3:,:,:].clone()
            CAB = torch.cat((ABC[:,:,2:,:],ABC[:,:,0:1,:],ABC[:,:,1:2,:]),dim=-2)
            ABC = ABC.view(batch_size,wedge_num * 3, sample_num,3)
            CAB = CAB.view(batch_size, wedge_num * 3, sample_num, 3)
            H_ss = torch.where(residual.ge(0), ABC,CAB)
            I_ss = torch.where(residual.lt(0), ABC,CAB)

            pill_sihouette_dist_min = torch.min(self.vec_len(P.view(batch_size, 1, point_num, 1, 3) - pill_ss.view(batch_size, wedge_num*3, 1, sample_num, 3)).squeeze(-1),
                                                dim=-1, keepdim=True)[0]  # bactch_size*(wedge_num*3)*sample_num

            H_sihouette_dist = torch.min(self.vec_len(P.view(batch_size, 1, point_num, 1, 3) - H_ss.view(batch_size, wedge_num*3, 1, sample_num, 3)).squeeze(-1),
                                         dim=-1, keepdim=True)[0]  # bactch_size*(wedge_num*3)*sample_num

            I_sihouette_dist = torch.min(self.vec_len(P.view(batch_size, 1, point_num, 1, 3) - I_ss.view(batch_size, wedge_num*3, 1, sample_num, 3)).squeeze(-1),
                                         dim=-1, keepdim=True)[0]  # bactch_size*(wedge_num*3)*point_num
            # 位于轮廓或者对称的垂足上
            # 计算垂足
            N = H + n_symmetry * H_size
            FP_symmetry_len = torch.abs(torch.sum((P - N) * n_symmetry, dim=-1, keepdim=True))

            back_dist = torch.where(FP_symmetry_len <= pill_sihouette_dist_min, FP_symmetry_len, pill_sihouette_dist_min)
            FP_len = torch.where(direction_pill < 0, back_dist, FP_len)
            HP_len = torch.where(direction_HP < 0, H_sihouette_dist, HP_len)
            IP_len = torch.where(direction_IP < 0, I_sihouette_dist, IP_len)


        dist = FP_len.clone()
        dist = torch.where(R < 0, HP_len, dist)
        dist = torch.where(R > 1, IP_len, dist)
        dist = torch.where(incircle, HP_len, dist)
        return torch.abs(dist)


    # error version
    def Point2Pill_old(self, P, A, B, A_size, B_size):
        batch_size, point_num, _ = P.size()
        camera_unit_normal = torch.tensor([0, 0, -1.]).to(P.device).view(1, 1, 1, 3)

        bone = B - A
        bone_num = bone.size(1)
        bone_len = torch.sqrt(torch.sum(bone*bone, dim=-1, keepdim=True)+1e-8).view(batch_size,bone_num,1)


        AP = P.view(batch_size,1,-1,3) - A.view(batch_size,bone_num,1,3)
        BP = P.view(batch_size,1,-1,3) - B.view(batch_size,bone_num,1,3)
        AP_len = torch.sqrt(torch.sum(AP * AP, dim=-1) + 1e-8)
        BP_len = torch.sqrt(torch.sum(BP * BP, dim=-1) + 1e-8)
        AC_len = torch.sum(AP*bone.view(batch_size,bone_num,1,3), dim=-1) / bone_len
        AC = bone / bone_len.view(batch_size,bone_num,1,1) + A
        PC = -AP + AC
        R = AC_len / bone_len #


        PC_len = torch.sqrt(AP_len*AP_len - AC_len*AC_len+ 1e-8)
        PC_len[R<0] = torch.sqrt(torch.sum(AP*AP,dim=-1))[R<0]
        PC_len[R>1] = torch.sqrt(torch.sum(BP*BP,dim=-1))[R>1]


        radius = (B_size- A_size).view(batch_size,bone_num,1) * R + A_size.view(batch_size,bone_num,1)
        radius[R < 0] = A_size.repeat(1,1,point_num)[R<0]
        radius[R > 1] = B_size.repeat(1, 1, point_num)[R > 1]

        PC_len  = PC_len - radius
        # F is the most close point in the silhouette, D is the foot of the P on the plane of circle
        P_direction = torch.sum(PC * camera_unit_normal, dim=-1) / PC_len
        P_direction[R<0] = (torch.sum(AP * camera_unit_normal, dim=-1) / AP_len)[R<0]
        P_direction[R>1] = (torch.sum(BP * camera_unit_normal, dim=-1) / BP_len)[R>0]
        # for point circle A
        DP_len = torch.sum(AP*camera_unit_normal, dim=-1, keepdim=True) / AP_len
        FD_len = torch.sqrt(AP_len*AP_len - DP_len * DP_len +1e-8) - A_size.view(batch_size,bone_num,1, 1)
        FP_len_A = torch.sqrt(FD_len*FD_len + DP_len*DP_len)
        # for point circle B
        DP_len = torch.sum(BP*camera_unit_normal, dim=-1, keepdim=True) / AP_len
        FD_len = torch.sqrt(BP_len*AP_len - DP_len * DP_len +1e-8) - B_size.view(batch_size,bone_num,1, 1)
        FP_len_B = torch.sqrt(FD_len * FD_len + DP_len * DP_len)
        # for pill AB
        return PC_len

    def Pill2Sihouette_nearDense(self, A, B, A_size, B_size, sr=64):
        device = A.device
        mesh_x =  (torch.arange(sr).unsqueeze(1).expand(sr, sr).float() + 0.4) / (sr - 1.0)
        mesh_y =  (torch.arange(sr).unsqueeze(0).expand(sr, sr).float() + 0.4) / (sr - 1.0)
        point_num = sr * sr

        l = torch.min(torch.cat((A[:,:,0:1]-A_size,B[:,:,0:1]-B_size),dim=-1),dim=-1,keepdim=True)[0]
        r = torch.max(torch.cat((A[:, :, 0:1] + A_size, B[:, :, 0:1] + B_size), dim=-1), dim=-1, keepdim=True)[0]
        t = torch.min(torch.cat((A[:,:, 1:2]-A_size,B[:,:,1:2]-B_size),dim=-1),dim=-1,keepdim=True)[0]
        d = torch.max(torch.cat((A[:, :, 1:2] + A_size, B[:, :, 1:2] + B_size), dim=-1), dim=-1, keepdim=True)[0]
        mesh_x = mesh_x.view(1, 1,point_num)  * (d - t) + t
        mesh_y = mesh_y.view(1, 1, point_num)* (r - l) + l
        P = torch.stack((mesh_y, mesh_x), dim=-1).to(device)


        batch_size, pill_num= A.size()[:2]
        A = A.view(batch_size,pill_num,1,3)
        B = B.view(batch_size, pill_num, 1, 3)
        A_size = A_size.view(batch_size, pill_num, 1, 1)
        B_size = B_size.view(batch_size, pill_num, 1, 1)

        #only consider 2d
        residual = A_size - B_size
        H = torch.where(residual >= 0, A[:,:,:,:2], B[:,:,:,:2])
        H_depth = torch.where(residual >= 0, A[:, :, :, 2:], B[:, :, :, 2:])
        I = torch.where(residual < 0, A[:,:,:,:2], B[:,:,:,:2])
        I_depth = torch.where(residual < 0, A[:, :, :, 2:], B[:, :, :, 2:])
        H_size = torch.where(residual >= 0, A_size, B_size)
        I_size = torch.where(residual < 0, A_size, B_size)
        residual_radius_abs = residual.abs()

        n,n_symmetry = self.CircleTangent(P,H,I,residual_radius_abs)

        # N,0 为H,I切点
        N = H + n * H_size
        O = I + n * I_size
        NO = O - N
        NP = P - N
        # F为P垂直于NO的垂足
        FP_len = torch.sum(NP*n,dim=-1,keepdim=True)
        NF = NP + FP_len * (-n)
        F = NF + N
        R = self.vec_len(NF) / self.vec_len(NO) * torch.sum((NF/self.vec_len(NF))*(NO/self.vec_len(NO)),dim=-1,keepdim=True)
        z = H_depth*(1-R) + R * I_depth
        pill_sihouette = torch.cat((F, z),dim=-1)

        HP = P-H
        HD = HP / self.vec_len(HP) * H_size
        D = HD + H
        H_sihouette = torch.cat((D, H_depth.repeat(1,1,point_num,1)), dim=-1)

        IP = P-I
        ID = IP / self.vec_len(IP) * I_size
        D = ID + I
        I_sihouette = torch.cat((D, I_depth.repeat(1,1,point_num,1)), dim=-1)

        pill_sihouette = torch.where(R < 0, H_sihouette, pill_sihouette)
        pill_sihouette = torch.where(R > 1, I_sihouette, pill_sihouette)

        return pill_sihouette

    def Pill2Sihouette_Sample(self, A, B, A_size, B_size, sample_point=64):
        device = A.device
        batch_size, pill_num = A.size()[:2]

        A = A.view(batch_size, pill_num, 1, 3)
        B = B.view(batch_size, pill_num, 1, 3)
        A_size = A_size.view(batch_size, pill_num, 1, 1)
        B_size = B_size.view(batch_size, pill_num, 1, 1)

        #only consider 2d
        residual = A_size - B_size
        H = torch.where(residual >= 0, A[:,:,:,:2], B[:,:,:,:2])
        H_depth = torch.where(residual >= 0, A[:, :, :, 2:], B[:, :, :, 2:])
        I = torch.where(residual < 0, A[:,:,:,:2], B[:,:,:,:2])
        I_depth = torch.where(residual < 0, A[:, :, :, 2:], B[:, :, :, 2:])
        H_size = torch.where(residual >= 0, A_size, B_size)
        I_size = torch.where(residual < 0, A_size, B_size)

        alpha = (torch.arange(sample_point) + 0.5) / (sample_point - 1.0)
        alpha = alpha.view(1,1,sample_point,1).repeat(batch_size,pill_num,1,1).to(device)

        HI = I[:,:,:,:2] - H[:,:,:,:2]
        n_y  = -HI[:,:,:,0:1] /  (HI[:,:,:,1:2]+1e-8)
        n = torch.cat((torch.ones_like(n_y).to(device),n_y),dim=-1)
        n = n / self.vec_len(n)
        n_symmetry = -n

        # N,0 为H,I切点
        N = H + n * H_size
        O = I + n * I_size
        z = (1- alpha) * H_depth + alpha*I_depth
        P1 = (1- alpha) * N + alpha*O
        P1 = torch.cat((P1,z),dim=-1)


        N = H + n_symmetry * H_size
        O = I + n_symmetry * I_size
        z = (1 - alpha) * H_depth + alpha * I_depth
        P2 = (1 - alpha) * N + alpha * O
        P2 = torch.cat((P2, z), dim=-1)

        HN = N-H
        theta_diff = torch.sum(HN*HI,dim=-1,keepdim=True) / self.vec_len(HN)/self.vec_len(HI)
        theta = (torch.arange(sample_point) + 0.5) / (sample_point - 1.0) * 2*np.pi
        theta = theta.view(1,1,sample_point,1).repeat(batch_size,pill_num,1,1).to(device)
        P3  = H + \
              H_size * torch.cat((torch.cos(theta),torch.zeros_like(theta)), dim=-1).to(device)+ \
              H_size * torch.cat((torch.zeros_like(theta),torch.sin(theta)), dim=-1).to(device)
        HP3 = P3 - H
        theta_ = torch.sum(HP3*HI,dim=-1,keepdim=True) / self.vec_len(HN)/self.vec_len(HI)
        P3 = torch.cat((P3, H_depth.repeat(1,1,sample_point,1)),dim=-1)
        P3 = torch.where(theta_<theta_diff, P3, torch.ones_like(P3).to(device))

        P4  = I + \
              I_size * torch.cat((torch.cos(theta),torch.zeros_like(theta)), dim=-1).to(device)+ \
              I_size * torch.cat((torch.zeros_like(theta),torch.sin(theta)), dim=-1).to(device)
        HP4 = P4 - I
        theta_ = torch.sum(HP4*HI,dim=-1,keepdim=True) / self.vec_len(HN) /self.vec_len(HI)
        P4 = torch.cat((P4, I_depth.repeat(1,1,sample_point,1)),dim=-1)
        P4 = torch.where(theta_>theta_diff, P4, -torch.ones_like(P4).to(device))
        return torch.cat((P1,P2,P3,P4),dim=-2).view(batch_size,pill_num,sample_point*4,3)

    def Wedge2Sihouette_nearDense(self, A, B,C, A_size, B_size,C_size, sr=64):
        device = A.device
        point_num = sr * sr

        mesh_x =  (torch.arange(sr).unsqueeze(1).expand(sr, sr).float() + 0.5) / (sr - 1.0)
        mesh_y =  (torch.arange(sr).unsqueeze(0).expand(sr, sr).float() + 0.5) / (sr - 1.0)
        l = torch.min(torch.cat((A[:,:,0:1]-A_size,B[:,:,0:1]-B_size,C[:,:,0:1]-C_size),dim=-1),dim=-1,keepdim=True)[0]
        r = torch.max(torch.cat((A[:, :, 0:1] + A_size, B[:, :, 0:1] + B_size,C[:,:,0:1]+C_size), dim=-1), dim=-1, keepdim=True)[0]
        t = torch.min(torch.cat((A[:,:, 1:2]-A_size,B[:,:,1:2]-B_size,C[:,:,1:2]-C_size),dim=-1),dim=-1,keepdim=True)[0]
        d = torch.max(torch.cat((A[:, :, 1:2] + A_size, B[:, :, 1:2] + B_size,C[:,:,1:2]+C_size), dim=-1), dim=-1, keepdim=True)[0]
        mesh_x = mesh_x.view(1, 1,point_num)  * (d - t) + t
        mesh_y = mesh_y.view(1, 1, point_num)* (r - l) + l
        P = torch.stack((mesh_y, mesh_x), dim=-1).to(device)


        batch_size, wedge_num= A.size()[:2]


        A = A.view(batch_size,wedge_num,1,3)
        B = B.view(batch_size, wedge_num, 1, 3)
        C = C.view(batch_size, wedge_num, 1, 3)
        A_size = A_size.view(batch_size, wedge_num, 1, 1)
        B_size = B_size.view(batch_size, wedge_num, 1, 1)
        C_size = C_size.view(batch_size, wedge_num, 1, 1)


        group_A = torch.stack((A, B ,C),dim=2).view(batch_size,wedge_num*3,1,3)
        group_B = torch.stack((C, A, B), dim=2).view(batch_size, wedge_num * 3, 1, 3)
        group_A_size = torch.stack((A_size, B_size ,C_size),dim=2).view(batch_size,wedge_num*3,1,1)
        group_B_size = torch.stack((C_size, A_size, B_size), dim=2).view(batch_size, wedge_num * 3, 1, 1)

        # only consider 2d
        center = torch.mean(group_A[:,:,:,:2].view(batch_size,wedge_num,3,1,2),dim=2,keepdim=True)
        edge_center = (group_A[:,:,:,:2] + group_B[:,:,:,:2]) / 2
        edge_out_normal = edge_center.view(batch_size,wedge_num,3,1,2) - center
        edge_out_unit_normal = edge_out_normal / self.vec_len(edge_out_normal)

        #only consider 2d
        residual = group_A_size - group_B_size
        H = torch.where(residual >= 0, group_A[:,:,:,:2], group_B[:,:,:,:2])
        H_depth = torch.where(residual >= 0, group_A[:, :, :, 2:], group_B[:, :, :, 2:])
        I = torch.where(residual < 0, group_A[:,:,:,:2], group_B[:,:,:,:2])
        I_depth = torch.where(residual < 0, group_A[:, :, :, 2:], group_B[:, :, :, 2:])
        H_size = torch.where(residual >= 0, group_A_size, group_B_size)
        I_size = torch.where(residual < 0, group_A_size, group_B_size)
        residual_radius_abs = residual.abs()

        n,n_symmetry = self.CircleTangent(P,H,I,residual_radius_abs)

        # N,0 为H,I切点
        N = H + n * H_size
        O = I + n * I_size
        NO = O - N
        NP = P - N
        # F为P垂直于NO的垂足
        FP_len = torch.sum(NP*n,dim=-1,keepdim=True)
        NF = NP + FP_len * (-n)
        F = NF + N
        R = self.vec_len(NF) / self.vec_len(NO) * torch.sum((NF/self.vec_len(NF))*(NO/self.vec_len(NO)),dim=-1,keepdim=True)

        z = H_depth*(1-R) + R * I_depth
        pill_sihouette = torch.cat((F, z),dim=-1)

        HP = P-H
        HD = HP / self.vec_len(HP) * H_size
        D = HD + H
        H_sihouette = torch.cat((D, H_depth.repeat(1,1,point_num,1)), dim=-1)

        IP = P-I
        ID = IP / self.vec_len(IP) * I_size
        D = ID + I
        I_sihouette = torch.cat((D, I_depth.repeat(1,1,point_num,1)), dim=-1)


        # 只考虑外部轮廓
        R = R.view(batch_size,wedge_num,3,point_num,1)
        pill_sihouette = pill_sihouette.view(batch_size,wedge_num,3,point_num,3)
        H_sihouette = H_sihouette.view(batch_size,wedge_num,3,point_num,3)
        I_sihouette = I_sihouette.view(batch_size,wedge_num,3,point_num,3)
        invaild = torch.sum(n.view(batch_size,wedge_num,3,point_num,2)* edge_out_unit_normal, dim=-1,keepdim=True).lt(0)
        # R_1 = (R[:,:,0:1,:,:].lt(0) & R[:,:,1:2,:,:].lt(0) & R[:,:,2:3,:,:].lt(0))
        # R_2 = (R[:, :, 0:1, :, :].gt(1) & R[:, :,1:2, :, :].gt(1) & R[:, :, 2:3, :, :].gt(1))
        wedge_sihouette = pill_sihouette.masked_fill(invaild, -1)
        wedge_sihouette = torch.where(R<0, H_sihouette, wedge_sihouette)
        wedge_sihouette = torch.where(R>1, I_sihouette, wedge_sihouette)

        return wedge_sihouette.view(batch_size,wedge_num,point_num*3,3)

    def Wedge2Sihouette_Sample(self, A, B, C, A_size, B_size, C_size, sample_point=64):
        device = A.device

        batch_size, wedge_num= A.size()[:2]

        A = A.view(batch_size,wedge_num,1,3)
        B = B.view(batch_size, wedge_num, 1, 3)
        C = C.view(batch_size, wedge_num, 1, 3)
        A_size = A_size.view(batch_size, wedge_num, 1, 1)
        B_size = B_size.view(batch_size, wedge_num, 1, 1)
        C_size = C_size.view(batch_size, wedge_num, 1, 1)


        group_A = torch.stack((A, B ,C),dim=2).view(batch_size,wedge_num*3,1,3)
        group_B = torch.stack((C, A, B), dim=2).view(batch_size, wedge_num * 3, 1, 3)
        group_A_size = torch.stack((A_size, B_size ,C_size),dim=2).view(batch_size,wedge_num*3,1,1)
        group_B_size = torch.stack((C_size, A_size, B_size), dim=2).view(batch_size, wedge_num * 3, 1, 1)

        # only consider 2d
        center = torch.mean(group_A[:,:,:,:2].view(batch_size,wedge_num,3,1,2),dim=2,keepdim=True)
        edge_center = (group_A[:,:,:,:2] + group_B[:,:,:,:2]) / 2
        edge_out_normal = edge_center.view(batch_size,wedge_num,3,1,2) - center
        edge_out_unit_normal = edge_out_normal / self.vec_len(edge_out_normal)
        edge_out_unit_normal = edge_out_unit_normal.view(batch_size,wedge_num*3,1,2)

        #only consider 2d
        residual = group_A_size - group_B_size
        H = torch.where(residual >= 0, group_A[:,:,:,:2], group_B[:,:,:,:2])
        H_depth = torch.where(residual >= 0, group_A[:, :, :, 2:], group_B[:, :, :, 2:])
        I = torch.where(residual < 0, group_A[:,:,:,:2], group_B[:,:,:,:2])
        I_depth = torch.where(residual < 0, group_A[:, :, :, 2:], group_B[:, :, :, 2:])
        H_size = torch.where(residual >= 0, group_A_size, group_B_size)
        I_size = torch.where(residual < 0, group_A_size, group_B_size)
        circle_index = torch.tensor([[0,2],[1,0],[2,1]]).view(1,1,3,2,1).to(device)
        circle_index_inverse = torch.tensor([[2,0], [0,1], [1,2]]).view(1, 1, 3, 2,1).to(device)

        alpha = (torch.arange(sample_point) + 0.5) / (sample_point - 1.0)
        alpha = alpha.view(1,1,sample_point,1).repeat(batch_size,3*wedge_num,1,1).to(device)


        HI = I[:,:,:,:2] - H[:,:,:,:2]
        n_y  = -HI[:,:,:,0:1] /  (HI[:,:,:,1:2]+1e-8)
        n = torch.cat((torch.ones_like(n_y).to(device),n_y),dim=-1)
        n = n / self.vec_len(n)
        n_symmetry = -n

        # N,0 为H,I切点
        N = H + n * H_size
        O = I + n * I_size
        z = (1- alpha) * H_depth + alpha*I_depth
        P1 = (1- alpha) * N + alpha*O
        P1 = torch.cat((P1,z),dim=-1)

        N_s = H + n_symmetry * H_size
        O_s = I + n_symmetry * I_size
        z = (1 - alpha) * H_depth + alpha * I_depth
        P2 = (1 - alpha) * N_s + alpha * O_s
        P2 = torch.cat((P2, z), dim=-1)

        direction = torch.sum(n*edge_out_unit_normal,dim=-1,keepdim=True)
        P_wedge = torch.where(direction.gt(0),P1,P2)

        HN = N-H
        theta_diff = torch.sum(HN*HI,dim=-1,keepdim=True) / self.vec_len(HN)/self.vec_len(HI)
        theta = (torch.arange(sample_point) + 0.5) / (sample_point - 1.0) * 2*np.pi
        theta = theta.view(1,1,sample_point,1).repeat(batch_size,wedge_num*3,1,1).to(device)
        P3  = H + \
              H_size * torch.cat((torch.cos(theta),torch.zeros_like(theta)), dim=-1).to(device)+ \
              H_size * torch.cat((torch.zeros_like(theta),torch.sin(theta)), dim=-1).to(device)
        HP3 = P3 - H
        theta_H = torch.sum(HP3*HI,dim=-1,keepdim=True) / self.vec_len(HN)/self.vec_len(HI)
        theta_H_vaild = torch.where(theta_H <= theta_diff, torch.ones_like(theta_H).to(device),torch.zeros_like(theta_H).to(device))

        P4  = I + \
              I_size * torch.cat((torch.cos(theta),torch.zeros_like(theta)), dim=-1).to(device)+ \
              I_size * torch.cat((torch.zeros_like(theta),torch.sin(theta)), dim=-1).to(device)
        HP4 = P4 - I
        theta_I = torch.sum(HP4*HI,dim=-1,keepdim=True) / self.vec_len(HN) /self.vec_len(HI)
        theta_I_vaild = torch.where(theta_I > theta_diff, torch.ones_like(theta_I).to(device),torch.zeros_like(theta_I).to(device))

        HI_index = torch.where(residual.view(batch_size,wedge_num,3,1,1) >= 0, circle_index, circle_index_inverse).float()
        theta_H_vaild = theta_H_vaild.view(batch_size, wedge_num, 3, sample_point)
        theta_I_vaild = theta_I_vaild.view(batch_size, wedge_num, 3, sample_point)
        theta_ = torch.stack((theta_H_vaild,theta_I_vaild),dim=-2)

        # for point A
        circle_valid = torch.where(HI_index.eq(0), theta_, torch.zeros_like(HI_index).to(device)).sum(dim=-2)
        circle_valid = circle_valid[:,:,0,:].eq(1) & circle_valid[:, :, 1, :].eq(1)
        theta = (torch.arange(sample_point) + 0.5) / (sample_point - 1.0) * 2*np.pi
        theta = theta.view(1,1,sample_point,1).repeat(batch_size,wedge_num,1,1).to(device)
        P  = A[:,:,:,0:2] + \
              A_size * torch.cat((torch.cos(theta),torch.zeros_like(theta)), dim=-1).to(device)+ \
              A_size * torch.cat((torch.zeros_like(theta),torch.sin(theta)), dim=-1).to(device)
        P = torch.cat((P, A[:,:,:,2:].repeat(1,1,sample_point,1)),dim=-1)
        PA = torch.where(circle_valid.view(batch_size,wedge_num,sample_point,1), P, -torch.ones_like(P).to(device))

        # for point B
        circle_valid = torch.where(HI_index.eq(1), theta_, torch.zeros_like(HI_index).to(device)).sum(dim=-2)
        circle_valid = circle_valid[:,:,1,:].eq(1) & circle_valid[:, :, 2, :].eq(1)
        theta = (torch.arange(sample_point) + 0.5) / (sample_point - 1.0) * 2*np.pi
        theta = theta.view(1,1,sample_point,1).repeat(batch_size,wedge_num,1,1).to(device)
        P  = B[:,:,:,0:2] + \
              B_size * torch.cat((torch.cos(theta),torch.zeros_like(theta)), dim=-1).to(device)+ \
              B_size * torch.cat((torch.zeros_like(theta),torch.sin(theta)), dim=-1).to(device)
        P = torch.cat((P, B[:,:,:,2:].repeat(1,1,sample_point,1)),dim=-1)
        PB = torch.where(circle_valid.view(batch_size,wedge_num,sample_point,1), P, -torch.ones_like(P).to(device))

        # for point C
        circle_valid = torch.where(HI_index.eq(2), theta_, torch.zeros_like(HI_index).to(device)).sum(dim=-2)
        circle_valid = circle_valid[:,:,0,:].eq(1) & circle_valid[:, :, 2, :].eq(1)
        theta = (torch.arange(sample_point) + 0.5) / (sample_point - 1.0) * 2*np.pi
        theta = theta.view(1,1,sample_point,1).repeat(batch_size,wedge_num,1,1).to(device)
        P  = C[:,:,:,0:2] + \
              C_size * torch.cat((torch.cos(theta),torch.zeros_like(theta)), dim=-1).to(device)+ \
              C_size * torch.cat((torch.zeros_like(theta),torch.sin(theta)), dim=-1).to(device)
        P = torch.cat((P, C[:,:,:,2:].repeat(1,1,sample_point,1)),dim=-1)
        PC = torch.where(circle_valid.view(batch_size,wedge_num,sample_point,1), P, -torch.ones_like(P).to(device))

        P_wedge = P_wedge.view(batch_size, wedge_num, 3 * sample_point, 3)
        P = torch.cat((P_wedge,PA,PB,PC),dim=-2)
        return P



    #返回切线的单位法向量以及对称的单位法向量（都是对于目标节点来说）
    # H为大圆，I为小圆，residual_radius为两园半径的差值
    def CircleTangent(self,P, H, I, residual_radius):
        device = P.device
        HI = I - H
        HI_len = self.vec_len(HI)
        HI_unit = HI / (HI_len+1e-8)
        # 此时I内含/内切于H,没有切点
        incircle = (HI_len - residual_radius).lt(0)
        HG = HI_unit * residual_radius
        GK_len = torch.sum(HI_len * HI_len - residual_radius * residual_radius, dim=-1, keepdim=True)
        GK_len = torch.where(incircle, torch.zeros_like(GK_len).to(device), GK_len)
        GK_len = torch.sqrt(GK_len + 1e-8)

        PW = -torch.sum((P - H)*HI_unit,dim=-1,keepdim=True) * HI_unit + (P - H)
        GK = PW / self.vec_len(PW) * GK_len
        HK_unit = (HG+GK)/ self.vec_len((HG+GK))

        GK_symmetry = (-PW / self.vec_len(PW)) * GK_len
        HK_unit_symmetry = (HG+GK_symmetry)/ self.vec_len((HG+GK_symmetry))

        return HK_unit, HK_unit_symmetry, incircle

    def TestPoint2Pill(self):
        batch_size = 8
        A = torch.tensor([0, 0., 0.]).view(1,1,1,3).repeat(batch_size,1,1,1)
        B = torch.tensor([0.6, 0., 0.]).view(1,1,1,3).repeat(batch_size,1,1,1)
        A_size = torch.tensor([0.2]).view(1,1,1,1).repeat(batch_size,1,1,1)
        B_size = torch.tensor([0.1]).view(1,1,1,1).repeat(batch_size,1,1,1)
        P = torch.tensor([[2,0,0], [2,1,0],[2,2,0],
                          [0, 0, 0], [0, 1, 0], [0, 2, 0],
                          [-1, 0, 0], [-1, 1, 0], [-1, 2, 0],
                          [0,0,-4],[0,0,-2],[0,0,-1],[0,0,0],[0,0,1],[0,0,2],[0,0,3],

                          [3, 1, 1], [3, 1, -1], [3, 1, 0],
                          [4,1,1],[4,1,-1],[4,1,0],
                          [7,2,0],
                          [-1,3,1],[-1,3,-1],[-1,3,0],
                          [3,-2,0]]).view(1,-1,3) / 10.0
        P_2d = P[:,:,0:2]
        dist = self.Pill2Depth(P_2d,A,B,A_size,B_size)
        return dist

    def TestPoint2Wedge(self):
        batch_size = 1
        sr = 128
        A = torch.tensor([0., 0., -1]).view(1, 1, 3).repeat(batch_size, 1, 1)/10
        B = torch.tensor([3., 0, -1.2]).view(1, 1, 3).repeat(batch_size, 1, 1)/10
        C = torch.tensor([0., 3, -1.5]).view(1, 1, 3).repeat(batch_size, 1, 1)/10
        A_size = torch.tensor([1. ]).view(1, 1, 1).repeat(batch_size, 1, 1)/10
        B_size = torch.tensor([2.]).view(1, 1, 1).repeat(batch_size, 1, 1)/10
        C_size = torch.tensor([3.]).view(1, 1, 1).repeat(batch_size, 1, 1)/10
        mesh_x = 2.0 * torch.arange(sr).unsqueeze(1).expand(sr, sr).float() / (sr - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(sr).unsqueeze(0).expand(sr, sr).float() / (sr - 1.0) - 1.0
        P = torch.stack((mesh_y,mesh_x), dim=-1).view(1,1,sr*sr,2).repeat(batch_size,1,1,1)
        # P = torch.tensor([[-1.,1.],[0.,1.],[0.1,1.],[1.,1.]]).view(1,1,-1,2).repeat(batch_size,1,1,1)/10.
        dist = self.Wedge2Depth(P,A,B,C,A_size,B_size,C_size)
        return dist

    def TestPill2Sihouette(self):
        batch_size = 8
        A = torch.tensor([0, 0.5, 0.5]).view(1,1,3).repeat(batch_size,1,1)
        B = torch.tensor([0, -0.2, 0.]).view(1,1,3).repeat(batch_size,1,1)
        A_size = torch.tensor([.2]).view(1,1,1).repeat(batch_size,1,1)
        B_size = torch.tensor([.1]).view(1,1,1).repeat(batch_size,1,1)
        sihouette = self.Pill2Sihouette_Sample(A,B,A_size,B_size,32)
        return sihouette

    def TestWedge2Sihouette(self):
        batch_size = 8
        A = torch.tensor([0., 0., 0.4]).view(1, 1, 3).repeat(batch_size, 1, 1)
        B = torch.tensor([0.6, 0, 0.9]).view(1, 1, 3).repeat(batch_size, 1, 1)
        C = torch.tensor([0.5, 0.7, 0]).view(1, 1, 3).repeat(batch_size, 1, 1)
        A_size = torch.tensor([.1]).view(1, 1, 1).repeat(batch_size, 1, 1)
        B_size = torch.tensor([.12]).view(1, 1, 1).repeat(batch_size, 1, 1)
        C_size = torch.tensor([.13]).view(1, 1, 1).repeat(batch_size, 1, 1)
        P = torch.tensor([[4,3,1], [4,3,-1],[4,3,0],
                          [3,1,1],[3,1,-1],[3,1,0],
                          [7,2,0],
                          [-1,3,1],[-1,3,-1],[-1,3,0],
                          [3,-2,0]]).view(1,-1,3).repeat(batch_size,1,1)
        dist = self.Wedge2Sihouette_Sample(A,B,C,A_size,B_size,C_size,64)
        return dist

    def vec_len(self, x):
        return torch.sqrt(torch.sum(torch.pow(x, 2),dim=-1,keepdim=True)+1e-8)

    def BoneLen(self, joint_xyz, dataset):
        bone_list = get_HandModel_bone(dataset)
        A = []
        B = []
        for pair in bone_list:
            A.append(joint_xyz[:, pair[0], :])
            B.append(joint_xyz[:, pair[1], :])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        bone_len = self.vec_len(A-B)
        return bone_len.squeeze(-1)

    # select feature
    def joint2depth(self, img, joint, feature_size, feature_paras, dataset):
        # softmax= torch.nn.Softmax2d()
        bone_group = get_sketch_group(dataset)
        boneDist = self.joint2boneDist(img,joint,feature_size,feature_paras[0],dataset)
        com_list = []
        for group in bone_group:
            grout_list = []
            for bone in group:
                grout_list.append(boneDist[:,bone,:,:])
            com_list.append(torch.max(torch.stack(grout_list, dim=1),dim=1,keepdim=True)[0])
        com_list = torch.cat(com_list, dim=1)
        background = 1 - torch.max(com_list,dim=1,keepdim=True)[0]
        com_list = torch.cat((com_list,background),dim=1)

        joint_group = get_joint_group(dataset)
        joint_num = joint.size(1)
        offset = self.joint2offset(joint, img, feature_paras[0]*1.5, feature_size)
        heatmap = offset[:,joint_num*3:,:,:]
        com_list2 = []
        for group in joint_group:
            grout_list = []
            for joint_index in group:
                grout_list.append(heatmap[:,joint_index:joint_index+1,:,:])
            background = 1 - torch.max(torch.cat(grout_list, dim=1),dim=1,keepdim=True)[0]
            grout_list.append(background)
            com_list2.append(torch.cat(grout_list, dim=1))
        com_list2 = torch.cat(com_list2, dim=1)
        return torch.cat([com_list,com_list2],dim=1)

    # select feature
    def joint2boneFeature(self, img, joint, feature_size, dataset):
        bone_group = get_sketch_group(dataset)
        bone_label = self.joint2boneHeatmap(img,joint,feature_size,dataset)
        com_list = []
        for group in bone_group:
            grout_list = []
            for bone in group:
                grout_list.append(bone_label[:,bone,:,:])
            com_list.append(torch.min(torch.stack(grout_list, dim=1),dim=1,keepdim=True)[0])
        finger_label= torch.cat(com_list, dim=1)
        hand_label = torch.min(finger_label,dim=1,keepdim=True)[0]
        DRM = torch.cat((finger_label,bone_label),dim=1)
        CRM = torch.cat((hand_label, finger_label), dim=1)
        KRM = bone_label
        return DRM, CRM, KRM

    # predict joint from pixel-wise estimation
    def feature2joint(self, img, pixel_pd, feature_types, feature_paras):
        for feature_index, feature_type in enumerate(feature_types):
            if feature_type == 'heatmap':
                device = img.device
                joint_uv = self.heatmap2joint_softmax(pixel_pd)
                joint_d = torch.zeros(joint_uv.size(0),joint_uv.size(1),1).to(device)
                joint = torch.cat((joint_uv,joint_d),dim=-1)
            elif feature_type == 'offset':
                # joint = self.offset2joint(pixel_pd, img, feature_paras[feature_index])
                joint = self.offset2joint_softmax(pixel_pd, img, feature_paras[feature_index])
            elif feature_type == 'weight_offset':
                joint = self.offset2joint_weight(pixel_pd, img, feature_paras[feature_index])
                # joint = self.offset2joint_weight_sample(pixel_pd, img, feature_paras[feature_index])
            elif feature_type == 'weight_offset_nosoftmax':
                joint = self.offset2joint_weight_nosoftmax(pixel_pd, img, feature_paras[feature_index])
            elif feature_type == 'heatmap_depth':
                joint = self.heatmap_depth2joint(pixel_pd, img)
            elif feature_type == 'heatmap_depthoffset':
                joint = self.heatmap_depthoffset2joint(pixel_pd, img)
            elif feature_type == 'plainoffset_depth':
                joint = self.plainoffset_depth2joint(img, pixel_pd, feature_paras[feature_index])
            elif feature_type == 'plainoffset_depthoffset':
                joint= self.plainoffset_depthoffset2joint(img, pixel_pd, feature_paras[feature_index])
            elif feature_type == 'weight_pos':
                joint = self.weight_pos2joint(pixel_pd)

        return joint


    # 2020/5/16
    ################### for HGCN  #################################
    def Joint2GCNbone(self, joint, dataset):
        bone_list = get_bone_id_setting(dataset)
        A = []
        B = []
        for pair in bone_list:
            A.append(joint[:, pair[0], :])
            B.append(joint[:, pair[1], :])
        A = torch.stack(A, dim=1)
        B = torch.stack(B, dim=1)
        bone_len = self.vec_len(A-B)
        bone_mean = (A+B)/2
        bone_offset = (A-B)/bone_len
        bone_info = torch.cat((bone_len, bone_mean, bone_offset), dim=-1)
        return bone_info
