import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures.meshes import Meshes


class depth_loss(torch.nn.Module):
    def __init__(self, beta=0.4, smooth=False):
        super(depth_loss, self).__init__()
        self.smooth = smooth
        self.smoothLoss = torch.nn.SmoothL1Loss(beta=beta)

    def forward(self, real, synth):
        if not self.smooth:
            diff = real - synth
            mask_1 = real.lt(0.99) & synth.lt(0.99)
            mask_2 = torch.rand_like(real).to(real.device).lt(1)
            select = torch.masked_select(diff, mask_1 & mask_2)
            return torch.abs(select).mean()
        else:
            mask_1 = real.lt(0.99) & synth.lt(0.99)
            mask_2 = torch.rand_like(real).to(real.device).lt(self.sample_rate)
            mask = mask_1 & mask_2
            select_real = torch.masked_select(real, mask)
            select_synth = torch.masked_select(synth, mask)
            return self.smoothLoss(select_synth, select_real)

    # def forward(self, real, synth):
    #     if not self.smooth:
    #         diff = real - synth
    #         mask_1 = real.lt(0.99) & synth.lt(0.99)
    #         mask_2 = torch.rand_like(real).to(real.device).lt(self.sample_rate)
    #         select = torch.masked_select(diff, mask_1 & mask_2)
    #         return self.l1Loss(select)
class surface_loss(torch.nn.Module):
    def __init__(self):
        super(surface_loss, self).__init__()
        self.img_size = 128
        self.paras = (588.03, 587.07, 320., 240.)
        self.flip = 1

    def forward(self, real, synth, verts, faces, center, M, cube):
        batch_size = real.size(0)
        pcl1 = self.Img2pcl(real, center, M, cube, verts)
        # pcl2 = self.Img2pcl(synth, center, M, cube, verts)
        # mesh = Meshes(verts=verts, faces=faces.unsqueeze(0).repeat(batch_size, 1, 1))
        # pcl2 = sample_points_from_meshes(mesh, 1024)
        loss, _ = chamfer_distance(pcl1, verts)
        return loss

    def Img2pcl(self, img, center, M, cube, verts=None, sample_num=1024):
        batch_size = img.size(0)
        device = img.device
        img_rs = F.interpolate(img, (self.img_size, self.img_size))
        mask = img_rs.le(0.99)
        mesh_x = 2.0 * torch.arange(self.img_size).unsqueeze(1).expand(self.img_size, self.img_size).float() / (self.img_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(self.img_size).unsqueeze(0).expand(self.img_size, self.img_size).float() / (self.img_size - 1.0) - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        # coords = torch.cat((coords, img_rs), dim=1)
        # return coords
        pcl = torch.cat((coords, img_rs), dim=1).view(batch_size, 3, self.img_size*self.img_size).permute(0, 2, 1)
        pcl = torch.split(pcl, 1)
        mask = torch.split(mask.view(batch_size, 1, self.img_size*self.img_size).permute(0, 2, 1), 1)
        pcl_valid = []
        for index in range(batch_size):
            temp = torch.masked_select(pcl[index], mask[index]).view(1, -1, 3)
            temp = self.uvd_nl2xyznl_tensor(temp, center[index], M[index], cube[index])
            temp = temp.squeeze(0)
            point_num = temp.size(0)
            if point_num < 2:
                # pcl_valid.append(torch.zeros(sample_num,3).to(device))
                return verts
            elif sample_num > point_num:
                mult = int(np.floor(sample_num/point_num))
                point_mult = temp.repeat(mult, 1)
                if sample_num % point_num == 0:
                    pcl_valid.append(point_mult)
                else:
                    pcl_index = torch.multinomial(torch.ones(point_num).to(device), sample_num-point_num*mult, False)
                    pcl_valid.append(torch.cat((point_mult, torch.index_select(temp, 0, pcl_index).view(-1,3)),dim=0))
            else:
                pcl_index = torch.multinomial(torch.ones(point_num).to(device), sample_num, False)
                pcl_valid.append(torch.index_select(temp, 0, pcl_index))
        return torch.stack(pcl_valid, dim=0)

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
        joints_trans_xy = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans_z = joints[:, :, 2:]
        return torch.cat((joints_trans_xy,joints_trans_z),dim=-1)

    def uvd_nl2xyznl_tensor(self, uvd, center, m, cube):
        batch_size, point_num, _ = uvd.size()
        device = uvd.device
        cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        M_t = m.to(device).view(batch_size, 1, 3, 3).repeat(1, point_num, 1, 1)
        M_inverse = torch.inverse(M_t)

        uv_unnormal = (uvd[:, :, 0:2] + 1) * (self.img_size / 2)
        d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
        uvd_unnormal = torch.cat((uv_unnormal,d_unnormal),dim=-1)
        uvd_world = self.get_trans_points(uvd_unnormal, M_inverse)
        xyz = self.pointsImgTo3D(uvd_world)
        xyz_noraml = (xyz - center_t) / (cube_size_t / 2.0)
        return xyz_noraml

    def pointsImgTo3D(self, point_uvd):
        point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
        point_xyz[:, :, 0] = (point_uvd[:, :, 0] - self.paras[2]) * point_uvd[:, :, 2] / self.paras[0]
        point_xyz[:, :, 1] = self.flip * (point_uvd[:, :, 1] - self.paras[3]) * point_uvd[:, :, 2] / self.paras[1]
        point_xyz[:, :, 2] = point_uvd[:, :, 2]
        return point_xyz