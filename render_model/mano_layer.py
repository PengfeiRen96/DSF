'''
    file:   MANO_SMPL.py
    date:   2019_07_15
    modifier: Seungryul Baek and Anil Armagan
    source:   This code is modified from SMPL.py of https://github.com/MandyMo/pytorch_HMR.
    If you use this code for your research, please cite:

    @article{sbaek_cvpr_2019,
      title={Pushing the envelope for RGB-based dense 3D hand pose estimation via neural rendering},
      author={Seungryul Baek and Kwang In Kim and Tae-Kyun Kim},
      journal={CVPR},
      year={2019}
    }
    and
    @article{zhang2019end,
      title={End-to-end Hand Mesh Recovery from a Monocular RGB Image},
      author={Zhang, Xiong and Li, Qiang and Zhang, Wenbo and Zheng, Wen},
      journal={arXiv preprint arXiv:1902.09305},
      year={2019}
    }
'''

import cv2
import numpy as np
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numbers

from torchvision.ops import RoIAlign

# joint mapping indices from mano to bighand
MANO2HANDS = [0, 13, 1, 4, 10, 7, 14, 15, 20, 2, 3, 16, 5, 6, 17, 11, 12, 19, 8, 9, 18]
MANO2MSRA = [
  0,
  1,2,3,16,
  4,5,6,17,
  10,11,12,19,
  7,8,9,18,
  13,14,15,20
]
# MANO2ICVL = [
#   0,
#   14, 15, 20,
#   1, 3, 16,
#   4, 6, 17,
#   10, 12, 19,
#   7, 9, 18
# ]
# MANO2ICVL = [
#   0,
#   13, 14, 15,
#   1, 2, 16,
#   4, 5, 17,
#   10, 11, 19,
#   7, 8, 18
# ]

MANO2ICVL = [
  0,
  13, 14, 15,
  1, 2, 3,
  4, 5, 6,
  10, 11, 12,
  7, 8, 9
]

MANO2NYU = [
    18, 8,
    19, 11,
    17, 5,
    16, 2,
    20, 15, 14,
    0,
    # 1, 3, 4, 6, 7, 9, 10, 12, 13,
]

HANDS2MANO = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 11, 14, 20, 17, 8]
class MANO_SMPL(nn.Module):
        def __init__(self, mano_pkl_path, dataset, scale=1000):
            super(MANO_SMPL, self).__init__()
            if 'msra' in dataset:
                self.transfer = MANO2MSRA
            elif 'icvl' in dataset:
                self.transfer = MANO2ICVL
            elif 'hands' in dataset:
                self.transfer = MANO2HANDS
            elif 'nyu' in dataset:
                self.transfer = MANO2NYU
            else:
                self.transfer = range(21)

            self.dataset = dataset
            # Load the MANO_RIGHT.pkl
            with open(mano_pkl_path, 'rb') as f:
                model = pickle.load(f, encoding='latin1')

            self.scale = scale
            self.faces = torch.from_numpy(np.array(model['f'], dtype=np.float)).float()
            self.wrist_faces = torch.from_numpy(np.array([[121, 214, 778],[214, 215, 778],[215, 279, 778],[279, 239, 778],
                                                          [239, 234, 778],[234,92, 778],[92,38, 778],[38,122, 778],[122,118, 778],[118,117, 778],
                                                          [117,119, 778],[119,120, 778],[120,108, 778],[108,79, 778],[79,78, 778],[78,121, 778]])).float()
            self.faces = torch.cat((self.faces, self.wrist_faces), dim=0)
            # check if cuda available
            self.is_cuda = False
            if torch.cuda.is_available():
                self.is_cuda = True

            np_v_template = np.array(model['v_template'], dtype=np.float)
            np_v_template = torch.from_numpy(np_v_template).float()

            self.size = [np_v_template.shape[0], 3]
            np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
            self.num_betas = np_shapedirs.shape[-1]
            np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
            # 用来恢复形状的矩阵
            np_shapedirs = torch.from_numpy(np_shapedirs).float()

            # Adding new joints for the fingertips. Original MANO model provide only 16 skeleton joints.
            np_J_regressor = model['J_regressor'].T.toarray()
            np_J_addition = np.zeros((778, 5))
            np_J_addition[333][0] = 1
            np_J_addition[444][1] = 1
            np_J_addition[672][2] = 1
            np_J_addition[555][3] = 1
            np_J_addition[744][4] = 1
            # [333, 444, 672, 555, 744]
            np_J_regressor = np.concatenate((np_J_regressor, np_J_addition), axis=1)
            np_J_regressor = torch.from_numpy(np_J_regressor).float()

            # 用来恢复姿态的矩阵,存储的实际上是经过PCA之后的参数，所以需要通过矩阵乘法去恢复
            np_hand_component = np.array(model['hands_components'], dtype=np.float)
            np_hand_component = torch.from_numpy(np_hand_component).float()

            np_hand_mean = np.array(model['hands_mean'], dtype=np.float)
            np_hand_mean = torch.from_numpy(np_hand_mean).float()

            # 由角度恢复形状
            np_posedirs = np.array(model['posedirs'], dtype=np.float)
            num_pose_basis = np_posedirs.shape[-1]
            np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
            np_posedirs = torch.from_numpy(np_posedirs).float()

            self.parents = np.array(model['kintree_table'])[0].astype(np.int32)
            # 每一个顶点对于目标关节点的权重
            np_weights = np.array(model['weights'], dtype=np.float)
            vertex_count = np_weights.shape[0]
            vertex_component = np_weights.shape[1]
            np_weights = np_weights.reshape([-1, vertex_count, vertex_component])
            vertex_seg = np.argmax(np_weights, axis=-1)
            np_weights = torch.from_numpy(np_weights).float().squeeze(0)

            self.vertex_seg = torch.from_numpy(vertex_seg).float().squeeze(0)
            self.vertex_joint_index_list = []
            for index in range(vertex_component):
                self.vertex_joint_index_list.append(self.vertex_seg.eq(index).nonzero().squeeze(-1))

            vertex_joint = []
            for index in range(vertex_component):
                vertex_joint.append(np_weights[:, index].gt(0.1).nonzero().squeeze())

            self.joint_faces = []
            for index in range(1, vertex_component):
                joint_face = []
                for face in self.faces:
                    if (face.unsqueeze(1) - vertex_joint[index].unsqueeze(0)).eq(0).sum()>0:
                        joint_face.append(face)
                self.joint_faces.append(torch.stack(joint_face, dim=0))

            # finger segment
            self.vertex_finger_index_list = []
            for index in range(5):
                self.vertex_finger_index_list.append(torch.cat((vertex_joint[3*index+1],vertex_joint[3*index+2],vertex_joint[3*index+3])))
            Joint2Finger = np.array([0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5])
            self.finger_seg = torch.from_numpy(Joint2Finger[vertex_seg[0]])
            self.finger_faces = []
            for index in range(5):
                finger_face = []
                for face in self.faces:
                    if (face.unsqueeze(1) - self.vertex_finger_index_list[index].unsqueeze(0)).eq(0).sum()>0:
                        finger_face.append(face)
                self.finger_faces.append(torch.stack(finger_face, dim=0))


            # self.faces = self.joint_faces[2]
            e3 = torch.eye(3).float()
            np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
            np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [1, 3, 3])
            self.base_rot_mat_x = torch.from_numpy(np_rot_x).float()

            if self.is_cuda:
                np_v_template = np_v_template.cuda()
                np_shapedirs = np_shapedirs.cuda()
                np_J_regressor = np_J_regressor.cuda()
                np_hand_component = np_hand_component.cuda()
                np_hand_mean = np_hand_mean.cuda()
                np_posedirs = np_posedirs.cuda()
                e3 = e3.cuda()
                np_weights = np_weights.cuda()
                self.base_rot_mat_x = self.base_rot_mat_x.cuda()
                self.faces = self.faces.cuda()
                for index in range(5):
                    self.finger_faces[index] = self.finger_faces[index].cuda()
                for index in range(vertex_component-1):
                    self.joint_faces[index] = self.joint_faces[index].cuda()
            # vertex importance
            # importance_joint = [2, 3, 4, 5, 8, 9, 11, 12, 14, 15]
            # self.vertex_weights = np_weights[:, :, importance_joint].mean(-1, keepdim=True)

            self.register_buffer('v_template', np_v_template)
            self.register_buffer('shapedirs', np_shapedirs)
            self.register_buffer('J_regressor', np_J_regressor)
            self.register_buffer('hands_comp', np_hand_component)
            self.register_buffer('hands_mean', np_hand_mean)
            self.register_buffer('posedirs', np_posedirs)
            self.register_buffer('e3', e3)
            self.register_buffer('weight', np_weights)            # vertex importance
            # importance_joint = [2, 3, 4, 5, 8, 9, 11, 12, 14, 15]
            # self.vertex_weights = np_weights[:, :, importance_joint].mean(-1, keepdim=True)


            self.cur_device = None
            self.rotate_base = False

            # finger
            self.child = [2, 3, 16, 5, 6, 17, 8, 9, 18, 11, 12, 19, 14, 15, 20]
            self.per_adj_shpere = 2
            self.interval_value = torch.linspace(0, 1, self.per_adj_shpere + 2)[:-1].reshape(1, 1, -1)
            self.interval = self.per_adj_shpere + 1

            # plam bone interval
            self.plam_per_adj_shpere = 4
            self.plam_interval_value = torch.linspace(0, 1, self.plam_per_adj_shpere + 2)[1:-1].reshape(1, 1, -1)
            self.plam_interval = self.plam_per_adj_shpere + 1

            # palm mask 手掌球体之间不做碰撞约束
            plam_shpere_n = 1 + 5*self.plam_per_adj_shpere
            finger_shpere_n = 15*(self.per_adj_shpere+1)
            plam_mask = torch.zeros([plam_shpere_n, plam_shpere_n])
            plam_mask_add = torch.ones([plam_shpere_n, finger_shpere_n])
            plam_mask = torch.cat((plam_mask,plam_mask_add), dim=1)

            # finger mask 同一个骨骼之间不做碰撞约束
            finger_mask = torch.ones(finger_shpere_n, plam_shpere_n+finger_shpere_n)
            # 考虑同一个手指之间的碰撞
            for finger_index in range(0, 15):
                finger_root_index = int(finger_index / 3) + 1
                # 指跟节点
                if finger_index in [0, 3, 6, 9, 12]:
                    for finger_bone_index in range(self.interval):
                        finger_mask[self.interval * finger_index + finger_bone_index, finger_root_index * self.plam_per_adj_shpere] = 0
                        plam_mask[finger_root_index*self.plam_per_adj_shpere, plam_shpere_n + self.interval * finger_index + finger_bone_index] = 0
                        shpere_index = plam_shpere_n + self.interval * finger_index
                        finger_mask[self.interval * finger_index + finger_bone_index, shpere_index:shpere_index+self.interval+3] = 0
                # 其他位置的关节点
                else:
                    shpere_index = plam_shpere_n + self.interval * finger_index
                    max_index = plam_shpere_n + (3*self.interval) * finger_root_index
                    for finger_bone_index in range(self.interval):
                        finger_mask[self.interval * finger_index + finger_bone_index, (shpere_index - self.interval):min(shpere_index + self.interval*2+1, max_index)] = 0
            # 特别的删除拇指指跟与掌心的碰撞
            thumb_root_index = 12*self.interval
            finger_mask[thumb_root_index:thumb_root_index+self.interval+1, :plam_shpere_n] = 0
            self.mask = torch.cat((plam_mask, finger_mask), dim=0)
            # 特别的删除拇指跟与掌心的碰撞
            self.mask[:plam_shpere_n, thumb_root_index+plam_shpere_n:plam_shpere_n+thumb_root_index+self.interval+1] = 0

        def get_sphere_radius(self, joints, mesh):
            # N*J
            batch_size, joint_num, _ = joints.size()
            device = joints.device
            joint_vertex = self.J_regressor.clone().gt(0).unsqueeze(0).repeat(batch_size, 1, 1).permute(0,2,1)
            # joint_center = joints.clone()
            joint_r = joints.unsqueeze(2) - mesh.unsqueeze(1)[:,:,:778,:]
            dis = torch.sqrt(torch.sum(joint_r*joint_r, dim=-1) + 1e-8)
            joint_r = torch.where(joint_vertex, dis, torch.ones_like(dis).to(device)*100)
            joint_r = torch.mean(torch.topk(joint_r, k=10, dim=-1,largest=False)[0], dim=-1)
            joint_r = torch.cat((joint_r[:, :16], joint_r[:,[3,6,9,12,15]]/1.5), dim=-1)

            # for plam radius
            plam_child = joint_r[:, [1, 4, 7, 10, 13]]
            plam_parent = torch.clamp(joint_r[:, 0:1] - 0.05, 0.01, 0.4)
            plam_bone = plam_child - plam_parent
            plam_shpere_r = plam_bone.reshape([batch_size, -1, 1]) * self.plam_interval_value.to(device) \
                          + plam_parent.reshape([batch_size, -1, 1])
            plam_shpere_r = torch.cat((plam_parent, plam_shpere_r.view(batch_size, -1)), dim=1)

            # for finger radius
            finger_child = joint_r[:, self.child]
            finger_parent = joint_r[:, 1:16]
            finger_shpere_r = finger_child - finger_parent
            finger_shpere_r = finger_shpere_r.reshape([batch_size, -1, 1]) * self.interval_value.to(device)\
                       + finger_parent.reshape([batch_size, -1, 1])

            shpere_r = torch.cat((plam_shpere_r.view(batch_size, -1), finger_shpere_r.view(batch_size, -1)), dim=1)

            # for plam center
            plam_child_c = joints[:, [1, 4, 7, 10, 13],:]
            plam_parent_c = joints[:, 0:1,:]
            plam_bone_c = plam_child_c - plam_parent_c
            plam_shpere_c = plam_bone_c.reshape([batch_size, -1, 1, 3]) * self.plam_interval_value.to(device).unsqueeze(-1) \
                          + plam_parent_c.reshape([batch_size, -1, 1, 3])
            plam_shpere_c = torch.cat((plam_parent_c, plam_shpere_c.view(batch_size,-1,3)),dim=1)

            # for finger center
            finger_child_c = joints[:, self.child, :]
            finger_parent_c = joints[:, 1:16, :]
            finger_shpere_c = finger_child_c - finger_parent_c
            finger_shpere_c = finger_shpere_c.reshape([batch_size, -1, 1, 3]) * self.interval_value.to(device).unsqueeze(-1) \
                       + finger_parent_c.reshape([batch_size, -1, 1, 3])

            shpere_c = torch.cat((plam_shpere_c.view(batch_size, -1, 3), finger_shpere_c.view(batch_size, -1, 3)), dim=1)

            return shpere_c, shpere_r

        def get_sphere(self, joints):
            # N*J
            batch_size, joint_num, _ = joints.size()
            device = joints.device
            # for plam center
            plam_child_c = joints[:, [1, 4, 7, 10, 13],:]
            plam_parent_c = joints[:, 0:1,:]
            plam_bone_c = plam_child_c - plam_parent_c
            plam_shpere_c = plam_bone_c.reshape([batch_size, -1, 1, 3]) * self.plam_interval_value.to(device).unsqueeze(-1) \
                          + plam_parent_c.reshape([batch_size, -1, 1, 3])
            plam_shpere_c = torch.cat((plam_parent_c, plam_shpere_c.view(batch_size,-1,3)),dim=1)

            # for finger center
            finger_child_c = joints[:, self.child, :]
            finger_parent_c = joints[:, 1:16, :]
            finger_shpere_c = finger_child_c - finger_parent_c
            finger_shpere_c = finger_shpere_c.reshape([batch_size, -1, 1, 3]) * self.interval_value.to(device).unsqueeze(-1) \
                       + finger_parent_c.reshape([batch_size, -1, 1, 3])

            shpere_c = torch.cat((plam_shpere_c.view(batch_size, -1, 3), finger_shpere_c.view(batch_size, -1, 3)), dim=1)

            return shpere_c

        def get_radius(self, joints, mesh):
            # N*J
            batch_size, joint_num, _ = joints.size()
            device = joints.device
            joint_vertex = self.J_regressor.clone().gt(0).unsqueeze(0).repeat(batch_size, 1, 1).permute(0,2,1)
            # joint_center = joints.clone()
            joint_r = joints.unsqueeze(2) - mesh.unsqueeze(1)[:,:,:778,:]
            dis = torch.sqrt(torch.sum(joint_r*joint_r, dim=-1) + 1e-8)
            joint_r = torch.where(joint_vertex, dis, torch.ones_like(dis).to(device)*100)
            joint_r = torch.mean(torch.topk(joint_r, k=10, dim=-1,largest=False)[0], dim=-1)
            joint_r = torch.cat((joint_r[:, :16], joint_r[:,[3,6,9,12,15]]/1.5), dim=-1)

            # for plam radius
            plam_child = joint_r[:, [1, 4, 7, 10, 13]]
            plam_parent = torch.clamp(joint_r[:, 0:1] - 0.05, 0.01, 0.4)
            plam_bone = plam_child - plam_parent
            plam_shpere_r = plam_bone.reshape([batch_size, -1, 1]) * self.plam_interval_value.to(device) \
                          + plam_parent.reshape([batch_size, -1, 1])
            plam_shpere_r = torch.cat((plam_parent, plam_shpere_r.view(batch_size, -1)), dim=1)

            # for finger radius
            finger_child = joint_r[:, self.child]
            finger_parent = joint_r[:, 1:16]
            finger_shpere_r = finger_child - finger_parent
            finger_shpere_r = finger_shpere_r.reshape([batch_size, -1, 1]) * self.interval_value.to(device)\
                       + finger_parent.reshape([batch_size, -1, 1])

            shpere_r = torch.cat((plam_shpere_r.view(batch_size, -1), finger_shpere_r.view(batch_size, -1)), dim=1)

            return shpere_r

        def calculate_coll(self, joints, meshs):
            device = joints.device
            batch_size, joint_num, _ = joints.size()
            shpere_c, shpere_r = self.get_sphere_radius(joints.clone(), meshs)

            dis = shpere_c.reshape([batch_size, -1, 1, 3]) - shpere_c.reshape([batch_size, 1, -1, 3])
            dis = torch.sqrt(torch.sum(torch.pow(dis, 2), dim=-1) + 1e-8)
            pair_r = shpere_r.reshape([batch_size, -1, 1]) + shpere_r.reshape([batch_size, 1, -1])
            error = torch.max(pair_r - dis, torch.zeros_like(dis).to(device))
            error = error * self.mask.to(device)
            batch_mask = error.sum(-1, keepdim=True).sum(-1, keepdim=True).lt(0.1).float()
            error = error * batch_mask
            return torch.mean(error.sum(-1))
            # return error.sum(-1).sum(-1)

        def calculate_PWE_coll(self, joints_PWE, joints, meshs):
            device = joints.device
            batch_size, joint_num, _ = joints.size()
            shpere_r = self.get_radius(joints.clone(), meshs)
            shpere_c = self.get_sphere(joints_PWE.clone())
            dis = shpere_c.reshape([batch_size, -1, 1, 3]) - shpere_c.reshape([batch_size, 1, -1, 3])
            dis = torch.sqrt(torch.sum(torch.pow(dis, 2), dim=-1) + 1e-8)
            pair_r = shpere_r.reshape([batch_size, -1, 1]) + shpere_r.reshape([batch_size, 1, -1])
            error = torch.max(pair_r - dis, torch.zeros_like(dis).to(device))
            error = error * self.mask.to(device)
            batch_mask = error.sum(-1, keepdim=True).sum(-1, keepdim=True).lt(0.1).float()
            error = error * batch_mask
            return torch.mean(error.sum(-1))


        # use shpere to calculate pcl seg 15
        def seg_pcl(self, joints, joints_mano, mesh, pcl):
            device = joints.device
            batch_size, joint_num, _ = joints.size()
            shpere_c, _ = self.get_sphere_radius(joints.clone(), mesh)
            _, shpere_r = self.get_sphere_radius(joints_mano.clone(), mesh)
            plam_spere_num = 1 + 5 * self.plam_per_adj_shpere
            finger_sphere_c = shpere_c[:, plam_spere_num:]
            finger_shpere_r = shpere_r[:, plam_spere_num:]

            finger_dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - finger_sphere_c.unsqueeze(1), 2).sum(dim=-1) + 1e-8)  # B P C
            finger_dis = torch.abs(finger_dis - finger_shpere_r.unsqueeze(1))
            finger_dis, finger_dis_min_id = torch.min(finger_dis, dim=-1)
            finger_dis_min_id = (finger_dis_min_id.float() / self.interval).long() + 1

            plam_sphere_c = shpere_c[:, :plam_spere_num]
            plam_shpere_r = shpere_r[:, :plam_spere_num]

            plam_dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - plam_sphere_c.unsqueeze(1), 2).sum(dim=-1) + 1e-8)  # B P C
            plam_dis = torch.abs(plam_dis - plam_shpere_r.unsqueeze(1))
            plam_dis, _ = torch.min(plam_dis, dim=-1)

            dis_min_id = torch.where(plam_dis<finger_dis, torch.zeros_like(finger_dis_min_id).to(device), finger_dis_min_id)
            return dis_min_id

        # pcl to mesh 15
        def calculate_point2mesh_distance(self, mesh, pcl, pcl_seg):
            batch_size, point_num, _ = pcl.size()
            device = pcl.device
            mesh_remove = mesh[:, :-1, :]
            joint_loss = []
            for index in range(15):
                vertex_num = self.vertex_joint_index_list[index].size(0)
                mesh_select = torch.index_select(mesh_remove, 1, self.vertex_joint_index_list[index])
                offset = torch.pow(pcl.unsqueeze(-2) - mesh_select.unsqueeze(1), 2).sum(dim=-1) # B P M
                offset = torch.where(pcl_seg.eq(index).unsqueeze(-1).repeat(1,1,vertex_num), offset, torch.ones_like(offset).to(device)*1e5)
                dis = torch.mean(torch.min(offset, dim=-1)[0], dim=-1)
                joint_loss.append(dis)
            return torch.stack(joint_loss, dim=-1) # B J

        # pcl to sphere 15
        def calculate_point2shpere_distance(self, joint, mesh, pcl, pcl_seg):
            batch_size, point_num, _ = pcl.size()
            device = pcl.device
            batch_size, joint_num, _ = joint.size()
            shpere_c, shpere_r = self.get_sphere_radius(joint.clone(), mesh)
            plam_spere_num = 1 + 5 * self.plam_per_adj_shpere
            joint_loss = []
            min_index = torch.zeros_like(pcl_seg).to(device)
            # 只需要考虑手指
            for index in range(15):
                shpere_c_select = shpere_c[:,plam_spere_num+index*self.interval:plam_spere_num+(index+1)*self.interval]
                shpere_r_select = shpere_r[:,plam_spere_num+index*self.interval:plam_spere_num+(index+1)*self.interval]
                dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - shpere_c_select.unsqueeze(1), 2).sum(dim=-1) + 1e-8) # B P C
                dis = torch.abs(dis - shpere_r_select.unsqueeze(1))
                dis = torch.where(pcl_seg.eq(index+1).unsqueeze(-1).repeat(1,1,self.interval), dis, torch.zeros_like(dis).to(device))
                dis, min_index_finger = torch.min(dis, dim=-1)
                pcl_num = dis.gt(0).sum(-1)
                loss = dis.sum(-1) / (pcl_num + 1e-8)
                loss = torch.where(pcl_num.eq(0), torch.zeros_like(loss).to(device), loss)
                min_index = torch.where(pcl_seg.eq(index+1),  min_index_finger+plam_spere_num+index*self.interval, min_index)
                joint_loss.append(loss)
            return torch.stack(joint_loss, dim=-1), min_index

        # use shpere to calculate pcl seg
        def seg_pcl_21(self, joints, joints_mano, mesh, pcl):
            device = joints.device
            batch_size, joint_num, _ = joints.size()
            shpere_c, _ = self.get_sphere_radius(joints.clone(), mesh)
            _, shpere_r = self.get_sphere_radius(joints_mano.clone(), mesh)
            plam_spere_num = 1 + 5 * self.plam_per_adj_shpere
            finger_sphere_c = shpere_c[:, plam_spere_num:]
            finger_shpere_r = shpere_r[:, plam_spere_num:]

            finger_dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - finger_sphere_c.unsqueeze(1), 2).sum(dim=-1) + 1e-8)  # B P C
            finger_dis = torch.abs(finger_dis - finger_shpere_r.unsqueeze(1))
            finger_dis, finger_dis_min_id = torch.min(finger_dis, dim=-1)
            id_mapping = torch.from_numpy(np.array([1,1,2,2,2,3,3,3,16,4,4,5,5,5,6,6,6,17,7,7,8,8,8,9,9,9,18,10,10,11,11,11,12,12,12,19,13,13,14,14,14,15,15,15,20])).to(device)
            finger_dis_min_id = id_mapping[finger_dis_min_id]

            plam_sphere_c = shpere_c[:, :plam_spere_num]
            plam_shpere_r = shpere_r[:, :plam_spere_num]

            plam_dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - plam_sphere_c.unsqueeze(1), 2).sum(dim=-1) + 1e-8)  # B P C
            plam_dis = torch.abs(plam_dis - plam_shpere_r.unsqueeze(1))
            plam_dis, _ = torch.min(plam_dis, dim=-1)

            dis_min_id = torch.where(plam_dis<finger_dis, torch.zeros_like(finger_dis_min_id).to(device), finger_dis_min_id)
            return dis_min_id

        # pcl to sphere
        def calculate_point2shpere_distance_21(self, joint, mesh, pcl, pcl_seg):
            batch_size, point_num, _ = pcl.size()
            device = pcl.device
            batch_size, joint_num, _ = joint.size()
            shpere_c, shpere_r = self.get_sphere_radius(joint.clone(), mesh)
            plam_spere_num = 1 + 5 * self.plam_per_adj_shpere
            joint_loss = []
            id_list = [[0,1],[2,3,4],[5,6,7],
                       [9,10],[11,12,13],[14,15,16],
                       [18,19],[20,21,22],[23,24,25],
                       [27,28],[29,30,31],[32,33,34],
                       [36,37],[38,39,40],[41,42,43],
                       [8],[17],[26],[35],[44]]

            for index in range(1, 21):
                joint_id = np.array(id_list[index-1]) + plam_spere_num
                shpere_c_select = shpere_c[:,joint_id]
                shpere_r_select = shpere_r[:,joint_id]
                dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - shpere_c_select.unsqueeze(1), 2).sum(dim=-1) + 1e-8) # B P C
                dis = torch.abs(dis - shpere_r_select.unsqueeze(1))
                dis = torch.where(pcl_seg.eq(index).unsqueeze(-1).repeat(1,1,dis.size(-1)), dis, torch.zeros_like(dis).to(device))
                dis = torch.min(dis, dim=-1)[0]
                pcl_num = dis.gt(0).sum(-1)
                loss = dis.sum(-1) / (pcl_num + 1e-8)
                loss = torch.where(pcl_num.eq(0), torch.zeros_like(loss).to(device), loss)
                joint_loss.append(loss)
            return torch.stack(joint_loss, dim=-1) # B J

        # use shpere to calculate pcl seg
        def seg_pcl_finger(self, joints, joints_mano, mesh, pcl):
            device = joints.device
            batch_size, joint_num, _ = joints.size()
            shpere_c, _ = self.get_sphere_radius(joints.clone(), mesh)
            _, shpere_r = self.get_sphere_radius(joints_mano.clone(), mesh)
            plam_spere_num = 1 + 5 * self.plam_per_adj_shpere
            finger_sphere_c = shpere_c[:, plam_spere_num:]
            finger_shpere_r = shpere_r[:, plam_spere_num:]

            finger_dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - finger_sphere_c.unsqueeze(1), 2).sum(dim=-1) + 1e-8)  # B P C
            finger_dis = torch.abs(finger_dis - finger_shpere_r.unsqueeze(1))
            finger_dis, finger_dis_min_id = torch.min(finger_dis, dim=-1)
            id_mapping = torch.from_numpy(np.array([1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5])).to(device)
            finger_dis_min_id = id_mapping[finger_dis_min_id]

            plam_sphere_c = shpere_c[:, :plam_spere_num]
            plam_shpere_r = shpere_r[:, :plam_spere_num]

            plam_dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - plam_sphere_c.unsqueeze(1), 2).sum(dim=-1) + 1e-8)  # B P C
            plam_dis = torch.abs(plam_dis - plam_shpere_r.unsqueeze(1))
            plam_dis, _ = torch.min(plam_dis, dim=-1)

            dis_min_id = torch.where(plam_dis<finger_dis, torch.zeros_like(finger_dis_min_id).to(device), finger_dis_min_id)
            return dis_min_id

        # pcl to sphere
        def calculate_point2shpere_distance_finger(self, joint, mesh, pcl, pcl_seg):
            batch_size, point_num, _ = pcl.size()
            device = pcl.device
            batch_size, joint_num, _ = joint.size()
            shpere_c, shpere_r = self.get_sphere_radius(joint.clone(), mesh)
            plam_spere_num = 1 + 5 * self.plam_per_adj_shpere
            joint_loss = []
            for index in range(5):
                shpere_c_select = shpere_c[:,(plam_spere_num+3*self.interval*index):(plam_spere_num+3*self.interval*(index+1))]
                shpere_r_select = shpere_r[:,(plam_spere_num+3*self.interval*index):(plam_spere_num+3*self.interval*(index+1))]
                dis = torch.sqrt(torch.pow(pcl.unsqueeze(-2) - shpere_c_select.unsqueeze(1), 2).sum(dim=-1) + 1e-8) # B P C
                dis = torch.abs(dis - shpere_r_select.unsqueeze(1))
                dis = torch.where(pcl_seg.eq(index+1).unsqueeze(-1).repeat(1,1,dis.size(-1)), dis, torch.zeros_like(dis).to(device))
                dis = torch.min(dis, dim=-1)[0]
                pcl_num = dis.gt(0).sum(-1)
                loss = dis.sum(-1) / (pcl_num + 1e-8)
                loss = torch.where(pcl_num.eq(0), torch.zeros_like(loss).to(device), loss)
                joint_loss.append(loss)
            return torch.stack(joint_loss, dim=-1) # B 5




        # beta:shape[Nx10] theta:pose[Nxpca_num]
        def forward(self, beta, theta, quat_or_euler, get_skin=False):
            # check if not tensor: wrap
            if not isinstance(beta, torch.Tensor):
                beta = torch.tensor(beta, dtype=torch.float)
            if not isinstance(theta, torch.Tensor):
                theta = torch.tensor(theta, dtype=torch.float)

            if self.is_cuda:
                beta = beta.cuda()
                theta = theta.cuda()

            num_batch = beta.shape[0]
            # 获取得到了目标形状和目标关节点的位置
            v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
            Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
            # 根据形状的得到的关节点的坐标
            J = torch.stack([Jx, Jy, Jz], dim=2)

            if quat_or_euler.shape[-1] == 3:
                # print(quat_or_euler.size())
                global_rot = self.batch_rodrigues(quat_or_euler).view(num_batch, 1, 3, 3)
                # global_rot = cv2.Rodrigues(quat_or_euler[0])[0][np.newaxis, np.newaxis]
                # global_rot = torch.tensor(global_rot, dtype=torch.float).repeat(num_batch,1,1,1)
                if self.is_cuda:
                    global_rot = global_rot.cuda()
                # Rs = self.batch_rodrigues(theta.view(-1, 3)).view(-1, 15, 3, 3)
                Rs = self.batch_rodrigues((torch.matmul(theta, self.hands_comp[:theta.size(-1)]) + self.hands_mean).view(-1,3)).view(-1, 15, 3, 3)
            else:
                if not isinstance(quat_or_euler, torch.Tensor):
                    quat_or_euler = torch.tensor(quat_or_euler, dtype=torch.float)
                if self.is_cuda:
                    quat_or_euler = quat_or_euler.cuda()
                global_rot = self.quat2mat(quat_or_euler).view(-1, 1, 3, 3)
                # 变化成角度的矩阵
                Rs = self.batch_rodrigues((torch.matmul(theta, self.hands_comp[:theta.size(-1)]) + self.hands_mean).view(-1,3)).view(-1, 15, 3, 3)

            pose_feature = (Rs[:, :, :, :]).sub(1.0, self.e3).view(-1, 135)
            # 形状+角度
            v_posed = v_shaped + torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
            # J_transformed 根据姿态得到新的关节点的位置
            # print(Rs.size(), global_rot.size())
            self.J_transformed, A = self.batch_global_rigid_transformation(torch.cat([global_rot, Rs], dim=1), J[:, :16, :], self.parents)

            # weight = self.weight[:num_batch]
            weight = self.weight.repeat(num_batch, 1, 1)
            W = weight.view(num_batch, -1, 16)
            T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

            ones_homo = torch.ones(num_batch, v_posed.shape[1], 1)
            if self.is_cuda:
                ones_homo = ones_homo.cuda()
            v_posed_homo = torch.cat([v_posed, ones_homo], dim=2)
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

            verts = v_homo[:, :, :3, 0]
            joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
            joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
            joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
            joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

            # 封住手腕
            wrist_vert = torch.mean(verts[:, [121,214,215,279,239,234,92,38,122,118,117,119,120,108,79,78],:],dim=1,keepdim=True)
            verts = torch.cat((verts, wrist_vert), dim=1)
            if get_skin:
                return verts, joints, Rs
            else:
                return joints

        def get_mano_vertices(self, quat_or_euler, pose, shape, cam, global_scale=None):
            """
            :param quat_or_euler: mano global rotation params in quaternion or euler representation [batch_size, 4 or 3]
            :param pose: mano articulation params [batch_size, 45]
            :param shape: mano shape params [batch_size, 10]
            :param cam: mano scale and translation params [batch_size, 4]
            :return: vertices: mano vertices Nx778x3,
                     joints: 3d joints in BigHand skeleton indexing Nx21x3
            """

            # apply parameters on the model
            verts, joints, Rs = self.forward(shape, pose, quat_or_euler, get_skin=True)

            # check if not tensor and cuda: wrap
            if not isinstance(cam, torch.Tensor):
                cam = torch.tensor(cam, dtype=torch.float)
            if self.is_cuda:
                cam = cam.cuda()

            scale = cam[:, 0].contiguous().view(-1, 1, 1)
            trans = cam[:, 1:].contiguous().view(cam.size(0), 1, -1)

            joints = joints * 1000
            verts = verts * 1000

            if not (global_scale is None):
                joints = joints * global_scale
                verts = verts * global_scale

            verts = verts*scale
            verts = verts+trans
            joints = joints*scale
            joints = joints+trans

            if self.dataset != 'icvl':
                return verts, joints
            else:
                select_joint = joints[:, self.transfer, :].clone()
                select_joint[:, 2, :] = (joints[:, 14, :] + joints[:, 15, :]) / 2
                select_joint[:, 5, :] = (joints[:, 2, :]+joints[:, 3, :]) / 2
                select_joint[:, 8, :] = (joints[:, 5, :] + joints[:, 6, :]) / 2
                select_joint[:, 11, :] = (joints[:, 11, :] + joints[:, 12, :]) / 2
                select_joint[:, 14, :] = (joints[:, 8, :] + joints[:, 9, :]) / 2

                select_joint[:, 3, :] = (joints[:, 20, :] + joints[:, 15, :]) / 2
                select_joint[:, 6, :] = (joints[:, 16, :]+joints[:, 3, :]) / 2
                select_joint[:, 9, :] = (joints[:, 17, :] + joints[:, 6, :]) / 2
                select_joint[:, 12, :] = (joints[:, 19, :] + joints[:, 12, :]) / 2
                select_joint[:, 15, :] = (joints[:, 18, :] + joints[:, 9, :]) / 2

                return verts, joints
            # [:, MANO2HANDS, :]
            # return verts, joints

        def quat2mat(self, quat):
            """Convert quaternion coefficients to rotation matrix.
            Args:
                quat: size = [B, 4] 4 <===>(w, x, y, z)
            Returns:
                Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
            """
            norm_quat = quat
            norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
            w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

            B = quat.size(0)

            w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
            wx, wy, wz = w * x, w * y, w * z
            xy, xz, yz = x * y, x * z, y * z

            rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                                  2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                                  2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)

            return rotMat

        def batch_rodrigues(self, theta):
            l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
            angle = torch.unsqueeze(l1norm, -1)
            normalized = torch.div(theta, angle)
            angle = angle * 0.5
            v_cos = torch.cos(angle)
            v_sin = torch.sin(angle)
            quat = self.quat2mat(torch.cat([v_cos, v_sin * normalized], dim=1))
            return quat

        def batch_global_rigid_transformation(self, Rs, Js, parent):
            N = Rs.shape[0]

            if self.rotate_base:
                root_rotation = torch.matmul(Rs[:, 0, :, :], self.base_rot_mat_x)
            else:
                root_rotation = Rs[:, 0, :, :]

            Js = torch.unsqueeze(Js, -1)

            # 构建旋转位移矩阵
            def make_A(R, t):
                R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
                ones_homo = torch.ones(N, 1, 1)
                if torch.cuda.is_available():
                    ones_homo = ones_homo.cuda()
                t_homo = torch.cat([t, ones_homo], dim=1)
                return torch.cat([R_homo, t_homo], 2)

            A0 = make_A(root_rotation, Js[:, 0])
            results = [A0]

            # result存储了全局的旋转矩阵
            for i in range(1, parent.shape[0]):
                j_here = Js[:, i] - Js[:, parent[i]]
                A_here = make_A(Rs[:, i], j_here)
                res_here = torch.matmul(results[parent[i]], A_here)
                results.append(res_here)

            results = torch.stack(results, dim=1)
            # new_J 旋转之后的关节点位置
            new_J = results[:, :, :3, 3]
            ones_homo = torch.zeros(N, 16, 1, 1)
            if self.is_cuda:
                ones_homo = ones_homo.cuda()
            Js_w0 = torch.cat([Js, ones_homo], dim=2)
            init_bone = torch.matmul(results, Js_w0)
            init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
            A = results - init_bone
            # 每个关节点对于初始的关节点的旋转和位移？
            return new_J, A


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)

    return rotMat


def batch_rodrigues(theta):
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = quat2mat(torch.cat([v_cos, v_sin * normalized], dim=1))
    return quat


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, kernel_size, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        self.meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        self.kernel_size = kernel_size

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input, sigma=1.7):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        kernel = 1
        sigma = [sigma]*2
        device = input.device
        for size, std, mgrid in zip(self.kernel_size, sigma, self.meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())

        return self.conv(input, weight=torch.Tensor(kernel).to(device))

import random


def RotationPoints(verts, joints, center3d, rot):
    if rot.size(-1) == 3:
        rotMat = batch_rodrigues(rot).unsqueeze(1)
    elif rot.size(-1) == 4:
        rotMat = quat2mat(rot).unsqueeze(1)
    center3d = center3d.unsqueeze(1)
    verts = verts - center3d
    joints = joints - center3d
    rot_verts = torch.matmul(rotMat, verts.unsqueeze(-1)).squeeze()
    rot_joints = torch.matmul(rotMat, joints.unsqueeze(-1)).squeeze()
    return rot_verts + center3d, rot_joints + center3d


def RotationNormalPoints(points, rot):
    if rot.size(-1) == 3:
        rotMat = batch_rodrigues(rot).unsqueeze(1)
    elif rot.size(-1) == 4:
        rotMat = quat2mat(rot).unsqueeze(1)
    rot_points = torch.matmul(rotMat, points.unsqueeze(-1)).squeeze()
    return rot_points


from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRasterizer,Textures,TexturesVertex,MeshRenderer,BlendParams,softmax_rgb_blend
)

from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes


class myShader(torch.nn.Module):
    def __init__(self, cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        texels = meshes.sample_textures(fragments)

        return texels


class Render(nn.Module):
    def __init__(self, mano_path, dataset, cam_para, image_size, crop_size=(128, 128)):
        super(Render, self).__init__()
        # if dataset == 'icvl':
        #     mano_path = mano_path + '/MANO_LEFT.pkl'
        # else:
        mano_path = mano_path + '/MANO_RIGHT.pkl'
        self.mano_layer = MANO_SMPL(mano_path, dataset)
        fx, fy, px, py = cam_para
        self.paras = cam_para
        R = torch.eye(3).unsqueeze(0)
        R[:, 0, 0] = -1
        R[:, 1, 1] = -1
        T = torch.zeros(3).unsqueeze(0)
        cameras = PerspectiveCameras(
            focal_length=((fx, fy),),  # (fx, fy)
            principal_point=((px, py),),  # (px, py)
            image_size=(image_size,),  # (imwidth, imheight)
            device="cuda",
            R=R.cuda(), T=T.cuda(),
        )
        raster_settings = RasterizationSettings(
            image_size=max(image_size),
            blur_radius=0,
            faces_per_pixel=1,
            # clip_barycentric_coords=True,
        )
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # point_raster_settings = PointsRasterizationSettings(
        #     image_size=int(min(image_size)*0.5),
        #     radius=0.003,
        #     points_per_pixel=10
        # )
        # self.point_rasterizer = PointsRasterizer(cameras=cameras, raster_settings=point_raster_settings)
        # self.point_renderer = PointsRenderer(rasterizer=self.point_rasterizer, compositor=AlphaCompositor())
        self.img_size = image_size
        self.crop_size = crop_size
        self.resize_roi = RoIAlign((image_size[1], image_size[0]), spatial_scale=1, sampling_ratio=1)
        self.crop_roi = RoIAlign(crop_size, spatial_scale=1, sampling_ratio=1)
        # self.bk_value = 0
        # self.bk_value = 0

        xx, yy = np.meshgrid(np.arange(crop_size[0]), np.arange(crop_size[0]))
        xx = 2 * (xx + 0.5) / crop_size[0] - 1.0
        yy = 2 * (yy + 0.5) / crop_size[0] - 1.0
        mesh = np.stack((xx, yy), axis=-1).reshape([1, -1, 2])
        self.xy_mesh = torch.from_numpy(mesh).float().cuda()

        xx, yy = np.meshgrid(np.arange(crop_size[0]), np.arange(crop_size[0]))
        padd = np.ones([crop_size[0], crop_size[0]])
        mesh = np.stack((xx, yy, padd), axis=-1).reshape([1, -1, 3])
        self.crop_mesh = torch.from_numpy(mesh).float().cuda()
        if dataset == 'nyu':
            self.depth_range = [500, 1200]
        if dataset == 'msra' or dataset == 'icvl':
            self.depth_range = [150, 600]

    def forward(self, model_paras,
                center3d, cube_size,
                augmentView=None, augmentShape=None, augmentCenter=None, augmentSize=None, mask=True):
        device = model_paras.device
        batch_size = model_paras.size(0)
        if model_paras.size(-1) == 63:
            quat_dim = 4
        else:
            quat_dim = 3

        quat = model_paras[:, :quat_dim]
        theta = model_paras[:, quat_dim:quat_dim+45]
        if not (augmentShape is None):
            beta = model_paras[:, quat_dim+45:quat_dim+45+10] + augmentShape
        else:
            beta = model_paras[:, quat_dim+45:quat_dim+45+10]
        cam = model_paras[:, quat_dim + 45 + 10:]

        hand_verts, hand_joints = self.mano_layer.get_mano_vertices(quat, theta, beta, cam)
        synth_center = hand_joints.mean(dim=1).unsqueeze(1).clone()
        hand_verts = hand_verts - synth_center
        hand_joints = hand_joints - synth_center

        if center3d is None:
            synth_depth = torch.rand([batch_size, 1]) * (self.depth_range[1] - self.depth_range[0]) + self.depth_range[0]
            center3d = torch.cat((torch.zeros([batch_size, 2]), synth_depth), dim=-1).to(device)

        hand_verts = hand_verts + center3d.unsqueeze(1)
        hand_joints = hand_joints + center3d.unsqueeze(1)

        if not (augmentView is None):
            hand_verts, hand_joints = RotationPoints(hand_verts, hand_joints, center3d, augmentView)
        if not (augmentCenter is None):
            center3d = center3d + augmentCenter
        if not (augmentSize is None):
            cube_size = cube_size * augmentSize
        center2d = self.points3DToImg(center3d.unsqueeze(1)).squeeze(1)

        meshes = Meshes(verts=hand_verts, faces=self.mano_layer.faces.unsqueeze(0).repeat(batch_size, 1, 1))
        fragments = self.rasterizer(meshes)
        ori_depth = fragments.zbuf
        ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device)*0, ori_depth)
        resize_depth = self.resize(ori_depth.permute(0, 3, 1, 2))

        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(center2d, cube_size)
        M = self.Offset2Trans(xstart, xend, ystart, yend)  # trans for
        cropped_img = self.warpPerspective(resize_depth, M)
        noraml_img = self.normalize_img(cropped_img, center2d, cube_size)

        noraml_joint = self.JointTrans(hand_joints, M, center2d, cube_size)
        noraml_verts = self.JointTrans(hand_verts, M, center2d, cube_size)

        noraml_joint_xyz = (hand_joints - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2
        normal_verts_xyz = (hand_verts - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2
        if mask:
            noraml_img = self.mask_img(noraml_img, noraml_joint, 0.15, 0.3)
        return noraml_img, noraml_joint, noraml_verts, noraml_joint_xyz, normal_verts_xyz, center3d, cube_size, M

    # 根据模型参数得到的是归一化的空间坐标
    def normal_render(self, model_paras, center3d, cube_size):
        batch_size = model_paras.size(0)
        quat = model_paras[:, :3]
        theta = model_paras[:, 3:3+45]
        beta = model_paras[:, 3+45:3+45+10]
        cam = model_paras[:, 3 + 45 + 10:]
        hand_verts, hand_joints = self.mano_layer.get_mano_vertices(quat, theta, beta, cam, global_scale=1/125)
        hand_verts = (hand_verts + 1) / 2 * cube_size.unsqueeze(1) + center3d.unsqueeze(1)
        hand_joints = (hand_joints + 1) / 2 * cube_size.unsqueeze(1) + center3d.unsqueeze(1)
        center2d = self.points3DToImg(center3d.unsqueeze(1)).squeeze(1)

        meshes = Meshes(verts=hand_verts, faces=self.mano_layer.faces.unsqueeze(0).repeat(batch_size, 1, 1))
        fragments = self.rasterizer(meshes)
        ori_depth = fragments.zbuf
        ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device)*0, ori_depth)
        resize_depth = self.resize(ori_depth.permute(0, 3, 1, 2))

        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(center2d, cube_size)
        M = self.Offset2Trans(xstart, xend, ystart, yend)  # trans for
        cropped_img = self.warpPerspective(resize_depth, M)

        noraml_img = self.normalize_img(cropped_img, center2d, cube_size)
        noraml_joint = self.JointTrans(hand_joints, M, center2d, cube_size)
        noraml_joint_xyz = (hand_joints - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2
        normal_verts_xyz = (hand_verts - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2

        return noraml_img, noraml_joint, noraml_joint_xyz, normal_verts_xyz

    # 根据模型参数得到的是归一化的空间坐标
    def render(self, model_paras, center3d, cube_size, M=None):
        batch_size = model_paras.size(0)
        quat = model_paras[:, :3]
        theta = model_paras[:, 3:3+45]
        beta = model_paras[:, 3+45:3+45+10]
        cam = model_paras[:, 3 + 45 + 10:]
        hand_verts, hand_joints = self.mano_layer.get_mano_vertices(quat, theta, beta, cam, global_scale=1/125)
        hand_verts = hand_verts * cube_size.unsqueeze(1) / 2 + center3d.unsqueeze(1)
        hand_joints = hand_joints * cube_size.unsqueeze(1) / 2 + center3d.unsqueeze(1)
        center2d = self.points3DToImg(center3d.unsqueeze(1)).squeeze(1)

        meshes = Meshes(verts=hand_verts, faces=self.mano_layer.faces.unsqueeze(0).repeat(batch_size, 1, 1))
        fragments = self.rasterizer(meshes)
        ori_depth = fragments.zbuf
        ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device)*0, ori_depth)
        resize_depth = self.resize(ori_depth.permute(0, 3, 1, 2))

        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(center2d, cube_size)
        M = self.Offset2Trans(xstart, xend, ystart, yend)  # trans for
        cropped_img = self.warpPerspective(resize_depth, M)

        noraml_img = self.normalize_img(cropped_img, center2d, cube_size)
        joint_uvd = self.JointTrans(hand_joints, M, center2d, cube_size)
        joint_xyz = (hand_joints - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2
        mesh_xyz = (hand_verts - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2

        return noraml_img, joint_uvd, joint_xyz, mesh_xyz


    def M_render(self, model_paras,
                center3d, cube_size, M=None, mask=True):

        batch_size = model_paras.size(0)
        if model_paras.size(-1) == 63:
            quat_dim = 4
        else:
            quat_dim = 3

        quat = model_paras[:, :quat_dim]
        theta = model_paras[:, quat_dim:quat_dim+45]
        beta = model_paras[:, quat_dim+45:quat_dim+45+10]
        cam = model_paras[:, quat_dim + 45 + 10:]
        hand_verts, hand_joints = self.mano_layer.get_mano_vertices(quat, theta, beta, cam)
        center2d = self.points3DToImg(center3d.unsqueeze(1)).squeeze(1)

        meshes = Meshes(verts=hand_verts, faces=self.mano_layer.faces.unsqueeze(0).repeat(batch_size, 1, 1))
        fragments = self.rasterizer(meshes)
        ori_depth = fragments.zbuf
        ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device)*0, ori_depth)
        resize_depth = self.resize(ori_depth.permute(0, 3, 1, 2))

        cropped_img = self.warpPerspective(resize_depth, M)
        noraml_img = self.normalize_img(cropped_img, center2d, cube_size)
        noraml_joint_uvd = self.JointTrans(hand_joints, M, center2d, cube_size)
        normal_verts_uvd = self.JointTrans(hand_verts, M, center2d, cube_size)

        noraml_joint_xyz = (hand_joints - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2
        normal_verts_xyz = (hand_verts - center3d.unsqueeze(1))/cube_size.unsqueeze(1)*2
        if mask:
            noraml_img = self.mask_img(noraml_img, noraml_joint_uvd, 0.15, 0.3)
        return noraml_img

    def comToBounds(self, com, size):
        fx, fy, fu, fv = self.paras
        zstart = com[:, 2] - size[:, 2] / 2.
        zend = com[:, 2] + size[:, 2] / 2.
        xstart = torch.floor((com[:, 0] * com[:, 2] / fx - size[:, 0] / 2.) / com[:, 2] * fx + 0.5).int()
        xend = torch.floor((com[:, 0] * com[:, 2] / fx + size[:, 0] / 2.) / com[:, 2] * fx + 0.5).int()
        ystart = torch.floor((com[:, 1] * com[:, 2] / fy - size[:, 1] / 2.) / com[:, 2] * fy + 0.5).int()
        yend = torch.floor((com[:, 1] * com[:, 2] / fy + size[:, 1] / 2.) / com[:, 2] * fy + 0.5).int()
        return xstart, xend, ystart, yend, zstart, zend

    def Offset2Trans(self, xstart, xend, ystart, yend):
        # resize to same size
        device = xstart.device
        b = xstart.size(0)
        wb = (xend - xstart)
        hb = (yend - ystart)

        sz0 = torch.where(wb.gt(hb), torch.ones_like(wb).to(device)*self.crop_size[0], (wb * self.crop_size[0] / hb).int())
        sz1 = torch.where(wb.gt(hb), (hb * self.crop_size[0] / wb).int(), torch.ones_like(wb).to(device)*self.crop_size[1])

        s = torch.where(wb.gt(hb), self.crop_size[0] / wb, self.crop_size[1] / hb)

        trans = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).to(device)
        trans[:, 0, 2] = -xstart
        trans[:, 1, 2] = -ystart
        scale = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).to(device)
        scale[:, 0, 0] = s
        scale[:, 1, 1] = s

        xstart = (torch.floor(self.crop_size[0] / 2. - sz0 / 2.)).int()
        ystart = (torch.floor(self.crop_size[1] / 2. - sz1 / 2.)).int()
        off = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).to(device)
        off[:, 0, 2] = xstart
        off[:, 1, 2] = ystart

        M = torch.matmul(off, torch.matmul(scale, trans))
        return M

    def get_mesh_xyz(self, model_paras):
        quat = model_paras[:, :3]
        theta = model_paras[:, 3:3 + 45]
        beta = model_paras[:, 3 + 45:3 + 45 + 10]
        cam = model_paras[:, 3 + 45 + 10:]
        hand_mesh, hand_joints = self.mano_layer.get_mano_vertices(quat, theta, beta, cam, global_scale=1 / 125)
        return hand_joints, hand_mesh

    def get_mesh_xyz_old(self, model_paras):
        quat = model_paras[:, :3]
        theta = model_paras[:, 3:3 + 45]
        beta = model_paras[:, 3 + 45:3 + 45 + 10]
        cam = model_paras[:, 3 + 45 + 10:]
        hand_mesh, hand_joints = self.mano_layer.get_mano_vertices(quat, theta, beta, cam, global_scale=1 / 125)
        hand_mesh = hand_mesh + 1
        hand_joints = hand_joints + 1
        return hand_joints, hand_mesh

    # hand_mesh in world corrd
    def mesh2img(self, hand_mesh, center3d, cube_size):
        batch_size = hand_mesh.size(0)
        center2d = self.points3DToImg(center3d.unsqueeze(1)).squeeze(1)
        meshes = Meshes(verts=hand_mesh, faces=self.mano_layer.faces.unsqueeze(0).repeat(batch_size, 1, 1))
        fragments = self.rasterizer(meshes)
        ori_depth = fragments.zbuf
        ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device)*0, ori_depth)
        resize_depth = self.resize(ori_depth.permute(0, 3, 1, 2))
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(center2d, cube_size)
        M = self.Offset2Trans(xstart, xend, ystart, yend)
        cropped_img = self.warpPerspective(resize_depth, M)
        noraml_img = self.normalize_img(cropped_img, center2d, cube_size)
        return noraml_img

    def getDepth(self, hand_verts, hand_joints, center3d, cube_size, M, rot=None):
        batch_size = center3d.size(0)
        center2d = self.points3DToImg(center3d.unsqueeze(1)).squeeze(1)
        if not (rot is None):
            hand_verts, hand_joints = RotationPoints(hand_verts, hand_joints, center3d, rot)
        # rotation joint
        meshes = Meshes(verts=hand_verts, faces=self.mano_layer.faces.unsqueeze(0).repeat(batch_size, 1, 1))
        fragments = self.rasterizer(meshes)
        ori_depth = fragments.zbuf
        ori_depth = torch.where(ori_depth.le(0), torch.ones_like(ori_depth).to(ori_depth.device)*0, ori_depth)
        resize_depth = self.resize(ori_depth.permute(0, 3, 1, 2))
        cropped_img = self.warpPerspective(resize_depth, M)
        noraml_img = self.normalize_img(cropped_img, center2d, cube_size)
        noraml_joint = self.JointTrans(hand_joints, M, center2d, cube_size)
        return noraml_img, noraml_joint

        # return ori_depth, noraml_img, noraml_joint

    def synth2real(self, noraml_img, noise=0.1, noise_patch=2, sigma=1.7, bk_value=0.95):
        device = noraml_img.device
        B, C, H, W = noraml_img.size()
        img_white_noise_scale = F.upsample_nearest(noise * torch.randn((B, C, H//noise_patch, W//noise_patch)).to(device), scale_factor=noise_patch) * noraml_img.lt(bk_value).float()
        noraml_img = noraml_img + img_white_noise_scale
        if sigma != 0:
            noraml_img = F.pad(noraml_img, (2, 2, 2, 2), mode='reflect')
            sigma = random.uniform(sigma, sigma)
            noraml_img = self.smoothing(noraml_img, sigma)
        return noraml_img

    def resize(self, img):
        batch_size = img.size(0)
        theta = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.float).to(img.device)
        target = torch.ones(batch_size, 1, self.img_size[1], self.img_size[0])
        grid = F.affine_grid(theta.unsqueeze(0).repeat(batch_size, 1, 1), target.size())
        output = F.grid_sample(img, grid, mode='nearest')
        return output

    def affine_grid(self, img, M):
        device = img.device
        b, c, h_ori, w_ori = img.size()
        h, w = self.crop_size[0],self.crop_size[1]
        M_inverse = torch.inverse(M).view(b, 1, 3, 3)
        mesh = self.crop_mesh.repeat(b, 1, 1)
        mesh_trans = torch.matmul(M_inverse, mesh.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        mesh_trans = mesh_trans.reshape(b, h*w, 2)
        # normal mesh_trans
        coeff = torch.Tensor([w_ori, h_ori]).to(device).view(1, 1, 2)
        normal_mesh_trans = (mesh_trans / coeff)*2-1
        return normal_mesh_trans.view(b, h, w, 2)

    def warpPerspective(self, img, M):
        grid = self.affine_grid(img, M)
        crop_img = F.grid_sample(img, grid, mode='nearest')
        return crop_img

    def ResizeRenderImg(self, img):
        device = img.device
        b = img.size(0)
        xstart = torch.zeros([b]).to(device)
        ystart = torch.zeros([b]).to(device)
        img_max_size = max(self.img_size)
        xend = torch.ones_like(xstart).to(device)*img_max_size
        yend = torch.ones_like(ystart).to(device)*img_max_size
        batch_index = torch.arange(0, b).float().to(device)
        rois = torch.stack((batch_index, xstart, ystart, xend, yend), dim=1)
        resize_img = self.resize_roi(img, rois)
        return resize_img

    def massCenter(self, img):
        # opencv HWC
        device = img.device
        b, c, h, w = img.size()
        x = torch.arange(0, h)
        y = torch.arange(0, w)
        xv, yv = torch.meshgrid([x, y])
        xv, yv = xv.view(1, 1, h, w).repeat(b,1,1,1).float().to(device), yv.view(1, 1, h, w).repeat(b,1,1,1).float().to(device)
        center = torch.cat((yv, xv, img), dim=1)
        center = center*img.gt(0).float()
        center = center.mean(-1).mean(-1) / img.gt(0).float().mean(-1).mean(-1)
        return center
        # return self.pointsImgTo3D(center.unsqueeze(1)).squeeze(1)

    def normalize_img(self, imgD, com, cube):
        z_min = com[:, 2] - cube[:, 2] / 2.
        z_max = com[:, 2] + cube[:, 2] / 2.
        z_max = z_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z_min = z_min.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        imgD = torch.where((imgD == -1) | (imgD == 0), z_max, imgD)
        imgD = torch.where(imgD.gt(z_max), z_max, imgD)
        imgD = torch.where(imgD.lt(z_min), z_min, imgD)
        imgD -= com[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        imgD /= (cube[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 2.)
        return imgD

    def JointTrans(self, joint, M, com, cube):
        device = joint.device
        b, j, _ = joint.size()
        joint_uvd = self.points3DToImg(joint)
        joint_trans = torch.cat((joint_uvd[:, :, 0:2], torch.ones([b, j, 1]).to(device)),dim=-1).unsqueeze(-1)
        joint_trans = torch.matmul(M.unsqueeze(1), joint_trans).squeeze(-1)
        joint_uv = joint_trans[:, :, 0:2] / self.crop_size[0] * 2 - 1
        joint_d = (joint_uvd[:, :, 2:] - com.unsqueeze(1)[:, :, 2:]) / (cube.unsqueeze(1)[:, :, 2:] / 2.0)
        return torch.cat((joint_uv, joint_d), dim=-1)

    def pointsImgTo3D(self, point_uvd):
        point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
        point_xyz[:, :, 0] = (point_uvd[:, :, 0] - self.paras[2]) * point_uvd[:, :, 2] / self.paras[0]
        point_xyz[:, :, 1] = (point_uvd[:, :, 1] - self.paras[3]) * point_uvd[:, :, 2] / self.paras[1]
        point_xyz[:, :, 2] = point_uvd[:, :, 2]
        return point_xyz

    def points3DToImg(self, joint_xyz):
        fx, fy, fu, fv = self.paras
        joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
        joint_uvd[:, :, 0] = (joint_xyz[:, :, 0] * fx / (joint_xyz[:, :, 2]+1e-8) + fu)
        joint_uvd[:, :, 1] = (joint_xyz[:, :, 1] * fy / (joint_xyz[:, :, 2]) + fv)
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    def mask_img(self, img, img_joint, mask_offset, mask_para, min_mask_num=3, max_mask_num=10):
        device = img.device
        mask_num = np.random.choice(np.arange(min_mask_num, max_mask_num), 1, replace=False)[0]
        b, j, _ = img_joint.size()
        joint_id = np.random.choice(np.arange(0, j), mask_num, replace=False)
        mask_uvd = img_joint[:, joint_id, :]
        uvd_offset = (torch.rand(mask_uvd.size()) - 0.5) * mask_offset * 2
        mask_uvd = mask_uvd + uvd_offset.to(device)
        mask_range = torch.rand([b, mask_num]).to(device) * mask_para
        mesh = self.xy_mesh.view(1, -1, 2).repeat(b, 1, 1)
        mesh = torch.cat((mesh, img.view(b, -1, 1)), dim=-1).view(b, 1, -1, 3)
        dis = torch.sqrt(torch.sum((mesh - mask_uvd.view(b, mask_num, 1, 3)) ** 2, dim=-1))
        mask = dis.lt(mask_range.view([b, mask_num, 1])).float()
        mask = ~mask.sum(1).gt(0)
        return torch.where(mask.view(b, 1, img.size(-2), img.size(-1)), img, torch.ones_like(img).to(device))

    def sobel_conv2d(self, im):
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        weight = torch.from_numpy(sobel_kernel).cuda()
        edge_detect = F.conv2d(im, weight)
        return edge_detect

    def PatchGaussian(self, img, patch_scale=0.125):
        batch_size, _, img_size, img_size = img.size()
        patch_size = int(img_size*patch_scale)
        noise = (torch.rand((batch_size, 1, patch_size, patch_size)).to(img.device) - 0.5) * 0.1
        mask = img.lt(0.99)
        img = img + mask * F.interpolate(noise, scale_factor=1/patch_scale)
        return img

