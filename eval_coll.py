import cv2
import pickle
import trimesh
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import trimesh


# part_mesh = []
# for part_index in range(15):
#     path = 'D:\\pycharm\\TIP2021\\data\\waterObjMap\\part%d.obj' % (part_index)
#     part_mesh.append(trimesh.load_mesh(path))
#
# index_tip = part_mesh[1]
# index_mip = part_mesh[2]
# # index_mip = trimesh.Trimesh(vertices=index_tip.vertices+0.5, faces=index_tip.faces)
# inter = index_mip.intersection(index_tip, engine="blender")
# print(inter.volume)
# print(index_tip.volume)
# print(index_mip.volume)
#
# vertex = inter.vertices
# face = inter.faces
# save_path = 'D:\\pycharm\\TIP2021\\data\\inter.obj'
# with open(save_path, 'w') as fp:
#     for v in vertex:
#         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
#     for face_index, f in enumerate(face):
#         fp.write('f %d %d %d\n' % (f[0]+1, f[1]+1, f[2]+1))


class MANO_SMPL(nn.Module):
    def __init__(self, mano_pkl_path):
        super(MANO_SMPL, self).__init__()
        with open(mano_pkl_path + '/MANO_RIGHT.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        self.faces = torch.from_numpy(np.array(model['f'], dtype=np.float)).float()
        self.wrist_faces = torch.from_numpy(
            np.array([[121, 214, 778], [214, 215, 778], [215, 279, 778], [279, 239, 778],
                      [239, 234, 778], [234, 92, 778], [92, 38, 778], [38, 122, 778], [122, 118, 778], [118, 117, 778],
                      [117, 119, 778], [119, 120, 778], [120, 108, 778], [108, 79, 778], [79, 78, 778],
                      [78, 121, 778]])).float()
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
        np_shapedirs = torch.from_numpy(np_shapedirs).float()

        # Adding new joints for the fingertips. Original MANO model provide only 16 skeleton joints.
        np_J_regressor = model['J_regressor'].T.toarray()
        np_J_addition = np.zeros((778, 5))
        np_J_addition[333][0] = 1
        np_J_addition[444][1] = 1
        np_J_addition[672][2] = 1
        np_J_addition[555][3] = 1
        np_J_addition[744][4] = 1
        np_J_regressor = np.concatenate((np_J_regressor, np_J_addition), axis=1)
        np_J_regressor = torch.from_numpy(np_J_regressor).float()

        np_hand_component = np.array(model['hands_components'], dtype=np.float)
        np_hand_component = torch.from_numpy(np_hand_component).float()

        np_hand_mean = np.array(model['hands_mean'], dtype=np.float)
        np_hand_mean = torch.from_numpy(np_hand_mean).float()

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        np_posedirs = torch.from_numpy(np_posedirs).float()

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)
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
                if (face.unsqueeze(1) - vertex_joint[index].unsqueeze(0)).eq(0).sum() > 0:
                    joint_face.append(face)
            self.joint_faces.append(torch.stack(joint_face, dim=0))

        # assign each face a semantic
        self.face_segment = []
        for face in self.faces:
            if 778 in face:
                self.face_segment.append(0)
            else:
                weight = np_weights[int(face[0]), :] + np_weights[int(face[1]), :] + np_weights[int(face[2]), :]
                self.face_segment.append(weight.argmax())
        self.face_parent = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

        # finger segment
        self.vertex_finger_index_list = []
        for index in range(5):
            self.vertex_finger_index_list.append(
                torch.cat((vertex_joint[3 * index + 1], vertex_joint[3 * index + 2], vertex_joint[3 * index + 3])))
        Joint2Finger = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        self.finger_seg = torch.from_numpy(Joint2Finger[vertex_seg[0]])
        self.finger_faces = []
        for index in range(5):
            finger_face = []
            for face in self.faces:
                if (face.unsqueeze(1) - self.vertex_finger_index_list[index].unsqueeze(0)).eq(0).sum() > 0:
                    finger_face.append(face)
            self.finger_faces.append(torch.stack(finger_face, dim=0))

        # self.faces = torch.cat((self.joint_faces[0],self.joint_faces[1],self.joint_faces[2]), dim=0)
        e3 = torch.eye(3).float()
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [1, 3, 3])
        self.base_rot_mat_x = torch.from_numpy(np_rot_x).float()

        # Load the MANO_PART.pkl
        with open(mano_pkl_path + 'MANO_PART.pkl', 'rb') as f:
            model_part = pickle.load(f)

        self.part_num = 15
        self.part_vertex_id_list = []
        self.part_face_list = []
        for i in range(self.part_num):
            self.part_vertex_id_list.append(model_part['v-%d' % i])
            self.part_face_list.append(model_part['f-%d' % i])

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
            for index in range(vertex_component - 1):
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
        self.register_buffer('weight', np_weights)  # vertex importance
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
        plam_shpere_n = 1 + 5 * self.plam_per_adj_shpere
        finger_shpere_n = 15 * (self.per_adj_shpere + 1)
        plam_mask = torch.zeros([plam_shpere_n, plam_shpere_n])
        plam_mask_add = torch.ones([plam_shpere_n, finger_shpere_n])
        plam_mask = torch.cat((plam_mask, plam_mask_add), dim=1)

        # finger mask 同一个骨骼之间不做碰撞约束
        finger_mask = torch.ones(finger_shpere_n, plam_shpere_n + finger_shpere_n)
        # 考虑同一个手指之间的碰撞
        for finger_index in range(0, 15):
            finger_root_index = int(finger_index / 3) + 1
            # 指跟节点
            if finger_index in [0, 3, 6, 9, 12]:
                for finger_bone_index in range(self.interval):
                    finger_mask[
                        self.interval * finger_index + finger_bone_index, finger_root_index * self.plam_per_adj_shpere] = 0
                    plam_mask[
                        finger_root_index * self.plam_per_adj_shpere, plam_shpere_n + self.interval * finger_index + finger_bone_index] = 0
                    shpere_index = plam_shpere_n + self.interval * finger_index
                    finger_mask[self.interval * finger_index + finger_bone_index,
                    shpere_index:shpere_index + self.interval + 2] = 0
            # 其他位置的关节点
            else:
                shpere_index = plam_shpere_n + self.interval * finger_index
                max_index = plam_shpere_n + (3 * self.interval) * finger_root_index
                for finger_bone_index in range(self.interval):
                    finger_mask[self.interval * finger_index + finger_bone_index,
                    shpere_index - self.interval:min(shpere_index + self.interval * 2 + 1, max_index)] = 0
        # 特别的删除拇指指跟与掌心的碰撞
        thumb_root_index = 12 * self.interval
        finger_mask[thumb_root_index:thumb_root_index + self.interval + 1, :plam_shpere_n] = 0
        self.mask = torch.cat((plam_mask, finger_mask), dim=0)
        # 特别的删除拇指指跟与掌心的碰撞
        self.mask[:plam_shpere_n,
        thumb_root_index + plam_shpere_n:plam_shpere_n + thumb_root_index + self.interval + 1] = 0

    def get_sphere_radius(self, joints, mesh):
        # N*J
        batch_size, joint_num, _ = joints.size()
        device = joints.device
        joint_vertex = self.J_regressor.clone().gt(0).unsqueeze(0).repeat(batch_size, 1, 1).permute(0, 2, 1)
        # joint_center = joints.clone()
        joint_r = joints.unsqueeze(2) - mesh.unsqueeze(1)[:, :, :778, :]
        dis = torch.sqrt(torch.sum(joint_r * joint_r, dim=-1) + 1e-8)
        joint_r = torch.where(joint_vertex, dis, torch.ones_like(dis).to(device) * 100)
        joint_r = torch.mean(torch.topk(joint_r, k=20, dim=-1, largest=False)[0], dim=-1)
        joint_r = torch.cat((joint_r[:, :16], joint_r[:, [3, 6, 9, 12, 15]] / 1.5), dim=-1)

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
        finger_shpere_r = finger_shpere_r.reshape([batch_size, -1, 1]) * self.interval_value.to(device) \
                          + finger_parent.reshape([batch_size, -1, 1])

        shpere_r = torch.cat((plam_shpere_r.view(batch_size, -1), finger_shpere_r.view(batch_size, -1)), dim=1)

        # for plam center
        plam_child_c = joints[:, [1, 4, 7, 10, 13], :]
        plam_parent_c = joints[:, 0:1, :]
        plam_bone_c = plam_child_c - plam_parent_c
        plam_shpere_c = plam_bone_c.reshape([batch_size, -1, 1, 3]) * self.plam_interval_value.to(device).unsqueeze(-1) \
                        + plam_parent_c.reshape([batch_size, -1, 1, 3])
        plam_shpere_c = torch.cat((plam_parent_c, plam_shpere_c.view(batch_size, -1, 3)), dim=1)

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
        plam_child_c = joints[:, [1, 4, 7, 10, 13], :]
        plam_parent_c = joints[:, 0:1, :]
        plam_bone_c = plam_child_c - plam_parent_c
        plam_shpere_c = plam_bone_c.reshape([batch_size, -1, 1, 3]) * self.plam_interval_value.to(device).unsqueeze(-1) \
                        + plam_parent_c.reshape([batch_size, -1, 1, 3])
        plam_shpere_c = torch.cat((plam_parent_c, plam_shpere_c.view(batch_size, -1, 3)), dim=1)

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
        joint_vertex = self.J_regressor.clone().gt(0).unsqueeze(0).repeat(batch_size, 1, 1).permute(0, 2, 1)
        # joint_center = joints.clone()
        joint_r = joints.unsqueeze(2) - mesh.unsqueeze(1)[:, :, :778, :]
        dis = torch.sqrt(torch.sum(joint_r * joint_r, dim=-1) + 1e-8)
        joint_r = torch.where(joint_vertex, dis, torch.ones_like(dis).to(device) * 100)
        joint_r = torch.mean(torch.topk(joint_r, k=20, dim=-1, largest=False)[0], dim=-1)
        joint_r = torch.cat((joint_r[:, :16], joint_r[:, [3, 6, 9, 12, 15]] / 1.5), dim=-1)

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
        finger_shpere_r = finger_shpere_r.reshape([batch_size, -1, 1]) * self.interval_value.to(device) \
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
        return error.sum(-1)
        # return error.sum(-1).sum(-1)

    def get_part_mesh(self, mesh):
        # waterlight mano part
        edge_vertex_id = [[274, 260, 258, 172, 169, 168, 138, 137, 263, 186],
                          [213, 87, 59, 58, 156, 49, 48, 166, 167, 225],
                          [298, 340, 341, 301, 300, 299, 294, 295, 296, 297],
                          [75, 246, 277, 269, 270, 187, 185, 262, 228, 288],
                          [389, 365, 363, 362, 373, 359, 358, 376, 377, 394],
                          [412, 411, 406, 407, 408, 409, 410, 452, 453, 413],
                          [606, 607, 596, 597, 612, 611, 627, 593, 592, 608],
                          [602, 603, 622, 617, 589, 587, 586, 599, 583, 582],
                          [641, 640, 639, 634, 635, 636, 637, 638, 680, 682],
                          [76, 141, 197, 198, 162, 163, 290, 276, 247],
                          [499, 504, 487, 486, 470, 471, 483, 474, 475, 477],
                          [523, 522, 517, 518, 519, 520, 521, 551, 565, 567],
                          [105, 286, 253, 252, 248, 123, 126, 266, 10, 30, 29],
                          [707, 708, 709, 710, 711, 753, 755, 714, 713, 712]]
        add_vertex = [mesh]
        for vertex_id in edge_vertex_id:
            add_vertex.append(mesh[vertex_id, :].mean(axis=0, keepdims=True))
        water_mesh = np.concatenate(add_vertex, axis=0)
        hand_part_meshs = []
        for i in range(self.part_num):
            vertex = water_mesh[self.part_vertex_id_list[i], :]
            face = self.part_face_list[i]
            hand_part_meshs.append(trimesh.Trimesh(vertices=vertex, faces=face))
        return hand_part_meshs

    def forward(self, beta, theta, quat_or_euler, get_skin=False):
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=torch.float)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float)

        if self.is_cuda:
            beta = beta.cuda()
            theta = theta.cuda()

        num_batch = beta.shape[0]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        if quat_or_euler.shape[-1] == 3:
            global_rot = self.batch_rodrigues(quat_or_euler).view(num_batch, 1, 3, 3)
            if self.is_cuda:
                global_rot = global_rot.cuda()
            Rs = self.batch_rodrigues(
                (torch.matmul(theta, self.hands_comp[:theta.size(-1)]) + self.hands_mean).view(-1, 3)).view(-1, 15, 3,
                                                                                                            3)
        else:
            if not isinstance(quat_or_euler, torch.Tensor):
                quat_or_euler = torch.tensor(quat_or_euler, dtype=torch.float)
            if self.is_cuda:
                quat_or_euler = quat_or_euler.cuda()
            global_rot = self.quat2mat(quat_or_euler).view(-1, 1, 3, 3)
            Rs = self.batch_rodrigues(
                (torch.matmul(theta, self.hands_comp[:theta.size(-1)]) + self.hands_mean).view(-1, 3)).view(-1, 15, 3,
                                                                                                            3)

        pose_feature = (Rs[:, :, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = v_shaped + torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        self.J_transformed, A = self.batch_global_rigid_transformation(torch.cat([global_rot, Rs], dim=1), J[:, :16, :],
                                                                       self.parents)

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

        wrist_vert = torch.mean(
            verts[:, [121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120, 108, 79, 78], :], dim=1,
            keepdim=True)
        verts = torch.cat((verts, wrist_vert), dim=1)
        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def get_mano_vertices(self, quat_or_euler, pose, shape, cam=None, global_scale=None):
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

        joints = joints * 1000
        verts = verts * 1000

        if not (global_scale is None):
            joints = joints * global_scale
            verts = verts * global_scale

        if not (cam is None):
            # check if not tensor and cuda: wrap
            if not isinstance(cam, torch.Tensor):
                cam = torch.tensor(cam, dtype=torch.float)
            if self.is_cuda:
                cam = cam.cuda()

            scale = cam[:, 0].contiguous().view(-1, 1, 1)
            trans = cam[:, 1:].contiguous().view(cam.size(0), 1, -1)

            verts = verts * scale
            verts = verts + trans
            joints = joints * scale
            joints = joints + trans
        return verts, joints


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

from aabbtree import AABB
from aabbtree import AABBTree


def aabb_coll(mesh_1, mesh_2):
    tree = AABBTree()

    vert1 = mesh_1.vertices
    x_min, x_max = vert1[:, 0].min(), vert1[:, 0].max()
    y_min, y_max = vert1[:, 1].min(), vert1[:, 1].max()
    z_min, z_max = vert1[:, 2].min(), vert1[:, 2].max()
    aabb1 = AABB([(x_min, x_max), (y_min, y_max),(z_min, z_max)])
    tree.add(aabb1, 'box 1')

    vert2 = mesh_2.vertices
    x_min, x_max = vert2[:, 0].min(), vert2[:, 0].max()
    y_min, y_max = vert2[:, 1].min(), vert2[:, 1].max()
    z_min, z_max = vert2[:, 2].min(), vert2[:, 2].max()
    aabb2 = AABB([(x_min, x_max), (y_min, y_max), (z_min, z_max)])

    return tree.does_overlap(aabb2)


def self_intersection_blender(part_mesh_list):
    parent_id = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13]
    inter_val = 0
    for s_index, s_mesh in enumerate(part_mesh_list):
        for t_index, t_mesh in enumerate(part_mesh_list[s_index:]):
            t_index = t_index + s_index
            if s_index == t_index or parent_id[s_index] == t_index or parent_id[t_index] == s_index:
                continue
            else:
                if aabb_coll(s_mesh, t_mesh):
                    inter_mesh = s_mesh.intersection(t_mesh, engine="blender")
                    if inter_mesh.area != 0:
                        inter_val += inter_mesh.volume
    return inter_val


def jointImgTo3D(uvd, paras, flip):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
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


def self_intersection(part_mesh_list, pitch=2):
    voxel_list = []
    for mesh in part_mesh_list:
        voxel_list.append(mesh.voxelized(pitch=pitch))
    parent_id = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13]
    inter_val = 0
    for s_index, s_mesh in enumerate(part_mesh_list):
        for t_index, t_voxel in enumerate(voxel_list[s_index:]):
            t_index = t_index + s_index
            if s_index == t_index or parent_id[s_index] == t_index or parent_id[t_index] == s_index:
                continue
            else:
                inside = s_mesh.contains(t_voxel.points)
                volume = inside.sum() * np.power(pitch, 3)
                inter_val += volume
    return inter_val

def intersect_vox(obj_mesh, hand_mesh, pitch=2):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


if __name__ == '__main__':
    #
    data_path = '/home/pfren/pycharm/TIP2021/checkpoint/nyu/Finetune-NoColl-v0/'
    mano_path = '/home/pfren/pycharm/TIP2021/MANO/'
    # view = 2
    for view_id in range(3):
        # Approximate Calculated Intersection Volume
        meshs = np.loadtxt(data_path + 'mesh_result_1_%d.txt' % (view_id)).reshape([-1, 779, 3])
        MANO = MANO_SMPL(mano_path)
        print('load finish !!')
        inter_val_list = []
        for idx, mesh in enumerate(meshs):
            part_mesh_list = MANO.get_part_mesh(mesh)
            inter_val = self_intersection(part_mesh_list)
            inter_val_list.append(inter_val)
            if idx % 50 == 0:
                print(idx)
        inter_val = np.stack(inter_val_list, axis=0)
        np.savetxt(data_path + 'coll_vox_pitch2_view%d.txt' % view_id, inter_val, fmt='%.6f')

        # Fine-grained Calculated Intersection Volume
        coll_vox_2 = np.loadtxt(data_path + 'coll_vox_pitch2_view%d.txt' % view_id)
        coll_id = np.where(coll_vox_2 > 0)[0]
        meshs = np.loadtxt(data_path + 'mesh_result_1_%d.txt' % view_id).reshape([-1, 779, 3])
        MANO = MANO_SMPL(mano_path)
        print('load finish !!')
        coll_vox_1 = np.zeros_like(coll_vox_2)

        for idx in coll_id:
            part_mesh_list = MANO.get_part_mesh(meshs[idx])
            inter_val = self_intersection(part_mesh_list, pitch=1)
            coll_vox_1[idx] = inter_val
            print(idx)

        np.savetxt(data_path + 'coll_vox_pitch1_view%d.txt' % view_id, coll_vox_1, fmt='%.6f')