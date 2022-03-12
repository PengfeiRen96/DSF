import os
import shutil
import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from model.backbone import MANO_OCR, MANO_OCR_stage
from render_model.render_loss import depth_loss, surface_loss
from metric.meshLoss import ICPLoss, FingerICPLoss, JointICPLoss
from config import opt
from util import vis_tool
from util.generateFeature import GFM
from metric.losses import SmoothL1Loss
from prefetch_generator import BackgroundGenerator
from data import render_loader
from render_model.mano_layer import Render as Render
from tensorboardX import SummaryWriter
from render_model.transfer import define_G
from util import vis_3d
from pytorch3d.loss import point_mesh_distance
from pytorch3d import _C
import cv2


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.data_rt = self.config.root_dir + "/" + self.config.dataset
        if self.config.model_save == '':
            self.model_save = self.config.net + \
                              '_' + str(self.config.opt) + \
                              '_centerType' + self.config.center_type + \
                              '_coord_weight_' + str(self.config.coord_weight) + \
                              '_deconv_weight_' + str(self.config.deconv_weight) + \
                              '_step_size_' + str(self.config.step_size) + \
                              '_CubeSize_' + str(self.config.cube_size[0])

            self.model_save += '_'
            for index, feature in enumerate(self.config.feature_type):
                self.model_save += feature + '_' + str(self.config.feature_para[index])

            if self.config.finetune_dir != '':
                self.model_save = 'finetune_' + self.model_save
            if self.config.dataset == 'msra':
                self.model_dir = './checkpoint/' + self.config.dataset + '/' + self.model_save + '/' + str(
                    self.config.test_id)
            else:
                self.model_dir = './checkpoint/' + self.config.dataset + '/' + self.model_save
            self.model_dir += self.config.add_info
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.model_dir + '/img')
            os.makedirs(self.model_dir + '/debug')
            os.makedirs(self.model_dir + '/obj')
            os.makedirs(self.model_dir + '/mano')
            os.makedirs(self.model_dir + '/files')

        # save config
        with open(self.model_dir + '/config.txt', 'w') as f:
            for k, v in self.config.__class__.__dict__.items():
                if not k.startswith('_'):
                    print(str(k) + ":" + str(v))
                    f.writelines(str(k) + ":" + str(v) + '\n')

        # save core file
        shutil.copyfile('./train_render.py', self.model_dir+'/files/train_render.py')
        shutil.copyfile('./config.py', self.model_dir + '/files/config.py')
        shutil.copyfile('./model/backbone.py', self.model_dir + '/files/backbone.py')
        shutil.copyfile('./data/render_loader.py', self.model_dir + '/files/render_loader.py')
        shutil.copyfile('./render_model/mano_layer.py', self.model_dir + '/files/mano_layer.py')


        torch.cuda.set_device(0)
        cudnn.benchmark = True
        self.net_joint = 21
        self.net = MANO_OCR_stage(self.config.net, self.net_joint, self.config.stage_num == 2)
        # self.net = MANO_OCR(self.config.net, self.net_joint)
        print(self.net)
        self.net = self.net.cuda()

        # load ori transfer net
        self.transferNet = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
        self.set_requires_grad(self.transferNet, False)
        self.transferNet.eval()
        if self.config.tansferNet_pth != '':
            model_dict = torch.load(self.config.tansferNet_pth+'/latest_net_G_A.pth', map_location=lambda storage, loc: storage)
            self.transferNet.load_state_dict(model_dict)

        optimList = [{"params": self.net.parameters(), "initial_lr": self.config.lr}]
        # init optimizer
        if self.config.opt == 'sgd':
            self.optimizer = SGD(optimList, lr=self.config.lr, momentum=0.9, weight_decay=1e-4)
        elif self.config.opt == 'adam':
            self.optimizer = Adam(optimList, lr=self.config.lr)#1e-4
        elif self.config.opt == 'adamw':
            self.optimizer = AdamW(optimList, lr=self.config.lr, weight_decay=0.01)

        self.L1Loss = SmoothL1Loss(size_average=True).cuda()
        self.L2Loss = nn.MSELoss(reduction='mean')
        self.recon_l1 = nn.L1Loss(reduction='mean')
        self.depthLoss = depth_loss(smooth=False, beta=0.4)

        self.start_epoch = 0

        # load model
        if self.config.load_model != '':
            print('loading model from %s' % self.config.load_model)
            checkpoint = torch.load(self.config.load_model, map_location=lambda storage, loc: storage)

            model_checkpoint = checkpoint['model']
            model_dict = self.net.state_dict()
            for k, v in model_checkpoint.items():
                print(k)
            pretrained_dict = {k: v for k, v in model_checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            self.start_epoch = checkpoint['epoch'] + 1

        # fine-tune model
        if self.config.finetune_dir != '':
            print('loading model from %s' % self.config.finetune_dir)
            checkpoint = torch.load(self.config.finetune_dir, map_location=lambda storage, loc: storage)
            checkpoint_model = checkpoint['model']
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

        # init scheduler
        if self.config.scheduler == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size, gamma=0.1, last_epoch=self.start_epoch)
        elif self.config.scheduler == 'multi_step':
            self.scheduler = MultiStepLR(self.optimizer, self.config.step_size, 0.1, last_epoch=self.start_epoch)
        elif self.config.scheduler == 'auto':
            self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=2, min_lr=1e-8)

        if self.config.dataset == 'msra':
            if self.config.phase == 'train':
                self.trainData = render_loader.msra_loader(self.data_rt, 'train', test_persons=self.config.test_id,
                                                    aug_para=self.config.augment_para,
                                                             img_size=self.config.input_size,
                                                             center_type=self.config.center_type)
                self.trainLoader = DataLoaderX(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

            self.testData = render_loader.msra_loader(self.data_rt, 'test', test_persons=self.config.test_id,
                                           img_size=self.config.input_size, center_type=self.config.center_type)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        if self.config.dataset == 'nyu':
            if self.config.phase == 'train':
                self.trainData = render_loader.nyu_loader(self.data_rt, 'train', aug_para=self.config.augment_para,
                                                             img_size=self.config.input_size,
                                                             cube_size=self.config.cube_size,
                                                             center_type=self.config.center_type)
                # self.trainData = render_loader.nyu_loader_test(self.data_rt, aug_para=self.config.augment_para,
                #                                              img_size=self.config.input_size,
                #                                              cube_size=self.config.cube_size,
                #                                              center_type=self.config.center_type)
                # self.trainData = render_loader.nyu_loader_train_test(self.data_rt,  aug_para=self.config.augment_para,
                #                                              img_size=self.config.input_size,
                #                                              cube_size=self.config.cube_size,
                #                                              center_type=self.config.center_type)
                self.trainLoader = DataLoaderX(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
                # self.trainData_uncorr = render_loader.nyu_modelPara_loader(self.data_rt, 'train',
                #                                              img_size=self.config.input_size,
                #                                             cube_size=self.config.cube_size,
                #                                              center_type=self.config.center_type)

            self.testData = render_loader.nyu_loader(self.data_rt, 'test', type=self.config.test_img_type, view=0, img_size=self.config.input_size,
                                                        cube_size=self.config.cube_size,
                                                        center_type=self.config.center_type)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        if self.config.dataset == 'icvl':
            if self.config.phase == 'train':
                self.trainData = render_loader.flip_icvl_loader(self.data_rt, 'train', aug_para=self.config.augment_para,
                                                             img_size=self.config.input_size,
                                                             cube_size=self.config.cube_size,
                                                             center_type=self.config.center_type)
                self.trainLoader = DataLoaderX(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
            self.testData = render_loader.flip_icvl_loader(self.data_rt, 'test', img_size=self.config.input_size,
                                                        cube_size=self.config.cube_size,
                                                        center_type=self.config.center_type)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        if self.config.dataset == 'shrec':
            if self.config.phase == 'train':
                self.trainData = render_loader.shrec_loader(self.data_rt, aug_para=self.config.augment_para,
                                                             img_size=self.config.input_size,
                                                             cube_size=self.config.cube_size)
                self.trainLoader = DataLoaderX(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
                self.testData = self.trainData
            else:
                self.testData = render_loader.shrec_loader(self.data_rt, aug_para=[0, 0, 0], img_size=self.config.input_size, cube_size=self.config.cube_size)
                self.testLoader = DataLoaderX(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
                self.trainData = self.testData
        self.trainData_synth = render_loader.hands_modelPara_loader(self.config.root_dir + '/hands20/', 'train', cube_size=self.config.cube_size)
        self.trainLoader_synth = DataLoaderX(self.trainData_synth, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

        self.RenderNet = Render(self.config.mano_model_path, self.config.dataset, self.testData.paras, self.testData.ori_img_size).cuda()

        self.best_records = {}
        self.best_records["epoch"] = -1
        self.best_records["val_loss"] = 10000
        self.best_records["Error"] = 10000
        self.test_error = 10000
        self.min_error = 100
        self.GFM_ = GFM()
        # record data
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(self.model_dir, 'train.log'), level=logging.INFO)
        logging.info('======================================================')

        self.writer = SummaryWriter('runs/'+self.config.dataset+'-'+self.config.add_info)

    def train(self):
        self.phase = 'train'
        for epoch in range(self.start_epoch, self.config.max_epoch):
            self.net.train()
            if self.config.train_stage == 'Pretrain':
                iter_synth = self.trainLoader_synth.__iter__()
                main_loader = self.trainLoader_synth
            else:
                iter_synth = self.trainLoader_synth.__iter__()
                iter_real = self.trainLoader.__iter__()
                main_loader = self.trainLoader

            for ii in tqdm(range(main_loader.__len__())):
                model_para, cube_synth = iter_synth.__next__()
                model_para, cube_synth = model_para.cuda(), cube_synth.cuda()
                if self.config.train_stage == 'Pretrain':
                    pose_list, img_list, img_name, scalar_list, scalar_name = self.Pretrain(model_para, cube_synth)
                else:
                    img, xyz_gt, uvd_gt, center, M, cube = iter_real.__next__()
                    img, uvd_gt, xyz_gt = img.cuda(), uvd_gt.cuda(), xyz_gt.cuda()
                    center, M, cube = center.cuda(), M.cuda(), cube.cuda()
                    if self.config.stage_num == 1:
                        pose_list, img_list, img_name, scalar_list, scalar_name = \
                            self.Finetune(model_para, cube_synth, img, center, cube, M, xyz_gt)
                    else:
                        pose_list, img_list, img_name, scalar_list, scalar_name = \
                            self.FinetuneStage(model_para, cube_synth, img, center, cube, M, xyz_gt)

                iter_num = epoch*main_loader.__len__()+ii
                for write_index, name in enumerate(scalar_name):
                    self.writer.add_scalar(name, scalar_list[write_index], global_step=iter_num)
                draw_dataset = 'MANO'
                if (ii + 1) % 1 == 0:
                    iter_num = epoch*main_loader.__len__()+ii
                    for write_index, img_draw in enumerate(img_list):
                        if pose_list[write_index] is not None:
                            img_show = vis_tool.draw_2d_pose(img_draw[0], pose_list[write_index][0], draw_dataset)
                            self.writer.add_image(img_name[write_index], np.transpose(img_show, (2, 0, 1))/255.0, global_step=iter_num)
                        else:
                            self.writer.add_image(img_name[write_index], np.transpose(img_draw, (2, 0, 1))/255.0, global_step=iter_num)

            lof_info = 'Epoch#%d:' % (epoch)
            for write_index, error in enumerate(scalar_list):
                lof_info += scalar_name[write_index] + '_error: %.2f' % (error) + " "

            logging.info(lof_info)
            if 'eval' in self.config.net:
                save = {
                    "model": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }
            else:
                save = {
                    "model": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }

            torch.save(
                save,
                self.model_dir + "/latest.pth"
            )

            if self.config.test_during_train:
                test_error = self.test(epoch=epoch)

                if test_error <= self.min_error:
                    self.min_error = test_error
                    save = {
                        "model": self.net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(
                        save,
                        self.model_dir + "/best.pth"
                    )

            if self.config.scheduler == 'step':
                self.scheduler.step(epoch)
            elif self.config.scheduler == 'multi_step':
                self.scheduler.step()

    @torch.no_grad()
    def test(self, view=0, epoch=-1):
        '''
        计算模型测试集上的准确率
        '''
        self.result_file_list = []
        for file_index in range(self.config.stage_num*2):
            self.result_file_list.append(open(self.model_dir+'/result_'+str(file_index)+'_'+str(view)+'.txt', 'w'))
        self.mano_file = open(self.model_dir+'/MANO_result_'+str(file_index)+'_'+str(view)+'.txt', 'w')
        self.mesh_file = open(self.model_dir + '/mesh_result_' + str(file_index) + '_' + str(view) + '.txt', 'w')
        self.coll_file = open(self.model_dir + '/coll_' + str(file_index) + '_' + str(view) + '.txt', 'w')
        self.phase = 'test'
        self.net.eval()
        if self.config.dataset == 'nyu':
            self.testData = render_loader.nyu_loader(self.data_rt, 'test', type=self.config.test_img_type, view=view, img_size=self.config.input_size,
                                                        cube_size=self.config.cube_size,
                                                        center_type=self.config.center_type)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        error_list = [0] * 2 * self.config.stage_num
        batch_num = 0
        self.coll_loss = 0
        for ii, data in tqdm(enumerate(self.testLoader)):
            img, xyz_gt, uvd_gt, center, M, cube = data
            img, uvd_gt, xyz_gt = img.cuda(), uvd_gt.cuda(), xyz_gt.cuda()
            center, M, cube = center.cuda(), M.cuda(), cube.cuda()
            # joints_uvd, error = self.aug_test_iter(model_para, center, cube, M)
            error = self.test_iter(img, xyz_gt, center, cube, M, ii, view)
            batch_num += 1
            for jj in range(2 * self.config.stage_num):
                error_list[jj] += error[jj]
        print_info = ''
        mean_error = 0
        for jj in range(2 * self.config.stage_num):
            error_list[jj] = error_list[jj] / batch_num
            print_info += " [mean_Error %.2f]" % (error_list[jj])
            mean_error += error_list[jj]
        print(print_info)
        logging.info('Epoch#%d:' % (epoch) + print_info)
        return mean_error / 2 / self.config.stage_num

    def test_iter(self, img, xyz_gt, center, cube, M, batch_index, view):
        outputs = self.net(img, self.RenderNet, center, cube)
        error_list = []
        for index, output in enumerate(outputs):
            pixel_pd, mano_para = output
            all_joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type, self.config.feature_para)
            all_joint_xyz = self.testData.uvd_nl2xyznl_tensor(all_joint_uvd, center, M, cube)

            mano_all_joint_xyz_pd, mano_mesh_xyz_pd = self.RenderNet.get_mesh_xyz(mano_para)
            mano_all_joint_uvd_pd = self.testData.xyz_nl2uvdnl_tensor(mano_all_joint_xyz_pd, center, M, cube)


            joint_uvd = all_joint_uvd[:, self.RenderNet.mano_layer.transfer, :]
            joint_xyz = all_joint_xyz[:, self.RenderNet.mano_layer.transfer, :]
            mano_joint_uvd_pd = mano_all_joint_uvd_pd[:, self.RenderNet.mano_layer.transfer, :]
            mano_joint_xyz_pd = mano_all_joint_xyz_pd[:, self.RenderNet.mano_layer.transfer, :]
            joint_num = joint_uvd.size(1)

            error0 = self.xyz2error(joint_xyz[:, :joint_num - 1, :], xyz_gt[:, :joint_num - 1, :], center, cube,
                                    write_file=False,
                                    stage_index=0)
            error1 = self.xyz2error(mano_joint_xyz_pd[:, :joint_num - 1, :], xyz_gt[:, :joint_num - 1, :], center,
                                    cube, write_file=False,
                                    stage_index=1)

            error_list.append(error0)
            error_list.append(error1)

        if self.config.save_mesh:
            # coll = self.RenderNet.mano_layer.calculate_coll(mano_all_joint_xyz_pd, mano_mesh_xyz_pd.detach()).mean(-1)
            # np.savetxt(self.coll_file, coll.detach().cpu().numpy().reshape([img.size(0), -1]), fmt='%.8f')
            world_mesh = mano_mesh_xyz_pd * cube.unsqueeze(-2) / 2 + center.unsqueeze(-2)
            np.savetxt(self.mesh_file, world_mesh.detach().cpu().numpy().reshape([img.size(0), -1]), fmt='%.3f')
            np.savetxt(self.mano_file, mano_para.detach().cpu().numpy().reshape([img.size(0), -1]), fmt='%.3f')

        if self.config.save_obj:
            world_mesh = mano_mesh_xyz_pd * cube.unsqueeze(-2) / 2 + center.unsqueeze(-2)
            vis_tool.debug_mesh(world_mesh, self.RenderNet.mano_layer.faces, batch_index, self.model_dir+'/obj/', str(view))

        if self.config.save_result:
            joint_world = all_joint_xyz * cube.unsqueeze(-2) / 2 + center.unsqueeze(-2)
            np.savetxt(self.result_file_list[0], self.testData.points3DToImg(joint_world).detach().cpu().numpy().reshape([img.size(0), -1]), fmt='%.3f')
            mano_world_joint = mano_all_joint_xyz_pd * cube.unsqueeze(-2) / 2 + center.unsqueeze(-2)
            np.savetxt(self.result_file_list[1], self.testData.points3DToImg(mano_world_joint).detach().cpu().numpy().reshape([img.size(0), -1]), fmt='%.3f')

        return error_list

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def Pretrain(self, model_para, cube):
        device = model_para.device
        self.optimizer.zero_grad()
        batch_size = model_para.size(0)

        # render
        augmentShape = torch.randn([batch_size, 10]).to(device) * 3
        augmentCenter = (torch.rand([batch_size, 3]).to(device) - 0.5) * 40
        augmentSize = (1 + (torch.rand([batch_size, 1]).to(device) - 0.5) * 0.4) # 250 mm
        augmentView = torch.rand([model_para.size(0), 3]).to(device) * np.pi * 2 * 0

        # 62 3 10（shape） 45（theta） 4（scale trans）
        #
        img, joint_uvd_gt, mesh_uvd_gt, joint_xyz_gt, mesh_xyz_gt, center, cube, M = \
            self.RenderNet(model_para, None, cube,  augmentView=augmentView, augmentShape=augmentShape,
                           augmentCenter=augmentCenter, augmentSize=augmentSize, mask=self.config.mask)

        if not self.config.tansferNet_pth == '':
            img_transfer = self.transferNet(img)
        else:
            img_transfer = img

        outputs = self.net(img_transfer, self.RenderNet, center, cube)
        loss = 0
        scalar_list = []
        scalar_name = []
        vis_pose = [joint_uvd_gt]
        vis_list = [img]
        vis_name = ['label']
        for index in range(self.config.stage_num):
            pixel_pd, mano_para_pd = outputs[index]
            feature_size = pixel_pd.size(-1)

            # PWE
            # B X 4J X H X W
            # B X C X H X W
            pixel_gt = self.GFM_.joint2feature(joint_uvd_gt, img, self.config.feature_para, feature_size, self.config.feature_type)
            joint_uvd_pd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type, self.config.feature_para)
            joint_xyz_pd = self.trainData.uvd_nl2xyznl_tensor(joint_uvd_pd, center, M, cube)
            loss_pixel = self.L1Loss(pixel_pd, pixel_gt) * self.config.deconv_weight
            loss_coord = self.L1Loss(joint_uvd_pd, joint_uvd_gt) * self.config.coord_weight
            loss += (loss_pixel + loss_coord)

            # MPE
            mano_joint_xyz_s_pd, mesh_xyz_s_pd = self.RenderNet.get_mesh_xyz(mano_para_pd)
            mano_joint_uvd_pd = self.trainData.xyz_nl2uvdnl_tensor(mano_joint_xyz_s_pd, center, M, cube)
            joint_loss = self.L1Loss(mano_joint_xyz_s_pd, joint_xyz_gt) * self.config.coord_weight
            verts_loss = self.L1Loss(mesh_xyz_s_pd, mesh_xyz_gt) * self.config.coord_weight
            beta_loss = torch.mean(torch.pow(mano_para_pd[:, 48:58], 2)) * self.config.coord_weight * 10
            scale_loss = torch.mean(torch.abs(torch.min(mano_para_pd[:, 58], torch.zeros_like(mano_para_pd[:, 58]).to(device)))) * 0.1
            mano_loss = beta_loss + verts_loss + joint_loss + scale_loss
            loss += mano_loss

            # for TensorBoard
            error_pixel = self.xyz2error(joint_xyz_pd, joint_xyz_gt,center, cube)
            scalar_list.append(error_pixel)
            scalar_name.append('Pixel-Error_%d'%(index))
            error_mano = self.xyz2error(mano_joint_xyz_s_pd, joint_xyz_gt, center, cube)
            scalar_list.append(error_mano)
            scalar_name.append('MANO-Error_%d'%(index))
            scalar_list.append(scale_loss)
            scalar_name.append('scale-loss%d'%(index))

            vis_pose.append(joint_uvd_pd)
            vis_list.append(img_transfer)
            vis_name.append('PWE-%d'%(index))

            vis_pose.append(mano_joint_uvd_pd)
            vis_list.append(img_transfer)
            vis_name.append('MPE-%d'%(index))

        loss.backward()
        self.optimizer.step()
        return vis_pose, vis_list, vis_name, scalar_list, scalar_name

    def Finetune(self, model_para, cube, img_r, center_r, cube_r, M_r, xyz_gt_r):
        device = model_para.device
        batch_size = model_para.size(0)
        loss = 0
        self.optimizer.zero_grad()

        # for render img
        augmentShape = torch.randn([batch_size, 10]).to(device) * 3
        augmentCenter = (torch.rand([batch_size, 3]).to(device) - 0.5) * 40
        augmentSize = (1 + (torch.rand([batch_size, 1]).to(device) - 0.5) * 0.4)
        augmentView = torch.rand([model_para.size(0), 3]).to(device) * np.pi * 2

        img, joint_uvd_gt, mesh_uvd_gt, joint_xyz_gt, mesh_xyz_gt, center_s, cube_s, M_s = \
                self.RenderNet(model_para, None, cube,
                               augmentView=augmentView, augmentShape=augmentShape, augmentCenter=augmentCenter,
                               augmentSize=augmentSize, mask=self.config.mask)
        if self.config.tansferNet_pth !='':
            img_transfer = self.transferNet(img)
        else:
            img_transfer = img
        outputs = self.net(img_transfer, self.RenderNet, center_s, cube_s)

        # pixel loss
        pixel_pd, mano_para_pd = outputs[0]
        feature_size = pixel_pd.size(-1)
        pixel_gt = self.GFM_.joint2feature(joint_uvd_gt, img, self.config.feature_para, feature_size, self.config.feature_type)
        joint_uvd_pd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type, self.config.feature_para)
        joint_xyz_pd = self.trainData.uvd_nl2xyznl_tensor(joint_uvd_pd, center_s, M_s, cube_s)
        loss_pixel = self.L1Loss(pixel_pd[:, :pixel_gt.size(1), :, :], pixel_gt) * self.config.deconv_weight
        loss_coord = self.L1Loss(joint_uvd_pd, joint_uvd_gt) * self.config.coord_weight
        loss += (loss_pixel + loss_coord)

        mano_joint_xyz_s_pd, mesh_xyz_s_pd = self.RenderNet.get_mesh_xyz(mano_para_pd)
        joint_loss = self.L1Loss(mano_joint_xyz_s_pd, joint_xyz_gt) * self.config.coord_weight
        verts_loss = self.L1Loss(mesh_xyz_s_pd, mesh_xyz_gt) * self.config.coord_weight
        coll_loss = self.RenderNet.mano_layer.calculate_coll(mano_joint_xyz_s_pd, mesh_xyz_s_pd.detach())
        mano_loss = verts_loss + joint_loss + coll_loss*self.config.coll_weight
        loss += mano_loss

        ############################## for real img ############################
        batch_size = img_r.size(0)
        outputs = self.net(img_r, self.RenderNet,  center_r, cube_r)
        # pixel loss
        pixel_r_pd, mano_para_r_pd = outputs[0]
        joint_uvd_r_pd = self.GFM_.feature2joint(img_r, pixel_r_pd, self.config.feature_type, self.config.feature_para)
        joint_xyz_r_pd = self.trainData.uvd_nl2xyznl_tensor(joint_uvd_r_pd, center_r, M_r, cube_r)

        error_pixel_r_batch = self.xyz2error(
            joint_xyz_r_pd[:, self.RenderNet.mano_layer.transfer, :][:, :xyz_gt_r.size(1) - 1, :],
            xyz_gt_r[:, :12, :], center_r, cube_r, keep_batch=True)
        error_pixel_r = error_pixel_r_batch.mean(-1)


        ################# MANO loss ###############
        render_img_r, mano_joint_uvd_r_pd, mano_joint_xyz_r_pd, mesh_xyz_r_pd =\
            self.RenderNet.render(mano_para_r_pd, center_r, cube_r)
        coll_loss = self.RenderNet.mano_layer.calculate_coll(mano_joint_xyz_r_pd, mesh_xyz_r_pd.detach())

        error_mano_r_batch = self.xyz2error(
            mano_joint_xyz_r_pd[:, self.RenderNet.mano_layer.transfer, :][:, :xyz_gt_r.size(1) - 1, :],
            xyz_gt_r[:, :12, :], center_r, cube_r, keep_batch=True)
        error_mano_r = error_mano_r_batch.mean(-1)

        ################# model-to-data term ###############
        img_r_crop = self.trainData.crop_hand(img_r, mano_joint_xyz_r_pd.detach(), center_r, M_r, cube_r)
        render_img_r_crop = self.trainData.crop_hand(render_img_r, mano_joint_xyz_r_pd.detach(), center_r, M_r, cube_r)
        depth_loss_mask = img_r_crop.lt(0.99) | render_img_r_crop.lt(0.99)
        m2d_loss_batch = torch.abs(img_r_crop-render_img_r_crop).mean(-1).mean(-1) / (depth_loss_mask.float().mean(-1).mean(-1)+1e-8)
        m2d_loss = m2d_loss_batch.mean()

        ################# Part-aware data-to-model term  ###############################
        _, pcl_img = self.trainData.uvdImg2xyzImg(img_r_crop, center_r, M_r, cube_r)
        pcl_img = pcl_img.reshape(batch_size, 3, -1).permute(0, 2, 1)
        segment_img = self.RenderNet.mano_layer.seg_pcl(joint_xyz_r_pd, mano_joint_xyz_r_pd, mesh_xyz_r_pd, pcl_img)
        segment_img = torch.where(img_r_crop.lt(0.99).reshape(batch_size, -1), segment_img, torch.zeros_like(segment_img))
        segment_img = segment_img.reshape(batch_size, 1, 128, 128)
        joint_img_r = torch.where(segment_img.gt(0), img_r, torch.ones_like(img_r))

        joint_pcl = self.trainData.Img2pcl(joint_img_r, 128, center_r, M_r, cube_r, 2048)
        segment = self.RenderNet.mano_layer.seg_pcl(joint_xyz_r_pd, mano_joint_xyz_r_pd, mesh_xyz_r_pd, joint_pcl)
        pd2m_loss_joint = JointICPLoss(mesh_xyz_r_pd, joint_pcl, self.RenderNet.mano_layer.joint_faces, segment)
        pd2m_loss_batch = pd2m_loss_joint.mean(-1)
        pd2m_loss = pd2m_loss_batch.mean(-1)

        ################# Data-to-model term  ###############################
        pcl = self.trainData.Img2pcl(img_r_crop, 128, center_r, M_r, cube_r, 2048)
        d2m_loss_batch = ICPLoss(mesh_xyz_r_pd, pcl, self.RenderNet.mano_layer.faces)
        d2m_loss = d2m_loss_batch.mean(-1)

        id_to_color = vis_tool.get_segmentJointColor()
        segment_img_show = id_to_color[segment_img.squeeze(1).detach().cpu().numpy()]

        ################# P2M Loss ###############
        P2M_loss = self.L1Loss(mano_joint_uvd_r_pd, joint_uvd_r_pd.detach()) * self.config.coord_weight

        ################# M2P Loss ###############
        depth_loss_mask = (img_r_crop.lt(0.95) & render_img_r.lt(0.95)).float()
        depth_diff = torch.abs(img_r_crop - render_img_r) * depth_loss_mask
        depth_mask_value = depth_diff.sum(-1).sum(-1)/depth_loss_mask.sum(-1).sum(-1)
        depth_mask = depth_mask_value.lt(0.04).squeeze(-1)
        icp_mask = d2m_loss_batch.lt(1e-3)
        mano_mask = depth_mask & icp_mask
        joint_mask = pd2m_loss_joint.lt(1e-3)
        joint_add = np.array([2, 5, 8, 11, 14])
        joint_mask = torch.cat((torch.ones(batch_size, 1).to(device), joint_mask, joint_mask[:, joint_add]), dim=-1).gt(0)
        joint_mano_mask = mano_mask.unsqueeze(-1) & joint_mask
        joint_mano_mask = joint_mano_mask.detach().view(-1)
        joint_mano_mask = joint_mano_mask.gt(0).nonzero().squeeze()
        mano_joint_uvd_r_pd_select = torch.index_select(mano_joint_uvd_r_pd.view(-1, 3), dim=0, index=joint_mano_mask)
        joint_uvd_r_pd_select = torch.index_select(joint_uvd_r_pd.view(-1, 3), dim=0, index=joint_mano_mask)
        if joint_mano_mask.sum() == 0:
            M2P_loss = 0
        else:
            M2P_loss = self.L1Loss(joint_uvd_r_pd_select, mano_joint_uvd_r_pd_select.detach()) * self.config.coord_weight

        loss += P2M_loss
        loss += m2d_loss*0.1*self.config.model_weight
        loss += d2m_loss*self.config.model_weight
        loss += pd2m_loss*self.config.partICP_weight
        loss += M2P_loss*self.config.M2P_weight
        loss += coll_loss*self.config.coll_weight

        loss.backward()
        self.optimizer.step()

        return [joint_uvd_r_pd, joint_uvd_r_pd, mano_joint_uvd_r_pd, mano_joint_uvd_r_pd*0,               None], \
               [img_r,           img_r_crop,      render_img_r,       depth_loss_mask, segment_img_show[0]], \
                ['img_r',        'img_r_crop',    'mano_r',         'depth_select',      'segment'], \
               [error_pixel_r, error_mano_r,  m2d_loss,   pd2m_loss, P2M_loss, coll_loss, M2P_loss, d2m_loss], \
                ['PixelError', 'ManoError',   "m2d",      "pd2m",    "P2M",    "coll",    "M2P",    "d2m"]


    def FinetuneStage(self, model_para, cube, img_r, center_r, cube_r, M_r, xyz_gt_r):
        device = model_para.device
        batch_size = model_para.size(0)
        self.optimizer.zero_grad()

        ############################ for render img #############################################
        augmentShape = torch.randn([batch_size, 10]).to(device) * 3
        augmentCenter = (torch.rand([batch_size, 3]).to(device) - 0.5) * 40
        augmentSize = (1 + (torch.rand([batch_size, 1]).to(device) - 0.5) * 0.4)
        augmentView = torch.rand([model_para.size(0), 3]).to(device) * np.pi * 2

        img, joint_uvd_gt, mesh_uvd_gt, joint_xyz_gt, mesh_xyz_gt, center_s, cube_s, M_s = self.RenderNet(model_para, None, cube,
                                                                                              augmentView=augmentView,
                                                                                              augmentShape=augmentShape,
                                                                                              augmentCenter=augmentCenter,
                                                                                              augmentSize=augmentSize,
                                                                                              mask=self.config.mask)
        img_transfer = self.transferNet(img)
        outputs = self.net(img_transfer, self.RenderNet, center=center_s, cube=cube_s)
        dense_joints = []
        dense_errors = []
        mano_errors = []
        loss = 0
        for index in range(2):
            pixel_pd, mano_para_pd = outputs[index]
            feature_size = pixel_pd.size(-1)
            pixel_gt = self.GFM_.joint2feature(joint_uvd_gt, img, self.config.feature_para, feature_size,
                                               self.config.feature_type)
            joints_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type, self.config.feature_para)
            loss_pixel = self.L1Loss(pixel_pd, pixel_gt) * self.config.deconv_weight
            loss_coord = self.L1Loss(joints_uvd, joint_uvd_gt) * self.config.coord_weight
            loss += (loss_pixel + loss_coord)
            error = self.xyz2error(self.trainData.uvd_nl2xyznl_tensor(joints_uvd, center_s, M_s, cube_s),
                                   self.trainData.uvd_nl2xyznl_tensor(joint_uvd_gt, center_s, M_s, cube_s),
                                   center_s, cube_s)
            dense_joints.append(joints_uvd)
            dense_errors.append(error)

            mano_joint_xyz_s, mano_mesh_s = self.RenderNet.get_mesh_xyz(mano_para_pd)
            joint_loss = self.L1Loss(mano_joint_xyz_s, joint_xyz_gt) * self.config.coord_weight
            verts_loss = self.L1Loss(mano_mesh_s, mesh_xyz_gt) * self.config.coord_weight
            coll_loss = self.RenderNet.mano_layer.calculate_coll(mano_joint_xyz_s, mano_mesh_s.detach()).mean()*self.config.coll_weight
            mano_loss = verts_loss + joint_loss + coll_loss
            loss += mano_loss
            error_mano_s = self.xyz2error(mano_joint_xyz_s, joint_xyz_gt, center_s, cube_s)
            mano_errors.append(error_mano_s)

        ######################################## for real img final sup ########################################
        batch_size = img_r.size(0)
        outputs = self.net(img_r, self.RenderNet, center=center_r, cube=cube_r)

        dense_joints = []
        mano_joints = []
        mano_imgs = []
        P2M_losses = []
        m2d_losses = []
        d2m_losses = []
        pd2m_losses = []
        mano_errors = []
        pixel_errors = []

        pixel_pd_teacher, mano_para_pd_teacher = outputs[1]
        pixel_pd_teacher, mano_para_pd_teacher = pixel_pd_teacher.detach(), mano_para_pd_teacher.detach()
        joints_uvd_teacher = self.GFM_.feature2joint(img_r, pixel_pd_teacher, self.config.feature_type, self.config.feature_para).detach()
        joints_xyz_teacher = self.trainData.uvd_nl2xyznl_tensor(joints_uvd_teacher.detach(), center_r, M_r, cube_r)
        mano_joint_xyz_r, mano_mesh_r = self.RenderNet.get_mesh_xyz(mano_para_pd_teacher)
        mano_joints_xyz_teacher = mano_joint_xyz_r.detach()
        mano_mesh_teacher = mano_mesh_r.detach()

        # pre-process joint
        crop_img_r = self.trainData.crop_hand(img_r, mano_joints_xyz_teacher.detach(), center_r, M_r, cube_r)
        _, pcl_img = self.trainData.uvdImg2xyzImg(crop_img_r, center_r, M_r, cube_r)
        pcl_img = pcl_img.reshape(batch_size, 3, -1).permute(0, 2, 1)
        segment_img = self.RenderNet.mano_layer.seg_pcl(joints_xyz_teacher, mano_joints_xyz_teacher, mano_mesh_teacher, pcl_img)
        segment_img = torch.where(crop_img_r.lt(0.99).reshape(batch_size, -1), segment_img, torch.zeros_like(segment_img))
        segment_img = segment_img.reshape(batch_size, 1, 128, 128)
        joint_img_r = torch.where(segment_img.gt(0), crop_img_r, torch.ones_like(img_r))
        joint_pcl = self.trainData.Img2pcl(joint_img_r, 128, center_r, M_r, cube_r, 2048)
        segment = self.RenderNet.mano_layer.seg_pcl(joints_xyz_teacher, mano_joints_xyz_teacher, mano_mesh_teacher, joint_pcl)
        pcl = self.trainData.Img2pcl(crop_img_r, 128, center_r, M_r, cube_r, 2048)
        id_to_color = vis_tool.get_segmentJointColor()
        segment_img_show = id_to_color[segment_img.squeeze(1).detach().cpu().numpy()]

        # for PEW stage 1
        pixel_pd_r, mano_para_pd_r = outputs[0]
        joints_uvd_pd_r = self.GFM_.feature2joint(img_r, pixel_pd_r, self.config.feature_type, self.config.feature_para)
        loss_pixel = self.L1Loss(pixel_pd_r, pixel_pd_teacher) * self.config.deconv_weight
        loss_coord = self.L1Loss(joints_uvd_pd_r, joints_uvd_teacher) * self.config.coord_weight
        loss += (loss_pixel + loss_coord)
        dense_joints.append(joints_uvd_pd_r)
        joints_xyz_pd_r = self.trainData.uvd_nl2xyznl_tensor(joints_uvd_pd_r, center_r, M_r, cube_r)
        error_pixel_r = self.xyz2error(joints_xyz_pd_r[:,self.RenderNet.mano_layer.transfer,:][:, :joints_uvd_pd_r.size(1) - 1, :],
                                        xyz_gt_r[:, :12, :], center_r, cube_r)

        pixel_errors.append(error_pixel_r)

        # for MPE stage 1
        mano_img_r, mano_joint_uvd_r, mano_joint_xyz_r, mano_mesh_r = self.RenderNet.render(mano_para_pd_r, center_r, cube_r)
        error_mano_r = self.xyz2error(mano_joint_xyz_r[:, self.RenderNet.mano_layer.transfer, :][:, :joints_uvd_pd_r.size(1)-1, :], xyz_gt_r[:, :12, :], center_r, cube_r)
        mano_errors.append(error_mano_r)

        distill_joint_loss = self.L1Loss(mano_joint_xyz_r, joints_xyz_teacher) * self.config.coord_weight
        distill_verts_loss = self.L1Loss(mano_mesh_r, mano_mesh_teacher) * self.config.coord_weight
        coll_loss = self.RenderNet.mano_layer.calculate_coll(mano_joint_xyz_r, mano_mesh_r.detach()).mean()

        ################# Depth loss ###############
        mano_img_r_crop = self.trainData.crop_hand(mano_img_r, mano_joints_xyz_teacher.detach(), center_r, M_r, cube_r)
        depth_loss_mask = crop_img_r.lt(0.99) | mano_img_r_crop.lt(0.99)
        depth_diff = torch.abs(crop_img_r-mano_img_r_crop)*depth_loss_mask
        m2d_loss_batch = depth_diff.sum(-1).sum(-1) / (depth_loss_mask.float().sum(-1).sum(-1)+1e-8)
        m2d_loss = m2d_loss_batch.mean() * 0.1

        ################# ICP loss ###############
        part_d2m_loss_joint = JointICPLoss(mano_mesh_r, joint_pcl, self.RenderNet.mano_layer.joint_faces, segment)
        part_d2m_loss_batch = part_d2m_loss_joint.mean(-1)
        part_d2m_loss = part_d2m_loss_batch.mean(-1)

        d2m_loss_batch = ICPLoss(mano_mesh_r, pcl, self.RenderNet.mano_layer.faces)
        d2m_loss = d2m_loss_batch.mean(-1)

        loss += distill_joint_loss
        loss += distill_verts_loss
        loss += coll_loss*self.config.coll_weight
        loss += m2d_loss*self.config.model_weight
        loss += d2m_loss*self.config.model_weight
        loss += part_d2m_loss*self.config.partICP_weight
        mano_joints.append(mano_joint_uvd_r)
        mano_imgs.append(mano_img_r)

        ################# Stage 2 ###############
        pixel_pd_r, mano_para_pd_r = outputs[1]
        joints_uvd_pd_r = self.GFM_.feature2joint(img_r, pixel_pd_r, self.config.feature_type, self.config.feature_para)
        joints_xyz_pd_r = self.trainData.uvd_nl2xyznl_tensor(joints_uvd_pd_r, center_r, M_r, cube_r)
        error_pixel_r = self.xyz2error(joints_xyz_pd_r[:,self.RenderNet.mano_layer.transfer,:][:, :joints_uvd_pd_r.size(1) - 1, :],xyz_gt_r[:, :12, :], center_r, cube_r)
        pixel_errors.append(error_pixel_r)
        dense_joints.append(joints_uvd_pd_r)

        mano_img_r, mano_joint_uvd_r, mano_joint_xyz_r, mano_mesh_r = self.RenderNet.render(mano_para_pd_r, center_r, cube_r)
        error_mano_r = self.xyz2error(mano_joint_xyz_r[:, self.RenderNet.mano_layer.transfer, :][:, :joints_uvd_pd_r.size(1)-1, :], xyz_gt_r[:, :12, :], center_r, cube_r)
        mano_errors.append(error_mano_r)
        mano_joints.append(mano_joint_uvd_r)


        ################# P2M loss ###############
        P2M_loss = self.L1Loss(mano_joint_uvd_r, joints_uvd_teacher.detach()) * self.config.coord_weight
        coll_loss = self.RenderNet.mano_layer.calculate_coll(mano_joint_xyz_r, mano_mesh_r.detach()).mean()

        ################# m2d loss ###############
        mano_img_r_crop = self.trainData.crop_hand(mano_img_r, mano_joints_xyz_teacher.detach(), center_r, M_r, cube_r)
        depth_loss_mask = crop_img_r.lt(0.99) | mano_img_r_crop.lt(0.99)
        depth_diff = torch.abs(crop_img_r-mano_img_r_crop)*depth_loss_mask
        m2d_loss_batch = depth_diff.sum(-1).sum(-1) / (depth_loss_mask.float().sum(-1).sum(-1)+1e-8)
        m2d_loss = d2m_loss_batch.mean() * 0.1

        ################# d2m loss ###############
        pd2m_loss_joint = JointICPLoss(mano_mesh_r, joint_pcl, self.RenderNet.mano_layer.joint_faces, segment)
        pd2m_loss_batch = pd2m_loss_joint.mean(-1)
        pd2m_loss = pd2m_loss_batch.mean(-1)

        d2m_loss_batch = ICPLoss(mano_mesh_r, pcl, self.RenderNet.mano_layer.faces)
        d2m_loss = d2m_loss_batch.mean(-1)

        ################# M2P loss ###############
        M2P_depth_mask = crop_img_r.lt(0.99) & mano_img_r_crop.lt(0.99)
        depth_mask_batch = (torch.abs(crop_img_r-mano_img_r_crop)*M2P_depth_mask).sum(-1).sum(-1) / (depth_loss_mask.float().sum(-1).sum(-1)+1e-8)
        depth_mask = depth_mask_batch.lt(0.04).squeeze(-1)
        icp_mask = d2m_loss_batch.lt(1e-3)
        mano_mask = depth_mask & icp_mask
        joint_mask = pd2m_loss_joint.lt(1e-3)
        joint_add = np.array([2, 5, 8, 11, 14])
        joint_mask = torch.cat((torch.ones(batch_size, 1).to(device), joint_mask, joint_mask[:, joint_add]), dim=-1).gt(0)
        joint_mano_mask = mano_mask.unsqueeze(-1) & joint_mask
        joint_mano_mask = joint_mano_mask.detach().view(-1)
        joint_mano_mask = joint_mano_mask.gt(0).nonzero().squeeze()
        mano_joint_uvd_r_select = torch.index_select(mano_joint_uvd_r.view(-1,3), dim=0, index=joint_mano_mask)
        joints_uvd_pd_r_select = torch.index_select(joints_uvd_pd_r.view(-1,3), dim=0, index=joint_mano_mask)
        if joint_mano_mask.sum() == 0:
            M2P_loss = 0
        else:
            M2P_loss = self.L1Loss(joints_uvd_pd_r_select, mano_joint_uvd_r_select.detach()) * self.config.coord_weight

        loss += P2M_loss
        loss += coll_loss * self.config.coll_weight
        loss += m2d_loss * self.config.model_weight
        loss += d2m_loss * self.config.model_weight
        loss += pd2m_loss * self.config.partICP_weight
        loss += M2P_loss * self.config.M2P_weight

        P2M_losses.append(P2M_loss)
        d2m_losses.append(m2d_loss)
        m2d_losses.append(d2m_loss)
        pd2m_losses.append(pd2m_loss)
        mano_joints.append(mano_joint_uvd_r)
        mano_imgs.append(mano_img_r)

        loss.backward()
        self.optimizer.step()
        return [dense_joints[0], mano_joints[0], mano_joints[1],  dense_joints[1], None], \
               [img_r,           mano_imgs[0],   mano_imgs[1],    crop_img_r,      segment_img_show[0]], \
                ['img_r',        'mano_r',       'mano_r_refine', 'img_r_refine',  'segment', 'diff'], \
               [pixel_errors[0], pixel_errors[1], mano_errors[0],mano_errors[1], P2M_loss, m2d_loss, d2m_loss, pd2m_loss, M2P_loss,coll_loss], \
                ['PixelError0', 'PixelError1',  'MANOError0',   'MANOError1',  "P2M",  "m2d",  "d2m",'pd2m',  "M2P","coll"]

    @torch.no_grad()
    def xyz2error(self, output, joint, center, cube_size, write_file=False, stage_index=0, keep_batch=False, keep_joint=False):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        center = center.detach().cpu().numpy()
        cube_size = cube_size.detach().cpu().numpy()
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

        # output = rotatePoint2D(output, -self.config.angle)

        joint_xyz = output * cube_size / 2 + center
        joint_world_select = joint * cube_size / 2 + center
        # joint_xyz[:,:,2] = joint_xyz[:, :, 2] - 15

        if 'icvl' == self.config.dataset:
            icvl_bias = np.array([20, 22, 13.5, 7.5, 12.5, 12.5, 3, 12.5, 12.5, 8, 16, 12.5, 3, 13, 7.3, 6]).reshape([1,16])
            joint_xyz[:, :, 2] = joint_xyz[:, :, 2] - icvl_bias
        errors = (joint_xyz - joint_world_select) * (joint_xyz - joint_world_select)
        # np.savetxt(self.result_file_list[stage_index], np.sqrt(np.sum(errors, axis=2)).mean(-1).reshape([batchsize]), fmt='%.3f')
        if keep_joint:
            errors = np.sqrt(np.sum(errors, axis=2))
        elif keep_batch:
            errors = np.sqrt(np.sum(errors, axis=2)).mean(-1)
        else:
            if self.config.dataset =='msra':
                errors = (np.sqrt(np.sum(errors, axis=2))[:, 1:]).mean()
            else:
                errors = np.sqrt(np.sum(errors, axis=2)).mean()

        if self.phase == 'test' and write_file:
            if self.config.dataset =='icvl':
                joint_uvd = self.testData.joint3DToImg(joint_xyz).reshape([batchsize, joint_num, 3])
                joint_uvd[:, :, 0] = 320 - joint_uvd[:, :, 0]
                np.savetxt(self.result_file_list[stage_index],joint_uvd.reshape([batchsize, joint_num*3]), fmt='%.3f')
            else:
                np.savetxt(self.result_file_list[stage_index], self.testData.joint3DToImg(joint_xyz).reshape([batchsize, joint_num * 3]), fmt='%.3f')

        return errors


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


if __name__ == '__main__':
    Trainer = Trainer(opt)
    if Trainer.config.phase == 'train':
        Trainer.train()
        if Trainer.config.dataset == 'nyu':
            for view in range(3):
                Trainer.test(view)
        else:
            Trainer.test(0)
    else:
        if Trainer.config.dataset == 'nyu':
            Trainer.test(0)
            Trainer.test(1)
            Trainer.test(2)
        else:
            Trainer.test(0)