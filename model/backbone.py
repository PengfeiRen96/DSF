import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import BasicBlock, Bottleneck

BN_MOMENTUM = 0.1

resnet = {18: (BasicBlock, [2, 2, 2, 2]),
          50: (Bottleneck, [3, 4, 6, 3]),
          101: (Bottleneck, [3, 4, 23, 3]),
          152: (Bottleneck, [3, 8, 36, 3])
          }


def conv_bn_relu(in_planes, out_planes, kernel):
    return nn.Sequential(
        nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel,
                stride=1,
                padding=1,
                bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.1),
        nn.ReLU(inplace=True)
    )


def convtranspose_bn_relu(in_planes, out_planes, kernel):
    return nn.Sequential(
        nn.ConvTranspose2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.1),
        nn.ReLU(inplace=True)
    )


def offset2joint_softmax(offset, depth, kernel_size, scale=30):
    device = offset.device
    batch_size, joint_num, feature_size, feature_size = offset.size()
    joint_num = int(joint_num / 4)
    if depth.size(-1) != feature_size:
        depth = F.interpolate(depth, size=[feature_size, feature_size])
    offset_unit = offset[:, :joint_num * 3, :, :].contiguous()
    heatmap = offset[:, joint_num * 3:, :, :].contiguous()
    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size,feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size,feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
    mask = depth.lt(0.99).float().view(batch_size, 1, feature_size, feature_size)
    offset_mask = (offset_unit * mask).view(batch_size, joint_num, 3, -1)
    heatmap_mask = (heatmap * mask).view(batch_size, joint_num, -1)
    normal_heatmap = F.softmax(heatmap_mask * scale, dim=-1)

    dist = kernel_size - heatmap_mask * kernel_size
    joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * normal_heatmap.unsqueeze(2).repeat(1, 1, 3, 1),dim=-1)
    return joint


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


class MANO_OCR(nn.Module):
    def __init__(self, backbone, joint_num):
        super(MANO_OCR, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('_')[-1])
        block, layers = resnet[layers_num]
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # self.pre = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mano_regress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.inplanes, 3+45+10+4)
        )
        self.deconv_layer4 = convtranspose_bn_relu(self.inplanes, 256, 4)
        self.deconv_layer3 = convtranspose_bn_relu(256, 256, 4)
        self.deconv_layer2 = convtranspose_bn_relu(256, 256, 4)

        self.finals = nn.ModuleList()
        self.finals.append(nn.Conv2d(in_channels=256, out_channels=self.joint_num*3, kernel_size=1, stride=1))
        self.finals.append(nn.Conv2d(in_channels=256, out_channels=self.joint_num, kernel_size=1, stride=1))# distance
        # self.finals.append(nn.Conv2d(in_channels=256, out_channels=self.joint_num, kernel_size=1, stride=1))# weight

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # def forward(self, img, node, GFM, pcl_uvd, pcl_index, feature_para):
    def forward(self, img):
        device = img.device

        c0 = self.pre(img)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        mano_para = self.mano_regress(c4)
        x = self.deconv_layer4(c4)
        x = self.deconv_layer3(x)
        img_feature = self.deconv_layer2(x)

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)
        return [[img_result, mano_para]]


class MANO_OCR_stage(nn.Module):
    def __init__(self, backbone, joint_num, refine=False, coord='xyz'):
        super(MANO_OCR_stage, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        self.refine = refine
        self.coord_type = coord
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers_num = int(backbone.split('_')[-1])
        block, layers = resnet[layers_num]
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mano_regress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.inplanes, 3+45+10+4)
        )

        self.deconv_layer4 = convtranspose_bn_relu(self.inplanes, 256, 4)
        self.deconv_layer3 = convtranspose_bn_relu(256, 256, 4)
        self.deconv_layer2 = convtranspose_bn_relu(256, 256, 4)

        self.finals = nn.ModuleList()
        self.finals.append(nn.Conv2d(in_channels=256, out_channels=self.joint_num*3, kernel_size=1, stride=1))
        self.finals.append(nn.Conv2d(in_channels=256, out_channels=self.joint_num, kernel_size=1, stride=1))

        if self.refine:
            self.fusion = nn.Sequential(nn.Conv2d(256 + self.joint_num * 4 * 2 + 64, 256, 3, 1, 1), nn.BatchNorm2d(256),
                                        nn.ReLU())
            self.inplanes = 256
            self.layer1_s2 = self._make_layer(block, 64, layers[0])
            self.layer2_s2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3_s2 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4_s2 = self._make_layer(block, 512, layers[3], stride=2)
            self.mano_regress_s2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.inplanes, 3 + 45 + 10 + 4)
            )
            self.deconv_layer4_s2 = convtranspose_bn_relu(self.inplanes, 256, 4)
            self.deconv_layer3_s2 = convtranspose_bn_relu(256, 256, 4)
            self.deconv_layer2_s2 = convtranspose_bn_relu(256, 256, 4)

            self.finals_s2 = nn.ModuleList()
            self.finals_s2.append(nn.Conv2d(in_channels=256, out_channels=self.joint_num * 3, kernel_size=1, stride=1))
            self.finals_s2.append(nn.Conv2d(in_channels=256, out_channels=self.joint_num, kernel_size=1, stride=1))

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        if self.refine:
            for m in self.finals_s2.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, img, render=None, center=None, cube=None, M=None):
        device = img.device

        c0 = self.pre(img)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        mano_para = self.mano_regress(c4)
        x = self.deconv_layer4(c4)
        x = self.deconv_layer3(x)
        img_feature = self.deconv_layer2(x)
        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        if not self.refine:
            return [[img_result, mano_para]]
        else:
            mano_img, mano_joint_uvd, mano_joint_xyz, mesh_xyz = render.render(mano_para, center, cube)
            remap = joint2offset(mano_joint_uvd, mano_img, 0.8, 64)
            fusion_feature = torch.cat((c0, img_feature, img_result, remap), dim=1)

            # fusion_feature = torch.cat((c0, img_feature, img_result), dim=1)

            fusion_feature = self.fusion(fusion_feature)
            c1_s2 = self.layer1_s2(fusion_feature)
            c2_s2 = self.layer2_s2(c1_s2)
            c3_s2 = self.layer3_s2(c2_s2)
            c4_s2 = self.layer4_s2(c3_s2)
            mano_para_s2 = self.mano_regress_s2(c4_s2)
            x = self.deconv_layer4_s2(c4_s2)
            x = self.deconv_layer3_s2(x)
            img_feature_s2 = self.deconv_layer2_s2(x)
            img_result_s2 = torch.Tensor().to(device)
            for layer in self.finals_s2:
                temp = layer(img_feature_s2)
                img_result_s2 = torch.cat((img_result_s2, temp), dim=1)
            return [[img_result, mano_para], [img_result_s2, mano_para_s2]]

    def encoder(self, img):
        device = img.device
        c0 = self.pre(img)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        feat = self.pool(c4)

        x = self.deconv_layer4(c4)
        x = self.deconv_layer3(x)
        img_feature = self.deconv_layer2(x)
        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)
        joint = offset2joint_softmax(img_result, img, 0.8)
        return feat.squeeze(), joint

if __name__ == '__main__':
    # img = torch.rand([32, 14*4, 128, 128])
    # models = Evaluator(14)
    # print(models(img).size())
    img = torch.rand([32, 1, 128, 128])
    models = OCR_UNet('resnet_18', 14)
    models(img)
    # print(models(img)[0][1].size(0))