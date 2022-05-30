from torch import nn
import torch
import torch.nn.functional as F
import math
ReLU = nn.ReLU
Pool = nn.MaxPool2d


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
    joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * normal_heatmap.unsqueeze(2).repeat(1, 1, 3, 1),
        dim=-1)
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


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            # self.relu = ReLU(out_dim)
            self.relu = ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()

        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.relu1 = ReLU()
        self.relu1 = ReLU(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.relu2 = ReLU()
        # self.relu2 = ReLU(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        # self.relu3 = ReLU(int(out_dim / 2))
        self.relu3 = ReLU()
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, joint_num, inp_dim=256, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        self.nstack = nstack
        self.joint_num = joint_num
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 256),
            Residual(256, inp_dim)
        )
        # self.pre = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.BatchNorm2d(64, momentum=0.1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        self.hgs = nn.ModuleList()
        for i in range(nstack):
            self.hgs.append(Hourglass(4, inp_dim, bn, increase))
            # self.hgs.append(nn.Sequential(Conv(inp_dim, inp_dim, 1, bn=True, relu=True), Hourglass(4, inp_dim, bn, increase)))

        self.features = nn.ModuleList()
        for i in range(nstack):
            self.features.append(
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True))
            )

        self.outs_1 = nn.ModuleList([nn.Conv2d(in_channels=inp_dim, out_channels=self.joint_num * 3, kernel_size=1, stride=1, padding=0) for i in range(nstack)])
        self.outs_2 = nn.ModuleList([nn.Conv2d(in_channels=inp_dim, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0) for i in range(nstack)])
        self.outs_3 = nn.ModuleList([nn.Conv2d(in_channels=inp_dim, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0) for i in range(nstack)])

        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(self.joint_num * 5, inp_dim) for i in range(nstack - 1)])
        self.merge_all = nn.ModuleList([Merge(inp_dim*2, inp_dim) for i in range(nstack - 1)])
        self.nstack = nstack
        self.init_weights()
        # self.MGCNs = nn.ModuleList([MGCN(inp_dim, joint_num, dataset) for i in range(nstack)])

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

        for m in self.outs_1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.outs_2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        ## our posenet
        attention = 0
        x = self.pre(imgs)
        combined_hm_preds = []
        combined_feature = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            offset = self.outs_1[i](feature)
            dis = self.outs_2[i](feature)
            weight = self.outs_3[i](feature)
            preds = torch.cat((offset, dis, weight), dim=1)
            # preds = torch.cat((offset, dis), dim=1)
            # preds = offset
            combined_hm_preds.append(preds)
            combined_feature.append(feature)

            # joint = offset2joint_softmax(preds, imgs, 0.8).view(-1,self.joint_num,3)
            # # temp = joint2offset(joint, imgs, 0.8, 64)
            # gcn_out, attention = self.MGCNs[i](imgs, feature, preds, None)
            # preds = gcn_out[1]
            # feature = gcn_out[0]
            # combined_hm_preds += gcn_out[1:]

            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        # return combined_hm_preds, combined_feature, attention
        return combined_hm_preds, hg


if __name__ == '__main__':
    torch.cuda.set_device(0)
    batch_size = 4
    img_size = 96
    joint_num = 14
    # #
    img = torch.rand([batch_size, 1, img_size, img_size])
    model = PoseNet('hourglass_1', joint_num, 'nyu')
    print(model)
    # #
    output, feature = model(img)
    # #
    # # model = DKmodule(1,256)
    # # output = model(img,feature)
    # print(img.size())
    # print(output[0].size())
    # from ptflops import get_model_complexity_info
    # net = nn.ModuleList([
    #         nn.Sequential(
    #             Hourglass(4, 256, True, 0),
    #         ) for i in range(1)])

    net = PoseNet('hourglass_3', 21, 'hands20')
    # print(net(img))
    # macs, params = get_model_complexity_info(net, (1, 128, 128), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
