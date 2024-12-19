import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, vgg_diff):
    vgg_diff = torch.unsqueeze(vgg_diff, dim=0)
    gamma_overlap = net(warp1_tensor, warp2_tensor, vgg_diff)
    overlap = mask1_tensor * mask2_tensor
    nonoverlap = mask1_tensor - overlap
    gamma = gamma_overlap * overlap + nonoverlap
    lightened_image = torch.pow(warp1_tensor, gamma)

    out_dict = {}
    out_dict.update(gamma=gamma, lightened_image=lightened_image)
    return out_dict


class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation=dilation))
        blk.append(nn.ReLU(inplace=False))
        blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation=dilation))
        blk.append(nn.ReLU(inplace=False))
        self.layer = nn.Sequential(*blk)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layer(x)


class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation):
        super(UpBlock, self).__init__()
        self.halfChanelConv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation=dilation),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=(x2.size()[2], x2.size()[3]), mode='nearest')
        x1 = self.halfChanelConv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Network(nn.Module):
    def __init__(self, nclasses=3):
        super(Network, self).__init__()

        self.down1 = DownBlock(7, 16, 1, pool=False)
        self.down2 = DownBlock(16, 32, 2)
        self.down3 = DownBlock(32, 64, 3)
        self.down4 = DownBlock(64, 128, 4)
        self.down5 = DownBlock(128, 256, 5)

        self.up0 = UpBlock(256, 128, 4)
        self.up1 = UpBlock(128, 64, 3)
        self.up2 = UpBlock(64, 32, 2)
        self.up3 = UpBlock(32, 16, 1)
        self.out = nn.Sequential(
            nn.Conv2d(16, nclasses, kernel_size=1),
            # nn.BatchNorm2d(nclasses),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y, z):
        x1 = self.down1(torch.cat((x, y, z), 1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        res = self.up0(x5, x4)
        res = self.up1(res, x3)
        res = self.up2(res, x2)
        res = self.up3(res, x1)
        res = self.out(res)

        return res
