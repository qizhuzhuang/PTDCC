import torch
import torch.nn as nn
import torchvision.models as models


def cal_smooth_term_adjacent(gamma, overlap):
    padding_h = nn.ReplicationPad2d((0, 0, 0, 1))
    img_h = padding_h(gamma)
    padding_w = nn.ReplicationPad2d((0, 1, 0, 0))
    img_w = padding_w(gamma)

    diff_h = torch.abs(img_h[:, :, 0:-1, :] - img_h[:, :, 1:, :])
    diff_w = torch.abs(img_w[:, :, :, 0:-1] - img_w[:, :, :, 1:])

    loss = (torch.sum(diff_h * overlap) + torch.sum(diff_w * overlap)) / torch.sum(overlap)

    return loss


def vgg_feature(img1, img2):
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # if torch.cuda.is_available():
    vgg = vgg.cuda()

    feature = vgg.features[:4]
    feature1 = feature(img1)
    feature2 = feature(img2)
    diff = torch.sqrt(torch.mean((feature1 - feature2) ** 2, dim=1))
    return diff


def cal_color_term_overlap(img1, vgg_diff, lightened_img2, overlap):
    w0 = torch.exp(-torch.pow(vgg_diff / 1, 2))
    w1 = torch.unsqueeze(w0, dim=0)
    w = torch.cat((w1, w1, w1), dim=0)

    loss = torch.sum(w * torch.abs(lightened_img2 - img1) * overlap) / torch.sum(overlap)

    return loss


def total_loss(vgg_diff, img1, lightened_img2, gamma, overlap):
    w0 = torch.exp(-torch.pow(vgg_diff / 0.35, 2))
    w1 = torch.unsqueeze(w0, dim=0)
    w = torch.cat((w1, w1, w1), dim=1)

    padding_h = nn.ReplicationPad2d((0, 0, 0, 1))
    img_h = padding_h(gamma)
    padding_w = nn.ReplicationPad2d((0, 1, 0, 0))
    img_w = padding_w(gamma)

    diff_h = torch.abs(img_h[:, :, 0:-1, :] - img_h[:, :, 1:, :])
    diff_w = torch.abs(img_w[:, :, :, 0:-1] - img_w[:, :, :, 1:])

    gamma_loss = (1 - w) * (diff_w + diff_h) * overlap
    color_loss = 2 * w * torch.abs(lightened_img2 - img1) * overlap
    loss = (torch.sum(gamma_loss) + torch.sum(color_loss)) / torch.sum(overlap)

    return loss

