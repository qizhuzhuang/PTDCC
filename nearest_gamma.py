import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from network_overlap import build_model, Network
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def cal_nonoverlap_extend(nonoverlap, extend=0):
    laplace_4 = [[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]]
    kernel = torch.FloatTensor(laplace_4).expand(1, 1, 3, 3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)
    for _ in range(extend):
        boundary = F.conv2d(nonoverlap, weight, stride=1, padding=1)
        boundary = torch.abs(boundary)
        nonoverlap = boundary + nonoverlap
        nonoverlap = torch.where(nonoverlap > 1, torch.tensor(1.).cuda(), nonoverlap)

    return nonoverlap


# name = '000397'
# img1 = cv2.imread('/home/ubuntu/data3/Datasets/zyz/dataset/testing/warp1/' + name + '.jpg')
# img2 = cv2.imread('/home/ubuntu/data3/Datasets/zyz/dataset/testing/warp2/' + name + '.jpg')
# mask1 = cv2.imread('/home/ubuntu/data3/Datasets/zyz/dataset/testing/mask1/' + name + '.png') / 255
# mask2 = cv2.imread('/home/ubuntu/data3/Datasets/zyz/dataset/testing/mask2/' + name + '.png') / 255
# stitch_mask1 = cv2.imread('/home/ubuntu/data3/Datasets/zyz/dataset/testing_nonoverlap/stitch_mask1/' + name + '.jpg') / 255
# stitch_mask2 = cv2.imread('/home/ubuntu/data3/Datasets/zyz/dataset/testing_nonoverlap/stitch_mask2/' + name + '.jpg') / 255

img1 = cv2.imread('./testing/warp1/warp1.jpg')
img2 = cv2.imread('./testing/warp2/warp2.jpg')
mask1 = cv2.imread('./testing/mask1/mask1.png') / 255
mask2 = cv2.imread('./testing/mask2/mask2.png') / 255
stitch_mask1 = cv2.imread('./testing/learn_mask1.jpg') / 255
stitch_mask2 = cv2.imread('./testing/learn_mask2.jpg') / 255
overlap = mask1 * mask2
nonoverlap = mask1 - overlap

warp1 = img1.astype(dtype=np.float32)
warp1 = warp1 / 255
warp1 = np.transpose(warp1, [2, 0, 1])

warp2 = img2.astype(dtype=np.float32)
warp2 = warp2 / 255
warp2 = np.transpose(warp2, [2, 0, 1])

mask1 = mask1.astype(dtype=np.float32)
mask1 = mask1
mask1 = np.transpose(mask1, [2, 0, 1])

mask2 = mask2.astype(dtype=np.float32)
mask2 = mask2
mask2 = np.transpose(mask2, [2, 0, 1])

warp1_tensor = torch.tensor(warp1)
warp2_tensor = torch.tensor(warp2)
mask1_tensor = torch.tensor(mask1)
mask2_tensor = torch.tensor(mask2)

vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
# vgg = vgg.cuda()
feature = vgg.features[:4] + vgg.features[5:9]
vgg_diff = torch.sqrt(torch.mean((feature(warp1_tensor) - feature(warp2_tensor)) ** 2, dim=0))

warp1_tensor = torch.unsqueeze(warp1_tensor, dim=0).cuda()
warp2_tensor = torch.unsqueeze(warp2_tensor, dim=0).cuda()
mask1_tensor = torch.unsqueeze(mask1_tensor, dim=0).cuda()
mask2_tensor = torch.unsqueeze(mask2_tensor, dim=0).cuda()
vgg_diff = torch.unsqueeze(vgg_diff, dim=0).cuda()

net = Network()
net = net.cuda()
model_path = './model_other/epoch100_model.pth'
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['model'])

with torch.no_grad():
    batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, vgg_diff)

gamma_img1 = batch_out['gamma'].cpu().detach()
gamma_img1 = gamma_img1[0].numpy().transpose(1, 2, 0)

gamma_res = np.zeros_like(gamma_img1, dtype=np.float32)

mask = 1 - overlap[:, :, 0]
distance, indices = distance_transform_edt(mask, return_indices=True)
for i in range(gamma_res.shape[0]):
    for j in range(gamma_res.shape[1]):
        gamma_res[i, j] = gamma_img1[indices[0, i, j], indices[1, i, j]]

N = 1
for n in range(N):
    mask_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(mask.astype(dtype=np.float32)), dim=0), dim=0)
    mask_tensor = mask_tensor.cuda()
    mask_extend = cal_nonoverlap_extend(mask_tensor, 1)
    mask_extend = mask_extend[0, 0].cpu().detach().numpy()

    distance, indices = distance_transform_edt(mask_extend, return_indices=True)
    for row in range(gamma_res.shape[0]):
        for col in range(gamma_res.shape[1]):
            gamma_res[row, col] = gamma_res[row, col] + gamma_img1[indices[0, row, col], indices[1, row, col]]

gamma_res = gamma_res / (N + 1)
gamma_res = gamma_res * nonoverlap + gamma_img1 * overlap + np.ones_like(img1) - nonoverlap - overlap
lightened_img2 = np.power(img1 / 255, gamma_res) * 255
lightened_res = lightened_img2 * stitch_mask1 + img2 * stitch_mask2

cv2.imwrite('lightened_img2.jpg', lightened_img2.astype(np.uint8))
cv2.imwrite('lightened_res.jpg', lightened_res.astype(np.uint8))
