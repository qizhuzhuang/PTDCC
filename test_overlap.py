import argparse
import torch
from torch.utils.data import DataLoader
from network_overlap import build_model, Network
from dataset_overlap import *
import os
import numpy as np
import cv2


def test(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False)
    img_names = test_data.datas['warp1']['image']

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    model_path = './model_overlap/epoch100_model.pth'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model'])
    print('load model from {}!'.format(model_path))

    path_learned_gamma = './gamma_overlap/'
    if not os.path.exists(path_learned_gamma):
        os.makedirs(path_learned_gamma)

    path_lightened = './lightened_image_overlap/'
    if not os.path.exists(path_lightened):
        os.makedirs(path_lightened)

    print("##################start testing#######################")
    net.eval()
    for i, batch_value in enumerate(test_loader):
        warp1_tensor = batch_value[0]
        warp2_tensor = batch_value[1]
        mask1_tensor = batch_value[2]
        mask2_tensor = batch_value[3]
        vgg_diff = batch_value[4]

        with torch.no_grad():
            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, vgg_diff)

        name_num = (img_names[i].split('\\')[-1]).split('.')[0]
        gamma = batch_out['gamma']
        torch.save(gamma.to(torch.device('cpu')), path_learned_gamma + name_num + ".pth")

        lightened = batch_out['lightened_image'] * mask1_tensor
        lightened_image = (lightened[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)
        lightened_image = np.uint8(lightened_image)
        path_image = path_lightened + name_num + ".jpg"
        cv2.imwrite(path_image, lightened_image)

        print('i = {}'.format(i+1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='F:/work/stitch_work/dataset/testing_ill/')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)

    test(args)