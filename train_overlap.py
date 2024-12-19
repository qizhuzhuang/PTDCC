import argparse
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network_overlap import build_model, Network
from dataset_overlap import TrainDataset
from loss_overlap import *
import cv2
import numpy as np
import torchvision.models as models


# path of project
# last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# path to save the summary files
SUMMARY_DIR = './summary_overlap/'
writer = SummaryWriter(log_dir=SUMMARY_DIR)

# path to save the model files
MODEL_DIR = './model_overlap/'

# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=0, shuffle=True,
                              drop_last=False)

    # define the network
    net = Network()

    if torch.cuda.is_available():
        net = net.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    start_epoch = 0
    glob_iter = 0
    print('training from stratch!')

    print("##################start training#######################")
    score_print_fre = 300
    net.train()

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))

        sigma_total_loss = 0.
        sigma_gamma_loss = 0.
        sigma_color_loss = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            warp1_tensor = batch_value[0]
            warp2_tensor = batch_value[1]
            mask1_tensor = batch_value[2]
            mask2_tensor = batch_value[3]
            vgg_diff = batch_value[4]

            # forward, backward, update weights\           optimizer.zero_grad()
            overlap = mask1_tensor * mask2_tensor
            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, vgg_diff)
            learned_gamma = batch_out['gamma']
            lightened_img = batch_out['lightened_image']

            # loss
            gamma_loss = 10 * cal_smooth_term_adjacent(learned_gamma, overlap)

            color_loss = 100 * cal_color_term_overlap(warp2_tensor, vgg_diff, lightened_img, overlap)
            # loss = total_loss(vgg_diff, warp1_tensor, lightened_img, learned_gamma, overlap)
            loss = gamma_loss + color_loss
            loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            sigma_gamma_loss += gamma_loss.item()
            sigma_color_loss += color_loss.item()
            sigma_total_loss += loss.item()

            print(glob_iter)
            # print loss etc.
            if i % score_print_fre == 0 and i != 0:
                average_total_loss = sigma_total_loss / score_print_fre
                average_gamma_loss = sigma_gamma_loss / score_print_fre
                average_color_loss = sigma_color_loss / score_print_fre

                sigma_total_loss = 0.
                sigma_gamma_loss = 0.
                sigma_color_loss = 0.

                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  lr={:.8f}".format(
                        epoch + 1, args.max_epoch, i + 1, len(train_loader), average_total_loss, optimizer.state_dict()['param_groups'][0]['lr']))

                # visualization
                writer.add_image("input1", cv2.cvtColor(np.uint8((warp1_tensor[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)), cv2.COLOR_BGR2RGB), glob_iter, dataformats="HWC")
                writer.add_image("input2", cv2.cvtColor(np.uint8((warp2_tensor[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)), cv2.COLOR_BGR2RGB), glob_iter, dataformats="HWC")
                writer.add_image("lightened_image", cv2.cvtColor(np.uint8((lightened_img[0] * 255).cpu().detach().numpy().transpose(1, 2, 0)), cv2.COLOR_BGR2RGB), glob_iter, dataformats="HWC")
                writer.add_image("gamma_s", torch.unsqueeze(learned_gamma[0, 0] / learned_gamma[0, 0].max(), 0), glob_iter)
                writer.add_image("gamma_v", torch.unsqueeze(learned_gamma[0, 1] / learned_gamma[0, 1].max(), 0), glob_iter)
                writer.add_scalar('total loss', average_total_loss, glob_iter)
                writer.add_scalar('average_gamma_loss', average_gamma_loss, glob_iter)
                writer.add_scalar('average_color_loss', average_color_loss, glob_iter)

            glob_iter += 1

        scheduler.step()
        # save model
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.max_epoch:
            filename = 'epoch' + str(epoch + 1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                     "glob_iter": glob_iter}
            torch.save(state, model_save_path)

    print("##################end training#######################")


if __name__ == "__main__":
    print('<==================== setting arguments ===================>\n')

    # nl: create the argument parser
    parser = argparse.ArgumentParser()
    # nl: add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='/home/ubuntu/data3/Datasets/zyz/dataset/training_ill/')

    # nl: parse the arguments
    args = parser.parse_args()
    print(args)

    print('<==================== jump into training function ===================>\n')
    # nl: rain
    train(args)
