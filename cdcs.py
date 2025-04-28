import cv2
import numpy as np
import os
import pickle


def cal_boundary(mask):
    mask_1 = np.where(mask[:, :, 0] > 0.5, 1., 0.)
    boundary = np.pad(mask_1[:, 0:-1] - mask_1[:, 1:], ((0, 0), (0, 1)), 'constant', constant_values=0)
    return boundary


def cal_color_diff(stitch_mask, img, k=5):
    boundary = cal_boundary(stitch_mask)
    color_diff = 0
    # 总和大于0，说明stitch_mask在左侧，则每一行最右侧为1的点为分界点，将该点左侧的4列设为1，右侧的5列设为-1
    if np.sum(boundary) > 0:
        coordinate = np.where(boundary == 1)
        rows = np.unique(coordinate[0])
        for row in rows:
            col = int(coordinate[1][np.where(coordinate[0] == row)[0][-1]])
            color = np.mean(img[row, col - (k - 1): col + 1], axis=0) - np.mean(img[row, col + 1: col + (k + 1)], axis=0)
            color_diff += np.sqrt(np.mean(color ** 2))
    # 总和小于0，说明stitch_mask在右侧，则每一行最左侧为-1的点为分界点，将该点左侧的4列设为-1，右侧的5列设为1
    else:
        coordinate = np.where(boundary == -1)
        rows = np.unique(coordinate[0])
        for row in rows:
            col = int(coordinate[1][np.where(coordinate[0] == row)[0][0]])
            color = np.mean(img[row, col - (k - 1): col + 1], axis=0) - np.mean(img[row, col + 1: col + (k + 1)], axis=0)
            color_diff += np.sqrt(np.mean(color ** 2))

    return color_diff / len(rows)


# ori_res_folder = 'F:/work/stitch_work/dataset/res/ori_res/'
ori_res_folder = 'F:/work/stitch_work/dataset/ablation/lightened_image_nocolor&gamma/'
ori_res_files = os.listdir(ori_res_folder)
# global_res_folder = 'F:/work/stitch_work/dataset/res/global_gamma_res/'
global_res_folder = 'F:/work/stitch_work/dataset/ablation/lightened_image_nocolor/'
global_res_files = os.listdir(global_res_folder)
# gcc_res_folder = 'F:/work/stitch_work/dataset/res/GCC_res/'
gcc_res_folder = 'F:/work/stitch_work/dataset/ablation/lightened_image_nogamma/'
gcc_res_files = os.listdir(global_res_folder)
# hhm_res_folder = 'F:/work/stitch_work/dataset/res/HHM_res/'
hhm_res_folder = 'F:/work/stitch_work/deep_seam/overlap_nonoverlap/overlap_by_vgg/lightened_res_overlap/'
hhm_res_files = os.listdir(hhm_res_folder)
lightened_res_folder = 'F:/work/stitch_work/dataset/res/nearest_res/'
lightened_res_files = os.listdir(lightened_res_folder)
mask_folder = 'F:/work/stitch_work/dataset/testing_nonoverlap/stitch_mask1/'
mask_files = os.listdir(mask_folder)

types = ['easy', 'moderate', 'hard', 'average']
for type in types:
    folder_lightened_img2 = './test_images/' + type + '/'
    files_lightened_img2 = os.listdir(folder_lightened_img2)

    res = np.zeros((len(files_lightened_img2) + 1, 5))
    for i in range(len(files_lightened_img2)):
        file_num = int(files_lightened_img2[i].split('.')[0]) - 1
        ori_res = cv2.imread(ori_res_folder + ori_res_files[file_num])
        global_res = cv2.imread(global_res_folder + global_res_files[file_num])
        gcc_res = cv2.imread(gcc_res_folder + gcc_res_files[file_num])
        hhm_res = cv2.imread(hhm_res_folder + hhm_res_files[file_num])
        lightened_res = cv2.imread(lightened_res_folder + lightened_res_files[file_num])
        mask = cv2.imread(mask_folder + mask_files[file_num]) / 255

        res[i, 0] = cal_color_diff(mask, ori_res)
        res[i, 1] = cal_color_diff(mask, global_res)
        res[i, 2] = cal_color_diff(mask, gcc_res)
        res[i, 3] = cal_color_diff(mask, hhm_res)
        res[i, 4] = cal_color_diff(mask, lightened_res)

    res[-1] = np.mean(res[:len(files_lightened_img2)], axis=0)

    print(type + ': {}, {}, {}, {}, {}'.format(res[-1, 0], res[-1, 1], res[-1, 2], res[-1, 3], res[-1, 4]))
