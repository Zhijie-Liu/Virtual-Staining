from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
# from U_net.unet_model_batchnorm import UNet
# from model import Generator, Discriminator, Generator512, Discriminator512
from model import Generator, Discriminator
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os, itertools
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse
import cv2



gpu = 3
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
# parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
# parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
# parser.add_argument('--input_size', type=int, default=256, help='input size')
# parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
# parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
# parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=8000, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
# parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()

re_im_size = 512
toimg_size = 512
img_size = 32
net_img_size = 512
step = int(net_img_size/img_size)

transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((re_im_size, re_im_size)),
    # transforms.RandomCrop(crop_im_size, padding=0),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.ToTensor(),
    # transforms.Normalize(mean=0.5, std=0.5)
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform_to_image = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((re_im_size_2, re_im_size_2)),
    # transforms.RandomCrop(crop_im_size, padding=0),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),

    # transforms.Normalize(mean=0.5, std=0.5)
    transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
    transforms.ToPILImage(),
    transforms.Resize((toimg_size, toimg_size))
])


def get_probability_matrix(net, cuda, nrow, ncol, img_mosaic):
    target = np.zeros(shape=(nrow * img_size, ncol * img_size, 3))
    for i in range(0, nrow - step + 1):
        for j in range(0, ncol - step + 1):
            img_temp = img_mosaic[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            img = transform(Image.fromarray(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.to(cuda)
            output = net(img)
            fake_img = transform_to_image(output[0])
            fake_num = np.array(fake_img, dtype=np.float32)
            target[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size, :] += fake_num
            # target.paste(fake_img, (j * img_size, i * img_size))

            print(i, j)

    return target


if __name__ == '__main__':
    select = 0
    if select == 0:
        img_path = '0504-3/'
        # net_path = core_lzj.get_file()
        net_path = 'virtual staning/united network/mapping.pkl'
        # net2_path = 'united/low to high net.pkl'
    # elif select == 1:
    #     img_dir = core_lzj.get_file()
    #     net_path = core_lzj.get_file()
    # elif select == 2:
    #     img_dir = core_lzj.get_file()
    #     net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    # elif select == 3:
    #     img_dir = 'check/027h-2_18_21'
    #     net_path = core_lzj.get_file()
    else:
        img_path = []
        net_path = []
        net2_path = []
        core_lzj.exit_program()

    img_list = core_lzj.each_img_specify(img_path, '.tif')
    device, init_flag = core_lzj.cuda_init(gpu)
    # img_basename = img_dir.split('/')[-1]
    G_A = Generator(3, params.ngf, 3, params.num_resnet)

    if torch.cuda.is_available():
        G_A.to(device)

    # img = get_one(h_location=0, w_location=0, path='probability/194w_99_108', basename='194w_99_108')
    G_A.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['G_A'])
    G_A.eval()

    img_sub_path = os.path.join(img_path, 'mapping32')
    core_lzj.check_folder_existence(img_sub_path)
    for img_dir in img_list:
        print(img_dir)
        img_name = os.path.basename(img_dir).split('.')[0]
        img_raw = cv2.imread(img_dir)
        img_nrow, img_ncol = int(img_raw.shape[0] / img_size), int(img_raw.shape[1] / img_size)
        # img_nrow, img_ncol = int(img_dir.split('_')[-2]), int(img_dir.split('_')[-1])


        target = get_probability_matrix(net=G_A, cuda=device, nrow=img_nrow, ncol=img_ncol, img_mosaic=img_raw)
        dim_row = np.repeat(np.concatenate((np.arange(1, step + 1, 1), np.ones(img_nrow - step * 2) * step, np.arange(step, 0, -1))), img_size)
        dim_col = np.repeat(np.concatenate((np.arange(1, step + 1, 1), np.ones(img_ncol - step * 2) * step, np.arange(step, 0, -1))), img_size)
        adjust_matrix = dim_row.reshape(-1, 1) * dim_col.reshape(1, -1)
        adjust_matrix_3 = np.expand_dims(adjust_matrix, 2).repeat(3, axis=2)
        target_uint8 = (target/adjust_matrix_3).astype(np.uint8)
        img = Image.fromarray(target_uint8)

        img_save_path = os.path.join(img_sub_path, img_name + '_SRS2HE.png')
        img.save(img_save_path)
        # target.save('mosaic/test3.tif')
