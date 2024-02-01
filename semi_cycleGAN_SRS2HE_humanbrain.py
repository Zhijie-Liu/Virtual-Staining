from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
# from U_net.unet_model_batchnorm import UNet
# from model import Generator, Discriminator, Generator512, Discriminator512
from model import Generator, Discriminator
from torchvision import transforms
import matplotlib.pyplot as plt
import os, itertools
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse


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
parser.add_argument('--num_epochs', type=int, default=300, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
# parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()

re_im_size = 512

transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((re_im_size, re_im_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(256, padding=0),
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
    transforms.Resize((512, 512))
])

if __name__ == "__main__":
    # base_name = 'hela'
    base_name = 'data semi'
    s_SRS_path = base_name + '/strongly/SRS'
    s_SRH_path = base_name + '/strongly/SRH'
    w_SRS_path = base_name + '/weakly/humanbrain_SRS'
    w_HE_path = base_name + '/weakly/humanbrain_HE'

    data_s_SRS = core_lzj.SemiCycleGanDataset(path=s_SRS_path, transform=transform)
    data_s_SRH = core_lzj.SemiCycleGanDataset(path=s_SRH_path, transform=transform)
    data_w_SRS = core_lzj.FolderDataset(path=w_SRS_path, transform=transform)
    data_w_HE = core_lzj.FolderDataset(path=w_HE_path, transform=transform)

    # protein = base_name + '/protein'

    # dataset = core_lzj.DCGANHelaDataset(fs_dir=fs, lipids_dir=lipids, transform=transform)
    # data_A = core_lzj.UnetDatasetSRS(path=srs, transform=transform)
    # data_B = core_lzj.UnetDatasetHE(path=he, transform=transform)
    # print('dataset length is', data_s_SRS.__len__())

    # train_data_A = Subset(data_A, list(range(0, 2500)))
    # train_data_B = Subset(data_B, list(range(0, 2500)))
    # valid_data_A = Subset(data_A, list(range(2500, 3000)))
    # valid_data_B = Subset(data_B, list(range(2500, 3000)))
    print('train_data_s_SRS length is ', data_s_SRS.__len__())
    print('train_data_s_SRH length is ', data_s_SRH.__len__())
    print('train_data_w_SRS length is ', data_w_SRS.__len__())
    print('train_data_w_HE length is ', data_w_HE.__len__())
    # print('valid_dataset_A length is', valid_data_A.__len__())
    # print('valid_dataset_B length is', valid_data_B.__len__())
    train_loader_s_SRS = DataLoader(data_s_SRS, batch_size=2, shuffle=False, num_workers=8)
    train_loader_s_SRH = DataLoader(data_s_SRH, batch_size=2, shuffle=False, num_workers=8)
    train_loader_w_SRS = DataLoader(data_w_SRS, batch_size=2, shuffle=True, num_workers=8)
    train_loader_w_HE = DataLoader(data_w_HE, batch_size=2, shuffle=True, num_workers=8)
    # valid_loader_A = DataLoader(valid_data_A, batch_size=2, shuffle=False, num_workers=8)
    # valid_loader_B = DataLoader(valid_data_B, batch_size=2, shuffle=False, num_workers=8)
    print("train_batch_s_SRS is", len(train_loader_s_SRS))
    print("train_batch_s_SRH is", len(train_loader_s_SRH))
    print("train_batch_w_SRS is", len(train_loader_w_SRS))
    print("train_batch_w_HE is", len(train_loader_w_HE))
    # print("test_batch_A is", len(valid_loader_A))
    # print("test_batch_B is", len(valid_loader_B))

    G_A = Generator(3, params.ngf, 3, params.num_resnet)
    G_B = Generator(3, params.ngf, 3, params.num_resnet)
    D_A = Discriminator(3, params.ndf, 1)
    D_B = Discriminator(3, params.ndf, 1)
    G_A.normal_weight_init(mean=0.0, std=0.02)
    G_B.normal_weight_init(mean=0.0, std=0.02)
    D_A.normal_weight_init(mean=0.0, std=0.02)
    D_B.normal_weight_init(mean=0.0, std=0.02)

    # Loss function
    MSE_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    # BCE_loss = nn.BCEWithLogitsLoss()

    # optimizers
    G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=params.lrG,
                                   betas=(params.beta1, params.beta2))
    D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))
    D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

    # Training GAN
    # save_initial_epoch = 8

    epoch_str_valid = 'Epoch {}. Train Loss G_A: {:.6f}, Train Loss cycle_A: {:.6f}, Train Loss G_B: {:.6f}, Train Loss cycle_B: {:.6f}, Train Loss G: {:.6f}, Train Loss D_A: {:.6f}, Train Loss D_B: {:.6f}, ' \
                        'Valid Loss G_A: {:.6f}, Valid Loss cycle_A: {:.6f}, Valid Loss G_B: {:.6f}, Valid Loss cycle_B: {:.6f}, Valid Loss G: {:.6f}, Valid Loss D_A: {:.6f}, Valid Loss D_B: {:.6f}'
    epoch_str_notvalid = 'Epoch {}. Train Loss G_A: {:.6f}, Train Loss cycle_A: {:.6f}, Train Loss G_B: {:.6f}, Train Loss cycle_B: {:.6f}, Train Loss G: {:.6f}, Train Loss D_A: {:.6f}, Train Loss D_B: {:.6f}'
    epoch_str_train = 'Epoch {}. Train Loss G_A: {:.6f}, Train Loss cycle_A: {:.6f}, Train Loss G_B: {:.6f}, Train Loss cycle_B: {:.6f}, Train Loss G: {:.6f}, Train Loss D_A: {:.6f}, Train Loss D_B: {:.6f}, step {}/{}'
    epoch_str_trained = 'Epoch {}. Train Loss G_A: {:.6f}, Train Loss cycle_A: {:.6f}, Train Loss G_B: {:.6f}, Train Loss cycle_B: {:.6f}, Train Loss G: {:.6f}, Train Loss D_A: {:.6f}, Train Loss D_B: {:.6f}, ' \
                        'Valid Loss G_A: {:.6f}, Valid Loss cycle_A: {:.6f}, Valid Loss G_B: {:.6f}, Valid Loss cycle_B: {:.6f}, Valid Loss G: {:.6f}, Valid Loss D_A: {:.6f}, Valid Loss D_B: {:.6f}, step {}/{}'
    time_str_pre = ' Time {:02d}:{:02d}:{:02d}'

    time_all = core_lzj.get_time()
    model_name = 'semi-humanbrain-SRS2HE'
    tgaloss, tcaloss, tgbloss, tcbloss, tgloss, tdaloss, tdbloss = [], [], [], [], [], [], []
    # vgaloss, vcaloss, vgbloss, vcbloss, vgloss, vdaloss, vdbloss = [], [], [], [], [], [], []

    save_path = './time_' + time_all + '_' + model_name + '/'
    file = save_path + model_name + '_epochs_' + str(params.num_epochs) + '.dat'
    core_lzj.check_folder_existence(save_path)
    f = open(file, 'w')

    device, init_flag = core_lzj.cuda_init(gpu)
    if torch.cuda.is_available():
        G_A.to(device)
        G_B.to(device)
        D_A.to(device)
        D_B.to(device)
        print('GPU is ok')

    print('start to train the model:')

    for epoch in range(params.num_epochs):
        G_A_loss_train = 0
        cycle_A_loss_train = 0
        G_B_loss_train = 0
        cycle_B_loss_train = 0
        G_loss_train = 0

        D_A_loss_train = 0
        D_B_loss_train = 0

        # G_A_loss_valid = 0
        # cycle_A_loss_valid = 0
        # G_B_loss_valid = 0
        # cycle_B_loss_valid = 0
        # G_loss_valid = 0
        #
        # D_A_loss_valid = 0
        # D_B_loss_valid = 0


        # train_loss1, train_loss2 = 0, 0
        #
        # valid_loss1, valid_loss2 = 0, 0
        train_step = 0
        # valid_step = 0
        prev_time = datetime.now()
        G_A.train()
        G_B.train()
        D_A.train()
        D_B.train()
        s_SRS_iter = iter(train_loader_s_SRS)
        s_SRH_iter = iter(train_loader_s_SRH)
        for w_SRS, w_HE in zip(train_loader_w_SRS, train_loader_w_HE): # fs lipid protein

            # input image data
            if torch.cuda.is_available():
                w_SRS = w_SRS.to(device, dtype=torch.float32)
                w_HE = w_HE.to(device, dtype=torch.float32)

            if epoch < 10:
                try:
                    s_SRS, s_SRH = next(s_SRS_iter), next(s_SRH_iter)
                except StopIteration:
                    s_SRS_iter = iter(train_loader_s_SRS)
                    s_SRH_iter = iter(train_loader_s_SRH)
                    s_SRS, s_SRH = next(s_SRS_iter), next(s_SRH_iter)

                if torch.cuda.is_available():
                    s_SRS = s_SRS.to(device, dtype=torch.float32)
                    s_SRH = s_SRH.to(device, dtype=torch.float32)

                real_A = s_SRS
                real_B = s_SRH
            else:

                real_A = w_SRS
                real_B = w_HE
            # Train generator G
            # A -> B
            fake_B = G_A(real_A)
            D_B_fake_decision = D_B(fake_B)
            G_A_loss = MSE_loss(D_B_fake_decision, torch.ones(D_B_fake_decision.size()).to(device, dtype=torch.float32))

            # forward cycle loss
            recon_A = G_B(fake_B)
            cycle_A_loss = L1_loss(recon_A, real_A) * params.lambdaA
            # test_B = G_A(w_SRS)
            # test_recon_A = G_B(test_B)
            # test_cycle = L1_loss(test_recon_A, w_SRS) * params.lambdaA
            # test_C = G_A(s_SRS)
            # test_recon_C = G_B(test_C)
            # test_cycle2 = L1_loss(test_recon_C, s_SRS) * params.lambdaA
            # B -> A
            fake_A = G_B(real_B)
            D_A_fake_decision = D_A(fake_A)
            G_B_loss = MSE_loss(D_A_fake_decision, torch.ones(D_A_fake_decision.size()).to(device, dtype=torch.float32))

            # backward cycle loss
            recon_B = G_A(fake_A)
            cycle_B_loss = L1_loss(recon_B, real_B) * params.lambdaB

            # Back propagation
            G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # Train discriminator D_A
            fake_A = G_B(real_B)
            D_A_real_decision = D_A(real_A)
            D_A_real_loss = MSE_loss(D_A_real_decision, torch.ones(D_A_real_decision.size()).to(device, dtype=torch.float32))
            D_A_fake_decision = D_A(fake_A)
            D_A_fake_loss = MSE_loss(D_A_fake_decision, torch.zeros(D_A_fake_decision.size()).to(device, dtype=torch.float32))

            # Back propagation
            D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
            D_A_optimizer.zero_grad()
            D_A_loss.backward()
            D_A_optimizer.step()

            # Train discriminator D_B
            fake_B = G_A(real_A)
            D_B_real_decision = D_B(real_B)
            D_B_real_loss = MSE_loss(D_B_real_decision, torch.ones(D_B_real_decision.size()).to(device, dtype=torch.float32))
            D_B_fake_decision = D_B(fake_B)
            D_B_fake_loss = MSE_loss(D_B_fake_decision, torch.zeros(D_B_fake_decision.size()).to(device, dtype=torch.float32))

            # Back propagation
            D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
            D_B_optimizer.zero_grad()
            D_B_loss.backward()
            D_B_optimizer.step()

            G_A_loss_train += G_A_loss.item()
            cycle_A_loss_train += cycle_A_loss.item()
            G_B_loss_train += G_B_loss.item()
            cycle_B_loss_train += cycle_B_loss.item()
            G_loss_train += G_loss.item()

            D_A_loss_train += D_A_loss.item()
            D_B_loss_train += D_B_loss.item()

            train_step += 1
            epoch_str = epoch_str_train.format(epoch + 1,
                                               G_A_loss_train / train_step,
                                               cycle_A_loss_train / train_step,
                                               G_B_loss_train / train_step,
                                               cycle_B_loss_train / train_step,
                                               G_loss_train / train_step,
                                               D_A_loss_train / train_step,
                                               D_B_loss_train / train_step,
                                               train_step, len(train_loader_w_SRS))
            print(epoch_str, end='\r')

        tgaloss.append(G_A_loss_train / len(train_loader_w_SRS))
        tcaloss.append(cycle_A_loss_train / len(train_loader_w_SRS))
        tgbloss.append(G_B_loss_train / len(train_loader_w_SRS))
        tcbloss.append(cycle_B_loss_train / len(train_loader_w_SRS))
        tgloss.append(G_loss_train / len(train_loader_w_SRS))
        tdaloss.append(D_A_loss_train / len(train_loader_w_SRS))
        tdbloss.append(D_B_loss_train / len(train_loader_w_SRS))

        # if (epoch + 1) == int(save_initial_epoch):
        if (epoch + 1) % 3 == 0:
            # save_initial_epoch *= 1.2
            name = model_name + '_epochs_' + str(epoch + 1) + '.pkl'
            path = save_path + name
            torch.save({
                'epoch': epoch + 1,
                'G_A': G_A.state_dict(),
                'G_B': G_B.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'D_A_optimizer': D_A_optimizer.state_dict(),
                'D_B_optimizer': D_B_optimizer.state_dict()
            }, path)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = time_str_pre.format(h, m, s)

        # if valid_loader_A is not None:
        #     G_A.eval()
        #     G_B.eval()
        #     D_A.eval()
        #     D_B.eval()
        #     with torch.no_grad():
        #         for real_A, real_B in zip(valid_loader_A, valid_loader_B):  # fs lipid protein
        #
        #             # input image data
        #             if torch.cuda.is_available():
        #                 real_A = real_A.to(device, dtype=torch.float32)
        #                 real_B = real_B.to(device, dtype=torch.float32)
        #
        #             fake_B = G_A(real_A)
        #             D_B_fake_decision = D_B(fake_B)
        #             G_A_loss = MSE_loss(D_B_fake_decision, torch.ones(D_B_fake_decision.size()).to(device, dtype=torch.float32))
        #
        #             # forward cycle loss
        #             recon_A = G_B(fake_B)
        #             cycle_A_loss = L1_loss(recon_A, real_A) * params.lambdaA
        #
        #             # B -> A
        #             fake_A = G_B(real_B)
        #             D_A_fake_decision = D_A(fake_A)
        #             G_B_loss = MSE_loss(D_A_fake_decision, torch.ones(D_A_fake_decision.size()).to(device, dtype=torch.float32))
        #
        #             # backward cycle loss
        #             recon_B = G_A(fake_A)
        #             cycle_B_loss = L1_loss(recon_B, real_B) * params.lambdaB
        #
        #             # Back propagation
        #             G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
        #
        #             # Train discriminator D_A
        #             D_A_real_decision = D_A(real_A)
        #             D_A_real_loss = MSE_loss(D_A_real_decision, torch.ones(D_A_real_decision.size()).to(device, dtype=torch.float32))
        #             D_A_fake_decision = D_A(fake_A)
        #             D_A_fake_loss = MSE_loss(D_A_fake_decision, torch.zeros(D_A_fake_decision.size()).to(device, dtype=torch.float32))
        #
        #             # Back propagation
        #             D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        #
        #             # Train discriminator D_B
        #             D_B_real_decision = D_B(real_B)
        #             D_B_real_loss = MSE_loss(D_B_real_decision, torch.ones(D_B_real_decision.size()).to(device, dtype=torch.float32))
        #             D_B_fake_decision = D_B(fake_B)
        #             D_B_fake_loss = MSE_loss(D_B_fake_decision, torch.zeros(D_B_fake_decision.size()).to(device, dtype=torch.float32))
        #
        #             # Back propagation
        #             D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        #
        #             G_A_loss_valid += G_A_loss.item()
        #             cycle_A_loss_valid += cycle_A_loss.item()
        #             G_B_loss_valid += G_B_loss.item()
        #             cycle_B_loss_valid += cycle_B_loss.item()
        #             G_loss_valid += G_loss.item()
        #
        #             D_A_loss_valid += D_A_loss.item()
        #             D_B_loss_valid += D_B_loss.item()
        #
        #             valid_step += 1
        #             epoch_str = epoch_str_trained.format(epoch + 1,
        #                                                  G_A_loss_train / len(train_loader_A),
        #                                                  cycle_A_loss_train / len(train_loader_A),
        #                                                  G_B_loss_train / len(train_loader_A),
        #                                                  cycle_B_loss_train / len(train_loader_A),
        #                                                  G_loss_train / len(train_loader_A),
        #                                                  D_A_loss_train / len(train_loader_A),
        #                                                  D_B_loss_train / len(train_loader_A),
        #                                                  G_A_loss_train / valid_step,
        #                                                  cycle_A_loss_train / valid_step,
        #                                                  G_B_loss_train / valid_step,
        #                                                  cycle_B_loss_train / valid_step,
        #                                                  G_loss_train / valid_step,
        #                                                  D_A_loss_train / valid_step,
        #                                                  D_B_loss_train / valid_step,
        #                                                  valid_step, len(valid_loader_A))
        #             print(epoch_str, end='\r')
        #
        #     vgaloss.append(G_A_loss_valid / len(valid_loader_A))
        #     vcaloss.append(cycle_A_loss_valid / len(valid_loader_A))
        #     vgbloss.append(G_B_loss_valid / len(valid_loader_A))
        #     vcbloss.append(cycle_B_loss_valid / len(valid_loader_A))
        #     vgloss.append(G_loss_valid / len(valid_loader_A))
        #     vdaloss.append(D_A_loss_valid / len(valid_loader_A))
        #     vdbloss.append(D_B_loss_valid / len(valid_loader_A))
        #
        #     epoch_str = epoch_str_valid.format(epoch + 1,
        #                                        G_A_loss_train / len(train_loader_A),
        #                                        cycle_A_loss_train / len(train_loader_A),
        #                                        G_B_loss_train / len(train_loader_A),
        #                                        cycle_B_loss_train / len(train_loader_A),
        #                                        G_loss_train / len(train_loader_A),
        #                                        D_A_loss_train / len(train_loader_A),
        #                                        D_B_loss_train / len(train_loader_A),
        #                                        G_A_loss_train / len(valid_loader_A),
        #                                        cycle_A_loss_train / len(valid_loader_A),
        #                                        G_B_loss_train / len(valid_loader_A),
        #                                        cycle_B_loss_train / len(valid_loader_A),
        #                                        G_loss_train / len(valid_loader_A),
        #                                        D_A_loss_train / len(valid_loader_A),
        #                                        D_B_loss_train / len(valid_loader_A))
        # else:
            # epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
            #              (epoch, train_loss / len(train_data),
            #               train_acc / len(train_data)))
        epoch_str = epoch_str_notvalid.format(epoch + 1,
                                              G_A_loss_train / len(train_loader_w_SRS),
                                              cycle_A_loss_train / len(train_loader_w_SRS),
                                              G_B_loss_train / len(train_loader_w_SRS),
                                              cycle_B_loss_train / len(train_loader_w_SRS),
                                              G_loss_train / len(train_loader_w_SRS),
                                              D_A_loss_train / len(train_loader_w_SRS),
                                              D_B_loss_train / len(train_loader_w_SRS))

        print(epoch_str + time_str)
        print(epoch_str + time_str, file=f, flush=True)

        plottgaloss, plottcaloss, plottgbloss,plottcbloss, plottgloss, plottdaloss, plottdbloss = np.array(tgaloss), np.array(tcaloss), np.array(tgbloss), np.array(tcbloss), np.array(tgloss), np.array(tdaloss), np.array(tdbloss)
        # plotvgaloss, plotvcaloss, plotvgbloss,plotvcbloss, plotvgloss, plotvdaloss, plotvdbloss = np.array(vgaloss), np.array(vcaloss), np.array(vgbloss), np.array(vcbloss), np.array(vgloss), np.array(vdaloss), np.array(vdbloss)
        plot_all = np.vstack((plottgaloss, plottcaloss, plottgbloss, plottcbloss, plottgloss, plottdaloss, plottdbloss))
        plotdata = pd.DataFrame(data=plot_all.T,
                                columns=['train loss G_A', 'train loss cycle_A', 'train loss G_B', 'train loss cycle_B',
                                         'train loss G', 'train loss D_A', 'train loss D_B'])

        # plot_all = np.vstack((plottgaloss, plottcaloss, plottgbloss,plottcbloss, plottgloss, plottdaloss, plottdbloss, plotvgaloss, plotvcaloss, plotvgbloss,plotvcbloss, plotvgloss, plotvdaloss, plotvdbloss))
        # plotdata = pd.DataFrame(data=plot_all.T, columns=['train loss G_A', 'train loss cycle_A', 'train loss G_B', 'train loss cycle_B', 'train loss G', 'train loss D_A', 'train loss D_B', 'valid loss G_A', 'valid loss cycle_A', 'valid loss G_B', 'valid loss cycle_B', 'valid loss G', 'valid loss D_A', 'valid loss D_B'])
        plotdata.to_csv(save_path + model_name + '.csv')

    f.close()
    core_lzj.cuda_empty_cache(init_flag)




