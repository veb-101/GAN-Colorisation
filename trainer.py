import gc
import torch

from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
from utils import init_model
from model import Discriminator, Generator_Res_Unet, Generator_Unet


class GANLoss(nn.Module):
    def __init__(self, gan_mode="vanilla", real_label=0.9, fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class MainModel(nn.Module):
    def __init__(
        self,
        net_G=None,
        net_D=None,
        opt_G=None,
        opt_D=None,
        scaler_G=None,
        scaler_D=None,
        device=None,
        lambda_L1=100.0,
    ):
        super().__init__()

        self.device = device
        if net_G:
            self.net_G = net_G.to(self.device)
        else:
            self.net_G = init_model(Generator_Res_Unet().get_model(), self.device)
        if net_D:
            self.net_D = net_D.to(self.device)
        else:
            self.net_D = init_model(Discriminator(input_channels=3), self.device)

        if scaler_G:
            self.scaler_G = scaler_G
        else:
            self.scaler_G = amp.GradScaler()

        if scaler_D:
            self.scaler_D = scaler_D
        else:
            self.scaler_D = amp.GradScaler()

        if opt_G:
            self.opt_G = opt_G
        else:
            self.opt_G = optim.Adam(
                self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2)
            )
        if opt_D:
            self.opt_D = opt_D
        else:
            self.opt_D = optim.Adam(
                self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2)
            )

        self.GANcriterion = GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion = nn.L1Loss().to(self.device)
        self.lambda_L1 = lambda_L1

    # @staticmethod
    # def set_requires_grad(model, requires_grad=True):
    #     for p in model.parameters():
    #         p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def optimize(self):
        torch.cuda.empty_cache()
        gc.collect()

        self.net_D.train()
        self.net_G.train()

        with amp.autocast():
            self.forward()  # generate fake images 2 channel

        fake_image = torch.cat(
            [self.L, self.fake_color], dim=1
        )  # combine image L channel and generated ab channels

        # Train Discriminator
        self.opt_D.zero_grad()

        with amp.autocast():
            # fake loss
            fake_preds = self.net_D(fake_image.detach())
            self.loss_D_fake = self.GANcriterion(fake_preds, False)
            # real loss
            real_image = torch.cat([self.L, self.ab], dim=1)
            real_preds = self.net_D(real_image)
            self.loss_D_real = self.GANcriterion(real_preds, True)
            # discriminator final loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2.0

        self.scaler_D.scale(self.loss_D).backward()
        self.scaler_D.step(self.opt_D)
        dis_scale_val = self.scaler_D.get_scale()
        self.scaler_D.update()
        skip_dis_lr_sched = dis_scale_val != self.scaler_D.get_scale()

        torch.cuda.empty_cache()
        gc.collect()

        # Train Generator
        self.opt_G.zero_grad()

        fake_image = torch.cat([self.L, self.fake_color], dim=1)

        with amp.autocast():
            fake_preds = self.net_D(fake_image)
            self.loss_G_GAN = self.GANcriterion(fake_preds, True)
            self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
            self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.scaler_G.scale(self.loss_G).backward()
        self.scaler_G.step(self.opt_G)

        scale_gen = self.scaler_G.get_scale()
        self.scaler_G.update()
        skip_gen_lr_sched = scale_gen != self.scaler_G.get_scale()

        torch.cuda.empty_cache()
        gc.collect()
        return (skip_gen_lr_sched, skip_dis_lr_sched)
