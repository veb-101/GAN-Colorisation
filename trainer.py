import gc
import torch
import torch.nn as nn
import torch.optim as optim
from utils import init_model
from model import Discriminator, Generator_Unet


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
        device=None,
        lr_G=2e-4,
        lr_D=2e-4,
        beta1=0.5,
        beta2=0.999,
        lambda_L1=100.0,
    ):
        super().__init__()

        self.device = device
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Generator_Unet(), self.device)
        else:
            self.net_G = net_G.to(self.device)

        self.net_D = init_model(Discriminator(input_channels=3), self.device)
        self.GANcriterion = GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    @staticmethod
    def set_requires_grad(model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2.0
        self.loss_D.backward()

    def optimize(self):
        torch.cuda.empty_cache()
        gc.collect()
        self.forward()
        self.net_D.train()
        MainModel.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        torch.cuda.empty_cache()
        gc.collect()

        self.net_G.train()
        MainModel.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
        torch.cuda.empty_cache()
        gc.collect()

