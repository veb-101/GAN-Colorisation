import gc
import torch

from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
from utils import init_model
from model import Discriminator, Generator_Res_Unet, Generator_Unet
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(
            self.loss_network(high_resolution), self.loss_network(fake_high_resolution)
        )
        return perception_loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode="vanilla", real_label=1.0, fake_label=0.0):
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
        perceptual_loss_factor=1.0,
        adversarial_loss_factor=1.0,
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
        self.PERcriterion = PerceptualLoss().to(self.device)
        self.GANcriterion = GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion = nn.L1Loss().to(self.device)
        self.perceptual_loss_factor = perceptual_loss_factor
        self.adversarial_loss_factor = adversarial_loss_factor
        self.lambda_L1 = lambda_L1

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

        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        real_image = torch.cat([self.L, self.ab], dim=1)

        # Train Discriminator
        self.opt_D.zero_grad()

        with amp.autocast():
            score_real = self.net_D(real_image)
            score_fake = self.net_D(fake_image.detach())
            # RaGan loss
            discriminator_rf = score_real - score_fake.mean(axis=0, keepdim=True)
            discriminator_fr = score_fake - score_real.mean(axis=0, keepdim=True)
            adversarial_loss_rf = self.GANcriterion(discriminator_rf, True)
            adversarial_loss_fr = self.GANcriterion(discriminator_fr, False)
            self.loss_D = (adversarial_loss_fr + adversarial_loss_rf) / 2

        self.scaler_D.scale(self.loss_D).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        torch.cuda.empty_cache()
        gc.collect()

        # Train Generator
        self.opt_G.zero_grad()

        fake_image = torch.cat([self.L, self.fake_color], dim=1)

        with amp.autocast():

            # Perceptual loss
            self.loss_G_per = (
                self.PERcriterion(real_image, fake_image) * self.perceptual_loss_factor
            )

            # RaGan loss
            score_real = self.net_D(real_image).detach()
            score_fake = self.net_D(fake_image)
            discriminator_rf = score_real - score_fake.mean(axis=0, keepdim=True)
            discriminator_fr = score_fake - score_real.mean(axis=0, keepdim=True)
            adversarial_loss_rf = self.GANcriterion(discriminator_rf, False)
            adversarial_loss_fr = self.GANcriterion(discriminator_fr, True)

            self.loss_G_GAN = (
                (adversarial_loss_fr + adversarial_loss_rf) / 2
            ) * self.adversarial_loss_factor
            # ----------------------

            self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
            self.loss_G = self.loss_G_per + self.loss_G_GAN + self.loss_G_L1

        self.scaler_G.scale(self.loss_G).backward()
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        torch.cuda.empty_cache()
        gc.collect()
