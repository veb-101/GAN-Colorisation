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
        config=None,
        device=None,
    ):
        super().__init__()

        self.perceptual_loss_factor = config["perceptual_loss_factor"]
        self.adversarial_loss_factor = config["adversarial_loss_factor"]
        self.lambda_L1 = config["lambda_L1"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        self.lr_G = config["lr_G"]
        self.lr_D = config["lr_D"]

        self.device = device
        if net_G:
            self.net_G = net_G.to(self.device)
        else:
            self.net_G = init_model(Generator_Unet().get_model(), self.device)
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
                self.net_G.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2)
            )
        if opt_D:
            self.opt_D = opt_D
        else:
            self.opt_D = optim.Adam(
                self.net_D.parameters(), lr=self.lr_D, betas=(self.beta1, self.beta2)
            )
        self.PERcriterion = PerceptualLoss().to(self.device)
        self.GANcriterion = GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion = nn.L1Loss().to(self.device)
        self.perceptual_loss_factor = self.perceptual_loss_factor
        self.adversarial_loss_factor = self.adversarial_loss_factor
        self.lambda_L1 = self.lambda_L1

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
            score_fake = self.net_D(fake_image).detach()

            # RaGan loss Real
            discriminator_rf = score_real - score_fake.mean(axis=0, keepdim=True)
            adversarial_loss_rf = self.GANcriterion(discriminator_rf, True) * 0.5

            # Fake
            score_fake = self.net_D(fake_image.detach())
            discriminator_fr = score_fake - score_real.mean(axis=0, keepdim=True)
            adversarial_loss_fr = self.GANcriterion(discriminator_fr, False) * 0.5

        self.scaler_D.scale(adversarial_loss_rf).backward(retain_graph=True)
        self.scaler_D.scale(adversarial_loss_fr).backward()
        self.scaler_D.step(self.opt_D)
        dis_scale_val = self.scaler_D.get_scale()
        self.scaler_D.update()
        skip_dis_lr_sched = dis_scale_val != self.scaler_D.get_scale()

        self.loss_D = (
            adversarial_loss_fr.detach().item() + adversarial_loss_rf.detach().item()
        )

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

            # L1 loss
            self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
            # total loss
            self.loss_G = self.loss_G_per + self.loss_G_GAN + self.loss_G_L1

        self.scaler_G.scale(self.loss_G).backward()
        self.scaler_G.step(self.opt_G)

        scale_gen = self.scaler_G.get_scale()
        self.scaler_G.update()
        skip_gen_lr_sched = scale_gen != self.scaler_G.get_scale()

        torch.cuda.empty_cache()
        gc.collect()
        return (skip_gen_lr_sched, skip_dis_lr_sched)
