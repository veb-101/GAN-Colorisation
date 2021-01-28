import os
import torch
import numpy as np
import torch.nn as nn
from skimage.color import lab2rgb
from torchvision.utils import save_image


def init_weights(net, init="norm", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and "Conv" in classname:
            if init == "norm":
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    # loss_D_fake = AverageMeter()
    # loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {
        # "loss_D_fake": loss_D_fake,
        # "loss_D_real": loss_D_real,
        "loss_D": loss_D,
        "loss_G_GAN": loss_G_GAN,
        "loss_G_L1": loss_G_L1,
        "loss_G": loss_G,
    }


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}", end=" ")
    print()


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    
    # print(len(rgb_imgs))
    return torch.from_numpy(np.clip(
        np.concatenate(rgb_imgs, axis=1), 0, 255))


def visualize(model, data, save_name=None):
    try:
        model.net_G.eval()
        with torch.no_grad():
            model.setup_input(data)
            model.forward()

        model.net_G.train()

        fake_color = model.fake_color
        real_color = model.ab

        L = model.L

    except:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.eval()
        with torch.no_grad():
            L, real_color = data["L"], data["ab"]
            L.to(device)
            real_color.to(device)
            fake_color = model(L)

    
    fake_imgs = lab_to_rgb(L, fake_color.detach())
    real_imgs = lab_to_rgb(L, real_color.detach())

    # print(fake_imgs.shape, real_imgs.shape)
    result_val = torch.cat((real_imgs, fake_imgs,), 0)
    # print(result_val.shape)
    save_image(
        result_val.permute(2, 0, 1), 
        save_name, 
        nrow=8, 
        normalize=False,
    )

