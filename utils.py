import os
import torch
import numpy as np
import torch.nn as nn
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
import torchvision.transforms as T

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
    return torch.from_numpy(np.clip(np.concatenate(rgb_imgs, axis=1), 0, 255))


def _psnr(ground, gen):
    score = psnr(ground, gen, data_range=ground.max() - ground.min())
    return round(score, 3)


def _ssim(ground, gen):
    score = ssim(gen, ground, data_range=ground.max() - ground.min(), multichannel=True)
    return round(score, 3)


def cal_img_metrics(generated, ground_truth):

    # generated = generated
    # ground_truth = ground_truth

    scores_PSNR = []
    scores_SSIM = []

    # gen = gen.permute(0, 2, 3, 1).numpy() * 255.0
    # ground = ground.permute(0, 2, 3, 1).numpy() * 255.0

    for i in range(len(ground_truth)):
        ground = ground_truth[i]
        gen = generated[i]

        # print(ground_truth.max() - ground_truth.min())
        psnr_ = _psnr(ground, gen)
        ssim_ = _ssim(ground, gen)

        scores_PSNR.append(psnr_)
        scores_SSIM.append(ssim_)

    return (
        round(sum(scores_PSNR) / len(scores_PSNR), 3),
        round(sum(scores_SSIM) / len(scores_SSIM), 3),
    )


def get_gray_scale(tensor, transformation):
    # print(tensor.size())
    tensor = transformation(tensor.permute(2, 0, 1))
    # print(tensor.size())
    return tensor.permute(1, 2, 0)


def visualize(model, data, save_name=None, device=None):
    gray_scale_transform = T.Grayscale(num_output_channels=3)
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
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            L, real_color = data["L"], data["ab"]
            L = L.to(device)
            real_color = real_color.to(device)
            fake_color = model(L)

    real_imgs = lab_to_rgb(L, real_color.detach())
    fake_imgs = lab_to_rgb(L, fake_color.detach())
    psnr_, ssim_ = cal_img_metrics(fake_imgs.clone().numpy(), real_imgs.clone().numpy())
    
    real_imgs = get_gray_scale(real_imgs, gray_scale_transform)
    # print(fake_imgs.shape, real_imgs.shape)
    
    result_val = torch.cat((real_imgs, fake_imgs,), 0)
    # print(result_val.shape)
    save_image(
        result_val.permute(2, 0, 1), save_name, nrow=8, normalize=False,
    )
    return psnr_, ssim_

