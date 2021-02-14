import torch
import torch.nn as nn
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18, resnet34
from fastai.vision.models.unet import DynamicUnet
from torch.nn.utils import spectral_norm
from torchsummary import summary
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channel: int, gain: int = 1):
        super().__init__()
        self.query = self._spectral_init(
            nn.Conv1d(in_channel, in_channel // 8, 1), gain=gain
        )
        self.key = self._spectral_init(
            nn.Conv1d(in_channel, in_channel // 8, 1), gain=gain
        )
        self.value = self._spectral_init(
            nn.Conv1d(in_channel, in_channel, 1), gain=gain
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _spectral_init(self, module: nn.Module, gain: int = 1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return spectral_norm(module)

    def forward(self, input: torch.Tensor):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        return out


def _downscale_layers(
    in_filters=None,
    out_filters=None,
    kernel_size=4,
    stride_step=2,
    padding_size=1,
    bias=False,
    batch_norm=True,
):
    layer = []
    layer.append(
        spectral_norm(
            nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel_size,
                stride=stride_step,
                padding=padding_size,
                bias=bias,
            )
        )
    )
    if batch_norm:
        layer.append(nn.BatchNorm2d(out_filters))

    layer.append(nn.LeakyReLU(0.2, True))
    # layer.append(SelfAttention(out_filters, 1))
    return nn.Sequential(*layer)


def _upscale_layers(
    in_filters=None,
    out_filters=None,
    kernel_size=4,
    stride_step=2,
    padding_size=1,
    bias=False,
    batch_norm=True,
):
    layer = []

    layer.append(
        spectral_norm(
            nn.ConvTranspose2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel_size,
                stride=stride_step,
                padding=padding_size,
                bias=bias,
            )
        )
    )

    if batch_norm:
        layer.append(nn.BatchNorm2d(out_filters))

    layer.append(nn.ReLU(inplace=True))
    # layer.append(SelfAttention(out_filters, 1))

    return nn.Sequential(*layer)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, image_size=(128,)):
        super().__init__()
        self.img_size = image_size[0]
        self.conv0 = _downscale_layers(input_channels, 64, batch_norm=False)
        self.conv1 = _downscale_layers(64, 128)
        self.conv2 = _downscale_layers(128, 256)
        self.conv3 = _downscale_layers(256, 512, stride_step=1)
        # self.conv4 = _downscale_layers(512, 512)
        # self.conv5 = _downscale_layers(512, 512)
        self.last_layer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        out = self.last_layer(out)
        return out


class Generator_Unet(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, image_size=(256,)):
        super().__init__()

        self.image_size = image_size[0]
        self.conv0 = _downscale_layers(
            input_channels, 64, kernel_size=3, stride_step=1, batch_norm=False
        )
        self.conv1 = _downscale_layers(64, 64)
        self.conv2 = _downscale_layers(64, 128)
        self.conv3 = _downscale_layers(128, 256)
        self.conv4 = _downscale_layers(256, 512)
        self.conv5 = _downscale_layers(512, 512)
        # self.conv6 = _downscale_layers(512, 512)

        self.intermediate = _downscale_layers(512, 512)

        # self.deconv6 = _upscale_layers(512 * 1, 512)
        self.deconv5 = _upscale_layers(512 * 1, 512)
        self.deconv4 = _upscale_layers(512 * 2, 512)
        self.deconv3 = _upscale_layers(512 * 2, 256)
        self.deconv2 = _upscale_layers(256 * 2, 128)
        self.deconv1 = _upscale_layers(128 * 2, 64)
        self.deconv0 = _upscale_layers(64 * 2, 64)

        self.conv_penultimate = spectral_norm(
            nn.Conv2d(
                in_channels=64 * 2,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )

        self.conv_final = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):

        enc_0 = self.conv0(x)
        enc_1 = self.conv1(enc_0)
        enc_2 = self.conv2(enc_1)
        enc_3 = self.conv3(enc_2)
        enc_4 = self.conv4(enc_3)
        enc_5 = self.conv5(enc_4)

        output = self.intermediate(enc_5)

        output = self.deconv5(output)
        output = torch.cat((output, enc_5), dim=1)

        output = self.deconv4(output)
        output = torch.cat((output, enc_4), dim=1)

        output = self.deconv3(output)
        output = torch.cat((output, enc_3), dim=1)

        output = self.deconv2(output)
        output = torch.cat((output, enc_2), dim=1)

        output = self.deconv1(output)
        output = torch.cat((output, enc_1), dim=1)

        output = self.deconv0(output)
        output = torch.cat((output, enc_0), dim=1)

        output = self.conv_penultimate(output)
        output = self.batch_norm(output)
        output = self.leaky_relu(output)

        output = self.conv_final(output)
        output = self.tanh(output)
        return output


class Generator_Res_Unet(object):
    def __init__(self, n_input=1, n_output=2, size=256):
        self.input_channels = n_input
        self.output_channels = n_output
        self.image_size = size

    def get_model(self, pretrained=True):
        body = create_body(
            resnet34, pretrained=pretrained, n_in=self.input_channels, cut=-2
        )
        net_G = DynamicUnet(
            body, self.output_channels, (self.image_size, self.image_size)
        )
        return net_G


if __name__ == "__main__":
    im_size = 256
    # summary(Generator_Unet(image_size=(im_size,)).to("cuda"), (1, im_size, im_size))

    summary(Generator_Res_Unet().get_model(), (1, im_size, im_size))
    # summary(Discriminator(), (3, im_size, im_size))
