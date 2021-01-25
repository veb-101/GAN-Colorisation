import torch
import torch.nn as nn
from torchsummary import summary


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
        nn.Conv2d(
            in_channels=in_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            stride=stride_step,
            padding=padding_size,
            bias=bias,
        )
    )
    if batch_norm:
        layer.append(nn.BatchNorm2d(out_filters))

    layer.append(nn.LeakyReLU(0.2))
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
        nn.ConvTranspose2d(
            in_channels=in_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            stride=stride_step,
            padding=padding_size,
            bias=bias,
        )
    )

    if batch_norm:
        layer.append(nn.BatchNorm2d(out_filters))

    layer.append(nn.ReLU())

    return nn.Sequential(*layer)


class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, image_size=(256,)):
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
        self.conv6 = _downscale_layers(512, 512)

        self.intermediate = _downscale_layers(512, 512)

        self.deconv6 = _upscale_layers(512 * 1, 512)
        self.deconv5 = _upscale_layers(512 * 2, 512)
        self.deconv4 = _upscale_layers(512 * 2, 512)
        self.deconv3 = _upscale_layers(512 * 2, 256)
        self.deconv2 = _upscale_layers(256 * 2, 128)
        self.deconv1 = _upscale_layers(128 * 2, 64)
        self.deconv0 = _upscale_layers(64 * 2, 64)

        self.conv_penultimate = nn.Conv2d(
            in_channels=64 * 2,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv_final = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=1,
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
        enc_6 = self.conv6(enc_5)

        output = self.intermediate(enc_6)

        dec_6 = self.deconv6(output)
        dec_6 = torch.cat((dec_6, enc_6), dim=1)

        dec_5 = self.deconv5(dec_6)
        dec_5 = torch.cat((dec_5, enc_5), dim=1)

        dec_4 = self.deconv4(dec_5)
        dec_4 = torch.cat((dec_4, enc_4), dim=1)

        dec_3 = self.deconv3(dec_4)
        dec_3 = torch.cat((dec_3, enc_3), dim=1)

        dec_2 = self.deconv2(dec_3)
        dec_2 = torch.cat((dec_2, enc_2), dim=1)

        dec_1 = self.deconv1(dec_2)
        dec_1 = torch.cat((dec_1, enc_1), dim=1)

        dec_0 = self.deconv0(dec_1)
        dec_0 = torch.cat((dec_0, enc_0), dim=1)

        output = self.conv_penultimate(dec_0)
        output = self.batch_norm(output)
        output = self.leaky_relu(output)

        output = self.conv_final(output)
        output = self.tanh(output)
        return output


# class Discriminator(nn.Module):
#     def __init__(self, input_channels, image_size=(256,)):
#         self.img_size = image_size[0]

#         self.conv0 = _downscale_layers(input_channels, 64, batch_norm=False)
#         self.conv1 = _downscale_layers(64, 64)
#         self.conv2 = _downscale_layers(64, 128)
#         self.conv3 = _downscale_layers(128, 256)
#         self.conv4 = _downscale_layers(256, 512)
#         self.conv5 = _downscale_layers(512, 512)
#         self.conv6 = _downscale_layers(512, 512)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1 + 3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x_grey, x_rgb):
        out = torch.cat([x_grey, x_rgb], dim=1)
        return self.model(out)


if __name__ == "__main__":
    im_size = 256
    # summary(Generator(image_size=(im_size,)), (1, im_size, im_size))
    summary(Discriminator(), (4, 256, 256))
    # print(322.35 * 32)
    # print(122 * 32)
