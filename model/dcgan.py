import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Reference:
        https://arxiv.org/abs/1511.06434
    """
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.channels_img = channels_img
        self.features_d = features_d

        # Convolution output dimension formula
        # - output = \frac{n + 2p - f}{s} + 1
        self.net = nn.Sequential(
                # (batch_size x channels_img x 64 x 64)
                nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # (batch_size x features_d x 32 x 32)
                nn.Conv2d(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(features_d*2),
                nn.LeakyReLU(0.2),
                # (batch_size x features_d*2 x 16 x 16)
                nn.Conv2d(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(features_d*4),
                nn.LeakyReLU(0.2),
                # (batch_size x features_d*4 x 8 x 8)
                nn.Conv2d(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(features_d*8),
                nn.LeakyReLU(0.2),
                # (batch_szie x features_d*8 x 4 x 4)
                nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
                # (batch_size x 1 x 1 x 1)
                nn.Sigmoid()
                )

    def forward(self, x):
        assert tuple(x.shape[1:]) == (self.channels_img, 64, 64)
        return self.net(x)


class Generator(nn.Module):
    """
    Reference:
        https://arxiv.org/abs/1511.06434
    """
    def __init__(self, channels_noise, channels_img, features_g):
        super().__init__()
        self.channels_noise = channels_noise
        self.channels_img = channels_img
        self.features_g = features_g

        # TransposedConvolution output dimension formula
        # - output = s(n-1) + f - 2p
        self.net = nn.Sequential(
                # (batch_size x channels_noise x 1 x 1 )
                nn.ConvTranspose2d(channels_noise, features_g*16, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(features_g*16),
                nn.ReLU(),
                # (batch_size x features_g*16 x 4 x 4 )
                nn.ConvTranspose2d(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(features_g*8),
                nn.ReLU(),
                # (batch_size x features_g*8 x 8 x 8 )
                nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(features_g*4),
                nn.ReLU(),
                # (batch_size x features_g*4 x 16 x 16 )
                nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(features_g*2),
                nn.ReLU(),
                # (batch_size x features_g*2 x 32 x 32 )
                nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
                # (batch_size x channels_img x 64 x 64 )
                nn.Tanh()
                )

    def forward(self, x):
        assert tuple(x.shape[1:]) == (self.channels_noise, 1, 1)
        return self.net(x)
