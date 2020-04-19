import torch.nn.functional as F
from torch import nn
import torch


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv(c_in, c_out, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False),
        norm(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )


class conv_transpose(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d):
        super(conv_transpose, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(c_out)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.conv_t(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class HEAR(nn.Module):
    def __init__(self):
        super(HEAR, self).__init__()
        self.conv1 = conv(6, 64)
        self.conv2 = conv(64, 128)
        self.conv3 = conv(128, 256)
        self.conv4 = conv(256, 512)
        self.conv5 = conv(512, 512)

        self.conv_t1 = conv_transpose(512, 512)
        self.conv_t2 = conv_transpose(512, 256)
        self.conv_t3 = conv_transpose(256, 128)
        self.conv_t4 = conv_transpose(128, 64)

        self.conv6 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.apply(init_weights)

    def forward(self, dY_Yst):
        enc1 = self.conv1(dY_Yst)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)

        dec1 = self.conv5(enc4)
        dec2 = self.conv_t1(dec1, enc4)
        dec3 = self.conv_t2(dec2, enc3)
        dec4 = self.conv_t3(dec3, enc2)
        dec5 = self.conv_t4(dec4, enc1)

        y = F.interpolate(dec5, scale_factor=2, mode='bilinear', align_corners=True)
        y = self.conv6(y)

        return torch.tanh(y)
