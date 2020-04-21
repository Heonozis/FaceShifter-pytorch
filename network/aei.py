import torch.nn.functional as F
from network.add import ADDResBlk
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


# Multilayer Attributes Encoder
class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.conv1 = conv(3, 32)
        self.conv2 = conv(32, 64)
        self.conv3 = conv(64, 128)
        self.conv4 = conv(128, 256)
        self.conv5 = conv(256, 512)
        self.conv6 = conv(512, 512)

        self.conv_t1 = conv_transpose(512, 512)
        self.conv_t2 = conv_transpose(1024, 256)
        self.conv_t3 = conv_transpose(512, 128)
        self.conv_t4 = conv_transpose(256, 64)
        self.conv_t5 = conv_transpose(128, 32)

        self.apply(init_weights)

    def forward(self, Xt):
        enc1 = self.conv1(Xt)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        z_att1 = self.conv6(enc5)

        z_att2 = self.conv_t1(z_att1, enc5)
        z_att3 = self.conv_t2(z_att2, enc4)
        z_att4 = self.conv_t3(z_att3, enc3)
        z_att5 = self.conv_t4(z_att4, enc2)
        z_att6 = self.conv_t5(z_att5, enc1)

        z_att7 = F.interpolate(z_att6, scale_factor=2, mode='bilinear', align_corners=True)

        return z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7


class ADDGenerator(nn.Module):
    def __init__(self, c_id=512):
        super(ADDGenerator, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=c_id, out_channels=1024, kernel_size=2, stride=1, padding=0, bias=False)
        self.add1 = ADDResBlk(512, 512, 512, c_id)
        self.add2 = ADDResBlk(512, 512, 1024, c_id)
        self.add3 = ADDResBlk(512, 512, 512, c_id)
        self.add4 = ADDResBlk(512, 256, 256, c_id)
        self.add5 = ADDResBlk(256, 128, 128, c_id)
        self.add6 = ADDResBlk(128, 64, 64, c_id)
        self.add7 = ADDResBlk(64, 3, 64, c_id)

        self.apply(init_weights)

    def forward(self, z_att, z_id):
        x = self.conv_t(z_id.reshape(z_id.shape[0], -1, 1, 1))
        x = self.add1(x, z_att[0], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add2(x, z_att[1], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add3(x, z_att[2], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add4(x, z_att[3], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add5(x, z_att[4], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add6(x, z_att[5], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        Y_st = self.add7(x, z_att[6], z_id)

        return torch.tanh(Y_st)


class AEI_Net(nn.Module):
    def __init__(self, c_id=512):
        super(AEI_Net, self).__init__()
        self.encoder = MAE()
        self.generator = ADDGenerator(c_id=c_id)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        # with torch.no_grad():
        return self.encoder(X)
