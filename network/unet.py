import torch
import torch.nn as nn


class conv_unit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_unit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 卷积得到得feature map尺寸与原图尺寸一致: out_size = (in_size - kernal_size + 2*padding)/stride + 1
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5a = conv_block(filters[3], filters[4])
        self.Conv5b = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Softmax(dim=1)  # 在out_ch维度的sum = 1，activate的输入为：[bs, out_ch, H, W]

    def forward(self, x):  # [bs, in_ch, H, W]

        e1 = self.Conv1(x)  # [bs, n1, H, W]

        e2 = self.Maxpool1(e1)  # [bs, n1, H/2, W/2]
        e2 = self.Conv2(e2)  # [bs, n1*2, H/2, W/2]

        e3 = self.Maxpool2(e2)  # [bs, n1*2, H/4, W/4]
        e3 = self.Conv3(e3)  # [bs, n1*4, H/4, W/4]

        e4 = self.Maxpool3(e3)  # [bs, n1*4, H/8, W/8]
        e4 = self.Conv4(e4)  # [bs, n1*8, H/8, W/8]

        e5 = self.Maxpool4(e4)  # [bs, n1*8, H/16, W/16]
        e = self.Conv5a(e5)     # [bs, n1*16, H/16, W/16]
        e5 = self.Conv5b(e)

        d5 = self.Up5(e5)  # [bs, n1*8, H/8, W/8]
        d5 = torch.cat((e4, d5), dim=1)  # [bs, n1*16, H/8, W/8]
        d5 = self.Up_conv5(d5)  # [bs, n1*8, H/8, W/8]

        d4 = self.Up4(d5)  # [bs, n1*4, H/4, W/4]
        d4 = torch.cat((e3, d4), dim=1)  # [bs, n1*8, H/4, W/4]
        d4 = self.Up_conv4(d4)  # [bs, n1*4, H/4, W/4]

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)  # [bs, n1*2, H/2, W/2]

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # [bs, n1, H, W]

        d1 = self.Conv(d2)  # [bs, out_ch, H, W]
        out = self.active(d1)

        return e, out


if __name__ == '__main__':
    x = torch.randn((2, 1, 240, 240))
    unet = UNet(1, 4)
    encoder, seg = unet(x)