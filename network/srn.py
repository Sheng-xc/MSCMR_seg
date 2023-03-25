import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class neck_block(nn.Module):
    def __init__(self, in_ch):
        super(neck_block, self).__init__()

        self.bottle = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=3, stride=3, padding=1, bias=False),
            nn.Flatten(1),
            nn.Linear(100, 64, bias=False),

            nn.Linear(64, 100, bias=False),

            nn.Unflatten(1, (1, 10, 10)),
            nn.Upsample(scale_factor=3),
            nn.Conv2d(1, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch)
        )

    def forward(self, x):
        x = self.bottle(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConSR(nn.Module):
    def __init__(self, in_ch):
        super(ConSR, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4]

        self.conv1 = conv_block(in_ch, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])

        self.bottle = neck_block(filters[2])

        self.up3 = up_conv(filters[2], filters[1])
        self.up2 = up_conv(filters[1], filters[0])
        self.up1 = up_conv(filters[0], in_ch)

        self.activate = nn.Softmax(dim=1)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)

        e = self.bottle(e3)

        d3 = self.up3(e)
        d2 = self.up2(d3)
        d1 = self.up1(d2)
        d1 = self.activate(d1)

        return d1


if __name__ == '__main__':
    x = torch.randn((2, 4, 240, 240))
    srnet = ConSR(4)
    print(srnet(x))