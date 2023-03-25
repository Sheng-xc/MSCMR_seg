import torch.nn as nn
from .unet import UNet


class ConSC(nn.Module):
    def __init__(self):
        super(ConSC, self).__init__()

        n1 = 16
        in_ch = n1 * 16

        self.reg = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_ch * 15 * 15, in_ch, bias=False),
            nn.Dropout(),
            nn.Linear(in_ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.reg(x)
        return x


if __name__ == '__main__':
    import torch
    x = torch.randn((2, 1, 240, 240))
    unet = UNet(in_ch=1, out_ch=4)
    e, seg = unet(x)
    consc = ConSC()
    pos = consc(e)
    print(pos)