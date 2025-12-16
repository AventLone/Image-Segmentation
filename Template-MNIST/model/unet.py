from .components import *


class UNet(nn.Module):
    """
    Full assembly of the parts to form the complete network
    """
    def __init__(self, config: dict):
        super().__init__()
        channels_num = config["ChannelsNum"]
        classes_num = config["ClassesNum"]
        bilinear = config["Bilinear"]

        self.inc = DoubleConv(channels_num, 64)
        self.down1 = Downsample(64, 128)
        self.down2 = Downsample(128, 256)
        self.down3 = Downsample(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Downsample(512, 1024 // factor)
        self.up1 = Upsample(1024, 512 // factor, bilinear)
        self.up2 = Upsample(512, 256 // factor, bilinear)
        self.up3 = Upsample(256, 128 // factor, bilinear)
        self.up4 = Upsample(128, 64, bilinear)
        self.outc = OutConv(64, classes_num)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
