import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs):
        return self.block(inputs)


class CourtHeatmapNet(nn.Module):
    """TrackNet-style encoder/decoder used by yastrebksv/TennisCourtDetector.

    Source architecture: https://github.com/yastrebksv/TennisCourtDetector/tree/e5cd4f1ce26b15361700d3d89e068cbf0e82749e
    """

    def __init__(self, out_channels=15):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.pool1(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.pool2(outputs)
        outputs = self.conv5(outputs)
        outputs = self.conv6(outputs)
        outputs = self.conv7(outputs)
        outputs = self.pool3(outputs)
        outputs = self.conv8(outputs)
        outputs = self.conv9(outputs)
        outputs = self.conv10(outputs)
        outputs = self.ups1(outputs)
        outputs = self.conv11(outputs)
        outputs = self.conv12(outputs)
        outputs = self.conv13(outputs)
        outputs = self.ups2(outputs)
        outputs = self.conv14(outputs)
        outputs = self.conv15(outputs)
        outputs = self.ups3(outputs)
        outputs = self.conv16(outputs)
        outputs = self.conv17(outputs)
        return self.conv18(outputs)
