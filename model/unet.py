import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义“双层卷积”基础块
class DoubleConv(nn.Module):
    """(Convolution => BatchNorm（批归一化） => ReLU) * 2"""
    # BatchNorm（批归一化） 的作用就是：在每一层卷积之后，强行把输出的数据拉回到均值为 0、方差为 1 的标准正态分布附近。
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 第一次卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 加上归一化，训练更稳
            nn.ReLU(inplace=True),
            # 第二次卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 定义“上采样”基础块
class Up(nn.Module):
    """上采样模块：(上采样 => 拼接 => 双层卷积)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 转置卷积：尺寸放大 2 倍 (stride=2)，同时通道数减半 (in_channels // 2)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 拼接后，通道数会变成 (上采样后的通道 + 左边过来的通道)
        # 在标准的 U-Net 中，这两个通道数通常是相等的，所以 DoubleConv 的输入是 out_channels * 2
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 来自下层的数据 (需要放大)
        # x2: 来自左边跳跃连接的数据 (零件)
        x1 = self.up(x1)

        # 核心步骤：拼接
        # dim=1 代表在“通道”这个维度上拼接
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


# 定义U-Net的下采样部分
class UNetEncoder(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()

        # Level 1: 初始输入 (512x512)
        self.inc = DoubleConv(n_channels, 64)

        # Level 2: 第一次下采样 (256x256)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )

        # Level 3: 第二次下采样 (128x128)
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )

        # Level 4: 第三次下采样 (64x64)
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )

        # Level 5: 第四次下采样 & 桥接层 (32x32)
        # 这是网络的最底部，特征最抽象，通道数达到 1024
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

    def forward(self, x):
        # 记录每一层的输出，用于后续上采样时的“跳跃连接”
        x1 = self.inc(x)  # [Batch, 64, 512, 512]
        x2 = self.down1(x1)  # [Batch, 128, 256, 256]
        x3 = self.down2(x2)  # [Batch, 256, 128, 128]
        x4 = self.down3(x3)  # [Batch, 512, 64, 64]

        # 最终的桥接特征
        x5 = self.down4(x4)  # [Batch, 1024, 32, 32]

        return x1, x2, x3, x4, x5


# 定义U-Net的上采样部分
class UNetDecoder(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        # 注意通道数的变化：
        self.up1 = Up(1024, 512)  # 下面1024变512，左边来512，拼接成1024，最后输出512
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 最终输出层：把 64 通道的特征图变成 1 个通道的分割图
        # 这里用 1x1 卷积，不改变尺寸，只改变深度
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        # x5 是最底部的 Bridge，x4~x1 是左边的零件
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


# 定义完整的 UNet 类
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(n_channels)
        self.decoder = UNetDecoder(n_classes)

    def forward(self, x):
        # 1. 走左半边，拿到所有层级的特征（零件）
        x1, x2, x3, x4, x5 = self.encoder(x)

        # 2. 走右半边，把零件传进去进行缝合
        logits = self.decoder(x1, x2, x3, x4, x5)

        return logits
