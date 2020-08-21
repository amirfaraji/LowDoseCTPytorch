import torch
import torch.nn as nn
import torch.nn.functional as F



class UnetConv2dBlock(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel, kernel_size=[3,3], stride_size=[1,1]):
        super().__init__()

        self.conv2dblock = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size[0], stride=stride_size[0], padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size[1], stride=stride_size[0], padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv2dblock(x)



class DownSample(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpooling = nn.MaxPool2d(2)

    def forward(self, x):
        return self.maxpooling(x)



class UpSample(nn.Module):

    def __init__(self, in_channel=None, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel , in_channel // 2, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x



    def __init__(self, in_channel=None, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel , in_channel // 2, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x