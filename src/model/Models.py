import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetComponents import *



class UNet(nn.Module):

    def __init__(self, in_channel, num_classes: int):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.metrics = 0 

        self.conv_block_1 = UnetConv2dBlock(in_channel, 64, 64)
        self.down1 = DownSample()
        self.conv_block_2 = UnetConv2dBlock(64, 128, 128)
        self.down2 = DownSample()
        self.conv_block_3 = UnetConv2dBlock(128, 256, 256)
        self.up1 = UpSample(256, False)
        self.conv_block_4 = UnetConv2dBlock(256, 256, 128)
        self.up2 = UpSample(128, False)
        self.conv_block_5 = UnetConv2dBlock(128, 128, 64)
        self.last_conv = nn.Conv2d(64, self.num_classes, 3, padding=1)
        self.out = torch.sigmoid


    def forward(self, x):
        x1 = self.conv_block_1(x)
        x2 = self.down1(x1)
        x2 = self.conv_block_2(x2)
        x3 = self.down2(x2)
        x3 = self.conv_block_3(x3)
        x = self.up1(x3, x2)
        x = self.conv_block_4(x)
        x = self.up2(x, x1)
        x = self.conv_block_5(x)
        x = self.last_conv(x)
        out = self.out(x)
        return out


class BiggerUnet(nn.Module):

    def __init__(self, in_channel, num_classes: int):
        super(BiggerUnet, self).__init__()
        self.num_classes = num_classes
        self.metrics = 0 

        self.conv_block_1 = UnetConv2dBlock(in_channel, 64, 64)
        self.down1 = DownSample()
        self.conv_block_2 = UnetConv2dBlock(64, 128, 128)
        self.down2 = DownSample()
        self.conv_block_3 = UnetConv2dBlock(128, 256, 256)
        self.down3 = DownSample()
        self.conv_block_4 = UnetConv2dBlock(256, 512, 512)
        self.up1 = UpSample(512, False)
        self.conv_block_5 = UnetConv2dBlock(512, 512, 256)
        self.up2 = UpSample(256, False)
        self.conv_block_6 = UnetConv2dBlock(256, 256, 128)
        self.up3 = UpSample(128, False)
        self.conv_block_7 = UnetConv2dBlock(128, 128, 64)
        self.last_conv = nn.Conv2d(64, self.num_classes, 3, padding=1)
        self.out = torch.sigmoid


    def forward(self, x):
        x1 = self.conv_block_1(x)
        x2 = self.down1(x1)
        x2 = self.conv_block_2(x2)
        x3 = self.down2(x2)
        x3 = self.conv_block_3(x3)
        x4 = self.down3(x3)
        x4 = self.conv_block_4(x4)
        x = self.up1(x4, x3)
        x = self.conv_block_5(x)
        x = self.up2(x, x2)
        x = self.conv_block_6(x)
        x = self.up3(x, x1)
        x = self.conv_block_7(x)
        x = self.last_conv(x)
        out = self.out(x)
        return out



class DeeperUNet(nn.Module):

    def __init__(self, in_channel, num_classes: int):
        super(DeeperUNet, self).__init__()
        self.num_classes = num_classes
        self.metrics = 0 

        self.conv_block_1_1 = UnetConv2dBlock(in_channel, 64, 64, kernel_size=[1,3])
        self.conv_block_1_2 = UnetConv2dBlock(64, 64, 64, kernel_size=[3,5])
        self.down1 = DownSample()
        self.conv_block_2_1 = UnetConv2dBlock(64, 128, 128, kernel_size=[1,3])
        self.conv_block_2_2 = UnetConv2dBlock(128, 128, 128, kernel_size=[3,5])
        self.down2 = DownSample()
        self.conv_block_3_1 = UnetConv2dBlock(128, 256, 256, kernel_size=[1,3])
        self.conv_block_3_2 = UnetConv2dBlock(256, 256, 256, kernel_size=[3,5])
        self.up1 = UpSample(256, False)
        self.conv_block_4_1 = UnetConv2dBlock(256, 256, 128, kernel_size=[1,3])
        self.conv_block_4_2 = UnetConv2dBlock(128, 128, 128, kernel_size=[3,5])
        self.up2 = UpSample(128, False)
        self.conv_block_5 = UnetConv2dBlock(128, 128, 64)
        self.last_conv = nn.Conv2d(64, self.num_classes, 3, padding=1)
        self.out = torch.sigmoid


    def forward(self, x):
        x1 = self.conv_block_1_1(x)
        x1 = self.conv_block_1_1(x1)
        x2 = self.down1(x1)
        x2 = self.conv_block_2_1(x2)
        x2 = self.conv_block_2_2(x2)
        x3 = self.down2(x2)
        x3 = self.conv_block_3_1(x3)
        x3 = self.conv_block_3_2(x3)
        x = self.up1(x3, x2)
        x = self.conv_block_4_1(x)
        x = self.conv_block_4_2(x)
        x = self.up2(x, x1)
        x = self.conv_block_5(x)
        x = self.last_conv(x)
        out = self.out(x)
        return out



class UnetPlusPlus(nn.Module):
    
    def __init__(self, in_channel, num_classes: int, deep_supervision: bool=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.metrics = 0 
        self.deep_supervision = deep_supervision

        num_filter = [32, 64, 128, 256, 512]

        self.down = DownSample()
        self.up = UpSample()

        self.conv_block_0_0 = UnetConv2dBlock(in_channel, num_filter[0], num_filter[0])
        self.conv_block_1_0 = UnetConv2dBlock(num_filter[0], num_filter[1], num_filter[1])
        self.conv_block_2_0 = UnetConv2dBlock(num_filter[1], num_filter[2], num_filter[2])
        self.conv_block_3_0 = UnetConv2dBlock(num_filter[2], num_filter[3], num_filter[3])
        
        self.conv_block_0_1 = UnetConv2dBlock(num_filter[0]+num_filter[1], num_filter[0], num_filter[0])
        self.conv_block_1_1 = UnetConv2dBlock(num_filter[1]+num_filter[2], num_filter[1], num_filter[1])
        self.conv_block_2_1 = UnetConv2dBlock(num_filter[2]+num_filter[3], num_filter[2], num_filter[2])

        self.conv_block_0_2 = UnetConv2dBlock(2*num_filter[0]+num_filter[1], num_filter[0], num_filter[0])
        self.conv_block_1_2 = UnetConv2dBlock(2*num_filter[1]+num_filter[2], num_filter[1], num_filter[1])

        self.conv_block_0_3 = UnetConv2dBlock(3*num_filter[0]+num_filter[1], num_filter[0], num_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv_block_0_0(x)

        x1_0 = self.conv_block_1_0(self.down(x0_0))
        x0_1 = self.conv_block_0_1(self.up(x1_0, x0_0))
        
        x2_0 = self.conv_block_2_0(self.down(x1_0))
        x1_1 = self.conv_block_1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv_block_0_2(self.up(x1_1, torch.cat([x0_0, x0_1], 1)))

        x3_0 = self.conv_block_3_0(self.down(x2_0))
        x2_1 = self.conv_block_2_1(self.up(x3_0, x2_0))
        x1_2 = self.conv_block_1_2(self.up(x2_1, torch.cat([x1_0, x1_1],1)))
        x0_3 = self.conv_block_0_3(self.up(x1_2, torch.cat([x0_0, x0_1, x0_2], 1)))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output1, output2, output3]

        else:
            out = self.final(x0_3)
            return out

class BiggerUnetPlusPlus(nn.Module):
    
    def __init__(self, in_channel, num_classes: int, deep_supervision: bool=False):
        super(BiggerUnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.metrics = 0 
        self.deep_supervision = deep_supervision

        num_filter = [32, 64, 128, 256, 512]

        self.down = DownSample()
        self.up = UpSample()

        self.conv_block_0_0 = UnetConv2dBlock(in_channel, num_filter[0], num_filter[0])
        self.conv_block_1_0 = UnetConv2dBlock(num_filter[0], num_filter[1], num_filter[1])
        self.conv_block_2_0 = UnetConv2dBlock(num_filter[1], num_filter[2], num_filter[2])
        self.conv_block_3_0 = UnetConv2dBlock(num_filter[2], num_filter[3], num_filter[3])
        self.conv_block_4_0 = UnetConv2dBlock(num_filter[3], num_filter[4], num_filter[4])
        
        self.conv_block_0_1 = UnetConv2dBlock(num_filter[0]+num_filter[1], num_filter[0], num_filter[0])
        self.conv_block_1_1 = UnetConv2dBlock(num_filter[1]+num_filter[2], num_filter[1], num_filter[1])
        self.conv_block_2_1 = UnetConv2dBlock(num_filter[2]+num_filter[3], num_filter[2], num_filter[2])
        self.conv_block_3_1 = UnetConv2dBlock(num_filter[3]+num_filter[4], num_filter[3], num_filter[3])

        self.conv_block_0_2 = UnetConv2dBlock(2*num_filter[0]+num_filter[1], num_filter[0], num_filter[0])
        self.conv_block_1_2 = UnetConv2dBlock(2*num_filter[1]+num_filter[2], num_filter[1], num_filter[1])
        self.conv_block_2_2 = UnetConv2dBlock(2*num_filter[2]+num_filter[3], num_filter[2], num_filter[2])

        self.conv_block_0_3 = UnetConv2dBlock(3*num_filter[0]+num_filter[1], num_filter[0], num_filter[0])
        self.conv_block_1_3 = UnetConv2dBlock(3*num_filter[1]+num_filter[2], num_filter[1], num_filter[1])

        self.conv_block_0_4 = UnetConv2dBlock(4*num_filter[0]+num_filter[1], num_filter[0], num_filter[0])


        if self.deep_supervision:
            self.final1 = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv_block_0_0(x)

        x1_0 = self.conv_block_1_0(self.down(x0_0))
        x0_1 = self.conv_block_0_1(self.up(x1_0, x0_0))
        
        x2_0 = self.conv_block_2_0(self.down(x1_0))
        x1_1 = self.conv_block_1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv_block_0_2(self.up(x1_1, torch.cat([x0_0, x0_1], 1)))

        x3_0 = self.conv_block_3_0(self.down(x2_0))
        x2_1 = self.conv_block_2_1(self.up(x3_0, x2_0))
        x1_2 = self.conv_block_1_2(self.up(x2_1, torch.cat([x1_0, x1_1],1)))
        x0_3 = self.conv_block_0_3(self.up(x1_2, torch.cat([x0_0, x0_1, x0_2], 1)))

        x4_0 = self.conv_block_4_0(self.down(x3_0))
        x3_1 = self.conv_block_3_1(self.up(x4_0, x3_0))
        x2_2 = self.conv_block_2_2(self.up(x3_1, torch.cat([x2_0, x2_1],1)))
        x1_3 = self.conv_block_1_3(self.up(x2_2, torch.cat([x1_0, x1_1, x1_2],1)))
        x0_4 = self.conv_block_0_4(self.up(x1_3, torch.cat([x0_0, x0_1, x0_2, x0_3],1)))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            out = self.final(x0_4)
            return out