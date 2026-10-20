import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,  
            stride=stride,  
            padding=1,       
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None


        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        identity = x  


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

  
        out = out + identity
        out = self.relu(out)

        return out


class ResNetDepth(nn.Module):
    def __init__(self):
        super().__init__()

 
        self.in_channels = 64

  
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)



        self.up4 = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1)
        self.bn_up4 = nn.BatchNorm2d(256)

        self.up3 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(128)

        self.up2 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(64)
        self.up1 = nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1)  
        self.bn_up1 = nn.BatchNorm2d(32)

        self.up0 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  
        self.bn_up0 = nn.BatchNorm2d(16)

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )



    def _make_layer(self, out_channels, blocks, stride):

        layers = []

        layers.append(
            BasicBlock(self.in_channels, out_channels, stride)
        )

        self.in_channels = out_channels

  
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.in_channels, out_channels)
            )

        return nn.Sequential(*layers)


    def forward(self, x):
        input_size = x.shape[2:] 

        x = self.conv1(x)   
        x = self.bn1(x)
        x = self.relu(x)

        p1 = self.maxpool(x)  

        x_l1 = self.layer1(p1)    
        x_l2 = self.layer2(x_l1)   
        x_l3 = self.layer3(x_l2)   
        x_l4 = self.layer4(x_l3)    

        up_4 = F.interpolate(x_l4, size=x_l3.shape[2:], mode='bilinear', align_corners=True)
        merge_4 = torch.cat([up_4, x_l3], dim=1)
        out_4 = F.relu(self.bn_up4(self.up4(merge_4)))

        up_3 = F.interpolate(out_4, size=x_l2.shape[2:], mode='bilinear', align_corners=True)
        merge_3 = torch.cat([up_3, x_l2], dim=1)
        out_3 = F.relu(self.bn_up3(self.up3(merge_3)))

        up_2 = F.interpolate(out_3, size=x_l1.shape[2:], mode='bilinear', align_corners=True)
        merge_2 = torch.cat([up_2, x_l1], dim=1)
        out_2 = F.relu(self.bn_up2(self.up2(merge_2)))

        up_1 = F.interpolate(out_2, size=p1.shape[2:], mode='bilinear', align_corners=True)
        merge_1 = torch.cat([up_1, p1], dim=1)
        out_1 = F.relu(self.bn_up1(self.up1(merge_1)))

        up_0 = F.interpolate(out_1, size=input_size, mode='bilinear', align_corners=True)
        out_0 = F.relu(self.bn_up0(self.up0(up_0)))

        out = self.final_conv(out_0)

        return out