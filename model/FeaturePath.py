import torch.nn as nn
import torch
import torch.nn.functional as func
import numpy as np
import torchvision



class DoubleConv(nn.Module):
    """(convolution => [BN] => LeakReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class TripleConv(nn.Module):
    """(convolution => [BN] => LeakReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)

class FeaturePath(nn.Module):
    def __init__(self, device, in_channels_img, filters=None):
        super().__init__()
        if filters is None:
            # 128 - 64 - 64 - 32 - 16 - 8
            filters = [(0, 2, 32), (1, 2, 32), (0, 2, 32), (1, 3, 32), (1, 3, 32), (1, 3, 32)]
        self.device = device
        self.module = nn.ModuleList()
        for step, (flag, num, channel) in enumerate(filters):
            if flag:
                self.module.append(nn.AvgPool2d(2))
            in_channels = in_channels_img if step == 0 else channel
            out_channels = channel
            layer = DoubleConv(in_channels, out_channels) if num == 2 else TripleConv(in_channels, out_channels)
            self.module.append(layer.to(device))
        
    def forward(self, x):
        features = []
        for m in self.module:
            x = m(x)
            if isinstance(m, DoubleConv) or isinstance(m, TripleConv):
                features.append(x)
        
        return features

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    FeatureModule = FeaturePath(device, 1)
    
    # Move input data to the same device as the model
    img = torch.rand(1, 1, 128, 128).to(device)
    
    features = FeatureModule(img)
    print(len(features))
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    