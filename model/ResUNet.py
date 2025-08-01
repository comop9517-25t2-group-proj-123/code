import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)

class ResUNet(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, depth=4, base_ch=64):
        super(ResUNet, self).__init__()
        self.depth = depth
        self.base_ch = base_ch
        
        # Compute channel progression
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.chs = chs
        
        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, chs[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(chs[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(depth):
            in_ch = chs[i] if i == 0 else chs[i-1]
            out_ch = chs[i]
            self.encoders.append(ResidualBlock(in_ch, out_ch, stride=1))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Bottleneck
        self.bottleneck = ResidualBlock(chs[-1], chs[-1] * 2)
        
        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        for i in reversed(range(depth)):
            up_in = chs[-1] * 2 if i == depth - 1 else chs[i + 1]
            up_out = chs[i]
            self.upconvs.append(UpConv(up_in, up_out))
            self.dec_blocks.append(ResidualBlock(up_out * 2, up_out))
        
        # Final convolution
        self.final_conv = nn.Conv2d(chs[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_conv(x)
        
        # Encoder path
        enc_feats = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            enc_feats.append(x)
            x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i in range(self.depth - 1, -1, -1):
            x = self.upconvs[self.depth - 1 - i](x)
            # Skip connection
            x = torch.cat([x, enc_feats[i]], dim=1)
            x = self.dec_blocks[self.depth - 1 - i](x)
        
        # Final output
        out = self.final_conv(x)
        return out