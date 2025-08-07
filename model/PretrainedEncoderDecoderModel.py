import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()

        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # Flatten spatial dimensions (H × W) to prepare for attention computation
        # Q: [B, HW, C'] where C' = in_channels // 8
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1) 
        proj_key = self.key(x).view(B, -1, H * W) 

        # Compute dot-product attention
        energy = torch.bmm(proj_query, proj_key)  
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, -1, H * W)  

        # Multiply attention with value (context aggregation)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # Reshape back to spatial dimensions: [B, C, H, W]
        out = out.view(B, C, H, W)

        # Residual connection: add scaled attention output to original input
        out = self.gamma * out + x
        
        return out


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Upsample input feature map by a factor of 2 using transpose convolution
        # Reduces in_channels → out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Apply self-attention on the skip connection from the encoder
        # This filters out irrelevant spatial regions before merging
        self.attn = SelfAttentionBlock(skip_channels)

        # After upsampling and attention, concatenate → [out_channels + skip_channels]
        # Apply two 3x3 conv layers with ReLU and BatchNorm to refine the merged features
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, skip):
        # Upsample decoder input (from previous layer)
        x = self.up(x)
        
        # Resize skip connection feature map to match x's spatial resolution (if needed)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        
        # Apply attention filtering to skip features before merging
        skip = self.attn(skip)

        # Concatenate along channel dimension: [B, out_channels + skip_channels, H, W]
        x = torch.cat([x, skip], dim=1)

        # Apply convolutional refinement
        return self.conv(x)


class PretrainedEncoderDecoderModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Load pretrained ResNet-34
        resnet = models.resnet34(pretrained=True)

        # Create a convolution that accepts 4 channels (R, G, B, NIR)
        self.input_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.input_conv.weight[:, :3] = resnet.conv1.weight
            self.input_conv.weight[:, 3] = resnet.conv1.weight[:, 0]  # Initialize NIR as Red

        self.bn1 = resnet.bn1
        # Activation function ReLu
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder to extract features from spatial and spectral images. 
        self.encoder1 = resnet.layer1  # Output: 64 channels
        self.encoder2 = resnet.layer2  # Output: 128 channels
        self.encoder3 = resnet.layer3  # Output: 256 channels
        self.encoder4 = resnet.layer4  # Output: 512 channels

        # Bridge between encoder and decoder for feature condensation, context capture, and decoder preparation
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        # Decoder with self attention block
        self.decoder4 = AttentionDecoderBlock(512, 256, 256)
        self.decoder3 = AttentionDecoderBlock(256, 128, 128)
        self.decoder2 = AttentionDecoderBlock(128, 64, 64)
        self.decoder1 = AttentionDecoderBlock(64, 64, 32)

        # Num class 1
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Apply convolution
        x = self.input_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Feature extractions
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        # Upsampling
        d4 = self.decoder4(center, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, x) 

        # Final convolution layer reduces channel to 1 
        out = self.final(d1)

        # Ensures final output matches target mask size (128x128)
        out = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)

        return out
