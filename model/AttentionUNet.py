"""
PyTorch implementation of "Attention U-Net: Learning Where to Look for the Pancreas." 
by Oktay et al applied for DRIVE blood vessels dataset. 

The paper describing the architecture is available at: https://arxiv.org/pdf/1804.03999.pdf
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, depth=4, base_ch=64):
        super(AttentionUNet, self).__init__()
        self.depth = depth
        self.base_ch = base_ch

        # Compute channel progression
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.chs = chs

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for out_ch in chs:
            self.encoders.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(chs[-1], chs[-1]*2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            up_in = chs[-1]*2 if i == depth-1 else chs[i+1]
            up_out = chs[i]
            self.upconvs.append(UpConv(up_in, up_out))
            self.attentions.append(AttentionBlock(F_g=up_out, F_l=chs[i], n_coefficients=up_out//2))
            self.dec_blocks.append(ConvBlock(up_out*2, up_out))

        # Final conv
        self.final_conv = nn.Conv2d(chs[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc_feats = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            enc_feats.append(x)
            x = self.pools[i](x)
        x = self.bottleneck(x)
        for i in range(self.depth-1, -1, -1):
            x = self.upconvs[self.depth-1-i](x)
            att = self.attentions[self.depth-1-i](x, enc_feats[i])
            x = torch.cat((att, x), dim=1)
            x = self.dec_blocks[self.depth-1-i](x)
        out = self.final_conv(x)
        return out