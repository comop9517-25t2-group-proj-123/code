import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.up(x)

class TransUNet(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, depth=4, base_ch=64, 
                 img_size=256, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super(TransUNet, self).__init__()
        self.depth = depth
        self.img_size = img_size
        self.patch_size = patch_size
        
        # CNN Encoder (first part of U-Net)
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.chs = chs
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for out_ch in chs[:-1]:  # Don't include the last layer
            self.encoders.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        # Transformer Encoder (replaces the bottleneck)
        transformer_input_size = img_size // (2 ** (depth - 1))
        self.patch_embed = PatchEmbed(
            img_size=transformer_input_size, 
            patch_size=patch_size, 
            in_channels=chs[-2], 
            embed_dim=embed_dim
        )
        
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Reshape transformer output back to feature map
        self.conv_trans = nn.Conv2d(embed_dim, chs[-1], kernel_size=1)
        
        # CNN Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        for i in reversed(range(depth - 1)):
            up_in = chs[i + 1]
            up_out = chs[i]
            self.upconvs.append(UpConv(up_in, up_out))
            self.dec_blocks.append(ConvBlock(up_out * 2, up_out))
        
        # Final convolution
        self.final_conv = nn.Conv2d(chs[0], n_classes, kernel_size=1)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # CNN Encoder
        enc_feats = []
        for i in range(self.depth - 1):
            x = self.encoders[i](x)
            enc_feats.append(x)
            x = self.pools[i](x)
        
        # Transformer Encoder
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        x = x + self.pos_embed
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to feature map
        n_patches_h = n_patches_w = int(math.sqrt(x.shape[1]))
        x = x.transpose(1, 2).reshape(B, -1, n_patches_h, n_patches_w)
        x = self.conv_trans(x)
        
        # Upsample to match encoder feature size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        # CNN Decoder
        for i in range(self.depth - 1):
            x = self.upconvs[i](x)
            # Skip connection
            skip = enc_feats[self.depth - 2 - i]
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)
        
        # Final output
        out = self.final_conv(x)
        return out