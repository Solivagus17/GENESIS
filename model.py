import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        return x + self.proj(out)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.residual(x)

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2, 4),
        attn_resolutions: tuple = (16, 8),
        num_res_blocks: int = 2
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        current_res = 64
        in_ch = base_channels
        down_channels = [base_channels]

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResnetBlock(in_ch, out_ch, time_dim)
                attn = AttentionBlock(out_ch) if current_res in attn_resolutions else nn.Identity()
                self.downs.append(nn.ModuleList([block, attn]))
                in_ch = out_ch
                down_channels.append(in_ch)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(in_ch))
                down_channels.append(in_ch)
                current_res //= 2

        # Bottleneck
        self.mid_block1 = ResnetBlock(in_ch, in_ch, time_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, time_dim)

        # Decoder
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = down_channels.pop()
                block = ResnetBlock(in_ch + skip_ch, out_ch, time_dim)
                attn = AttentionBlock(out_ch) if current_res in attn_resolutions else nn.Identity()
                self.ups.append(nn.ModuleList([block, attn]))
                in_ch = out_ch

            if i != 0:
                self.ups.append(Upsample(in_ch))
                current_res *= 2

        self.out_norm = nn.GroupNorm(min(32, in_ch), in_ch)
        self.out_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        skips = [h]

        for layer in self.downs:
            if isinstance(layer, Downsample):
                h = layer(h)
                skips.append(h)
            else:
                block, attn = layer
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for layer in self.ups:
            if isinstance(layer, Upsample):
                h = layer(h)
            else:
                block, attn = layer
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)

        return self.final_conv(self.out_act(self.out_norm(h)))

def get_model(image_size: int = 64, device: torch.device = None) -> UNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Attention at the two smallest spatial resolutions
    # 64px → (16, 8) | 128px → (32, 16)
    attn_resolutions = (image_size // 4, image_size // 8)

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=96,
        channel_mults=(1, 2, 2, 4),
        attn_resolutions=attn_resolutions,
        num_res_blocks=2
    ).to(device)
    return model

if __name__ == "__main__":
    model = get_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {params / 1e6:.2f}M")
