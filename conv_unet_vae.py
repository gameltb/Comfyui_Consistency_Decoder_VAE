#!/usr/bin/env python3
"""
Cleaned up reimplementation of public_diff_vae.ConvUNetVAE,
thanks to https://gist.github.com/mrsteyk/74ad3ec2f6f823111ae4c90e168505ac.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    def __init__(self, n_time=1024, n_emb=320, n_out=1280) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_time, n_emb)
        self.f_1 = nn.Linear(n_emb, n_out)
        self.f_2 = nn.Linear(n_out, n_out)

    def forward(self, x) -> torch.Tensor:
        x = self.emb(x)
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels=7, out_channels=320) -> None:
        super().__init__()
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(x)


class ImageUnembedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=6) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(F.silu(self.gn(x)))


class ConvResblock(nn.Module):
    def __init__(self, in_features=320, out_features=320) -> None:
        super().__init__()
        self.f_t = nn.Linear(1280, out_features * 2)

        self.gn_1 = nn.GroupNorm(32, in_features)
        self.f_1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

        self.gn_2 = nn.GroupNorm(32, out_features)
        self.f_2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)

        skip_conv = in_features != out_features
        self.f_s = (
            nn.Conv2d(in_features, out_features, kernel_size=1, padding=0)
            if skip_conv
            else nn.Identity()
        )

    def forward(self, x, t):
        x_skip = x
        t = self.f_t(F.silu(t))
        t = t.chunk(2, dim=1)
        t_1 = t[0].unsqueeze(dim=2).unsqueeze(dim=3) + 1
        t_2 = t[1].unsqueeze(dim=2).unsqueeze(dim=3)

        gn_1 = F.silu(self.gn_1(x))
        f_1 = self.f_1(gn_1)

        gn_2 = self.gn_2(f_1)

        return self.f_s(x_skip) + self.f_2(F.silu(gn_2 * t_1 + t_2))


# Also ConvResblock
class Downsample(nn.Module):
    def __init__(self, in_channels=320) -> None:
        super().__init__()
        self.f_t = nn.Linear(1280, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        avg_pool2d = F.avg_pool2d(gn_1, kernel_size=(2, 2), stride=None)
        f_1 = self.f_1(avg_pool2d)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.avg_pool2d(x_skip, kernel_size=(2, 2), stride=None)


# Also ConvResblock
class Upsample(nn.Module):
    def __init__(self, in_channels=1024) -> None:
        super().__init__()
        self.f_t = nn.Linear(1280, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        upsample = F.upsample_nearest(gn_1, scale_factor=2)
        f_1 = self.f_1(upsample)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.upsample_nearest(x_skip, scale_factor=2)


class ConvUNetVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_image = ImageEmbedding()
        self.embed_time = TimestepEmbedding()

        down_0 = nn.ModuleList(
            [
                ConvResblock(320, 320),
                ConvResblock(320, 320),
                ConvResblock(320, 320),
                Downsample(320),
            ]
        )
        down_1 = nn.ModuleList(
            [
                ConvResblock(320, 640),
                ConvResblock(640, 640),
                ConvResblock(640, 640),
                Downsample(640),
            ]
        )
        down_2 = nn.ModuleList(
            [
                ConvResblock(640, 1024),
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
                Downsample(1024),
            ]
        )
        down_3 = nn.ModuleList(
            [
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
            ]
        )
        self.down = nn.ModuleList(
            [
                down_0,
                down_1,
                down_2,
                down_3,
            ]
        )

        self.mid = nn.ModuleList(
            [
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
            ]
        )

        up_3 = nn.ModuleList(
            [
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                Upsample(1024),
            ]
        )
        up_2 = nn.ModuleList(
            [
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 + 640, 1024),
                Upsample(1024),
            ]
        )
        up_1 = nn.ModuleList(
            [
                ConvResblock(1024 + 640, 640),
                ConvResblock(640 * 2, 640),
                ConvResblock(640 * 2, 640),
                ConvResblock(320 + 640, 640),
                Upsample(640),
            ]
        )
        up_0 = nn.ModuleList(
            [
                ConvResblock(320 + 640, 320),
                ConvResblock(320 * 2, 320),
                ConvResblock(320 * 2, 320),
                ConvResblock(320 * 2, 320),
            ]
        )
        self.up = nn.ModuleList(
            [
                up_0,
                up_1,
                up_2,
                up_3,
            ]
        )

        self.output = ImageUnembedding()

    def forward(self, x, t, features) -> torch.Tensor:
        x = torch.cat([x, F.upsample_nearest(features, scale_factor=8)], dim=1)
        t = self.embed_time(t)
        x = self.embed_image(x)

        skips = [x]
        for down in self.down:
            for block in down:
                x = block(x, t)
                skips.append(x)

        for i in range(2):
            x = self.mid[i](x, t)

        for up in self.up[::-1]:
            for block in up:
                if isinstance(block, ConvResblock):
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)

        return self.output(x)


def rename_state_dict_key(k):
    k = k.replace("blocks.", "")
    for i in range(5):
        k = k.replace(f"down_{i}_", f"down.{i}.")
        k = k.replace(f"conv_{i}.", f"{i}.")
        k = k.replace(f"up_{i}_", f"up.{i}.")
        k = k.replace(f"mid_{i}", f"mid.{i}")
    k = k.replace("upsamp.", "4.")
    k = k.replace("downsamp.", "3.")
    k = k.replace("f_t.w", "f_t.weight").replace("f_t.b", "f_t.bias")
    k = k.replace("f_1.w", "f_1.weight").replace("f_1.b", "f_1.bias")
    k = k.replace("f_2.w", "f_2.weight").replace("f_2.b", "f_2.bias")
    k = k.replace("f_s.w", "f_s.weight").replace("f_s.b", "f_s.bias")
    k = k.replace("f.w", "f.weight").replace("f.b", "f.bias")
    k = k.replace("gn_1.g", "gn_1.weight").replace("gn_1.b", "gn_1.bias")
    k = k.replace("gn_2.g", "gn_2.weight").replace("gn_2.b", "gn_2.bias")
    k = k.replace("gn.g", "gn.weight").replace("gn.b", "gn.bias")
    return k


def rename_state_dict(sd, embedding):
    sd = {rename_state_dict_key(k): v for k, v in sd.items()}
    sd["embed_time.emb.weight"] = embedding["weight"]
    return sd


if __name__ == "__main__":
    model = ConvUNetVAE()
    import safetensors.torch

    cd_orig = safetensors.torch.load_file("consistency_decoder.safetensors")
    embedding = safetensors.torch.load_file("embedding.safetensors")
    print(model.load_state_dict(rename_state_dict(cd_orig, embedding)))
