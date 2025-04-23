# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
# https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, activation='leaky_relu'):
        super().__init__()
        self.conv_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.relu_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.relu_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.relu_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.relu_fn(x)
        return x


def to_3d(x):
    if len(x.shape) == 3:
        return rearrange(x, 'b c n -> b n c')
    else:
        return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w=None):
    if w is None:
        return rearrange(x, 'b n c -> b c n')
    else:
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 3:
            h = x.shape[-1]
            w = None
        else:
            h, w = x.shape[-2:]

        return to_4d(self.body(to_3d(x)), h, w)


class Mutual_Attention2D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention2D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(torch.cat((x, y, y), dim=1))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class CrossTransformerBlock2D(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(CrossTransformerBlock2D, self).__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(dim * 2, dim),
        )
        self.norm1x = LayerNorm(dim, LayerNorm_type)
        self.norm1y = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention2D(dim, num_heads, bias)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = FeedForward2D(dim, ffn_expansion_factor, bias)

    def forward(self, x, y, z):
        assert x.shape == y.shape and x.shape == z.shape, 'wrong shape!, {} {} {}'.format(x.shape, y.shape, z.shape)
        y2 = self.mlps(torch.cat([y, z], dim=1))
        x = x + self.attn(self.norm1x(x), self.norm1y(y2))
        # x = x + self.ffn(self.norm2(x))

        return x