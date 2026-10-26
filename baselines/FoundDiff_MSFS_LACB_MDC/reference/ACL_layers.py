import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, mlp_ratio, qkv_bias, norm_layer):
        super(ResBlock, self).__init__()

        self.main_block = BasicLayer(dim=embed_dim,
                               depth=depths,
                               num_heads=num_heads,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, 
                               norm_layer=norm_layer,
                               )
        

    def forward(self, x, x_shape):
        x = self.main_block(x, x_shape)
        return x
        

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0.01, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
    
class LinearAttention(nn.Module):
    r""" Linear Attention with LePE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, x_shape):
        b, n, c = x.shape
        h, w = x_shape[2:]
        # h = w = int(n ** 0.5)
        assert h * w == n, "Input feature has wrong size"

        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class LAMA(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, x_shape):
        B, L, C = x.shape
        H, W = x_shape[2:]
        
        assert H * W == L, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        x = self.attn(x,x_shape)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            LAMA(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x,x_shape):
        for blk in self.blocks:
            x = blk(x,x_shape)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        x = x.flatten(2).permute(0, 2, 1)
        return x

class DetailEnhancerBlock(nn.Module):
    def __init__(self, in_c, kernel_size1, kernel_size2, dilated=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=kernel_size1,stride=1, padding=dilated*(kernel_size1-1)//2,dilation=dilated,bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=kernel_size2,stride=1, padding=dilated*(kernel_size2-1)//2,dilation=dilated,bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU()
        )
        self.conv_out =  nn.Sequential(
            nn.Conv2d(in_channels=in_c*2, out_channels=in_c, kernel_size=1, bias=False),
        )
    def forward(self, x):
        path1 = self.conv1(x)
        path2 = self.conv2(x)
        out = self.conv_out(torch.cat((path1,path2),dim=1))

        return out + x 