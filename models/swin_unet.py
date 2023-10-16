from typing import Callable, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class MLP(nn.Sequential):
    """2-Layer MLP impl
    """
    def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      dropout: float = 0.
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        layers = nn.ModuleList([
          nn.Linear(self.in_features, hidden_features),
          act_layer(),
          nn.Dropout(dropout),
          nn.Linear(hidden_features, self.out_features),
          act_layer(),
          nn.Dropout(dropout)
        ])
        super(MLP, self).__init__(*layers)


def window_partition(x: torch.Tensor, win_size: int) -> torch.Tensor:
    """Window Partitioning Process

    :param x: (B, H, W, C)
    :param win_size: int - window size
    :return: (num_windows * B, win_size, win_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)


def window_reverse(windows: torch.Tensor, win_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse the window partition process

    :param windows: (num_windows * B, win_size, win_size, C)
    :param win_size: int - window size
    :param H: int - height of the image
    :param W: int - width of the image
    :return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H*W/win_size/win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window-based Self-Attention w/wo shifted window support

        :param dim: int - number of input channels.
        :param window_size: Tuple[int, int] - the height and width of the window.
        :param num_heads: int - number of attention heads.
        :param qkv_bias: bool - Learnable bias flag for the query, key and value. Default: True.
        :param qk_scale: float - Override default qk scale of head_dim ** -0.5 if set.
        :param attn_drop: float - Dropout ratio of attention weight. Default: 0.
        :param proj_drop: float - Dropout ratio of output. Default: 0.
    """
    def __init__(
      self,
      dim: int,
      window_size: Tuple[int, int],
      num_heads: int,
      qkv_bias: Optional[bool] = True,
      qk_scale: Optional[float] = None,
      attn_drop: Optional[float] = 0.,
      proj_drop: Optional[float] = 0.
    ) -> None:
        super(WindowAttention, self).__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
          torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: input feature maps, (num_windows * B, N, C)
        :param mask: (0/-inf) mask, (num_windows, Wh*Ww, Wh*Ww) or None
        :return: (num_windows * B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q *= self.scale
        attn = q @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
          self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
    
    def flops(self, N: int) -> float:
        """
        :param N: num_windows
        :return: flops in float
        """
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k).transpose(-2, -1)
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block

        :param dim: int - input channels dimension
        :param input_resolution: Tuple[int, int] - input resolution
        :param num_heads: int - number of attention heads
        :param window_size: int - window size
        :param shift_size: int - shift size for SW-MSA
        :param mlp_ratio: float - ratio of MLP hidden dim to embedding dim
        :param qkv_bias: bool - learnable bias for the query, key and value. Default: True
        :param qk_scale: float - override qk scale of head_dim ** -0.5 if set.
        :param drop: float - Dropout ratio of output. Default: 0.
        :param attn_drop: float - Dropout ratio of attention weight. Default: 0.
        :param drop_path: float - DropPath ratio. Default: 0.
        :param act_layer: nn.Module - activation layer. Default: nn.GELU
        :param norm_layer: nn.Module - norm layer. Default: nn.LayerNorm
    """
    def __init__(
      self,
      dim: int,
      input_resolution: Tuple[int, int],
      num_heads: int,
      window_size: int = 7,
      shift_size: int = 0,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      qk_scale: Optional[float] = None,
      drop: float = 0.,
      attn_drop: float = 0.,
      drop_path: float = 0.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super(SwinTransformerBlock, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        
        self.attn_norm = norm_layer(dim)
        self.attn = WindowAttention(
          dim,
          window_size=to_2tuple(self.window_size),
          num_heads=self.num_heads,
          qkv_bias=qkv_bias,
          qk_scale=qk_scale,
          attn_drop=attn_drop,
          proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.ffn = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, dropout=drop)
        
        attn_mask = None
        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
              slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            )
            w_slices = (
              slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        
        self.register_buffer('attn_mask', attn_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, H, W, C)
        :return: (B, H * W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f'{L} != {H*W}, input features has wrong size'
        
        shortcut = x
        x = self.attn_norm(x).view(B, H, W, C)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, ' \
               f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'
    
    def flops(self) -> float:
        flops = 0
        H, W = self.input_resolution
        # attn_norm
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        num_windows = H * W / self.window_size / self.window_size
        flops += num_windows * self.attn.flops(self.window_size * self.window_size)
        # FFN
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # ffn_norm
        flops += self.dim * H * W
        return flops


class BasicLayerDown(nn.Module):
    """One-stage Basic Swin Transformer Layer

        :param dim: int - input channels dimension
        :param input_resolution: Union[List[int], Tuple[int, int]] - input resolution
        :param depth: int - number of blocks
        :param num_heads: int - number of attention heads
        :param window_size: int - window size
        :param mlp_ratio: float - ratio of MLP hidden dim to embedding dim
        :param qkv_bias: bool - learnable bias for the query, key and value. Default: True
        :param qk_scale: float - override qk scale of head_dim ** -0.5 if set.
        :param drop: float - Dropout ratio of output. Default: 0.
        :param attn_drop: float - Dropout ratio of attention weight. Default: 0.
        :param drop_path: float - DropPath ratio. Default: 0.
        :param norm_layer: nn.Module - norm layer. Default: nn.LayerNorm.
        :param downsample: nn.Module - Down-sampler at the end of the layer. Default: None.
        :param use_checkpoint: bool - Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(
      self,
      dim: int,
      input_resolution: Union[List[int], Tuple[int, int]],
      depth: int,
      num_heads: int,
      window_size: int,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      qk_scale: Optional[float] = None,
      drop: float = 0.,
      attn_drop: float = 0.,
      drop_path: List[float] = 0.,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      downsample: Optional[Callable[..., nn.Module]] = None,
      use_checkpoint: bool = False
    ) -> None:
        super(BasicLayerDown, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
          SwinTransformerBlock(
            dim=self.dim,
            input_resolution=self.input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer
          ) for i in range(depth)
        ])
        
        self.downsample = downsample(
          self.input_resolution, dim=dim, norm_layer=norm_layer
        ) if downsample is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor - (B, H * W, C)
        :return: output - (B, H // 2 * W // 2, 2 * C)
        """
        for blk in self.blocks:  # 2x Successive SwinTransformerBlock
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'
    
    def flops(self) -> float:
        flops = sum(blk.flops() for blk in self.blocks)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayerUp(nn.Module):
    def __init__(
      self,
      dim: int,
      input_resolution: Union[List[int], Tuple[int, int]],
      depth: int,
      num_heads: int,
      window_size: int,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      qk_scale: Optional[float] = None,
      drop: float = 0.,
      attn_drop: float = 0.,
      drop_path: List[float] = 0.,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      upsample: Optional[Callable[..., nn.Module]] = None,
      use_checkpoint: bool = False
    ) -> None:
        super(BasicLayerUp, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
          SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer
          ) for i in range(depth)
        ])
        
        self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer) if upsample is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer

        :param input_resolution: Union[List[int], Tuple[int, int]] - the height and width of input feature maps
        :param dim: int - Dimension of previous output layer
        :param norm_layer: nn.Module - norm layer. Default: nn.LayerNorm
    """
    def __init__(
      self,
      input_resolution: Union[List[int], Tuple[int, int]],
      dim: int,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    ) -> None:
        super(PatchMerging, self).__init__()
        
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, H * W, C)
        :return: (B, H // 2 * W // 2, 2 * C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f'{L} != {H*W}, input features has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}x{W}) are not even'
        
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x
    
    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'
    
    def flops(self) -> float:
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H//2) * (W//2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    """Image to Patch Embedding

        :param img_size: int - input image resolution. Default: 224.
        :param patch_size: int - patch size. Default: 4.
        :param in_channels: int - input channels of image. Default: 3.
        :param embed_dim: int - output feature dimension. Default: 96.
        :param norm_layer: nn.Module - norm layer. Default: None.
    """
    def __init__(
      self,
      img_size: int = 224,
      patch_size: int = 4,
      in_channels: int = 3,
      embed_dim: int = 96,
      norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(PatchEmbed, self).__init__()
        
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input feature maps - (B, C, H, W)
        :return: (B, H // patch_size * W // patch_size, embed_dim)
        """
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, Ph * Pw, C)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_channels * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchExpand(nn.Module):
    def __init__(
      self,
      input_resolution: Union[List[int], Tuple[int, int]],
      dim: int,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    ) -> None:
        super(PatchExpand, self).__init__()
        
        self.input_resolution = input_resolution
        self.dim = dim
        
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        
        x = x.view(B, H, W, C)
        x = x.view(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand(nn.Module):
    def __init__(
      self,
      input_resolution: Union[List[int], Tuple[int, int]],
      dim: int,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    ) -> None:
        super(FinalPatchExpand, self).__init__()
        
        self.input_resolution = input_resolution
        self.dim = dim
        
        self.expand = nn.Linear(dim, dim * 16, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        
        x = x.view(B, H, W, C)
        x = x.view(B, H * 4, W * 4, C // 16)
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


class Encoder(nn.Module):
    def __init__(
      self,
      input_resolution: Union[List[int], Tuple[int, int]] = (56, 56),
      embed_dim: int = 96,
      depths: List[int] = (2, 2, 2),
      num_heads: List[int] = (3, 6, 12),
      window_size: int = 7,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      qk_scale: Optional[float] = None,
      drop_rate: float = 0.,
      attn_drop_rate: float = 0.,
      drop_path_rate: List[float] = 0.,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      use_checkpoint: bool = False
    ) -> None:
        super(Encoder, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerDown(
              dim=int(embed_dim * 2 ** i_layer),
              input_resolution=(
                input_resolution[0] // (2 ** i_layer),
                input_resolution[1] // (2 ** i_layer),
              ),
              depth=depths[i_layer],
              num_heads=num_heads[i_layer],
              window_size=window_size,
              mlp_ratio=mlp_ratio,
              qkv_bias=qkv_bias,
              qk_scale=qk_scale,
              drop=drop_rate,
              attn_drop=attn_drop_rate,
              drop_path=drop_path_rate[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
              norm_layer=norm_layer,
              downsample=PatchMerging,
              use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            self.input_resolution = layer.input_resolution
            self.embed_dim = layer.dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        
        return x, x_downsample


class Bottleneck(nn.Sequential):
    def __init__(
      self,
      input_resolution: Union[List[int], Tuple[int, int]] = (7, 7),
      embed_dim: int = 768,
      depth: int = 2,
      num_heads: int = 24,
      window_size: int = 7,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      qk_scale: Optional[float] = None,
      drop_rate: float = 0.,
      attn_drop_rate: float = 0.,
      drop_path_rate: List[float] = 0.1,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      use_checkpoint: bool = False
    ) -> None:
        self.input_resolution = input_resolution
        self.embed_dim = embed_dim
        super(Bottleneck, self).__init__(
          *[
            BasicLayerDown(
              dim=embed_dim,
              input_resolution=input_resolution,
              depth=depth,
              num_heads=num_heads,
              window_size=window_size,
              mlp_ratio=mlp_ratio,
              qkv_bias=qkv_bias,
              qk_scale=qk_scale,
              drop=drop_rate,
              attn_drop=attn_drop_rate,
              drop_path=drop_path_rate,
              norm_layer=norm_layer,
              downsample=None,
              use_checkpoint=use_checkpoint
            ),
          ]
        )


class SkipConnection(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super(SkipConnection, self).__init__()
        out_features = out_features or in_features // 2
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Decoder(nn.Module):
    def __init__(
      self,
      input_resolution: Union[List[int], Tuple[int, int]] = (56, 56),
      embed_dim: int = 96,
      depths: List[int] = (2, 2, 2),
      num_heads: List[int] = (3, 6, 12),
      window_size: int = 7,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      qk_scale: Optional[float] = None,
      drop_rate: float = 0.,
      attn_drop_rate: float = 0.,
      drop_path_rate: List[float] = 0.,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      use_checkpoint: bool = False
    ) -> None:
        super(Decoder, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        for i_layer in reversed(range(self.num_layers)):
            layer = BasicLayerUp(
              dim=int(embed_dim * 2 ** i_layer),
              input_resolution=(input_resolution[0] // 2 ** i_layer, input_resolution[1] // 2 ** i_layer),
              depth=depths[i_layer],
              num_heads=num_heads[i_layer],
              window_size=window_size,
              mlp_ratio=mlp_ratio,
              qkv_bias=qkv_bias,
              qk_scale=qk_scale,
              drop=drop_rate,
              attn_drop=attn_drop_rate,
              drop_path=drop_path_rate[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
              norm_layer=norm_layer,
              upsample=PatchExpand if i_layer != 0 else None,
              use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            self.skip_connections.append(SkipConnection(in_features=2 * layer.dim))
    
    def forward(self, x: torch.Tensor, downsamples: torch.Tensor) -> torch.Tensor:
        for idx, (layer, skip) in enumerate(zip(self.layers, self.skip_connections)):
            x = torch.cat([downsamples[idx], x], dim=-1)
            x = layer(skip(x))
        return x


class SwinUNet(nn.Module):
    def __init__(
      self,
      img_size: int = 224,
      patch_size: int = 4,
      in_channels: int = 3,
      num_classes: int = 2,
      embed_dim: int = 96,
      depths: List[int] = (2, 2, 2, 2),
      num_heads: List[int] = (3, 6, 12, 24),
      window_size: int = 7,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      qk_scale: Optional[float] = None,
      drop_rate: float = 0.,
      attn_drop_rate: float = 0.,
      drop_path_rate: float = 0.1,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      ape: bool = False,
      patch_norm: bool = True,
      use_checkpoint: bool = False,
      final_upsample: bool = True,
      **kwargs: Any
    ) -> None:
        super(SwinUNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        
        # Split image into non-overlapping patches, aka. Patch Partition + Linear Embeddings
        self.patch_embed = PatchEmbed(
          img_size=img_size,
          patch_size=patch_size,
          in_channels=in_channels,
          embed_dim=embed_dim,
          norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        self.input_resolution = self.patch_embed.patches_resolution
        
        # Absolute Positional Embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic Depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.encoder = Encoder(
          input_resolution=self.input_resolution,
          embed_dim=self.embed_dim,
          depths=depths[:-1],
          num_heads=num_heads[:-1],
          window_size=window_size,
          mlp_ratio=self.mlp_ratio,
          qkv_bias=qkv_bias,
          qk_scale=qk_scale,
          drop_rate=drop_rate,
          attn_drop_rate=attn_drop_rate,
          drop_path_rate=dpr[:-depths[-1]],
          norm_layer=norm_layer,
          use_checkpoint=use_checkpoint,
        )
        self.encoder_norm = norm_layer(self.num_features)
        
        self.bottleneck = Bottleneck(
          input_resolution=(self.encoder.input_resolution[0] // 2, self.encoder.input_resolution[1] // 2),
          embed_dim=self.encoder.embed_dim * 2,
          depth=depths[-1],
          num_heads=num_heads[-1],
          window_size=window_size,
          mlp_ratio=self.mlp_ratio,
          qkv_bias=qkv_bias,
          qk_scale=qk_scale,
          drop_rate=drop_rate,
          attn_drop_rate=attn_drop_rate,
          drop_path_rate=dpr[-depths[-1]:],
          norm_layer=norm_layer,
          use_checkpoint=use_checkpoint,
        )
        self.bottleneck_norm = norm_layer(self.num_features)
        
        self.first_expand = PatchExpand(
          input_resolution=self.bottleneck.input_resolution, dim=self.bottleneck.embed_dim
        )
        
        self.decoder = Decoder(
          input_resolution=self.input_resolution,
          embed_dim=self.embed_dim,
          depths=depths[:-1],
          num_heads=num_heads[:-1],
          window_size=window_size,
          mlp_ratio=mlp_ratio,
          qkv_bias=qkv_bias,
          qk_scale=qk_scale,
          drop_rate=drop_rate,
          attn_drop_rate=attn_drop_rate,
          drop_path_rate=dpr[:-depths[-1]],
          norm_layer=norm_layer,
          use_checkpoint=use_checkpoint
        )
        self.decoder_norm = norm_layer(self.embed_dim)
        
        self.last_expand = FinalPatchExpand(input_resolution=self.input_resolution, dim=self.embed_dim)
        self.output_proj = nn.Conv2d(
          in_channels=self.embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False
        )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.patch_embed(x)
        if self.ape:
            x += self.absolute_pos_embed
        x = self.pos_drop(x)
        
        x, x_downsample = self.encoder(x)
        x = self.encoder_norm(x)
        return x, x_downsample
    
    def forward_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(x)
        x = self.bottleneck_norm(x)
        return x
    
    def forward_decoder(self, x: torch, downsamples: List[torch.Tensor]) -> torch.Tensor:
        x = self.decoder(x, downsamples[::-1])
        x = self.decoder_norm(x)
        return x
    
    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f'{L} != {H*W}, input features has wrong size'
        
        if self.final_upsample:
            x = self.last_expand(x)
            x = x.view(B, H * 4, W * 4, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.output_proj(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x_downsample = self.forward_encoder(x)
        x = self.forward_bottleneck(x)
        x = self.first_expand(x)
        x = self.forward_decoder(x, x_downsample)
        x = self.forward_output(x)
        return x
    
    def flops(self) -> float:
        flops = 0
        flops += self.patch_embed.flops()
        flops += self.encoder.flops()
        flops += self.bottleneck.flops()
        flops += self.decoder.flops()
        return flops


def main():
    device = torch.device('mps')
    di = torch.randn(1, 3, 224, 224).to(device)
    su = SwinUNet().to(device)
    with torch.no_grad():
        out = su(di)
        print(out.shape)


if __name__ == "__main__":
    main()
