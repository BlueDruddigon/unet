from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from torch.utils import checkpoint as checkpoint


class PatchEmbed(nn.Module):
    def __init__(
      self,
      img_size: Union[int, List[int], Tuple[int, int]] = 224,
      patch_size: int = 4,
      in_chans: int = 3,
      embed_dim: int = 96,
      norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        :param img_size: image size. Default: 224
        :param patch_size: patch size. Default: 4
        :param in_chans: number of input channels. Default: 3
        :param embed_dim: embedding dimension. Default: 96
        :param norm_layer: normalization layer. Default: None
        """
        super(PatchEmbed, self).__init__()
        
        self.img_size = to_2tuple(img_size) if isinstance(img_size, int) else img_size
        self.patch_size = to_2tuple(patch_size)
        
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input feature map - (B, C, H, W)
        :return: feature map - (B, H * W, embed_dim)
        """
        _, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], 'input feature map has wrong size'
        assert C == self.in_chans, 'input feature map has wrong in_chans'
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class PatchExpand(nn.Module):
    def __init__(
      self,
      img_size: Union[int, List[int], Tuple[int, int]] = 224,
      patch_size: int = 4,
      in_chans: int = 3,
      embed_dim: int = 96,
      norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        :param img_size: image size. Default: 224
        :param patch_size: patch size. Default: 4
        :param in_chans: number of input channels. Default: 3
        :param embed_dim: embedding dimension. Default: 96
        :param norm_layer: normalization layer. Default: None
        """
        super(PatchExpand, self).__init__()
        
        self.img_size = to_2tuple(img_size) if isinstance(img_size, int) else img_size
        self.patch_size = to_2tuple(patch_size)
        
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        
        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, output_padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    
    def forward(self, x: torch.Tensor, is_last: bool = False) -> torch.Tensor:
        """
        :param x: input feature map - (B, L, C)
        :param is_last: whether is the last expanding layer. Default: False
        :return: output feature map - (B, newH * newW, embed_dim)
        """
        H, W = self.img_size
        B, _, _ = x.shape
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        if is_last:
            return self.proj(x)
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class MLP(nn.Sequential):
    """2-layer perceptron impl"""
    def __init__(
      self,
      in_features: int,
      out_features: Optional[int] = None,
      hidden_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      dropout: float = 0.
    ):
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        super(MLP, self).__init__(
          *[
            nn.Linear(self.in_features, self.hidden_features),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_features, self.out_features),
            act_layer(),
            nn.Dropout(dropout)
          ]
        )


class FocalModulation(nn.Module):
    def __init__(
      self,
      dim: int,
      input_resolution: Union[int, List[int], Tuple[int, int]],
      focal_level: int = 3,
      focal_window: int = 3,
      focal_factor: int = 2,
      modulator_bias: bool = True,
      dropout_rate: float = 0.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      use_post_norm_in_modulation: bool = False,
      normalize_modulator: bool = False,
    ) -> None:
        """
        :param dim: input feature dimension.
        :param input_resolution: input feature resolution.
        :param focal_level: number of focal's context aggregation level. Default: 3
        :param focal_window: focal window size at the first level. Default: 3
        :param focal_factor: factor for kernel size's multiplication. Default: 2
        :param modulator_bias: flag whether using bias for modulator or not. Default: True
        :param dropout_rate: dropout rate. Default: 0.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.LayerNorm
        :param use_post_norm_in_modulation: whether using post-norm in the modulation or not. Default: False
        :param normalize_modulator: whether to normalize the modulator's value or not. Default: False
        """
        super(FocalModulation, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        
        self.use_post_norm_in_modulation = use_post_norm_in_modulation
        self.normalize_modulator = normalize_modulator
        
        # first projection to turn the feature map into new feature space, eq: `Z_0=f(X)`
        self.f = nn.Linear(dim, 2*dim + self.focal_level + 1, bias=modulator_bias)
        # summarize the modulator across channels
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=modulator_bias)
        
        if self.use_post_norm_in_modulation:
            self.norm = norm_layer(dim)
        self.act = act_layer()
        
        # Hierarchical Contextualization with different kernel_sizes
        self.kernel_sizes = []
        self.focal_layers = nn.ModuleList()
        for k in range(self.focal_level):
            kernel_size = k*focal_factor + self.focal_window
            self.focal_layers.append(
              nn.Sequential(
                nn.Conv2d(
                  dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False
                ),
                nn.GELU(),
              )
            )
            self.kernel_sizes.append(kernel_size)
        
        # Post output projection
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout_rate))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input feature map - (B, H, W, C)
        :return: output feature map - (B, H, W, C)
        """
        C = x.shape[-1]
        
        # Pre-linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        # Light-weight Linear projection
        query, ctx, gates = torch.split(x, [C, C, self.focal_level + 1], dim=1)
        
        # Context Aggregation
        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all += ctx * gates[:, level:level + 1]
        # Average Pooling at last-level feature map to get global context
        global_ctx = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all += global_ctx * gates[:, self.focal_level:]
        
        # normalize the context
        if self.normalize_modulator:
            ctx_all /= (self.focal_level + 1)
        
        # Focal Modulation
        modulator = self.h(ctx_all)
        x_out = query * modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_post_norm_in_modulation:
            x_out = self.norm(x_out)
        
        # Post output projection
        x = self.out_proj(x_out)
        return x


class FMModule(nn.Module):
    def __init__(
      self,
      dim: int,
      input_resolution: Union[int, List[int], Tuple[int, int]],
      mlp_ratio: float = 4.,
      focal_level: int = 3,
      focal_window: int = 3,
      focal_factor: int = 2,
      modulator_bias: bool = True,
      use_layerscale: bool = False,
      layerscale_value: float = 1e-4,
      drop_path_rate: float = 0.1,
      dropout_rate: float = 0.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      use_post_norm: bool = False,
      use_post_norm_in_modulation: bool = False,
      normalize_modulator: bool = False,
    ) -> None:
        """
        :param dim: input feature dimension.
        :param input_resolution: input feature resolution.
        :param mlp_ratio: ratio of MLP. Default: 4.
        :param focal_level: number of focal's context aggregation level. Default: 3
        :param focal_window: focal window size at the first level. Default: 3
        :param focal_factor: factor for kernel size's multiplication. Default: 2
        :param modulator_bias: flag whether using bias for modulator or not. Default: True
        :param use_layerscale: whether using layer scale or not. Default: False
        :param layerscale_value: initial value of layerscale. Default: 1e-4
        :param drop_path_rate: stochastic drop path rate. Default: 0.1
        :param dropout_rate: dropout rate. Default: 0.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.LayerNorm
        :param use_post_norm: whether to use post-norm or not. Default: False
        :param use_post_norm_in_modulation: whether using post-norm in the modulation or not. Default: False
        :param normalize_modulator: whether to normalize the modulator's value or not. Default: False
        """
        super(FMModule, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.use_post_norm = use_post_norm
        
        # Focal Modulation
        self.modulation = FocalModulation(
          dim=self.dim,
          input_resolution=self.input_resolution,
          focal_level=focal_level,
          focal_window=focal_window,
          focal_factor=focal_factor,
          modulator_bias=modulator_bias,
          dropout_rate=dropout_rate,
          act_layer=act_layer,
          norm_layer=norm_layer,
          use_post_norm_in_modulation=use_post_norm_in_modulation,
          normalize_modulator=normalize_modulator
        )
        self.modulation_norm = norm_layer(dim)
        
        # DropPath
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # FFN
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.ffn = MLP(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, dropout=dropout_rate)
        self.ffn_norm = norm_layer(dim)
        
        self.alpha = 1.
        self.beta = 1.
        if use_layerscale:
            self.alpha = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)
            self.beta = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input feature map - (B, H x W, C)
        :return: output tensor - (B, L, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f'{L} != {H*W}, input features have wrong size'
        shortcut = x
        x = x.view(B, H, W, C)
        
        # Focal Modulation (replacement of SW/W-MSA)
        x = x if self.use_post_norm else self.modulation_norm(x)
        x = self.modulation(x)
        x = self.modulation_norm(x) if self.use_post_norm else x
        
        # Stochastic Drop Path
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.alpha * x)
        
        # FFN
        x = self.ffn_norm(self.ffn(x)) if self.use_post_norm else self.ffn(self.ffn_norm(x))
        x += self.drop_path(self.beta * x)
        
        return x


class BasicLayer(nn.Module):
    def __init__(
      self,
      dim: int,
      input_resolution: Union[int, List[int], Tuple[int, int]],
      depth: int,
      mlp_ratio: float = 4.,
      drop_path: List[float] = 0.1,
      dropout_rate: float = 0.,
      focal_level: int = 3,
      focal_window: int = 3,
      modulator_bias: bool = True,
      use_layerscale: bool = False,
      layerscale_value: float = 1e-4,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      downsample: Optional[Callable[..., nn.Module]] = None,
      upsample: Optional[Callable[..., nn.Module]] = None,
      use_post_norm: bool = True,
      use_post_norm_in_modulation: bool = False,
      normalize_modulator: bool = False,
      use_checkpoint: bool = False,
    ) -> None:
        """
        :param dim: input feature dimension.
        :param input_resolution: input feature resolution.
        :param depth: number of blocks.
        :param mlp_ratio: ratio of MLP. Default: 4.
        :param focal_level: number of focal's context aggregation level. Default: 3
        :param focal_window: focal window size at the first level. Default: 3
        :param modulator_bias: flag whether using bias for modulator or not. Default: True
        :param use_layerscale: whether using layer scale or not. Default: False
        :param layerscale_value: initial value of layerscale. Default: 1e-4
        :param drop_path: stochastic drop path rate. Default: 0.1
        :param dropout_rate: dropout rate. Default: 0.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.LayerNorm
        :param downsample: downsample at the end of the layer. Default: None
        :param upsample: upsample at the end of the layer. Default: None
        :param use_post_norm: whether to use post-norm after modulation or not. Default: False
        :param use_post_norm_in_modulation: whether using post-norm in the modulation or not. Default: False
        :param normalize_modulator: whether to normalize the modulator's value or not. Default: False
        :param use_checkpoint: whether to use checkpointing to save memory. Default: False
        """
        super(BasicLayer, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
          FMModule(
            dim=self.dim,
            input_resolution=self.input_resolution,
            mlp_ratio=self.mlp_ratio,
            focal_level=focal_level,
            focal_window=focal_window,
            modulator_bias=modulator_bias,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop_path_rate=drop_path[i] if isinstance(drop_path, list) else drop_path,
            dropout_rate=dropout_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_post_norm=use_post_norm,
            use_post_norm_in_modulation=use_post_norm_in_modulation,
            normalize_modulator=normalize_modulator
          ) for i in range(depth)
        ])
        
        self.downsample = downsample(
          img_size=self.input_resolution,
          patch_size=2,
          in_chans=self.dim,
          embed_dim=self.dim * 2,
          norm_layer=norm_layer
        ) if downsample is not None else None
        
        self.upsample = upsample(
          img_size=self.input_resolution,
          patch_size=2,
          in_chans=self.dim,
          embed_dim=self.dim // 2,
          norm_layer=norm_layer
        ) if upsample is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor - (B, L, C)
        :return: tensor - (B, L, C)
        """
        B, _, _ = x.shape
        H, W = self.input_resolution
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        
        # Downsample if available
        if self.downsample is not None:
            x = x.transpose(1, 2).view(B, -1, H, W)
            x = self.downsample(x)
        # Upsample if available
        elif self.upsample is not None:
            x = self.upsample(x)
        
        return x


class Encoder(nn.Module):
    def __init__(
      self,
      img_size: Union[int, List[int], Tuple[int, int]] = 224,
      patch_size: int = 4,
      in_chans: int = 3,
      embed_dim: int = 96,
      depths: List[int] = (4, 4, 4),
      mlp_ratio: float = 4.,
      drop_path: List[int] = 0.1,
      dropout_rate: float = 0.,
      focal_levels: List[int] = (3, 3, 3),
      focal_windows: List[int] = (3, 3, 3),
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
      use_patch_norm: bool = True,
      use_post_norm: bool = True,
      use_post_norm_in_modulation: bool = False,
      normalize_modulator: bool = False,
      use_checkpoint: bool = False,
    ) -> None:
        """
        :param img_size: image size. Default: 224
        :param patch_size: patch size. Default: 4
        :param in_chans: number of input channels. Default: 3
        :param embed_dim: embedding dimension. Default: 96
        :param depths: list of depths. Default: (4, 4, 4).
        :param mlp_ratio: ratio of MLP. Default: 4.
        :param focal_levels: list of focal levels. Default: (3, 3, 3)
        :param focal_windows: list of focal windows. Default: (3, 3, 3)
        :param drop_path: stochastic drop path rate. Default: 0.1
        :param dropout_rate: dropout rate. Default: 0.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.LayerNorm
        :param use_patch_norm: whether to use norm layer in patch embedding. Default: True
        :param use_post_norm: whether to use post-norm after modulation or not. Default: False
        :param use_post_norm_in_modulation: whether using post-norm in the modulation or not. Default: False
        :param normalize_modulator: whether to normalize the modulator's value or not. Default: False
        :param use_checkpoint: whether to use checkpointing to save memory. Default: False
        """
        super(Encoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        self.patch_embed = PatchEmbed(
          img_size=img_size,
          patch_size=patch_size,
          in_chans=in_chans,
          embed_dim=embed_dim,
          norm_layer=norm_layer if use_patch_norm else None
        )
        self.pos_drop = nn.Dropout(dropout_rate)
        input_resolution = self.patch_embed.patches_resolution
        self.input_resolution = input_resolution
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
              dim=embed_dim * 2 ** i_layer,
              input_resolution=(input_resolution[0] // (2 ** i_layer), input_resolution[1] // (2 ** i_layer)),
              depth=depths[i_layer],
              mlp_ratio=self.mlp_ratio,
              drop_path=drop_path[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
              dropout_rate=dropout_rate,
              focal_level=focal_levels[i_layer],
              focal_window=focal_windows[i_layer],
              act_layer=act_layer,
              norm_layer=norm_layer,
              downsample=PatchEmbed,
              upsample=None,
              use_post_norm=use_post_norm,
              use_post_norm_in_modulation=use_post_norm_in_modulation,
              normalize_modulator=normalize_modulator,
              use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            self.embed_dim = layer.dim
            self.input_resolution = layer.input_resolution
        
        self.norm = norm_layer(self.embed_dim * 2) if norm_layer is not None else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        :param x: input image tensor - (B, C, H, W)
        :return: final and list of hierarchical feature maps - (B, Ph * Pw, C)
        """
        x = self.pos_drop(self.patch_embed(x))
        x_downsamples = []
        for layer in self.layers:
            x_downsamples.append(x)
            x = layer(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, x_downsamples


class Bottleneck(nn.Sequential):
    def __init__(
      self,
      embed_dim: int,
      input_resolution: Union[int, List[int], Tuple[int, int]],
      depth: int = 4,
      mlp_ratio: float = 4.,
      drop_path: List[int] = 0.1,
      dropout_rate: float = 0.,
      focal_level: int = 3,
      focal_window: int = 3,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      use_post_norm: bool = True,
      use_post_norm_in_modulation: bool = False,
      normalize_modulator: bool = False,
      use_checkpoint: bool = False,
    ) -> None:
        """
        :param embed_dim: previous embed dimension of the encoder.
        :param input_resolution: input feature resolution.
        :param depth: number of blocks. Default: 4
        :param mlp_ratio: ratio of MLP. Default: 4.
        :param focal_level: number of focal's context aggregation level. Default: 3
        :param focal_window: focal window size at the first level. Default: 3
        :param drop_path: stochastic drop path rate. Default: 0.1
        :param dropout_rate: dropout rate. Default: 0.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.LayerNorm
        :param use_post_norm: whether to use post-norm after modulation or not. Default: False
        :param use_post_norm_in_modulation: whether using post-norm in the modulation or not. Default: False
        :param normalize_modulator: whether to normalize the modulator's value or not. Default: False
        :param use_checkpoint: whether to use checkpointing to save memory. Default: False
        """
        self.embed_dim = embed_dim * 2
        self.input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.mlp_ratio = mlp_ratio
        
        super(Bottleneck, self).__init__(
          *[
            BasicLayer(
              dim=self.embed_dim,
              input_resolution=self.input_resolution,
              depth=depth,
              mlp_ratio=self.mlp_ratio,
              drop_path=drop_path,
              dropout_rate=dropout_rate,
              focal_level=focal_level,
              focal_window=focal_window,
              act_layer=act_layer,
              norm_layer=norm_layer,
              downsample=None,
              upsample=None,
              use_post_norm=use_post_norm,
              use_post_norm_in_modulation=use_post_norm_in_modulation,
              normalize_modulator=normalize_modulator,
              use_checkpoint=use_checkpoint
            ),
            norm_layer(self.embed_dim)
          ]
        )


class SkipConnection(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        """
        :param in_features: input feature dimension
        :param out_features: output feature dimension. Default: None
        """
        super(SkipConnection, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features or in_features // 2
        
        self.linear = nn.Linear(self.in_features, self.out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor to be projected - (B, L, C)
        :return: output tensor - (B, L, C // 2)
        """
        return self.linear(x)


class Decoder(nn.Module):
    def __init__(
      self,
      embed_dim: int,
      input_resolution: Union[int, List[int], Tuple[int, int]],
      num_classes: int,
      depths: List[int] = (1, 1, 1),
      mlp_ratio: float = 4.,
      drop_path: List[int] = 0.1,
      dropout_rate: float = 0.,
      focal_levels: List[int] = (3, 3, 3),
      focal_windows: List[int] = (3, 3, 3),
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
      use_patch_norm: bool = True,
      use_post_norm: bool = True,
      use_post_norm_in_modulation: bool = False,
      normalize_modulator: bool = False,
      use_checkpoint: bool = False,
    ) -> None:
        """
        :param embed_dim: previous embed dimension of the encoder.
        :param input_resolution: input feature resolution.
        :param num_classes: number of output classes.
        :param depths: list of the decoder's depths. Default: (1, 1, 1)
        :param mlp_ratio: ratio of MLP. Default: 4.
        :param focal_levels: list of the decoder's focal levels. Default: (3, 3, 3)
        :param focal_windows: list of the decoder's focal windows. Default: (3, 3, 3)
        :param drop_path: stochastic drop path rate. Default: 0.1
        :param dropout_rate: dropout rate. Default: 0.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.LayerNorm
        :param use_patch_norm: whether to use normalization layer in the patch-expanding layer. Default: True
        :param use_post_norm: whether to use post-norm after modulation or not. Default: False
        :param use_post_norm_in_modulation: whether using post-norm in the modulation or not. Default: False
        :param normalize_modulator: whether to normalize the modulator's value or not. Default: False
        :param use_checkpoint: whether to use checkpointing to save memory. Default: False
        """
        super(Decoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.input_resolution = to_2tuple(input_resolution) if isinstance(input_resolution, int) else input_resolution
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        for i_layer in range(self.num_layers, -1, -1):
            embed_dim = int(self.embed_dim * 2 ** i_layer)
            if i_layer == self.num_layers:
                layer = PatchExpand(
                  img_size=self.input_resolution,
                  patch_size=2,
                  in_chans=embed_dim,
                  embed_dim=embed_dim // 2,
                  norm_layer=norm_layer if use_patch_norm else None
                )
            else:
                layer = BasicLayer(
                  dim=embed_dim,
                  input_resolution=(self.input_resolution[0] * 2, self.input_resolution[1] * 2),
                  depth=depths[i_layer],
                  mlp_ratio=self.mlp_ratio,
                  drop_path=drop_path[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                  dropout_rate=dropout_rate,
                  focal_level=focal_levels[i_layer],
                  focal_window=focal_windows[i_layer],
                  act_layer=act_layer,
                  norm_layer=norm_layer,
                  downsample=None,
                  upsample=PatchExpand if i_layer > 0 else None,
                  use_post_norm=use_post_norm,
                  use_post_norm_in_modulation=use_post_norm_in_modulation,
                  normalize_modulator=normalize_modulator,
                  use_checkpoint=use_checkpoint
                )
                self.input_resolution = layer.input_resolution
            
            self.layers.append(layer)
            skip_in_features = 2 * self.embed_dim * 2 ** i_layer
            self.skip_connections.append(
              SkipConnection(skip_in_features) if i_layer < self.num_layers else nn.Identity()
            )
        
        # the decoder's normalization layer
        self.norm = norm_layer(self.embed_dim) if norm_layer is not None else None
        # last patch-expanding layer
        self.last_expand = PatchExpand(
          img_size=self.input_resolution,
          patch_size=4,
          in_chans=embed_dim,
          embed_dim=embed_dim,
          norm_layer=norm_layer if use_patch_norm else None
        )
        # output projection layer to get classification probs
        self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x: torch.Tensor, downsamples: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: input feature map for decoder - (B, L, C)
        :param downsamples: list of the previous downsample results - (length: 3)
        :return: output probabilities. (B, num_classes, imgH, imgW)
        """
        for idx, (layer, skip_connection) in enumerate(zip(self.layers, self.skip_connections)):
            if idx == 0:
                x = layer(x)
            else:
                # Minus 1 because of downsamples just contains 3-depths in encoder
                x = torch.cat([downsamples[idx - 1], x], dim=-1)
                x = layer(skip_connection(x))
        
        # Normalize the output if available
        if self.norm is not None:
            x = self.norm(x)
        
        # Last Expansion with kernel_size=4 and stride=4
        x = self.last_expand(x, is_last=True)
        # Output Projection Probs
        x = self.output(x)
        
        return x


class FocalUNet(nn.Module):
    def __init__(
      self,
      img_size: Union[int, List[int], Tuple[int, int]] = 224,
      patch_size: int = 4,
      in_chans: int = 3,
      num_classes: int = 2,
      embed_dim: int = 96,
      depths: List[int] = (4, 4, 4, 4),
      decoder_depths: List[int] = (1, 1, 1),
      mlp_ratio: float = 4.,
      drop_path_rate: float = 0.1,
      dropout_rate: float = 0.,
      focal_levels: List[int] = (3, 3, 3, 3),
      focal_windows: List[int] = (3, 3, 3, 3),
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
      use_patch_norm: bool = True,
      use_post_norm: bool = False,
      use_post_norm_in_modulation: bool = False,
      normalize_modulator: bool = False,
      use_checkpoint: bool = False,
    ) -> None:
        """
        :param img_size: image size. Default: 224
        :param patch_size: patch size. Default: 4
        :param in_chans: number of input channels. Default: 3
        :param num_classes: number of output classes. Default: 2
        :param embed_dim: embedding dimension. Default: 96
        :param depths: list of encoder's depths. Default: (4, 4, 4, 4).
        :param decoder_depths: list of decoder's depths. Default: (1, 1, 1)
        :param mlp_ratio: ratio of MLP. Default: 4.
        :param focal_levels: list of focal levels. Default: (3, 3, 3, 3)
        :param focal_windows: list of focal windows. Default: (3, 3, 3, 3)
        :param drop_path_rate: stochastic drop path rate. Default: 0.1
        :param dropout_rate: dropout rate. Default: 0.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.LayerNorm
        :param use_patch_norm: whether to use norm layer in patch embedding. Default: True
        :param use_post_norm: whether to use post-norm after modulation or not. Default: False
        :param use_post_norm_in_modulation: whether using post-norm in the modulation or not. Default: False
        :param normalize_modulator: whether to normalize the modulator's value or not. Default: False
        :param use_checkpoint: whether to use checkpointing to save memory. Default: False
        """
        super(FocalUNet, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_classes = num_classes
        self.num_features = embed_dim * 2 ** (self.num_layers - 1)
        self.mlp_ratio = mlp_ratio
        self.use_post_norm = use_post_norm
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.encoder = Encoder(
          img_size=img_size,
          patch_size=patch_size,
          in_chans=in_chans,
          embed_dim=embed_dim,
          depths=depths[:-1],
          mlp_ratio=mlp_ratio,
          drop_path=dpr[:-depths[-1]],
          dropout_rate=dropout_rate,
          focal_levels=focal_levels,
          focal_windows=focal_windows,
          act_layer=act_layer,
          norm_layer=norm_layer,
          use_patch_norm=use_patch_norm,
          use_post_norm=use_post_norm,
          use_post_norm_in_modulation=use_post_norm_in_modulation,
          normalize_modulator=normalize_modulator,
          use_checkpoint=use_checkpoint
        )
        
        self.bottleneck = Bottleneck(
          embed_dim=self.encoder.embed_dim,
          input_resolution=self.encoder.input_resolution,
          depth=depths[-1],
          mlp_ratio=mlp_ratio,
          drop_path=dpr[-depths[-1]:],
          dropout_rate=dropout_rate,
          focal_level=focal_levels[-1],
          focal_window=focal_windows[-1],
          act_layer=act_layer,
          norm_layer=norm_layer,
          use_post_norm=use_post_norm,
          use_post_norm_in_modulation=use_post_norm_in_modulation,
          normalize_modulator=normalize_modulator,
          use_checkpoint=use_checkpoint
        )
        
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, sum(decoder_depths))]
        
        self.decoder = Decoder(
          embed_dim=embed_dim,
          input_resolution=self.bottleneck.input_resolution,
          num_classes=self.num_classes,
          depths=decoder_depths,
          mlp_ratio=mlp_ratio,
          drop_path=dpr,
          dropout_rate=dropout_rate,
          focal_levels=focal_levels[:-1],
          focal_windows=focal_windows[:-1],
          act_layer=act_layer,
          norm_layer=norm_layer,
          use_patch_norm=use_patch_norm,
          use_post_norm=use_post_norm,
          use_post_norm_in_modulation=use_post_norm_in_modulation,
          normalize_modulator=normalize_modulator,
          use_checkpoint=use_checkpoint
        )
    
    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        :param x: input image tensor - (B, C, H, W)
        :return: encoded feature map and list of down-sampled feature maps - (B, Ph * Pw, C)
        """
        return self.encoder(x)
    
    def forward_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: encoded feature map - (B, L, C)
        :return: output tensor - (B, L, C)
        """
        return self.bottleneck(x)
    
    def forward_decoder(self, x: torch, downsamples: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: encoded feature map - (B, L, C)
        :param downsamples: down-sampled feature maps - (3, (B, L, C))
        :return: output tensor - (B, num_classes, img_size[0], img_size[1])
        """
        return self.decoder(x, downsamples)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input image tensor - (B, C, H, W)
        :return: output probabilities - (B, num_classes, img_size[0], img_size[1])
        """
        x, downsamples = self.forward_encoder(x)
        x = self.forward_bottleneck(x)
        x = self.forward_decoder(x, downsamples[::-1])
        return x
