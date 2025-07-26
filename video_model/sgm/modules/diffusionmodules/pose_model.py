from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch as th
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
import os
from einops import rearrange
import math

from ...util import append_dims

import pdb

def f(tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    shape = tuple(tensor.shape)
    grad_fn = tensor.grad_fn if tensor.grad_fn else "None"
    return f"tensor{shape} x∈[{min_val:.3f}, {max_val:.3f}] μ={mean:.3f} σ={std:.3f} grad {grad_fn}"

class PoseEncoderSequential(nn.Sequential):
    def forward(self, x, emb):
        for module in self:
            x = module(x, emb)
        return x

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> SiLU
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        emb_channels=1280,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        diffusion_step_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("Total number of parameters in action diffusion model: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            emb: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        emb: (B,1280), diffusion timestep
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        global_feature = self.diffusion_step_encoder(emb) # returns torch.Size([b, 256])
        # pdb.set_trace()
        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

class Conv3DSimple(nn.Conv3d):
    def __init__(
        self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, temporal_stride: int = 1, stride: int = 1, padding: int = 1
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(temporal_stride, stride, stride),
            padding=padding,
            bias=False,
        )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        emb_planes: int = 1280,
        conv_builder: Callable[..., nn.Module] = Conv3DSimple,
        temporal_stride: int = 1,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        num_groups: int = 10,
    ) -> None:

        super().__init__()

        bottleneck_planes = planes // self.expansion

        self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_planes,
                    bottleneck_planes,
                ),
            )

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, bottleneck_planes, kernel_size=1, bias=False), nn.GroupNorm(num_groups, bottleneck_planes), nn.SiLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(bottleneck_planes, bottleneck_planes, bottleneck_planes, temporal_stride, stride), nn.GroupNorm(num_groups, bottleneck_planes), nn.SiLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(bottleneck_planes, planes, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups*self.expansion, planes),
        )
        self.silu = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        residual = x

        out = self.conv1(x)

        emb_out = self.emb_layers(emb).type(out.dtype)

        emb_out = append_dims(emb_out, out.ndim)
        emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
        # print(torch.allclose(emb_out[:,:,3:4,:,:], emb_out[:,:,4:5,:,:], rtol=0.001, atol=0.001))
        emb_out = emb_out[:, :, :out.shape[2], :, :]    # reduce dimension in temporal channel

        out = out + emb_out

        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.silu(out)

        return out

class BottleneckExp2(Bottleneck):
    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        emb_planes: int = 1280,
        conv_builder: Callable[..., nn.Module] = Conv3DSimple,
        temporal_stride: int = 1,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        num_groups: int = 10,
    ) -> None:
        super().__init__(
            inplanes,
            planes,
            emb_planes,
            conv_builder,
            temporal_stride,
            stride,
            downsample,
            num_groups,
        )

class VideoResNet(nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        conv_maker: Conv3DSimple,
        emb_planes: int = 1280,
        num_groups: int = 10,
        
    ) -> None:
        super().__init__()

        self.layer1_1 = self._make_layer(block, conv_maker, 320, 320, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer2_1_cat = self._make_layer(block, conv_maker, 640, 320, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer2_1 = self._make_layer(block, conv_maker, 640, 640, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer3_1_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer3_1 = self._make_layer(block, conv_maker, 1280, 1280, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer4_1_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer4_1 = self._make_layer(block, conv_maker, 1920, 1280, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer5_1_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer5_1 = self._make_layer(block, conv_maker, 1920, 1280, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer6_1 = self._make_layer(block, conv_maker, 1280, 864, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=8)
        self.layer7_1 = self._make_layer(block, conv_maker, 864, 576, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=6)

        self.layer1_2 = self._make_layer(block, conv_maker, 320, 320, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer2_2_cat = self._make_layer(block, conv_maker, 640, 320, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer2_2 = self._make_layer(block, conv_maker, 640, 640, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer3_2_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer3_2 = self._make_layer(block, conv_maker, 1280, 1280, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer4_2_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer4_2 = self._make_layer(block, conv_maker, 1920, 1280, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer5_2_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer5_2 = self._make_layer(block, conv_maker, 1920, 1280, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer6_2 = self._make_layer(block, conv_maker, 1280, 864, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=8)
        self.layer7_2 = self._make_layer(block, conv_maker, 864, 576, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=6)

        self.layer1_3 = self._make_layer(block, conv_maker, 320, 320, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer2_3_cat = self._make_layer(block, conv_maker, 640, 320, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer2_3 = self._make_layer(block, conv_maker, 640, 640, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer3_3_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer3_3 = self._make_layer(block, conv_maker, 1280, 1280, emb_planes, blocks=2, temporal_stride=1, stride=2, num_groups=num_groups)
        self.layer4_3_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer4_3 = self._make_layer(block, conv_maker, 1920, 1280, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer5_3_cat = self._make_layer(block, conv_maker, 1280, 640, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer5_3 = self._make_layer(block, conv_maker, 1920, 1280, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer6_3 = self._make_layer(block, conv_maker, 1280, 864, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=8)
        self.layer7_3 = self._make_layer(block, conv_maker, 864, 576, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=6)

        self.layer8 = self._make_layer(block, conv_maker, 1728, 1280, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=num_groups)
        self.layer9 = self._make_layer(block, conv_maker, 1280, 960, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=8)
        self.layer10 = self._make_layer(block, conv_maker, 960, 720, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=6)
        self.layer11 = self._make_layer(block, conv_maker, 720, 540, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=5)
        self.layer12 = self._make_layer(block, conv_maker, 540, 420, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=5)
        self.layer13 = self._make_layer(BottleneckExp2, conv_maker, 420, 420, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=7)
        self.layer14 = self._make_layer(BottleneckExp2, conv_maker, 420, 420, emb_planes, blocks=2, temporal_stride=1, stride=1, num_groups=7)
        self.layer15 = self._make_layer(BottleneckExp2, conv_maker, 420, 420, emb_planes, blocks=4, temporal_stride=2, stride=1, num_groups=7)
        self.layer16 = self._make_layer(BottleneckExp2, conv_maker, 420, 420, emb_planes, blocks=4, temporal_stride=2, stride=1, num_groups=7)
        self.layer17 = self._make_layer(BottleneckExp2, conv_maker, 420, 420, emb_planes, blocks=4, temporal_stride=2, stride=1, num_groups=7)

        self.norm = nn.GroupNorm(num_groups=14, num_channels=420)
    
    def forward(self, inputs: List[th.Tensor], emb: th.Tensor, num_pose_frames) -> th.Tensor:

        emb = rearrange(emb, "(b t) ... -> b t ...", t=num_pose_frames).contiguous()

        x1 = self.layer1_1(inputs[0]['frames_1'], emb)
        x_cat = self.layer2_1_cat(inputs[1]['frames_1'], emb)
        x1 = torch.cat([x1, x_cat], dim=1)
        x1 = self.layer2_1(x1, emb)
        x_cat = self.layer3_1_cat(inputs[2]['frames_1'], emb)
        x1 = torch.cat([x1, x_cat], dim=1)
        x1 = self.layer3_1(x1, emb)
        x_cat = self.layer4_1_cat(inputs[3]['frames_1'], emb)
        x1 = torch.cat([x1, x_cat], dim=1)
        x1 = self.layer4_1(x1, emb)
        x_cat = self.layer5_1_cat(inputs[4]['frames_1'], emb)
        x1 = torch.cat([x1, x_cat], dim=1)
        x1 = self.layer5_1(x1, emb)
        x1 = self.layer6_1(x1, emb)
        x1 = self.layer7_1(x1, emb)

        x2 = self.layer1_2(inputs[0]['frames_2'], emb)
        x_cat = self.layer2_2_cat(inputs[1]['frames_2'], emb)
        x2 = torch.cat([x2, x_cat], dim=1)
        x2 = self.layer2_2(x2, emb)
        x_cat = self.layer3_2_cat(inputs[2]['frames_2'], emb)
        x2 = torch.cat([x2, x_cat], dim=1)
        x2 = self.layer3_2(x2, emb)
        x_cat = self.layer4_2_cat(inputs[3]['frames_2'], emb)
        x2 = torch.cat([x2, x_cat], dim=1)
        x2 = self.layer4_2(x2, emb)
        x_cat = self.layer5_2_cat(inputs[4]['frames_2'], emb)
        x2 = torch.cat([x2, x_cat], dim=1)
        x2 = self.layer5_2(x2, emb)
        x2 = self.layer6_2(x2, emb)
        x2 = self.layer7_2(x2, emb)

        x3 = self.layer1_3(inputs[0]['frames_3'], emb)
        x_cat = self.layer2_3_cat(inputs[1]['frames_3'], emb)
        x3 = torch.cat([x3, x_cat], dim=1)
        x3 = self.layer2_3(x3, emb)
        x_cat = self.layer3_3_cat(inputs[2]['frames_3'], emb)
        x3 = torch.cat([x3, x_cat], dim=1)
        x3 = self.layer3_3(x3, emb)
        x_cat = self.layer4_3_cat(inputs[3]['frames_3'], emb)
        x3 = torch.cat([x3, x_cat], dim=1)
        x3 = self.layer4_3(x3, emb)
        x_cat = self.layer5_3_cat(inputs[4]['frames_3'], emb)
        x3 = torch.cat([x3, x_cat], dim=1)
        x3 = self.layer5_3(x3, emb)
        x3 = self.layer6_3(x3, emb)
        x3 = self.layer7_3(x3, emb)
        
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.layer8(x, emb)
        x = self.layer9(x, emb)
        x = self.layer10(x, emb)
        x = self.layer11(x, emb)
        x = self.layer12(x, emb)
        x = self.layer13(x, emb)
        x = self.layer14(x, emb)
        x = self.layer15(x, emb)
        x = self.layer16(x, emb)
        x = self.layer17(x, emb)

        x = self.norm(x)

        return x

    def _make_layer(
        self,
        block: Bottleneck,
        conv_builder: Conv3DSimple,
        inplanes: int,
        planes: int,
        emb_planes: int,
        blocks: int,
        temporal_stride: int,
        stride: int = 1,
        num_groups: int = 10,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or temporal_stride != 1 or inplanes != planes:
            ds_stride = (temporal_stride, stride, stride)
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=ds_stride, bias=False),
                nn.GroupNorm(num_groups*block.expansion, planes),
            )
        layers = []
        layers.append(block(inplanes, planes, emb_planes, conv_builder, temporal_stride, stride, downsample, num_groups))

        for i in range(1, blocks):
            layers.append(block(planes, planes, emb_planes, conv_builder, num_groups=num_groups))

        return PoseEncoderSequential(*layers)
