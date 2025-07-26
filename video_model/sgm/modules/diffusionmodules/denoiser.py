from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization

from einops import rearrange

import pdb

class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def compute_scaing(self, input, sigma):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)  # c_in shape is torch.Size([28, 1, 1, 1]), c_out shape is torch.Size([28, 1, 1, 1]), c_skip is shape torch.Size([28, 1, 1, 1]), c_noise shape is torch.Size([28])
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        return c_in, c_noise, c_out, c_skip

    def forward(
        self,
        network: nn.Module,     # the default network is OpenAIWrapper
        input: Dict,    # input shape is torch.Size([28, 4, 72, 128])
        sigma: Dict,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:

        video_c_in, video_c_noise, video_c_out, video_c_skip = self.compute_scaing(input['noised_video_input'], sigma['video_sigmas'])
        pose_c_in, pose_c_noise, pose_c_out, pose_c_skip = self.compute_scaing(input['noised_pose_input'], sigma['pose_sigmas'])

        network_input = {
            'video_input': input['noised_video_input'] * video_c_in,
            'pose_input': input['noised_pose_input'] * pose_c_in
        }

        pose_unet_c_noise = rearrange(
                pose_c_noise, "(b t) ... -> b t ...",
                t=additional_model_inputs['num_pose_frames'])
        pose_unet_c_noise = pose_unet_c_noise[:, :1]
        pose_unet_c_noise = rearrange(                                
                pose_unet_c_noise, "b t ... -> (b t) ...",
                t=1)    # correct timestep tensor shape for pose prediction unet

        c_noise = {
            'video_c_noise': video_c_noise,
            'pose_c_noise': pose_c_noise,
            'pose_unet_c_noise': pose_unet_c_noise
        }

        output = network(network_input, c_noise, cond, **additional_model_inputs)
        
        scaled_output = {
            'video_output': output['video_output'] * video_c_out + input['noised_video_input'] * video_c_skip,
            'pose_output': output['pose_output'] * pose_c_out + input['noised_pose_input'] * pose_c_skip
        }

        return scaled_output


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(
            discretization_config
        )
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
