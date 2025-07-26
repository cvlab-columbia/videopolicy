from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser

import numpy as np
from einops import rearrange
from rich import print

import torch.distributed as dist

import pdb


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        harmonize_sigmas: bool = False,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        use_action_loss: bool = False,
    ):
        super().__init__()

        self.harmonize_sigmas = harmonize_sigmas
        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

        self.use_action_loss = use_action_loss

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: Dict,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict, # {'crossattn': tensor[25, 1, 1024] x∈[-6.828, 6.785] μ=0.012 σ=0.533, 'vector': tensor[25, 768] x∈[-1.000, 1.000] μ=0.416 σ=0.572, 'concat': tensor[25, 4, 40, 56] x∈[-23.484, 24.769] μ=-0.078 σ=5.810}
        input: Dict,    # tensor[25, 4, 40, 56] x∈[-5.240, 5.343] μ=0.099 σ=1.147
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        # input = (28, 4, 32, 48) of float32 in [~-10.1, ~8.8].
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        video_sigmas = self.sigma_sampler(input['video'].shape[0]).to(input['video'])   # returns torch.Size([b*25])
        
        # NOTE: B and T dimensions are combined in tensors throughout most of the codebase.
        # old_sigmas = (28) of float32 in [~0.23, ~137.4] with all unique random values.

        # Fix 2/14: noise levels should be consistent across video frames!
        if self.harmonize_sigmas:
            old_sigmas = video_sigmas
            r_sigmas = rearrange(
                old_sigmas, "(b t) ... -> b t ...",
                t=additional_model_inputs['num_video_frames'])
            video_sigmas = r_sigmas[..., 0:1].broadcast_to(r_sigmas.shape).reshape(old_sigmas.shape)  # use the first sigma of a batch across the batch

        # new_sigmas = (28) of float32 with B sequences of T repeating values.
        pose_sigmas = video_sigmas.clone()
        pose_sigmas = rearrange(
                pose_sigmas, "(b t) ... -> b t ...",
                t=additional_model_inputs['num_video_frames'])
        # pose_sigmas = pose_sigmas[:, :additional_model_inputs['num_pose_frames']]
        pose_sigmas = pose_sigmas[:, :1].repeat(1,additional_model_inputs['num_pose_frames'] )
        pose_sigmas = rearrange(                                # returns tensor[b*8]
                pose_sigmas, "b t ... -> (b t) ...",
                t=additional_model_inputs['num_pose_frames'])
        
        video_noise = torch.randn_like(input['video'])     # return torch.Size([b*25, 4, 40, 56])
        pose_noise = torch.randn_like(input['pose'])        # return tensor[b*8, 6]
        
        video_sigmas_bc = append_dims(video_sigmas, input['video'].ndim)     # return torch.Size([b*25, 1, 1, 1])
        noised_video_input = self.get_noised_input(video_sigmas_bc, video_noise, input['video'])

        pose_sigmas_bc = append_dims(pose_sigmas, input['pose'].ndim)       # return tensor[b*8, 1]
        noised_pose_input = self.get_noised_input(pose_sigmas_bc, pose_noise, input['pose'])    # return tensor[b*8, 6]

        noised_input = {
            'noised_video_input': noised_video_input,
            'noised_pose_input': noised_pose_input
        }

        sigmas = {
            'video_sigmas': video_sigmas,
            'pose_sigmas': pose_sigmas
        }

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        
        video_w = append_dims(self.loss_weighting(video_sigmas), input['video'].ndim)    # return torch.Size([b*25, 1, 1, 1])
        video_loss_value = self.get_loss(model_output['video_output'], input['video'], video_w)

        if self.use_action_loss:

            pose_w = append_dims(self.loss_weighting(pose_sigmas), input['pose'].ndim)
            pose_loss_value = self.get_loss(model_output['pose_output'], input['pose'], pose_w)

            loss_value = torch.cat((video_loss_value, pose_loss_value), dim=0)
        else:
            loss_value = video_loss_value

        return loss_value

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1  # [25, 4, 40, 56] -> [25, 8960] -> [25]
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")