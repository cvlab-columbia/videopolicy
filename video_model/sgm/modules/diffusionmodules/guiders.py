import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange, repeat

from ...util import append_dims, default

import pdb

logpy = logging.getLogger(__name__)


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        pass


class VanillaCFG(Guider):
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class IdentityGuider(Guider):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out


class LinearPredictionGuider(Guider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        num_pose_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames
        self.num_pose_frames = num_pose_frames
        self.video_scale = torch.linspace(min_scale, max_scale, num_frames).unsqueeze(0)
        self.pose_scale = torch.linspace(min_scale, max_scale, num_pose_frames).unsqueeze(0)

        additional_cond_keys = default(additional_cond_keys, [])
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def __call__(self, x: dict, sigma: dict) -> dict:

        video_output = self._forward(x['video_output'], self.video_scale, self.num_frames)
        pose_output = self._forward(x['pose_output'], self.pose_scale, self.num_pose_frames)

        guided = {
            'video_output': video_output,
            'pose_output': pose_output
        }

        return guided

    def _forward(self, x, scale, num_frames):
        x_u, x_c = x.chunk(2)

        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
        scale = repeat(scale, "1 t -> b t", b=x_u.shape[0])
        scale = append_dims(scale, x_u.ndim).to(x_u.device)

        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")

    def prepare_inputs(
        self, x: dict, s: dict, c: dict, uc: dict
    ) -> Tuple[dict, dict, dict]:
        c_out = dict()
        for k in c:     # c is the dict dict_keys(['crossattn', 'vector', 'concat'])
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:  # self.additional_cond_keys is empty
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        x_cloned = {key: value.clone() for key, value in x.items()}
        s_cloned = {key: value.clone() for key, value in s.items()}

        x_cloned['noised_video_input'] = torch.cat([x_cloned['noised_video_input']] * 2)
        x_cloned['noised_pose_input'] = torch.cat([x_cloned['noised_pose_input']] * 2)
        
        s_cloned['video_sigmas'] = torch.cat([s_cloned['video_sigmas']] * 2)
        s_cloned['pose_sigmas'] = torch.cat([s_cloned['pose_sigmas']] * 2)

        return x_cloned, s_cloned, c_out
