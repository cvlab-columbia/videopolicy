import torch
import torch.nn as nn
from packaging import version

import pdb

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: dict, t: dict, c: dict, **kwargs
    ) -> torch.Tensor:
        x["video_input"] = torch.cat((x["video_input"], c.get("concat", torch.Tensor([]).type_as(x["video_input"]))), dim=1) # x.shape [25, 4, 40, 56], c['concat'].shape [25, 4, 40, 56], c['crossattn'].shape [25, 1, 1024], c['vector'].shape [25, 768]
        return self.diffusion_model(
            x,  
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )
