"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.models.splatfacto import (
    RGB2SH,
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)


@dataclass
class NVSMask3dModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: NVSMask3dModel)


class NVSMask3dModel(SplatfactoModel):
    """Template Model."""

    config: NVSMask3dModelConfig

    def populate_modules(self):
        super().populate_modules()
    #     self.gauss_params["means"].requires_grad = False

    # def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
    #     # 排除means参数
    #     param_groups = {
    #         name: [param]
    #         for name, param in self.gauss_params.items()
    #         if name != "means"
    #     }
    #     return param_groups
    # 
    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
