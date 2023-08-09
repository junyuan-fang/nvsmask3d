"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from torch import Tensor

from nerfstudio.fields.nerfacto_field import (
    NerfactoField,
)  # for subclassing NerfactoField


class TemplateNerfField(NerfactoField):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images)

    # TODO: Override any potential methods to implement your own field.
    # or subclass from base Field and define all mandatory methods.
