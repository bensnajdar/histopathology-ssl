import torch
# noinspection PyPackageRequirements
import torchvision.transforms.functional as functional
# noinspection PyPackageRequirements
from torchvision import transforms as torch_transforms
from typing import Sequence, Tuple, Optional
import random


# copied from: https://github.com/pytorch/vision/issues/566#issuecomment-535854734
class RotateTransform:
    def __init__(self, angles: Sequence[int], use_flip=True):
        self.angles = angles
        self.flip = None
        if use_flip:
            self.flip = torch_transforms.RandomHorizontalFlip(p=0.5)

    def __call__(self, x):
        angle = random.choice(self.angles)
        if angle:
            return functional.rotate(x, angle)
        if self.flip:
            x = self.flip(x)
        return x


class RandomGaussianBlur:
    def __init__(self, kernel_size, p=0.5, sigma=(0.1, 2.0)):
        self.blur = torch_transforms.RandomApply(
            [torch_transforms.GaussianBlur(kernel_size, sigma)],
            p=p
        )

    def __call__(self, x):
        return self.blur(x)


class Noise:
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        return x + torch.randn_like(x) * self.strength


class ColorDistort:
    def __init__(self, jitter_strength, jitter_p=0.8, grayscale_p=0.2):
        self.grayscale = torch_transforms.RandomGrayscale(p=grayscale_p)
        self.color_jitter = torch_transforms.RandomApply(
            [torch_transforms.ColorJitter(
                brightness=0.8*jitter_strength,
                saturation=0.8*jitter_strength,
                contrast=0.8*jitter_strength,
                hue=0.2*jitter_strength,
            )],
            p=jitter_p
        )

    def __call__(self, x):
        return self.grayscale(self.color_jitter(x))


class Clamp:
    def __call__(self, x):
        return x.clamp(0.0, 1.0)


class Cutout:
    def __init__(self, size: Optional[Tuple[int, int]] = None, color: float = 0.5, quadratic: bool = True):
        """
        Creates a new Cutout augmentation.

        :param size: A tuple (min, max) defining the size of the cutout. The actual size of the cutout is than randomly
                     sampled for each image (min <= size <= max). If not given, it defaults to (0, img_size).
        :param color: The brightness of the gray color that is used for the cutout. 0.0 means black and 1.0 means white.
                      Defaults to 0.5.
        :param quadratic: Whether the cutout should be quadratic. Defaults to True.
        """
        self.size = size
        self.color = color
        self.quadratic = quadratic

    def __call__(self, img):
        size = self.size
        if size is None:
            size = (0, min(img.size()[1], img.size()[2]))
        size_y = random.randint(size[0], size[1])
        pos_y = random.randint(0, img.size()[1] - size_y)
        if self.quadratic:
            size_x = size_y
        else:
            size_x = random.randint(size[0], size[1])
        pos_x = random.randint(0, img.size()[2] - size_x)
        img = img.detach().clone()
        img[:, pos_y:pos_y+size_y, pos_x:pos_x+size_x] = self.color
        return img
