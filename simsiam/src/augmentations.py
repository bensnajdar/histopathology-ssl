from typing import Optional
import torchvision.transforms as transforms
from .init import get_dataset_img_size, get_norm_values


__all__ = [
    'light_stack',
    'plain',
    'moco_v2',
    'medium_stack'
]


# def light_stack_96(dataset, normalize: bool = False):
#     img_size = int(96) #get_dataset_img_size(dataset)

#     transform_pre_norm = transforms.Compose([
#         transforms.RandomApply([
#             transforms.ColorJitter(
#                 brightness=[0.9, 1.1],
#                 contrast=0,
#                 saturation=[0.7, 1.8],
#                 hue=0
#             )  # not strengthened
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#     ])

#     transform_post_norm = transforms.Compose([
#         transforms.RandomApply([
#             transforms.RandomCrop(img_size * .9),
#             transforms.Resize(img_size)
#         ], p=0.5),
#         transforms.RandomApply([
#             transforms.Resize(int(img_size * (2**0.5) + 0.9999)),
#             transforms.RandomRotation(180)
#         ], p=0.8),
#         transforms.CenterCrop(img_size),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.5)
#     ])

#     if normalize:
#         mean, std = get_norm_values(dataset)
#         return transforms.Compose([transforms.ToTensor(),transforms.Resize(int(96)),transform_pre_norm, transforms.Normalize(mean, std), transform_post_norm])
#     else:
#         return transforms.Compose([transforms.ToTensor(),transforms.Resize(int(96)),transform_pre_norm, transform_post_norm])

# def plain_96(dataset, normalize: bool = False):
#     img_size = int(96) #get_dataset_img_size(dataset)

#     augmentations = [transforms.ToTensor()]
#     augmentations.append(transforms.Resize(img_size))
#     if normalize:
#         mean, std = get_norm_values(dataset)
#         augmentations.append(transforms.Normalize(mean, std))
#     augmentations.append(transforms.CenterCrop(img_size))
#     return transforms.Compose(augmentations)

def light_stack(dataset, normalize: bool = False, img_size: Optional[int] = None):
    if img_size is None:
        img_size = get_dataset_img_size(dataset)

    transform_pre_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=[0.9, 1.1],
                contrast=0,
                saturation=[0.7, 1.8],
                hue=0
            )  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])

    transform_post_norm = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomCrop(img_size * .9),
            transforms.Resize(img_size)
        ], p=0.5),
        transforms.RandomApply([
            transforms.Resize(int(img_size * (2**0.5) + 0.9999)),
            transforms.RandomRotation(180)
        ], p=0.8),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])

    if normalize:
        mean, std = get_norm_values(dataset)
        return transforms.Compose([transform_pre_norm, transforms.Normalize(mean, std), transform_post_norm])
    else:
        return transforms.Compose([transform_pre_norm, transform_post_norm])


def medium_stack(dataset, normalize: bool = False, img_size: Optional[int] = None):
    if img_size is None:
        img_size = get_dataset_img_size(dataset)

    transform_pre_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])

    transform_post_norm = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomCrop(img_size * .9),
            transforms.Resize(img_size)
        ], p=0.5),
        transforms.RandomApply([
            transforms.Resize(int(img_size * (2**0.5) + 0.9999)),
            transforms.RandomRotation(180)
        ], p=0.8),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=img_size // 20 * 2 + 1, sigma=(0.1, 2.0))
        ], p=0.5)
    ])

    if normalize:
        mean, std = get_norm_values(dataset)
        return transforms.Compose([transform_pre_norm, transforms.Normalize(mean, std), transform_post_norm])
    else:
        return transforms.Compose([transform_pre_norm, transform_post_norm])


def plain(dataset, normalize: bool = False, img_size: Optional[int] = None):
    if img_size is None:
        img_size = get_dataset_img_size(dataset)

    augmentations = [transforms.ToTensor()]
    if normalize:
        mean, std = get_norm_values(dataset)
        augmentations.append(transforms.Normalize(mean, std))
    augmentations.append(transforms.Resize(img_size))
    augmentations.append(transforms.CenterCrop(img_size))
    return transforms.Compose(augmentations)


def moco_v2(dataset, normalize: bool = False, img_size: Optional[int] = None):
    if img_size is None:
        img_size = get_dataset_img_size(dataset)

    augmentations = [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=img_size // 20 * 2 + 1, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.RandomHorizontalFlip()
    ]

    if normalize:
        mean, std = get_norm_values(dataset)
        augmentations.append(transforms.Normalize(mean, std))

    return transforms.Compose(augmentations)
