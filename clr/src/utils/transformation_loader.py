from torchvision import transforms as torch_transforms
from data.transforms import RandomGaussianBlur, RotateTransform, Noise, Clamp, ColorDistort, Cutout

DEFAULT_COLOR_JITTER_BRIGHTNESS = 0.2
DEFAULT_COLOR_JITTER_SATURATION = 0.2
DEFAULT_COLOR_JITTER_HUE = 0.2
DEFAULT_RANDOM_RESIZED_CROP_SCALE = (0.08, 1.0)


def _entry_to_transform(entry, slide_size):
    if entry == 'color_distort':
        return ColorDistort(
            jitter_strength=1.0,
            jitter_p=0.8,
            grayscale_p=0.2,
        )
    elif entry == 'random_crop':
        return torch_transforms.RandomResizedCrop(
                size=slide_size,
                scale=DEFAULT_RANDOM_RESIZED_CROP_SCALE
            )
    elif entry == 'cutout':
        return Cutout(size=(40, 50))
    elif entry == 'blur':
        return RandomGaussianBlur(kernel_size=23, p=0.5)
    elif entry == 'rotate':
        return RotateTransform([0, 90, 180, 270])
    elif entry == 'noise':
        return Noise(strength=0.2)
    elif entry == 'affine':
        return torch_transforms.RandomAffine(
            degrees=(-180, 180), translate=None,
            scale=(0.7, 1.3), shear=(-10, 10, -10, 10)
        )
    elif entry == 'color_jitter':
        raise ValueError('color jitter was replaced by color_distort')
    elif entry == 'color_drop':
        raise ValueError('color drop was replaced by color_distort')
    else:
        raise ValueError('Could not load transform "{}"'.format(entry))


def description_to_transform(augmentations, slide_size, use_rotation, use_clamp, image_rescale_size):
    transforms_list = [torch_transforms.ToTensor()]
    if use_rotation:
        transforms_list.append(RotateTransform([0, 90, 180, 270]))
    for entry in augmentations:
        if use_rotation and entry == 'rotate':
            raise AssertionError('got rotate as augmentation and entry')
        transforms_list.append(_entry_to_transform(entry, slide_size))

    if use_clamp:
        transforms_list.append(Clamp())

    if image_rescale_size is not None:
        transforms_list.append(torch_transforms.Resize(image_rescale_size))

    return torch_transforms.Compose(transforms_list)


def make_unique(l):
    unique = []
    for i in l:
        if i not in unique:
            unique.append(i)
    return unique


def load_transforms(augmentations, slide_size, use_rotation=False, use_clamp=True, image_rescale_size=None):
    augmentations = make_unique(augmentations)
    return [
        description_to_transform(augmentations, slide_size, use_rotation, use_clamp, image_rescale_size),
        description_to_transform(augmentations, slide_size, use_rotation, use_clamp, image_rescale_size)
    ]


if __name__ == '__main__':
    transforms = load_transforms(('color_distort', 'random_crop', 'blur', 'rotate', 'noise', 'affine'), (244, 244))
    for transform in transforms:
        print(transform)
