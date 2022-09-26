from typing import List, Any

from torch.utils.data.dataset import Dataset


class AugmentationWrapper(Dataset):
    """
    Wraps a given dataset and adds augmentations to the getitem method.
    The wrapped datasets __getitem__() method has to return a dictionary {<image-key>: image, ...} or a tuple
    (image, label).
    AugmentationWrapper will always return a dictionary. If the wrapped dataset returns a dictionary AugmentationWrapper
    will forward the same content but add new augmented images to it with the keys ['augmented0', 'augmented1', ...].
    In case the wrapped dataset returns a tuple AugmentationWrapper returns a dictionary of the following form:
    {
        'image': input_tuple[0], 'label': input_tuple[1],
        'augmented0': first_augmented_image, 'augmented1': second_augmented_image, ...
    }
    """
    def __init__(
            self, dataset: Dataset, transforms: List[Any], image_key: str = 'image',
            augmented_key: str = 'augmented{index}'
    ):
        """
        Creates a new AugmentationWrapper.
        Args:
            dataset: The dataset to wrap
            transforms: A list of transformations. len(transforms) defines the number of keys added to the
                        result of __getitem__().
            image_key: The key of the wrapped dataset to access the image to augment. Defaults to 'image'
            augmented_key: The key that will be added to the result of __getitem__(). The string {index} will be
                           replaced with the index of the augmentation. Defaults to 'augmented{index}'.
        """
        self.dataset = dataset
        self.transforms = transforms
        self.image_key = image_key
        self.augmented_key = augmented_key

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)

        if isinstance(item, dict):
            try:
                image = item[self.image_key]
            except KeyError:
                raise AssertionError(
                    'item does not contain \"image\" key. If this dataset uses a different key specify it in '
                    'AugmentationWrapper(dataset, image_key=<your-key>)'
                )
            for index, transform in enumerate(self.transforms):
                augmented_key = self.augmented_key.format(index=index)
                item[augmented_key] = transform(image)
        elif isinstance(item, tuple):
            image = item[0]
            label = item[1]
            item = {
                'image': image,
                'label': label,
            }
            for index, transform in enumerate(self.transforms):
                augmented_key = self.augmented_key.format(index=index)
                item[augmented_key] = transform(image)
        else:
            raise TypeError(
                'Can only wrap datasets that return dictionaries or tuples. Found: {}'.format(type(item).__name__)
            )

        return item

    def __len__(self):
        # noinspection PyTypeChecker
        return len(self.dataset)
