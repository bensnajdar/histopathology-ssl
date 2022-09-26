"""
Lizard Dataset described here: https://arxiv.org/pdf/2108.11195.pdf
"""
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import h5py
from PIL import Image

import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

DATASET_LOCATION = Path('/data/ldap/histopathologic/original_read_only/Lizard')
LABELS_DIR = Path('labels/Labels')

LABELS = ['Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Eosinophil', 'Connective tissue']
POSSIBLE_TISSUE_TYPES = ['pannuke', 'glas', 'consep', 'crag', 'dpath']


def imread(image_path):
    with Image.open(image_path) as image:
        # noinspection PyTypeChecker
        return np.array(image)


class Snapshot:
    """
    A Snapshot stands for a subregion in an image. It only saves positional information and no data itself
    (no image or label data):
    - The sample name: if the image file is named "images1/consep_10.png" the sample name is consep_10.
    - The image directory: either images1 or images2
    - The position of the subimage inside the image
    """

    def __init__(self, sample_name: str, image_directory: str, position: np.ndarray):
        self.sample_name = sample_name
        self.image_directory = Path(image_directory)
        self.position = position

    def get_image_path(self) -> Path:
        return self.image_directory / f'{self.sample_name}.png'

    def get_label_path(self) -> Path:
        return LABELS_DIR / f'{self.sample_name}.mat'

    def __repr__(self):
        return f'Snapshot(sample_name={self.sample_name} position=(y={self.position[0]}, x={self.position[1]})'


class ClassificationSnapshot(Snapshot):
    def __init__(self, sample_name: str, image_directory: str, position: np.ndarray, label: int):
        super().__init__(sample_name, image_directory, position)
        self.label = label


class LizardClassificationDataset(Dataset):
    """
    This dataset creates a classification sample for every annotated nuclei of the Lizard Dataset. The nuclei to
    classify is centered in the image.
    """
    def __init__(
        self, data_dir: Path, image_size: np.ndarray, snapshots: List[ClassificationSnapshot], use_cache: bool = False
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.snapshots = snapshots
        self.use_cache = use_cache
        self.image_cache = {}

    @staticmethod
    def from_data_dir(
        data_dir: Path, image_size: np.ndarray, min_label_limit: int = 0, tissue_types: List[str] or None = None,
        use_cache: bool = False
    ):
        """
        Args:
            data_dir: The directory to search images and labels in. This directory should have the three subdirectories
                      "labels/Labels", "images1", "images2"
            image_size: The size of the images returned by __getitem__ as [height, width]
            min_label_limit: All labels that appear less than <min_label_limit> times are filtered out.
                             If set to 0 (the default), all labels are used.
                             Be aware that this alters the meaning of a class index.
                             If a class with index I is filtered out, all class indices > I are reduced by 1.
            tissue_types: The tissue_types that are allowed for this dataset. Others are filtered out.
            use_cache: Whether to cache the loaded images
        """
        # check tissue types
        if tissue_types:
            for tissue_type in tissue_types:
                if tissue_type not in POSSIBLE_TISSUE_TYPES:
                    raise ValueError(
                        '"{}" is not a valid tissue type. Valid tissue types are: {}'.format(
                            tissue_type, POSSIBLE_TISSUE_TYPES
                        )
                    )

        snapshots = LizardClassificationDataset._define_snapshots(
            data_dir, image_size, min_label_limit, tissue_types
        )
        return LizardClassificationDataset(data_dir, image_size, snapshots, use_cache)

    @staticmethod
    def split(dataset, split_ratio: float, seed: int = 42) -> Tuple:
        num_snapshots = len(dataset)
        snapshots_dict = {}  # maps sample_names to snapshots
        for snapshot in dataset.snapshots:
            if snapshot.sample_name not in snapshots_dict:
                snapshots_dict[snapshot.sample_name] = []
            snapshots_dict[snapshot.sample_name].append(snapshot)
        split_index = int(num_snapshots * split_ratio)
        first_set = []
        second_set = []
        snapshots_lists_sorted = sorted(snapshots_dict.values(), key=lambda sl: sl[0].sample_name)
        random.Random(seed).shuffle(snapshots_lists_sorted)
        for snapshot_list in snapshots_lists_sorted:
            if len(first_set) >= split_index:
                second_set.extend(snapshot_list)
            else:
                first_set.extend(snapshot_list)
        return (
            LizardClassificationDataset(dataset.data_dir, dataset.image_size, first_set, dataset.use_cache),
            LizardClassificationDataset(dataset.data_dir, dataset.image_size, second_set, dataset.use_cache),
        )

    @staticmethod
    def _filter_label_limit(snapshots, min_label_limit: int) -> List[ClassificationSnapshot]:
        """
        Filters out snapshots with rare labels and changes labels.
        """
        # filter labels
        # count labels
        label_count = [0] * len(LABELS)
        for snapshot in snapshots:
            label_count[snapshot.label] += 1
        label_assignment = {}  # maps old labels to new labels

        # create label assignment
        next_available_label = 0
        for label, label_c in enumerate(label_count):
            if label_c >= min_label_limit:
                label_assignment[label] = next_available_label
                next_available_label += 1

        filtered_snapshots = []
        for snapshot in snapshots:
            if label_count[snapshot.label] >= min_label_limit:  # if label of snapshot is sufficiently present
                snapshot.label = label_assignment[snapshot.label]
                filtered_snapshots.append(snapshot)

        return filtered_snapshots

    @staticmethod
    def _define_snapshots(
        data_dir: Path, image_size: np.ndarray, min_label_limit: int, tissue_types: List[str] or None
    ) -> List[ClassificationSnapshot]:
        """
        Returns a list of snapshots to use for this dataset. Images will be searched in '<data_dir>/images1/*.png' and
        '<data_dir>/images2/*.png'.
        """
        image1_files = sorted([f for f in (data_dir / 'images1').iterdir() if str(f).endswith('.png')])
        image2_files = sorted([f for f in (data_dir / 'images2').iterdir() if str(f).endswith('.png')])
        image_files = image1_files + image2_files

        snapshots = []
        for image_file in image_files:
            snapshots.extend(
                LizardClassificationDataset.snapshots_from_image_file(image_file, image_size, tissue_types)
            )

        # filter labels
        if min_label_limit:
            snapshots = LizardClassificationDataset._filter_label_limit(snapshots, min_label_limit)

        return snapshots

    @staticmethod
    def snapshots_from_image_file(
        filename: Path, image_size: np.ndarray, tissue_types: List[str] or None
    ) -> List[ClassificationSnapshot]:
        sample_name = filename.stem
        # if tissue type is not mentioned -> no snapshots
        if tissue_types is not None and sample_name.split(sep='_')[0] not in tissue_types:
            return []
        label_path = filename.parent.parent / 'labels' / 'Labels' / '{}.mat'.format(sample_name)
        image_dir = filename.parent.stem

        # load label data
        label_data = sio.loadmat(str(label_path))
        inst_map = label_data['inst_map']
        classes = label_data['class']
        centroids = label_data['centroid']
        # noinspection PyTypeChecker
        nuclei_id: list = np.squeeze(label_data['id']).tolist()
        unique_values = np.unique(inst_map).tolist()[1:]  # remove 0 from begin; I checked and 0 is always at begin

        snapshots = []
        for value in unique_values:
            idx = nuclei_id.index(value)

            # checked classes. It is always a list with exactly one element.
            # We subtract 1 as 0 is background label and not present in dataset
            class_label = classes[idx][0] - 1
            centroid = centroids[idx]
            x, y = centroid
            snapshot_position = np.array([y, x], dtype=int) - (image_size // 2)
            snapshot = ClassificationSnapshot(sample_name, image_dir, snapshot_position, class_label)
            snapshots.append(snapshot)

        return snapshots

    def read_image(self, image_path):
        if self.use_cache:
            image = self.image_cache.get(image_path)
            if image is None:
                image = imread(str(image_path))
                self.image_cache[image_path] = image
        else:
            image = imread(str(image_path))
        return image

    def get_padded_subimage(self, snapshot) -> np.ndarray:
        image_path = self.data_dir / snapshot.get_image_path()
        image = self.read_image(str(image_path))

        half_image_size = self.image_size // 2
        padded_image = np.pad(image, [half_image_size, half_image_size, (0, 0)])

        position = snapshot.position + half_image_size  # the subimage position in the padded image

        return padded_image[
            position[0]:position[0]+self.image_size[0],
            position[1]:position[1]+self.image_size[1]
        ]

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Gets the dataset sample with the given index. Every sample contains N instances of nuclei.
        A sample is a dictionary with the following keys:
          - image: An image with shape [height, width, 3]
          - labels: A List[N] of class numbers 0 <= label <= 5
          - boxes: A List[N, 4] containing the bounding boxes of the sample as [top, left, bottom, right]
        """
        snapshot = self.snapshots[index]

        subimage = self.get_padded_subimage(snapshot)

        return {
            'image': subimage,
            'label': snapshot.label,
        }

    def __len__(self):
        return len(self.snapshots)


def show_images(images):
    new_images = []
    for index, img in enumerate(images):
        new_images.append(img)
        if index != 0:
            new_images.append(np.zeros((images[0].shape[0], 1, 3), dtype=np.uint8))
    # image = np.concatenate(new_images, axis=1)
    # imshow(image)
    plt.show()


def show_label_counts(dataset):
    counter = [0] * 6
    for sample in dataset:
        counter[sample['label']] += 1
    print(counter)


def save_dataset_as_h5(dataset: LizardClassificationDataset, target_location):
    image_list = []
    label_list = []

    for index in tqdm(range(len(dataset)), total=len(dataset)):
        sample = dataset[index]
        label_list.append(sample['label'])
        image_list.append(sample['image'].astype(np.uint8))

    os.makedirs(Path(target_location).parent, exist_ok=True)
    hf = h5py.File(target_location, 'w')
    hf.create_dataset('image', data=image_list)
    hf.create_dataset('label', data=label_list)
    hf.close()


def convert_datasets():
    dataset = LizardClassificationDataset.from_data_dir(
        data_dir=DATASET_LOCATION,
        image_size=np.array([224, 224]),
        min_label_limit=10000,
        # tissue_types=['consep'],
        use_cache=True,
    )
    # show_label_counts(dataset)
    train_eval_set, test_set = dataset.split(dataset, 0.85)
    train_set, eval_set = train_eval_set.split(train_eval_set, 0.82)  # this leads to another 15% for validation
    save_dataset_as_h5(train_set, '/data/ldap/histopathologic/processed_read_only/lizard_classification/train.h5')
    save_dataset_as_h5(eval_set, '/data/ldap/histopathologic/processed_read_only/lizard_classification/eval.h5')
    save_dataset_as_h5(test_set, '/data/ldap/histopathologic/processed_read_only/lizard_classification/test.h5')
    show_label_counts(train_set)
    show_label_counts(eval_set)
    show_label_counts(test_set)
    print('train set:', len(train_set))
    print('eval set:', len(eval_set))
    print('test set:', len(test_set))


def main():
    convert_datasets()


if __name__ == '__main__':
    main()
