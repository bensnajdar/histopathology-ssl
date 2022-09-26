from typing import Callable, Optional
from cbmi_utils.pytorch.datasets.kather import Kather224x224, Kather96x96
import cbmi_utils.pytorch.datasets.kather as kather
from cbmi_utils.pytorch.datasets.pcam import PatchCamelyon
from cbmi_utils.pytorch.datasets.wsss4luad import WSSS4LUAD96x96
from cbmi_utils.pytorch.datasets.icpr import ICPR96x96Balanced, ICPR96x96Unbalanced
from cbmi_utils.pytorch.datasets.midog import Midog224x224
from cbmi_utils.pytorch.datasets.colossal import ColossalSet224x224
from cbmi_utils.pytorch.datasets.crush import Crush96x96
from cbmi_utils.pytorch.datasets.tcga import tcga_h5_224_split, tcga_get_norm_values_split
from cbmi_utils.pytorch.datasets.lizard import LizardClassification
from .simsiam import SimSiamWideResNetPAWs, SimSiamResNetPAWs, SimSiamPT
from .simtriplet import SimTriplet, SimTripletPT, SimTripWideResNetPAWs

DS_NOT_FOUND = 'not found -- only following sets are implemented Kather, PatchCamelyon, WSSS4LUAD, ICPR, ICPR-BAL, MIDOG'


def init_model(method, encoder, feature_dim, pred_dim):
    if method == 'simsiam':
        if 'wide' in encoder:
            return SimSiamWideResNetPAWs(feature_dim=feature_dim, pred_dim=pred_dim, encoder=encoder)
        elif 'efficient' in encoder:
            return SimSiamPT(base_encoder=encoder, feature_dim=feature_dim, pred_dim=pred_dim)
        else:
            return SimSiamResNetPAWs(feature_dim=feature_dim, pred_dim=pred_dim, encoder=encoder)
    if method == 'simtriplet':
        if 'efficient' in encoder:
            return SimTripletPT(base_encoder=encoder, feature_dim=feature_dim, pred_dim=pred_dim)
        elif 'wide' in encoder:
            return SimTripWideResNetPAWs(feature_dim=feature_dim, pred_dim=pred_dim, encoder=encoder)
        else:
            return SimTriplet(feature_dim=feature_dim, pred_dim=pred_dim, encoder=encoder)
    else:
        raise NotImplementedError(f"Can't initialize a model with method {method}")


def init_data(ds: str, sub_set: str = 'train', transform: Optional[Callable] = None, split_size: Optional[float] = None, num_samples: int = 1):
    assert sub_set in ['train', 'valid', 'test']

    ds = ds.lower()
    if ds == 'kather224':
        dataset = Kather224x224.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'kather96':
        dataset = Kather96x96.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif "kather_h5" in ds:
        dataset = kather.__dict__[ds](sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif "kather_if" in ds:
        dataset = kather.__dict__[ds](sub_set=sub_set, transform=transform)
    elif ds == 'patchcamelyon':
        dataset = PatchCamelyon.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'wsss4luad':
        dataset = WSSS4LUAD96x96.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'icpr':
        dataset = ICPR96x96Unbalanced.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'icpr-bal':
        dataset = ICPR96x96Balanced.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'midog':
        dataset = Midog224x224.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'colossal':
        dataset = ColossalSet224x224.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'crush96':
        dataset = Crush96x96.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'tcga':
        dataset = tcga_h5_224_split(sub_set=sub_set, transform=transform, num_samples=num_samples)
    elif ds == 'lizard':
        dataset = LizardClassification.from_avocado(sub_set=sub_set, transform=transform, num_samples=num_samples)
    else:
        raise NotImplementedError(f'{ds} {DS_NOT_FOUND}')

    if split_size is not None and split_size != 1.0:
        dataset = dataset.get_split(split_size)
        dataset.classes = dataset.dataset.classes
    return dataset


def get_norm_values(ds: str):
    ds = ds.lower()
    if any(substr in ds for substr in ("kather_h5_224", "kather224", "kather_if_224")):
        mean, std = Kather224x224.normalization_values()
    elif any(substr in ds for substr in ("kather_h5_96", "kather96")):
        mean, std = Kather96x96.normalization_values()
    elif ds == 'patchcamelyon':
        mean, std = PatchCamelyon.normalization_values()
    elif ds == 'icpr':
        # TODO use from cbmi_utils
        # mean, std = ICPR96x96Unbalanced.normalization_values()
        mean = [0.7815, 0.4431, 0.6381]
        std = [0.1396, 0.1893, 0.1483]
    elif ds == 'icpr-bal':
        # TODO use from cbmi_utils
        # mean, std = ICPR96x96Balanced.normalization_values()
        mean = [0.7802, 0.4520, 0.6357]
        std = [0.1533, 0.2005, 0.1622]
    elif ds == 'midog':
        mean, std = Midog224x224.normalization_values()
    elif ds == 'colossal':
        mean, std = ColossalSet224x224.normalization_values()
    elif ds == 'crush96':
        mean, std = Crush96x96.normalization_values()  # to-do correct for crush
    elif ds == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif ds == 'tcga':
        mean, std = tcga_get_norm_values_split(tissue_type='all', sub_set='train')
    elif ds == 'lizard':
        mean, std = LizardClassification.normalization_values()
    else:
        raise NotImplementedError(f'Request of normalization constants of {ds} is not implemented!')

    return mean, std


def get_dataset_img_size(ds: str):
    ds_with_size_224 = ('kather224', 'midog', 'colossal', 'imagenet', 'kather_h5_224', 'tcga', 'lizard')
    ds_with_size_96 = ('patchcamelyon', 'icpr', 'icpr-bal', 'wss4luad', 'kather96', 'kather_h5_96' 'crush96')

    ds = ds.lower()
    if any(substr in ds for substr in ds_with_size_224):
        return 224
    elif any(substr in ds for substr in ds_with_size_96):
        return 96
    else:
        raise NotImplementedError(f'{ds} {DS_NOT_FOUND}')
