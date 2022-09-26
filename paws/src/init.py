from typing import Callable, Optional
from cbmi_utils.pytorch.datasets.kather import Kather96x96, Kather224x224
from cbmi_utils.pytorch.datasets.pcam import PatchCamelyon
from cbmi_utils.pytorch.datasets.wsss4luad import WSSS4LUAD96x96
from cbmi_utils.pytorch.datasets.icpr import ICPR96x96Balanced, ICPR96x96Unbalanced
from cbmi_utils.pytorch.datasets.midog import Midog224x224
from cbmi_utils.pytorch.datasets.colossal import ColossalSet224x224


DS_NOT_DOUND = 'not found -- only following sets are implemented Kather, PatchCamelyon, WSSS4LUAD, ICPR, ICPR-BAL, MIDOG'



def init_data_cbmi_utils(ds: str, sub_set: str = 'train', transform: Optional[Callable] = None, split_size: Optional[float] = None):
    assert sub_set in ['train', 'valid', 'test']
    
    ds = ds.lower()
    if ds == 'kather':
        dataset = Kather224x224.from_avocado(sub_set=sub_set, transform=transform)
    elif ds == 'kather96x96':
        dataset = Kather96x96.from_avocado(sub_set=sub_set, transform=transform)
    elif ds == 'patchcamelyon':
        dataset = PatchCamelyon.from_avocado(sub_set=sub_set, transform=transform)
    elif ds == 'wsss4luad':
        dataset = WSSS4LUAD96x96.from_avocado(sub_set=sub_set, transform=transform)
    elif ds == 'icpr':
        dataset = ICPR96x96Unbalanced.from_avocado(sub_set=sub_set, transform=transform)
    elif ds == 'icpr-bal':
        dataset = ICPR96x96Balanced.from_avocado(sub_set=sub_set, transform=transform)
    elif ds == 'midog':
        dataset = Midog224x224.from_avocado(sub_set=sub_set, transform=transform)
    elif ds == 'colossal':
        dataset = ColossalSet224x224.from_avocado(transform=transform)
    else:
        raise NotImplementedError(f'{ds} {DS_NOT_DOUND}')

    if split_size is not None and split_size != 1.0:
        dataset = dataset.get_split(split_size)

    return dataset

def get_class_names(ds:str):
    ds = ds.lower()
    if ds == 'kather':
        return ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    elif ds == 'patchcamelyon':
        return ['no_tumor', 'tumor']
    elif ds == 'wsss4luad':
        return ['tumor','stroma','normal']
    elif ds == 'midog':
        return ['non_mitosis', 'mitosis']
    


def get_dataset_img_size(ds: str):
    ds_with_size_224 = ['kather', 'midog','colossal'] 
    ds_with_size_99 = ['patchcamelyon', 'icpr', 'icpr-bal', 'wss4luad']

    ds = ds.lower()   
    if ds in ds_with_size_224:
        return 224
    elif ds in ds_with_size_99:
        return 99
    else:
        raise NotImplementedError(f'{ds} {DS_NOT_DOUND}')