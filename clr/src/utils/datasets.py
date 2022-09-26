from cbmi_utils.pytorch.datasets import *
from cbmi_utils.pytorch.datasets import tcga
from cbmi_utils.pytorch.datasets import lizard


def get_dataset(dataset_name: str, global_parameters, subset: str):
    if dataset_name == 'vgh':
        dataset = vgh_nki.VGHNKI(
            root=global_parameters['vgh_data_location'],
            sub_set=subset,
        )
    elif dataset_name == 'icpr':
        dataset = icpr.ICPR96x48Unbalanced(
            root=global_parameters['icpr_data_location'],
            sub_set=subset,
        )
    elif dataset_name == 'pcam':
        dataset = pcam.PatchCamelyon(
            root=global_parameters['pcam_data_location'],
            sub_set=subset,
        )
    elif dataset_name == 'kather':
        dataset = kather.kather_h5_224_norm_split_90_10(
            sub_set=subset,
        )
    elif dataset_name == 'wsss4luad':
        dataset = wsss4luad.WSSS4LUAD96x96(
            root=global_parameters['wsss4luad_data_location'],
            sub_set=subset,
        )
    elif dataset_name == 'colossal':
        dataset = colossal.ColossalSet224x224(
            root=global_parameters['colossal_data_location'],
            sub_set=subset,
        )
    elif dataset_name == 'crush':
        dataset = crush.Crush96x96(
            root=global_parameters['crush_data_location'],
            sub_set=subset,
        )
    elif dataset_name == 'tcga':
        dataset = tcga.tcga_h5_224_split(
            tissue_type='all',
            sub_set=subset,
        )
    elif dataset_name == 'lizard':
        dataset = lizard.LizardClassification.from_avocado(
            sub_set=subset,
        )
    else:
        raise ValueError('Could not find dataset \"{}\"'.format(dataset_name))

    return dataset


def get_num_classes(dataset_name) -> int:
    if dataset_name in ('vgh', 'pcam', 'kather', 'colossal'):
        raise ValueError('There are no classes for {} dataset'.format(dataset_name))
    elif dataset_name in ('icpr', 'tcga'):
        return 2
    elif dataset_name == 'wsss4luad':
        return 6  # bincount of labels gives: array([ 4815,  6770,  7380, 26965,    20,     5])
    else:
        raise ValueError('Could not find dataset \"{}\"'.format(dataset_name))
