from typing import Dict


GLOBAL_PARAMETERS = {
    'mnist': {
        'image_shape': {
            'width': 28,
            'height': 28,
            'channels': 1,
        }
    },
    'cifar': {
        'image_shape': {
            'width': 32,
            'height': 32,
            'channels': 3,
        }
    },
    'contrastive': {
        'image_shape': {
            'width': 96,
            'height': 96,
            'channels': 3,
        },
        'icpr_data_location': '/data/ldap/histopathologic/processed/icpr_mitosis',
        'pcam_data_location': '/data/ldap/histopathologic/original_read_only/PCam/PCam',
        'kather_data_location': '/data/ldap/histopathologic/processed_read_only/Kather_96',
        'vgh_data_location': '/data/ldap/histopathologic/original_read_only/vgh_nki/vgh_nki/he/patches_h224_w224',
        'wsss4luad_data_location': '/data/ldap/histopathologic/processed_read_only/WSSS4LUAD_96',
        'colossal_data_location': '/data/ldap/histopathologic/processed_read_only/Histo_PreTrainDataset_224',
        'crush_data_location': '/data/ldap/histopathologic/processed_read_only/Crush_96',
        'num_workers': 2,
    }
}


def get_global_parameters() -> Dict:
    return GLOBAL_PARAMETERS
