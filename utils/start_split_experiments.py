import argparse
import copy
import os

from fetch_experiments import get_credentials, DET_MASTER
from determined.experimental import Determined
from ruamel.yaml import YAML


yaml = YAML(typ='safe')
MODEL_HPARAM_NAMES = ['encoder', 'model_name', 'model']
FREEZE_ENCODER = {
    'kather_h5_224_norm_split_90_10': 900,
    'patchcamelyon': 2620,
    'lizard': 2972,
}

CLASS_NUMBER = {
    'kather_h5_224_norm_split_90_10': 9,
    'patchcamelyon': 2,
    'lizard': 4,
}

BATCH_SIZES = {
    'kather_h5_224_norm_split_90_10': {
        1.0: {
            'batch_size': 256,
            'batches_total': 7020,
            'batches_per_epoch': 351
        },
        0.08: {
            'batch_size': 32,
            'batches_total': 4500,
            'batches_per_epoch': 255
        },
        0.008: {
            'batch_size': 32,
            'batches_total': 4500,
            'batches_per_epoch': 255
        },
        0.002: {
            'batch_size': 32,
            'batches_total': 4500,
            'batches_per_epoch': 255
        }
    },
    'patchcamelyon': {
        1.0: {
            'batch_size': 256,
            'batches_total': 20480,
            'batches_per_epoch': 1024
        },
        0.08: {
            'batch_size': 32,
            'batches_total': 13100,
            'batches_per_epoch': 655
        },
        0.008: {
            'batch_size': 32,
            'batches_total': 13100,
            'batches_per_epoch': 655
        },
        0.002: {
            'batch_size': 32,
            'batches_total': 13100,
            'batches_per_epoch': 655
        }
    },
    'lizard': {  # Train set has 297245 samples
        1.0: {
            'batch_size': 256,
            'batches_total': 23220,  # 1161 * 20 = 23220
            'batches_per_epoch': 1161,  # 297245 * 1.0 / 256 = 1161
        },
        0.08: {
            'batch_size': 32,
            'batches_total': 14860,  # 743 * 20 = 14869
            'batches_per_epoch': 743,  # 297245 * 0.08 / 32 = 743
        },
        0.008: {
            'batch_size': 32,
            'batches_total': 14860,  # taken from 0.08 split size
            'batches_per_epoch': 743,
        },
        0.002: {
            'batch_size': 32,
            'batches_total': 14860,
            'batches_per_epoch': 743,
        }
    },
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='experiment config file (.yaml)')
    parser.add_argument('model_dir', type=str, help='file or directory containing model definition')
    return parser.parse_args()


def get_any(d: dict, keys: list) -> any:
    for key in keys:
        if key in d:
            return d[key]
    raise ValueError('Could not find keys \"{}\" in {}'.format(keys, d))


def main():
    credentials = get_credentials()
    args = get_args()
    d = Determined(master=DET_MASTER, user=credentials['username'], password=credentials['password'])

    if not os.path.isfile(args.config_path):
        raise ValueError('Could not find config path {}'.format(args.config_path))

    with open(args.config_path, 'r') as configfile:
        exp_config = yaml.load(configfile)

    splits = BATCH_SIZES[exp_config['hyperparameters']['dataset']]
    
    experiment_ids = []
    for split in splits:
        experiment_copy = copy.deepcopy(exp_config)
        experiment_copy['reproducibility']['experiment_seed'] = 1643118217  # set experiment seed
        experiment_copy['searcher']['max_length']['batches'] = splits[split]['batches_total']
        experiment_copy['min_validation_period']['batches'] = splits[split]['batches_per_epoch']
        experiment_copy['hyperparameters']['plot_embedding_each'] = splits[split]['batches_total']
        experiment_copy['hyperparameters']['global_batch_size'] = splits[split]['batch_size']
        experiment_copy['hyperparameters']['split_size'] = split
        experiment_copy['hyperparameters']['number_classes'] = CLASS_NUMBER[exp_config['hyperparameters']['dataset']]
        experiment_copy['hyperparameters']['freeze_encoder'] = {
            'type': 'categorical',
            'vals': [True, FREEZE_ENCODER[exp_config['hyperparameters']['dataset']]]
        }
        experiment_copy['name'] = f"{experiment_copy['hyperparameters']['method']}::{experiment_copy['hyperparameters']['encoder']}::split_reduce::{str(split)}"

        experiment_ref = d.create_experiment(experiment_copy, args.model_dir)
        experiment_ids.append(experiment_ref.id)

    print('experiment ids: {}'.format(' '.join(map(str, experiment_ids))))


if __name__ == '__main__':
    main()
