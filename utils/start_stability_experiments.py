import argparse
import copy
import os

from fetch_experiments import get_credentials, DET_MASTER
from determined.experimental import Determined
from ruamel.yaml import YAML


yaml = YAML(typ='safe')
MODEL_HPARAM_NAMES = ['encoder', 'model_name', 'model']
EXPERIMENT_SEEDS = {
    'resnet18': [1643366501, 1643366507, 1643366513, 1643366518, 1643366523],
    'resnet50': [1643403513, 1645520637, 1645521312, 1643403528, 1643403533],
    'wideresnet': [1643404500, 1643404508, 1643404513, 1643404518, 1643404524],
    'wide_resnet28w2': [1643404500, 1643404508, 1643404513, 1643404518, 1643404524],
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

    experiment_seeds = EXPERIMENT_SEEDS[get_any(exp_config['hyperparameters'], MODEL_HPARAM_NAMES)]
    experiment_ids = []
    for experiment_seed in experiment_seeds:
        experiment_copy = copy.deepcopy(exp_config)
        experiment_copy['reproducibility']['experiment_seed'] = experiment_seed  # set experiment seed

        experiment_ref = d.create_experiment(experiment_copy, args.model_dir)
        experiment_ids.append(experiment_ref.id)

    print('experiment ids: {}'.format(' '.join(map(str, experiment_ids))))


if __name__ == '__main__':
    main()
