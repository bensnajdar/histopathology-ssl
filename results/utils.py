import sys
import os.path
import torch
import torch.nn as nn
from pathlib import Path
# -- internal
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
import simsiam.src.augmentations as augs
from finetune_trial import init_paws_model, init_simclr_model, init_simsiam_model

torch.manual_seed(42)  # fixed seed for class mapping layer


MAPPINGS = {
    "dataset": {
        'kather': 'kather',
        'kather_h5_224_norm': 'kather',
        'kather_norm_224': 'kather',
        'kather_h5_224_norm_split_90_10': 'kather',
        'patchcamelyon': 'pcam',
        'pcam': 'pcam',
    },
    "freeze": {
        'True': 'enc_frozen',
        True: 'enc_frozen',
        1660: 'enc_finetuned',
        900: 'enc_finetuned',
        2620: 'enc_finetuned',
    },
    "encoder_dataset": {
        "56c542af-8021-46f0-9361-6d496bf5b653": "pcam",
        "029ff228-4ff2-463c-a209-0ec705db8473": "pcam",
        "b0906050-a06e-4a0c-8211-df917fc2e9bb": "pcam",
        "e5189813-b253-41bb-abb3-69db51835a53": "pcam",
        "56c542af-8021-46f0-9361-6d496bf5b653": "pcam",
        "029ff228-4ff2-463c-a209-0ec705db8473": "pcam",
        "b0906050-a06e-4a0c-8211-df917fc2e9bb": "pcam",
        "e5189813-b253-41bb-abb3-69db51835a53": "pcam",
        "54c83e3c-0886-47de-a3b7-b6f3fb90445f": "kather",
        "887c86b8-7878-496f-b37e-768c26172c18": "kather",
        "e74c1b83-dd08-4413-8bd2-75554e553006": "kather",
        "02f7c373-2ada-4e98-977e-a522f9eca676": "kather",
        "54c83e3c-0886-47de-a3b7-b6f3fb90445f": "kather",
        "887c86b8-7878-496f-b37e-768c26172c18": "kather",
        "e74c1b83-dd08-4413-8bd2-75554e553006": "kather",
        "02f7c373-2ada-4e98-977e-a522f9eca676": "kather",
    }
}


def get_model(checkpoint_id: str, checkpoint_hparam: dict, state_dict: dict = None):
    # -- HACK: Old SimCLR checkpoints do not rely on state_dicts
    OLD_SIM_CLR_CONFIGS = [
        '9026c9f5-266a-42fb-b5ad-0f3c415fecaf',
        '082ad5a6-bff0-4f80-b653-601d327e01e2',
        'bdfc1c5c-649a-4d1c-b7a0-e0d9b9c23199',
        '9d3cfdab-9047-44ae-b62d-7f5108eeca70'
    ]
    if checkpoint_id in OLD_SIM_CLR_CONFIGS:
        state_dict = None

    # -- Init
    if checkpoint_hparam['method'] == 'paws':
        model = init_paws_model(
            model_name=checkpoint_hparam['model_name'],
            output_dim=checkpoint_hparam['output_dim'],
            use_pred=True,
            number_classes=checkpoint_hparam['number_classes'],
            state_dict=None,
            freeze_encoder=checkpoint_hparam['freeze_encoder'],
            dropout_rate=checkpoint_hparam['dropout_rate'],
            pred_head_structure=checkpoint_hparam['pred_head_structure'],
        )
    elif checkpoint_hparam['method'] == 'simclr':
        model = init_simclr_model(
            encoder=checkpoint_hparam['encoder'],
            class_num=checkpoint_hparam['number_classes'],
            state_dict=state_dict,
            freeze_encoder=checkpoint_hparam['freeze_encoder'],
            keep_enc_fc=checkpoint_hparam['keep_enc_fc'],
            pred_head_structure=checkpoint_hparam['pred_head_structure'],
            pred_head_features=checkpoint_hparam['pred_head_features'],
        )
    elif checkpoint_hparam['method'] == 'simsiam' or checkpoint_hparam['method'] == 'simtriplet':
        model = init_simsiam_model(
            method=checkpoint_hparam['method'],
            encoder=checkpoint_hparam['encoder'],
            class_num=checkpoint_hparam['number_classes'],
            pred_head_features=checkpoint_hparam['pred_head_features'],
            state_dict=None,
            freeze_encoder=checkpoint_hparam['freeze_encoder'],
            keep_enc_fc=checkpoint_hparam['keep_enc_fc'],
            pred_head_structure=checkpoint_hparam['pred_head_structure']
        )

    if state_dict:
        model.load_state_dict(state_dict[0])

    return model


def get_image_data_list(xai_config, dataset):
    CLASS_LABEL = {
        'kather': ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'],
        'pcam': ['no_tumor', 'tumor'],
        'lizard': ['Epithelial', 'Lymphocyte', 'Plasma', 'Connective-tissue']
    }

    dataset = MAPPINGS['dataset'][dataset]
    data_list = []

    base_path = xai_config[dataset]['base_path']
    for class_name in xai_config[dataset]['data'].keys():
        for image in xai_config[dataset]['data'][class_name]:
            data_list.append(Path(base_path, class_name, image))
    classes = CLASS_LABEL[dataset]

    return data_list, classes, dataset


def get_augmentations(chekpoint_params: str):
    augmentation_stack = chekpoint_params['test_augmentation']
    augmentations = augs.__dict__[augmentation_stack](
        chekpoint_params['dataset'],
        chekpoint_params['normalize_data'],
        chekpoint_params['img_rescale_size']
    )
    return augmentations


class XAI_model_wrapper(nn.Module):
    def __init__(self, base_model, model_name):
        super(XAI_model_wrapper, self).__init__()

        self.fma = None  # feature map activations
        self.fmg = None  # fma gradients
        self.base_model = base_model

        for param in self.base_model.parameters():
            param.requires_grad = True

        if (model_name == 'resnet50'):
            self.last_conv = self.base_model.layer4[-1].conv3
        elif (model_name == 'resnet18'):
            self.last_conv = self.base_model.layer4[-1].conv2
        elif (model_name == 'wide_resnet28w2'):
            self.last_conv = self.base_model.layer3[-1].conv2
        else:
            raise NotImplementedError(f"Network structure {model_name} is not supported")

        # -- hooks
        self.last_conv.register_forward_hook(self.fma_hook)
        self.last_conv.register_backward_hook(self.fmg_hook)

    def fma_hook(self, module, input, output):
        self.fma = output

    def fmg_hook(self, module, grad_input, grad_output):
        self.fmg = grad_output

    def get_fmg_weights(self):
        fmg_size = self.fmg[0].shape[-1]
        return nn.AvgPool2d(kernel_size=fmg_size)(self.fmg[0])

    def forward(self, x):
        x = self.base_model(x)
        if hasattr(self.base_model, 'class_mapping'):
            x = self.base_model.class_mapping(x)
        return x
