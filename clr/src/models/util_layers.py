from collections import OrderedDict

from torch import nn


def projection_head(input_shape=2048, num_outputs=128, head_width=2048):
    return nn.Sequential(
        nn.Linear(input_shape, head_width//2),
        nn.BatchNorm1d(head_width//2),
        nn.ReLU(),
        nn.Linear(head_width//2, num_outputs),
        nn.BatchNorm1d(num_outputs)
    )


def classifier(num_outputs, input_shape=2048, head_width=2048):
    return nn.Sequential(
        nn.Linear(input_shape, head_width),
        nn.ReLU(),
        nn.Linear(head_width, num_outputs),
        nn.LogSoftmax(dim=1),
    )


# taken from simsiam code
def finetune_classifier(input_size, class_num, hidden_feat, base_structure: str = None):
    if base_structure == 'three_layer_mlp':
        pred_head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_feat // 2)),
            ('bn1', nn.BatchNorm1d(hidden_feat // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(hidden_feat // 2, hidden_feat // 4)),
            ('bn2', nn.BatchNorm1d(hidden_feat // 4)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(hidden_feat // 4, class_num))
        ]))
    elif base_structure == 'two_layer_mlp':
        pred_head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_feat // 2)),
            ('bn1', nn.BatchNorm1d(hidden_feat // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(hidden_feat // 2, class_num))
        ]))
    elif base_structure == 'one_layer_mlp':
        pred_head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, class_num))
        ]))
    else:
        pred_head = nn.Sequential(
            nn.Linear(input_size, hidden_feat, bias=False),
            nn.BatchNorm1d(hidden_feat),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feat, hidden_feat, bias=False),
            nn.BatchNorm1d(hidden_feat),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feat, class_num)
        )

    return pred_head
