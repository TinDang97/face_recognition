# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        self.feature_size = -1

    def forward(self, x):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

    def load_model(self, f='pretrain.model'):
        with open(f) as f:
            pretrained_dict = torch.load(f)
            model_dict = self.state_dict()
            model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
            self.load_state_dict(model_dict)
