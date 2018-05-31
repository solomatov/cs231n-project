import torch.nn as nn


def init_weights(module, std=0.02):
    for m in module.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, std)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)

