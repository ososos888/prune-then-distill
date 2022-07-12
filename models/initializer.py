import torch.nn as nn


# Collection of model initializers
def xavier_normal_weight(model, layer='linear'):
    for name, module in model.named_modules():
        if layer == 'linear':
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
        elif layer == 'conv':
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
        else:
            raise ValueError(f"Please check your layer type: linear or conv. (Your input is {layer})")


def xavier_normal_bias(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.bias)


def xavier_uniform_weight(model, layer='linear'):
    for name, module in model.named_modules():
        if layer == 'linear':
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        if layer == 'conv':
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
        else:
            raise ValueError(f"Please check your layer type: linear or conv. (Your input is {layer})")


def xavier_uniform_bias(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.bias)


def zeros_weight(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)


def zeros_bias(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)


def normal_weight(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)


def normal_bias(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.bias)
