import torch.nn as nn
import torch.optim as optim


def get_lossfunction(type=None):
    """
    This function gets the optimizer for the selected model.
    INPUT:
        type(:obj:`str`):
            Determine the type of lossfunction.(None or MSELoss)
    OUTPUT:
        criterion(:obj:`torch.nn.modules.loss.lossfunction`):
            Pytorch lossfunction class.
    """
    if type is None:
        criterion = nn.CrossEntropyLoss()
    elif type == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"{type} is not a valid lossfunction type.")

    return criterion


def get_optimizer(opt, model, mode='pre'):
    """
    This function gets the optimizer for the selected model.
    INPUT:
        opt(:obj:`argparse.Namespace`):
            parser argument of main.py.
        model(:obj:`torch.nn.Module`):
            Model is pytorch model class.
        mode(:obj:`str`):
            Mode depending on the module used. A total of 3 modes can be entered.
                1) Pre: for pre-training
                2) post: for Post-pruning training
                3) kd: for knowledge distillation
    OUTPUT:
        optimizer(:obj:`torch.optim.optimizer`):
            Pytorch optim.optimizer(type) class.
    """
    if mode == 'pre':
        optimizer = opt.optimizer
        lr = opt.lr
        weight_decay = opt.weight_decay
    elif mode == 'post':
        optimizer = opt.post_optimizer
        lr = opt.post_lr
        weight_decay = opt.post_weight_decay
    elif mode == 'kd':
        optimizer = opt.kd_optimizer
        lr = opt.kd_lr
        weight_decay = opt.kd_weight_decay
    else:
        raise ValueError(f"Please check the mode option (Please choose in (pre, post, kd) not {mode})")

    if optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == 'nesterov':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)


def get_scheduler(opt, optimizer, mode='pre'):
    """
    This function gets the scheduler for the selected optimizer. 'lr_scheduler.MultiStepLR' is one of several scheduler
    functions provided by 'torch.optim'. 'milestones'(one of the required options for 'lr_scheduler.MultiStepLR') are
    points where the scheduler sets the epoch point to apply the gamma value to lr.
    INPUT:
        opt(:obj:`str`):
            Model type.
        optimizer(:obj:`torch.optim.optimizer`):
            Pytorch optim.optimizer(type) class.
        mode(:obj:`str`):
            Mode depending on the module used. A total of 3 modes can be entered.
                1) Pre: for pre-training
                2) post: for Post-pruning training
                3) kd: for knowledge distillation
    OUTPUT:
        scheduler(:obj:`torch.optim.lr_scheduler`):
            Pytorch optim.lr_scheduler.(type) class.
    """
    if mode == 'pre':
        lr_drops = opt.lr_drops
        lr_drop_rate = opt.lr_drop_rate
    elif mode == 'post':
        lr_drops = opt.post_lr_drops
        lr_drop_rate = opt.post_lr_drop_rate
    elif mode == 'kd':
        lr_drops = opt.kd_lr_drops
        lr_drop_rate = opt.kd_lr_drop_rate
    else:
        raise ValueError(f"Please check the mode option (Please choose in (pre, post, kd) not {mode})")

    return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_drops, gamma=lr_drop_rate)
