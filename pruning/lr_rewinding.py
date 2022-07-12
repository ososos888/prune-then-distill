import copy
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune

from training import train_runner
from datasets.data_utils import get_dataloader


def iter_pruning_freq_determiner(pruning_ratio):
    """
    Set the number of iterative pruning using the entered pruning_ratio.
    0 < pruning ratio < 1: The number of iterative pruning is set so that the closest result to the pruning ratio can be
     obtained.
    the pruning ratio >= 1: Repeat pruning as much as the value of the set pruning ratio.
    INPUT:
        pruning_ratio(:obj:`float')
            Rate or number of pruning
    OUTPUT:
        counter(:obj:`int')
            Number of pruning iterations.
    """
    if pruning_ratio < 1:
        counter = 0
        while (1 - np.power(0.8, counter)) <= pruning_ratio:
            counter += 1
            if counter >= 1000:
                print("iter_pruning_freq_determinerpruning error! please check pruning_ratio or this function")
                break

        counter_lower = np.abs(pruning_ratio-(1-np.power(0.8, counter-1)))
        counter_upper = np.abs(pruning_ratio-(1-np.power(0.8, counter)))

        if counter_lower <= counter_upper:
            return counter-1
        else:
            return counter
    elif pruning_ratio >= 1:
        return pruning_ratio


def get_pruning_module_list(model_for_pruning):
    """
    Selecting the module to be pruned from the copied model.
        INPUT:
            model_for_pruning(:obj:`torch.models`):
                A model copied for a prune.
        OUTPUT:
            model_for_pruning(:obj:`torch.models`):
                A model copied for a prune.
            pruning_module_list(:obj:`list`):
                List of module information for global l1pruning (2-dim(modules, module name)).
            pruning_modulename_list(:obj:`list`):
                List of names of modules to be pruned.
    """
    pruning_module_list = []
    pruning_modulename_list = []
    for name, module in model_for_pruning.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) is True:
            for mask_name, mask_param in model_for_pruning.named_buffers():
                if f"{name}.weight_mask" in mask_name:
                    pruning_module_list.append((module, 'weight'))
                    pruning_modulename_list.append(mask_name)
                    # module.weight.data = module.weight.data.mul_(mask_param)
                    # torch.nn.init.ones_(mask_param)
    return model_for_pruning, pruning_module_list, pruning_modulename_list


def mask_copy(model, model_for_pruning, pruning_modulename_list):
    """
    The pruning mask saved in the copied model is copied to the original model. Search using the name of the mask stored
    in the list.
    INPUT:
        model(:obj:`torch.models`):
            Pytorch model to prune.
        model_for_pruning(:obj:`torch.models`):
            A model copied for a prune. Pruning is applied.
        pruning_modulename_list(:obj:`list`):
            A list in which the name of the pruning list is stored.
    OUTPUT:
        model(:obj:`torch.models`):
            The model with the pruning applied mask copied.
    """
    for name_copiedmodel, mask_copiedmodel in model_for_pruning.named_buffers():
        if name_copiedmodel in pruning_modulename_list:
            for name_origmodel, mask_origmodel in model.named_buffers():
                if name_copiedmodel == name_origmodel:
                    mask_origmodel.data = mask_copiedmodel.data.clone().detach()
    return model


def pruning_loop(platform, model):
    """
    Learning rate rewind pruning loop. Create a mask by copying an existing model to use the Pytorch pruning library.
    INPUT:
        platform(:obj:`class`):
            This class have main arg.parser value and logger.
        model(:obj:`torch.models`):
            Pytorch model to prune.
    """
    pruning_epoch = iter_pruning_freq_determiner(platform.opt.pruning_ratio)
    train_loader, val_loader = get_dataloader(platform.opt.dataset, platform.opt.post_batch_size,
                                              platform.opt.validation_size, platform.opt.dataloader_seed,
                                              loader_type="train_val")

    for i in range(pruning_epoch):
        # pruning
        model_for_pruning = copy.deepcopy(model)
        model_for_pruning, pruning_module_list, pruning_modulename_list = get_pruning_module_list(model_for_pruning)

        # sparsity = 1 - np.power(0.8, i + 1)
        sparsity = 0.2
        # l1 global unstructure pruning with pytorch pruning
        prune.global_unstructured(pruning_module_list,
                                  pruning_method=prune.L1Unstructured,
                                  amount=sparsity)

        # Copies the pruned mask stored in the copied model for pruning to the original model.
        model = mask_copy(model, model_for_pruning, pruning_modulename_list)

        # fine-tuning
        train_runner.run(platform, model=model, mode='post', train_loader=train_loader, val_loader=val_loader)

        # Save model
        model_temp = copy.deepcopy(model)
        torch.save(model_temp.state_dict(), os.path.join(platform.model_path['result_path'],
                                                         f"post_model_{np.power(0.8, i + 1):.2f}.pth")
                   )


def run(platform, model):
    pruning_loop(platform, model)
