import json
import os
import torch


def set_cuda(GPU_num):
    """
    This function is set the CUDA state.
    INPUT:
        GPU_num(:obj:`str`):
            Number of GPU to use(0, 1, 2...).
    OUTPUT:
        device(:obj:`torch.device`):
            Pythorch class for CUDA setting values.
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{GPU_num}')
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device


def make_dir(path):
    """
    This function make folder for save model and result data.
    INPUT:
        path(:obj:`str`):
            Save path of experimental results.
    """
    if os.path.exists(path) is False:
        os.makedirs(path)
    else:
        raise ValueError("Error: folder already exists. Delete folder or set other --experiment_name")


def save_result_data(model, path, mode="pre"):
    """
    This function saves the accuracy and loss data stored in the model as a json file in the result folder.
    INPUT:
        model(:obj:`torch.models`):
            It is called to save the result data in the model.
        path(:obj:`str`):
            Save path of experimental results.
        mode:obj:`str`):
            A string for creating a prefix in the json file to be saved.
    """
    with open(f"{path}/{mode}_train_trainset_loss.json", "w") as json_file:
        json.dump(model.train_trainset_loss_arr, json_file, indent=4)
    with open(f"{path}/{mode}_train_testset_accu.json", "w") as json_file:
        json.dump(model.train_testset_accuracy_arr, json_file, indent=4)
    with open(f"{path}/{mode}_train_testset_loss.json", "w") as json_file:
        json.dump(model.train_testset_loss_arr, json_file, indent=4)
    with open(f"{path}/{mode}_val_loss.json", "w") as json_file:
        json.dump(model.val_loss_arr, json_file, indent=4)
    with open(f"{path}/{mode}_val_accu.json", "w") as json_file:
        json.dump(model.val_accuracy_arr, json_file, indent=4)


def result_data_initializer(model):
    """
    This function initialize the accuracy and loss data stored in the model.
    INPUT:
        model(:obj:`torch.models`):
            It is called to save the result data in the model.
    """
    model.train_trainset_loss_arr = [0]
    model.train_testset_loss_arr = [0]
    model.train_testset_accuracy_arr = [0]
    model.val_loss_arr = []
    model.val_accuracy_arr = []
