import os
import torch
import torch.nn as nn
from datasets.data_utils import get_dataloader
from tqdm import tqdm
from training.train_utils import get_optimizer, get_scheduler, get_lossfunction
from training.Tester import Tester
from utils.base_utils import save_result_data, result_data_initializer


def train(model, device, train_loader, optimizer, criterion):
    """
    This is a general train function in pytorch.
    INPUT:
        model(:obj:`pytorch.class`):
            Model is pytorch model class.
        device(:obj:`torch.device`):
            Model is pytorch model class.
        dataloader(:obj:`torch.utils.data.DataLoader`):
            Data loader for train data set. Use only trainset.
        optimizer(:obj:`torch.optim.optimizer`):
            Pytorch optim.optimizer(type) class.
        criterion(:obj:`torch.nn.modules.loss.lossfunction`):
            Pytorch lossfunction class.
    OUTPUT:
        training_loss(:obj:`float`):
            Training loss.
    """

    model.train()
    training_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, label)
        training_loss += loss.item()

        loss.backward()
        optimizer.step()

    training_loss /= len(train_loader.dataset)

    #model = model.module
    model.train_trainset_loss_arr.append(training_loss)

    return training_loss


def run(platform, model, epochs, batch_size, mode):
    """
    This function is a training loop(platform.opt.epochs times).
    INPUT:
        platform(:obj:`class`):
            This class have main arg.parser value and logger.
        model(:obj:`torch.models`):
            Pytorch model to prune.
        epochs(:obj:`int`):
            Training epochs.
        mode(:obj:`str`):
            Mode depending on the module used. A total of 3 modes can be entered.
                1) Pre: for pre-training
                2) post: for Post-pruning training
    """
    train_loader, val_loader = get_dataloader(platform.opt.dataset, batch_size,
                                              platform.opt.validation_size, platform.opt.dataloader_seed,
                                              loader_type="train_val")
    test_loader = get_dataloader(platform.opt.dataset, batch_size, loader_type="test")
    criterion = get_lossfunction()
    optimizer = get_optimizer(platform.opt, model, mode=mode)
    scheduler = get_scheduler(platform.opt, optimizer, mode=mode)
    tester = Tester(platform, criterion, val_loader, test_loader)

    # Initial accuracy check
    tester.run(model, test_mode="val_eval")

    # Training loop start
    platform.logger.logger.info(f"\n{mode}-Training start!")
    for epoch in tqdm(range(epochs)):

        training_loss = train(model, platform.device, train_loader, optimizer, criterion)
        tester.run(model, training_loss=training_loss, epoch=epoch+1, test_mode="val_eval")
        tester.run(model, test_mode="test_eval")
        scheduler.step()

    # Load the validation model with the highest accuracy and test.
    if mode == 'pre' and len(val_loader) is not 0:
        model.load_state_dict(torch.load(tester.val_path))
        os.remove(tester.val_path)
    tester.run(model, test_mode="test_eval")
    tester.logging_table(platform.logger)
    platform.logger.logger.info(f"Training done!\n"
                                f"Best validation accuracy: {tester.best_val_accu} (epoch: {tester.best_val_epoch})\n"
                                f"Test accuracy: {model.train_testset_accuracy_arr.pop():.4f}\n"
                                f"Test loss: {model.train_testset_loss_arr.pop():.8f}\n")

    # Save accuracy array
    if mode == 'pre':
        save_result_data(model, platform.model_path['result_path'], mode="pre")
    elif mode == 'post':
        save_result_data(model, platform.model_path['result_path'], mode="post")
    result_data_initializer(model)
