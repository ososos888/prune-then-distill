import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.data_utils import get_dataloader
from tqdm import tqdm
from training.train_utils import get_optimizer, get_scheduler
from training.Tester import Tester
from utils.base_utils import save_result_data, result_data_initializer


def train_kd(model, teacher_model, platform, train_loader, optimizer, criterion_kd):
    """
    This is a kd train function. The difference from the normal train is that it uses the teacher model and the kd-loss
    function because it uses kd. The teacher model must be changed to eval mode, and torch.no_grad() is used while
    obtaining the output data of the teacher model.
    INPUT:
        model(:obj:`pytorch.class`):
            Model is pytorch model class.
        teacher_model(:obj:`torch.class`):
            Teacher model is pytorch model class.
        platform(:obj:`class`):
            This class is input to use the pytorch cuda device and the alpha and temperature of kd.
        dataloader(:obj:`torch.utils.data.DataLoader`):
            Data loader for train data set. Use only trainset.
        optimizer(:obj:`torch.optim.optimizer`):
            Pytorch optim.optimizer(type) class.
        criterion_kd(:obj:`torch.nn.modules.loss.lossfunction`):
            KD_lossfunction.
    OUTPUT:
        training_loss(:obj:`float`):
            Training loss.
    """
    model.train()
    teacher_model.eval()
    training_loss = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(platform.device), label.to(platform.device)
        optimizer.zero_grad()

        outputs = model(data)
        with torch.no_grad():
            output_teacher = teacher_model(data).to(platform.device)
        loss = criterion_kd(outputs, label, output_teacher, platform.opt)
        training_loss += loss.item()

        loss.backward()
        optimizer.step()

    training_loss /= len(train_loader.dataset)

    model.train_trainset_loss_arr.append(training_loss)

    return training_loss


# kd loss fn. from https://github.com/peterliht/knowledge-distillation-pytorch
def loss_fn_kd(outputs, labels, teacher_outputs, opt):
    alpha = opt.kd_alpha
    t = opt.kd_temp
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/t, dim=1),
                                                  F.softmax(teacher_outputs/t, dim=1))*(alpha*t*t) + \
        F.cross_entropy(outputs, labels)*(1.-alpha)
    return kd_loss


def run(platform, model, teacher_model):
    """
    This function runs the vanilla kd loop.
    INPUT:
        platform(:obj:`class`):
            This class have main arg.parser value and logger.
        model(:obj:`pytorch.class`):
            Model is pytorch model class.
        teacher_model(:obj:`torch.class`):
            Teacher model is pytorch model class.
    """
    # Get dataloader
    train_loader, val_loader = get_dataloader(platform.opt.dataset, platform.opt.kd_batch_size,
                                              platform.opt.validation_size, platform.opt.dataloader_seed,
                                              loader_type="train_val")
    test_loader = get_dataloader(platform.opt.dataset, platform.opt.kd_batch_size, loader_type="test")

    # Hyperparameter setup
    optimizer = get_optimizer(platform.opt, model, mode='kd')
    scheduler = get_scheduler(platform.opt, optimizer, mode='kd')
    criterion = nn.CrossEntropyLoss()
    criterion_kd = loss_fn_kd
    tester = Tester(platform, criterion, val_loader, test_loader)

    # KD loop
    tester.run(model, test_mode='val_eval')
    for epoch in tqdm(range(platform.opt.kd_epochs)):
        training_loss = train_kd(model, teacher_model, platform, train_loader, optimizer, criterion_kd)
        tester.run(model, training_loss=training_loss, epoch=epoch+1, test_mode="val_eval")
        tester.run(model, test_mode="test_eval")
        scheduler.step()

    # Load the validation model with the highest accuracy and test.
    if len(val_loader) is not 0:
        model.load_state_dict(torch.load(tester.val_path))
        os.remove(tester.val_path)
    tester.run(model, test_mode="test_eval")
    tester.logging_table(platform.logger)
    platform.logger.logger.info(f"Training done!\n"
                                f"Best validation accuracy: {tester.best_val_accu} (epoch: {tester.best_val_epoch})\n"
                                f"Test accuracy: {model.train_testset_accuracy_arr.pop():.4f}\n"
                                f"Test loss: {model.train_testset_loss_arr.pop():.8f}\n")

    # Save result
    torch.save(model.state_dict(), os.path.join(platform.model_path["result_path"], "kd_model.pth"))
    save_result_data(model, platform.model_path['result_path'], mode="kd")
    result_data_initializer(model)
