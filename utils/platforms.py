import argparse
import json
import os
import torch
import numpy as np

from datetime import datetime
from utils.loggers import Logger
from utils.base_utils import set_cuda, make_dir


class Platform:
    def __init__(self, mode=None, logger_type=None):
        self.opt = self.get_parser()
        self.start_setup(mode)
        self.model_path = self.get_model_path()
        make_dir(self.model_path["result_path"])
        self.device = set_cuda(self.opt.GPU_num)
        self.logger = Logger(self.opt, self.model_path["result_path"], mode, logger_type)

        # Save hyperparameter to .json file
        with open(f"{self.model_path['result_path']}/hyperparameter.json", "w") as json_file:
            json.dump(vars(self.opt), json_file, indent=4)

    def get_parser(self):
        # Make instance
        parser = argparse.ArgumentParser(description='KD-Pruning',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Basic hyperparameters
        base_args = parser.add_argument_group('Base hyperparameters')
        base_args.add_argument("--experiment_name", default=None, type=str, help="Check the test file name")
        base_args.add_argument("--seed", default=None, type=int, help="Set random seed. If the value is 0, do not apply")
        base_args.add_argument("--dataloader_seed", default=None, type=int,
                               help="Set random seed. If the value is 0, do not apply")
        base_args.add_argument("--GPU_num", default=None, type=int, help="Number of GPU to use")
        base_args.add_argument("--hparam_fname", default=None, type=str, nargs='*',
                               help="To use hyperparameter saved as json, enter the file in experiments/hyparam and"
                                    "enter the file name without the extension.")
        base_args.add_argument("--model_type", default=None, type=str, help="Select model type")
        base_args.add_argument("--dataset", default=None, type=str, choices=('cifar10',
                                                                             'cifar100',
                                                                             'imagenet'),
                               help="Select Dataset type")
        base_args.add_argument("--batch_size", default=None, type=int, help="Size of mini-batch")
        base_args.add_argument("--validation_size", default=None, type=int, help="Validation size")

        # Pre Training hyperparameters
        pre_args = parser.add_argument_group('Pre training hyperparameters')
        pre_args.add_argument("--pre_train", action='store_true', help="")
        pre_args.add_argument("--epochs", default=None, type=int)
        pre_args.add_argument("--lr", default=None, type=float, help="Learning rate")
        pre_args.add_argument("--lr_drops", default=None, type=int, nargs='*', help="Learning rate")
        pre_args.add_argument("--lr_drop_rate", default=None, type=float, help="Learning rate")
        pre_args.add_argument("--optimizer", default=None, type=str, choices=('adam',
                                                                              'sgd',
                                                                              'momentum'),
                              help="Choose optimizer")
        pre_args.add_argument("--weight_decay", default=None, type=float, help="Parameter for optimizer")

        # Pruning hyperparameters
        prune_args = parser.add_argument_group('Pruning hyperparameters')
        prune_args.add_argument("--pruning", action='store_true', help="")
        prune_args.add_argument("--pre_model_name", default=None, type=str, help="Name of pre pre-trained model")
        prune_args.add_argument("--pruner", default=None, type=str, choices=('l1norm', 'random', 'lr_rewinding'),
                                help="Select pruning type")
        prune_args.add_argument("--pruning_ratio", default=None, type=float,
                                help="ratio of weights to be removed by pruning")
        prune_args.add_argument("--post_batch_size", default=None, type=int, help="Size of mini-batch")
        prune_args.add_argument("--post_epochs", default=None, type=int)
        prune_args.add_argument("--post_lr", default=None, type=float, help="Learning rate")
        prune_args.add_argument("--post_lr_drops", default=None, type=int, nargs='*', help="Learning rate")
        prune_args.add_argument("--post_lr_drop_rate", default=None, type=float, help="Learning rate")
        prune_args.add_argument("--post_optimizer", default=None, type=str, choices=('adam',
                                                                                     'sgd',
                                                                                     'momentum',
                                                                                     'neterov'),
                                help="Choose optimizer")
        prune_args.add_argument("--post_weight_decay", default=None, type=float, help="Parameter for optimizer")

        # KD hyperparameters
        kd_args = parser.add_argument_group('KD hyperparameters')
        kd_args.add_argument("--kd", action='store_true', help="")
        kd_args.add_argument("--kd_type", default=None, type=str, choices=('vanilla'),
                             help="Select kd type")
        kd_args.add_argument("--teacher_model_name", default=None, type=str,
                             help="Put the teacher model in `experiments/teacher_model` and enter the name of the"
                                  " model. The name of the file must be entered in the format model_dataset_prune.pth."
                                  "ex) (vgg-11 model trained with cifar10 and had 50% weight) vgg-11_cifar10_50.pth")
        kd_args.add_argument("--student_model_type", default=None, type=str, help="Select student model type(test)")
        kd_args.add_argument("--kd_epochs", default=None, type=int)
        kd_args.add_argument("--kd_batch_size", default=None, type=int, help="Size of mini-batch")
        kd_args.add_argument("--kd_lr", default=None, type=float, help="Learning rate")
        kd_args.add_argument("--kd_lr_drops", default=None, type=int, nargs='*', help="Learning rate")
        kd_args.add_argument("--kd_lr_drop_rate", default=None, type=float, help="Learning rate")
        kd_args.add_argument("--kd_optimizer", default=None, type=str, choices=('adam',
                                                                                'sgd',
                                                                                'momentum'),
                             help="Choose optimizer")
        kd_args.add_argument("--kd_weight_decay", default=None, type=float, help="Parameter for optimizer")
        kd_args.add_argument("--kd_alpha", default=None, type=float)
        kd_args.add_argument("--kd_temp", default=None, type=float)

        # Save to args
        opt = parser.parse_args()

        return opt

    def start_setup(self, mode):
        """
        Set the basic settings according to the general experiment mode and test (unittest) mode.
        INPUT:
            mode(:obj:`str`):
                Variable for setting mode. Receives `None` or test(`str` when unittest) value as input.
            logger_type(:obj:`str`):
                Variable for setting mode. Receives `None` or an arbitrary `str` value as input.
        """
        if mode is None:
            self.hparam_setup()
            self.set_seed()
        elif mode == 'test':
            self.opt.result_fname = "unittest"
        else:
            raise ValueError("Please check mode option value")
        if self.opt.experiment_name is None:
            self.opt.experiment_name = self.get_result_path_name()

    def get_result_path_name(self):
        """
        If experiment_name is not entered, it returns the save path name generated as a random number.
        This returns the save path in the form of YYMMDD_hhmmss.
        OUTPUT:
            save_path(:obj:`str`):
                Name of save path.
        """
        save_path = datetime.today().strftime("%y%m%d_%H%M%S")

        return save_path

    def get_model_path(self):
        """
        This function returns the save and load paths of models to be used for each module by referring to the parser
        value.
        OUTPUT:
            model_path(:obj:`dict'):
                Save and load paths of models.
        """
        model_path = {}
        model_path["result_path"] = os.path.join("result/data", self.opt.experiment_name)

        # Pre train - save
        if self.opt.pre_train is True:
            model_path["pre_train_model_path"] = model_path["result_path"]

        if self.opt.pruning is True:
            # Pruning - load
            if self.opt.pre_train is True:
                model_path["trained_model_path"] = os.path.join(model_path["result_path"], "pre_model.pth")
            else:
                model_path["trained_model_path"] = f"experiments/trained_model/{self.opt.pre_model_name}.pth"
                self.model_check(model_path["trained_model_path"])
            # Pruning - save
            model_path["pruned_model_path"] = os.path.join(model_path["result_path"], "post_model.pth")

        # KD - load
        if self.opt.kd is True:
            if self.opt.pruning is True:
                model_path["teacher_model_path"] = model_path["pruned_model_path"]
            else:
                model_path["teacher_model_path"] = f"experiments/teacher_model/{self.opt.teacher_model_name}.pth"
                self.model_check(model_path["teacher_model_path"])

        return model_path

    def model_check(self, PATH):
        """
        Checks the existence of the file.
        INPUT:
            mode(:obj:`str`):
                Saved model path.
        """
        if os.path.isfile(PATH) is False:
            raise ValueError(f"Pre trained model is not exist! (Put in experiments/pre_model/{PATH})")

    def set_seed(self):
        """
        This function is setting the random seed. Seed is set only opt.seed is not 0
        """
        if self.opt.seed is not None:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed_all(self.opt.seed)
            np.random.seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def hparam_setup(self):
        """
        This function is automatically determines the stored hyperparameter according to the set opt.model_type.
        """
        # Get file path
        param_folder_path = "experiments/hyperparam"
        param_fpath = []

        if self.opt.hparam_fname is None:
            if self.opt.pre_train is True:
                param_fpath.append("train_default")
            if self.opt.pruning is True:
                param_fpath.append("prune_default")
            if self.opt.kd is True:
                param_fpath.append("kd_default")
        else:
            param_fpath = self.opt.hparam_fname

        for index, path in enumerate(param_fpath):
            path = os.path.join(param_folder_path, f"{path}.json")
            param_fpath[index] = path

        # Check file exists
        for path in param_fpath:
            if os.path.exists(path) is False:
                raise ValueError(f"{self.opt.hparam_fname}.json is not exists!")

        # Get parser data
        for path in param_fpath:
            with open(path, "rt") as f:
                hparam = argparse.Namespace()
                hparam.__dict__.update(json.load(f))
            for key in vars(hparam).keys():
                if self.opt.__dict__[key] is None:
                    self.opt.__dict__[key] = hparam.__dict__[key]

        # Get teacher model type.
        if self.opt.kd is True:
            if self.opt.pruning is False:
                teacher_model_name = self.opt.teacher_model_name.split('_')
                self.opt.teacher_model_type = teacher_model_name[0]
            else:
                self.opt.teacher_model_type = self.opt.model_type
