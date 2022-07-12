import os
import torch
import torch.nn as nn

from prettytable import PrettyTable


class Tester:
    def __init__(self, platform, criterion, val_loader, test_loader):
        self.criterion = criterion
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.platform = platform
        self.table = None
        self.table_initialize()
        self.best_val_accu = 0
        self.best_val_epoch = 0
        self.val_path = os.path.join(platform.model_path["result_path"], "val_weight_temp.pth")
        if len(val_loader) == 0:
            self.save_val = False
            self.val_loader = self.test_loader
        else:
            self.save_val = True

    def test(self, model, dataloader):
        """
        This is a general test function in pytorch.
        INPUT:
            model(:obj:`pytorch.class`):
                Model is pytorch model class.
        """
        #model = nn.DataParallel(model)
        model.eval()
        correct = 0
        test_loss = 0
        total = 0
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(self.platform.device), label.to(self.platform.device)
                outputs = model(data)
                test_loss += self.criterion(outputs, label).item()
                correct += torch.sum(torch.eq(label, outputs.argmax(dim=1))).item()
                total += label.size(0)

        test_loss /= total
        accuracy = correct / total

        return accuracy, test_loss

    def logging_per_epoch(self, accuracy, test_loss, training_loss, epoch):
        """
        This function logs the test results of the current epoch. If tr_loss and epoch are not received, the
        corresponding value is output as 0.
        INPUT:
            accuracy(:obj:`pytorch.class`):
                Model is pytorch model class.
            test_loss(:obj:`float`):
                The value to be recorded in the log.
            training_loss(:obj:`float`):
                The value to be recorded in the log.
            epoch(:obj:`int`):
                Epoch level. The value to be recorded in the log.
        """
        self.table.add_row([f"{epoch}",
                            f"{training_loss:.8f}" if isinstance(training_loss, float) else "nan",
                            f"{test_loss:.8f}",
                            f"{accuracy:.4f}"])

    def run(self, model, training_loss=0, epoch=0, test_mode=None):
        """
        This function performs 3 functions.
        1. tests the accuracy of the model.
        2. Save accuracy to self.test_accuracy_arr.
        3. Records a log of test results.
        INPUT:
            model(:obj:`pytorch.class`):
                Model is pytorch model class.
            training_loss(:obj:`float`):
                The value to be recorded in the log.
            epoch(:obj:`float`):
                 Epoch order. The value to be recorded in the log.
        """
        if test_mode == "val_eval":
            accuracy, loss = self.test(model, self.val_loader)
            if self.save_val is True:
                self.validation_check(model, accuracy, epoch)
            model.val_accuracy_arr.append(accuracy)
            model.val_loss_arr.append(loss)
            self.logging_per_epoch(accuracy, loss, training_loss, epoch)
        elif test_mode == "test_eval":
            accuracy, loss = self.test(model, self.test_loader)
            model.train_testset_accuracy_arr.append(accuracy)
            model.train_testset_loss_arr.append(loss)
        else:
            raise ValueError("Please check test_mode. you can use test_eval or val_eval")

    def validation_check(self, model, accuracy, epoch):
        """
        This function compares the accuracy of the model evaluated over the validation dataset with the previously
        highest accuracy. If it has the highest accuracy, that accuracy and model are saved.
         INPUT:
            model(:obj:`pytorch.class`):
                Model is pytorch model class.
            accuracy(:obj:`float`):
                Accuracy of model.
            epoch(:obj:`float`):
                 Epoch order. The value to be recorded in the log.
        """
        if accuracy > self.best_val_accu:
            self.best_val_accu = accuracy
            self.best_val_epoch = epoch
            torch.save(model.state_dict(), self.val_path)

    def table_initialize(self):
        """
        Prettytable initializer.
        """
        self.table = PrettyTable(['Epochs', 'Training_loss', 'Test_loss', 'Accuracy(val)'])
        self.table.align["Epochs"] = "l"
        self.table.align["Training_loss"] = "l"
        self.table.align["Test_loss"] = "l"
        self.table.align["Accuracy(val)"] = "l"

    def logging_table(self, logger):
        logger.logger.info(self.table)
