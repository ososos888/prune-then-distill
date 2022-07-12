import torch.nn as nn

from prettytable import PrettyTable


class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.remaining_params, self.total_params = 1, 1
        self.weight_ratio = self.remaining_params / self.total_params * 100
        self.table = None
        self.train_trainset_loss_arr = [0]
        self.train_testset_loss_arr = [0]
        self.train_testset_accuracy_arr = [0]
        self.val_loss_arr = []
        self.val_accuracy_arr = []

    def weight_counter(self):
        """
        This function Search nn.Linear or nn.Conv2D module and calculates the number of 1s in the mask.
        save the result as table(by prettytable).
        """
        self.table_initialize()
        self.remaining_params, self.total_params = 0, 0

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) is True:
                self.check_mask_weight(name, module)

        self.weight_ratio = self.remaining_params / self.total_params * 100
        self.table.add_row(['*total*',
                            f"{self.total_params:.0f}",
                            f"{self.remaining_params:.0f} / {self.total_params-self.remaining_params:.0f}",
                            f"{self.weight_ratio:.2f}"])

    def check_mask_weight(self, name, module):
        """
        Find mask in modeule.named_buffers(). And after counting the number of 1 in the parameter, record it in the
        table.
        INPUT:
            opt(:obj:`str`):
                Name of module.
            opt(:obj:`nn.Module`):
                Module(nn.Linear of nn.Conv2d) with mask.
        """
        for buf_name, buf_param in module.named_buffers():
            if "weight_mask" in buf_name:
                remaing_p = buf_param.detach().cpu().numpy().sum()
                total_p = buf_param.numel()
                self.remaining_params += remaing_p
                self.total_params += total_p
                self.table.add_row([name, f"{total_p:.0f}",
                                    f"{remaing_p:.0f} / {total_p-remaing_p:.0f}",
                                    f"{remaing_p/total_p*100:.2f}"])

    def table_initialize(self):
        """
        Prettytable initializer.
        """
        self.table = PrettyTable(['Layer', 'Total_Weight', 'Remaining/Pruned', 'Ratio(%)'])
        self.table.align["Layer"] = "l"
        self.table.align["Total_Weight"] = "l"
        self.table.align["Remaining/Pruned"] = "l"
        self.table.align["Ratio(%)"] = "l"

    def logging_table(self, logger):
        """
        Log the table
        INPUT:
            logger(:obj:`logging.RootLogger`):
                RootLogger variables.
        """
        logger.logger.info(self.table)
