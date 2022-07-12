import logging
import os

from prettytable import PrettyTable


class Logger:
    def __init__(self, opt, result_path, mode=None, logger_type=None):
        self.result_path = result_path
        self.mode = mode
        self.logger_type = logger_type

        if mode is None:
            self.logger = self.init_logging()
            self.parser_logging(opt, self.logger)
        elif mode == 'test':
            self.logger = self.init_logging(mode='test')
            self.logger.propagate = False

    def init_logging(self, mode=None):
        """
        This function initializes logging.
        INPUT:
            opt(:obj:`argparse.Namespace`):
                Parser argument of main.py
            logger_type(:obj:`str`):
                Variable for setting mode. Receives `None` or an arbitrary `str` value as input.
        OUTPUT:
            device(:obj:`torch.device`):
                Pythorch class for CUDA setting values.
        """
        if self.logger_type is not None:
            logger = logging.getLogger(self.logger_type)
        else:
            logger = logging.getLogger()

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')

        if mode != "test":
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(os.path.join(self.result_path, "test_log.log"), mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def close_logging_handler(self):
        """
        This function is close logging handler for new unittest function.
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def parser_logging(self, opt, logger):
        """
        Log the parser values.
        INPUT:
            logger(:obj:`logging.RootLogger`):
                RootLogger variables.
        """
        table = PrettyTable(["Item", "Value"])
        for arg in vars(opt):
            table.add_row([arg, getattr(opt, arg)])
        logger.info(table)
