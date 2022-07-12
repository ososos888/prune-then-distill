from pruning import prune_runner
from training import train_runner
from kd import kd_runner


def run(platform):
    """
    Main running function.
    platform(:obj:`class`):
        This class have main arg.parser value and logger.
    """

    # Pre training
    if platform.opt.pre_train is True:
        train_runner.run(platform, mode='pre')

    # Pruning
    if platform.opt.pruning is True:
        prune_runner.run(platform)

    # Knowledge distillation
    if platform.opt.kd is True:
        kd_runner.run(platform)
