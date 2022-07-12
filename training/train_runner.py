import os
import torch

from models.get_models import get_model
from training import train_model


def run(platform, **kwargs):
    """
    This function runs the training loop(platform.opt.epochs times).
    INPUT:
        platform(:obj:`class`):
            This class have main arg.parser value and logger.
        **kwargs(:obj:`dict`):
            This dictionary can take two variables, model and mode. If you choose mode="pre" you don't need to enter the
            model. However, if you select mode="post", you must enter the model='model' to be trained.
    """
    if "mode" not in kwargs:
        raise ValueError("Must put mode value (pre or post)")
    if kwargs["mode"] == "pre":
        model = get_model(platform.opt.model_type, platform.opt.dataset).to(platform.device)
        epochs = platform.opt.epochs
        batch_size = platform.opt.batch_size
    elif kwargs["mode"] == "post":
        model = kwargs["model"]
        epochs = platform.opt.post_epochs
        batch_size = platform.opt.post_batch_size
    model.weight_counter()
    model.logging_table(platform.logger)
    # Train runner
    train_model.run(platform, model, epochs, batch_size, kwargs["mode"])

    # Save model
    if kwargs["mode"] == 'pre':
        torch.save(model.state_dict(), os.path.join(platform.model_path["result_path"], "pre_model.pth"))
