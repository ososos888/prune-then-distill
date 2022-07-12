import os
import torch

from models.get_models import get_model
from pruning import lr_rewinding


def run(platform):
    """
    This function runs the pruning loop according to the setting of opt.
    INPUT:
        platform(:obj:`class`):
            Platform class with the main parameters of the program.
    """
    platform.logger.logger.info(f'Start pruning with {platform.opt.pruner}...')

    if platform.opt.pruner == "lr_rewinding":
        model = get_model(platform.opt.model_type, platform.opt.dataset).to(platform.device)
        model.load_state_dict(torch.load(platform.model_path["trained_model_path"]))
        model.weight_counter()
        model.logging_table(platform.logger)
        # Pruning
        lr_rewinding.run(platform, model)

    model.weight_counter()
    platform.logger.logger.info("Pruning Done!\n\n")

    # Save model
    torch.save(model.state_dict(), os.path.join(platform.model_path['result_path'], "post_model.pth"))
