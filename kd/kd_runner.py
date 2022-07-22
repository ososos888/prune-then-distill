import torch

from kd import kd_vanilla
from models.get_models import get_model


def run(platform):
    """
    This function runs the kd loop according to the setting of opt.
    INPUT:
        platform(:obj:`class`):
            Platform class with the main parameters of the program.
    """
    platform.logger.logger.info(f'Start knowledge distillation({platform.opt.kd_type})...')

    teacher_model = get_model(platform.opt.teacher_model_type, platform.opt.dataset).to(platform.device)
    teacher_model.load_state_dict(torch.load(platform.model_path["teacher_model_path"], map_location=platform.device))

    # Get student model
    if "custom" in platform.opt.student_model_type:
        model = get_model(platform.opt.student_model_type,
                          platform.opt.dataset,
                          plan=teacher_model.get_student_plan()
                          ).to(platform.device)
    else:
        model = get_model(platform.opt.student_model_type, platform.opt.dataset).to(platform.device)
    model.weight_counter()
    model.logging_table(platform.logger)
    if platform.opt.kd_type == 'vanilla':
        kd_vanilla.run(platform, model, teacher_model)
