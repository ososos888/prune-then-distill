import torch
from models import cifar_vgg, tinyimagenet_resnet, tinyimagenet_vgg, tinyimagenet_mobilenetv2


def get_model(model_type, dataset):
    """
    This function is getting model, initialize and save the original model.
    INPUT:
        platform(:obj:`class`):
            Platform class with the main parameters of the program. Use parser and save path.
        dataset(:obj:`str`):
            This value is used to check the input_shape and num_classes of the dataset.
    OUTPUT:
        model(:obj:`torch.nn.Module`):
            Model is pytorch model class. Pruned weight(zero-weight) gradients is fixed
    """

    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    elif dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    elif dataset == 'tiny_imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    elif dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000

    # Available model list

    cifar100_models = {
        'vgg11': cifar_vgg.vgg11_bn,
        'vgg19': cifar_vgg.vgg19_bn,
        'vgg19-rwd-cl1': cifar_vgg.vgg19_rwd_cl1,
        'vgg19-rwd-cl2': cifar_vgg.vgg19_rwd_cl2,
        'vgg19-rwd-st36': cifar_vgg.vgg19_rwd_st36,
        'vgg19-rwd-st59': cifar_vgg.vgg19_rwd_st59,
        'vgg19-rwd-st79': cifar_vgg.vgg19_rwd_st79,
        'vgg19dbl': cifar_vgg.vgg19dbl,
        'vgg19dbl-rwd-st36': cifar_vgg.vgg19dbl_rwd_st36,
        'vgg19dbl-rwd-st59': cifar_vgg.vgg19dbl_rwd_st59,
        'vgg19dbl-rwd-st79': cifar_vgg.vgg19dbl_rwd_st79,
    }

    tiny_imagenet_models = {
        'vgg16': tinyimagenet_vgg.vgg16_bn,
        'resnet18': tinyimagenet_resnet.resnet18,
        'resnet18-rwd-st36': tinyimagenet_resnet.resnet18_rwd_st36,
        'resnet18-rwd-st59': tinyimagenet_resnet.resnet18_rwd_st59,
        'resnet18-rwd-st79': tinyimagenet_resnet.resnet18_rwd_st79,
        'resnet18dbl': tinyimagenet_resnet.resnet18dbl,
        'resnet18dbl-rwd-st36': tinyimagenet_resnet.resnet18dbl_rwd_st36,
        'resnet18dbl-rwd-st59': tinyimagenet_resnet.resnet18dbl_rwd_st59,
        'resnet18dbl-rwd-st79': tinyimagenet_resnet.resnet18dbl_rwd_st79,
        'resnet50': tinyimagenet_resnet.resnet50,
        'resnet50-rwd-sp36': tinyimagenet_resnet.resnet50_rwd_sp36,
        'resnet50-rwd-sp59': tinyimagenet_resnet.resnet50_rwd_sp59,
        'resnet50-rwd-sp79': tinyimagenet_resnet.resnet50_rwd_sp79,
        'mobilenet-v2': tinyimagenet_mobilenetv2._mobilenet_v2,
    }

    models = {
        'cifar100': cifar100_models,
        'tiny_imagenet': tiny_imagenet_models
    }

    # Checks whether the input model-dataset is supported.
    if dataset not in models:
        raise ValueError(f"{dataset} is not supported.")
    if model_type not in models[dataset]:
        raise ValueError(f"{model_type} is not supported in {dataset}. Check the dataset or model_type.\n"
                         f"Supported model in {dataset}:\n"
                         f"{list(models[dataset].keys())}")

    model = models[dataset][model_type](input_shape, num_classes)

    return model
