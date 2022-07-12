import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datasets import custom_datasets


def get_dataloader(dataset_type, batch_size, validation_size=0, dataloader_seed=0, loader_type=None, length=None):
    """
    Daterloader function. There are three () options for the loader type. When selecting train, both trainset and valset
    should be received as return values. If synflow is selected, a length option value with a value of
    prune_dataset_ratio * num_classes must be entered for score calculation.
    INPUT:
        opt(:obj:`parser.argument`):
            Main parser argument class
        loader_type(:obj:`str`):
            Choose datalader type.(train, test, synflow)
        length(:obj:`int`):
            When loder_type is `synflow`, enter the length of data.(prune_dataset_ratio * num_classes)
    OUTPUT:
       dataloader(:obj:`torch.utils.data.DataLoader`):
            Return dataloader according to loader_type

    """
    if loader_type == "train":
        train = True
    elif loader_type == "train_val":
        train = True
    elif loader_type == "test":
        train = False
    elif loader_type == "synflow":
        train = True
        if length is None:
            raise ValueError("synflow dataloader must input the `length` option")
    else:
        raise ValueError("Please check the loader type")

    # Dataset (The calculated mean and std of each dataset)
    save_path = f'../datasets/{dataset_type}'
    if dataset_type == 'cifar10':
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR10(save_path, train=train, download=True, transform=transform)
    elif dataset_type == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR100(save_path, train=train, download=True, transform=transform)
    elif dataset_type == 'tiny_imagenet':
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        tiny_imagenet_path = '../datasets'
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=True)
        dataset = custom_datasets.TINYIMAGENET(tiny_imagenet_path, train=train, download=True, transform=transform)
    elif dataset_type == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            folder = os.path.join(save_path, 'train')
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            folder = os.path.join(save_path, 'val')

        dataset = datasets.ImageFolder(folder, transform=transform)

    # Dataloader
    kwargs = {}

    if loader_type == 'train_val':
        if dataset_type == 'tiny_imagenet':
            if validation_size in range(2, len(dataset)):
                split_line = validation_size
            elif 0 <= validation_size <= 1:
                split_line = round(len(dataset) * validation_size)

            else:
                raise ValueError(
                    f"Wrong validation size!({validation_size}) Must in 0 to 1(float) or 1 to dataset(int)")
            trainset, valset = torch.utils.data.random_split(dataset,
                                                             [len(dataset) - split_line, split_line],
                                                             generator=torch.Generator().manual_seed(dataloader_seed))
            train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       **kwargs)

            val_loader = torch.utils.data.DataLoader(dataset=valset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     **kwargs)

        else:
            if validation_size in range(2, len(dataset)):
                split_line = validation_size
            elif 0 <= validation_size <= 1:
                split_line = round(len(dataset) * validation_size)
            else:
                raise ValueError(
                    f"Wrong validation size!({validation_size}) Must in 0 to 1(float) or 1 to dataset(int)")
            trainset, valset = torch.utils.data.random_split(dataset,
                                                             [len(dataset) - split_line, split_line],
                                                             generator=torch.Generator().manual_seed(dataloader_seed))
            train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       **kwargs)

            val_loader = torch.utils.data.DataLoader(dataset=valset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     **kwargs)
        return train_loader, val_loader

    elif loader_type == 'train':
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   **kwargs)
        return train_loader

    elif loader_type == "synflow":
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 **kwargs)

        return dataloader

    elif loader_type == "test":
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 **kwargs)
        return dataloader

    else:
        raise ValueError("If loader_type is a train, it returns two dataloaders, train and val.")


# Function from synflow
def device(gpu):
    use_cuda = torch.cuda.is_available()
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")


def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)
