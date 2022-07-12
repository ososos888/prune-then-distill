# based on Synflow https://github.com/ganguli-lab/Synaptic-Flow
import math
import torch
import numpy as np
import torch.nn as nn

from datasets.data_utils import get_dataloader
from tqdm import tqdm


class Pruner:
    """Pruning class"""
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}
    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(device)
        output = model(input)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()


        nonlinearize(model, signs)


# pruning utility function
def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param


def prunable(module, batchnorm, residual):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (nn.Linear, nn.Conv2d))
    if batchnorm:
        isprunable |= isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (nn.Identity1d, nn.Identity2d))
    return isprunable


def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def prune_loop(model, loss, pruner, dataloader, device, sparsity, prune_epochs, scope, schedule,
               reinitialize=False, train_mode=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(prune_epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity ** ((epoch + 1) / prune_epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity) * ((epoch + 1) / prune_epochs)
        pruner.mask(sparse, scope)

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    sfscore = np.abs(remaining_params - total_params * sparsity)
    if sfscore >= 5:
        print("Warning: {} prunable parameters remaining, expected {}".format(remaining_params, total_params * sparsity))
        print(f"Score: {sfscore}")
        # quit()


def run(platform, model):
    """
    SynFlow pruning runner
    1. Get pruner class
    2. Set hyperparameters(Vanilla)
    3. Run pruning loop(Iterative pruning)
    INPUT:
        platform(:obj:`class`):
            This class have main arg.parser value and logger.
        model(:obj:`torch.models`):
            Pytorch model to prune.
    """
    # Get pruner class
    pruner = SynFlow(masked_parameters(model))

    # Set hyperparameters
    if platform.opt.dataset == 'cifar10':
        num_classes = 10
    elif platform.opt.dataset == 'cifar100':
        num_classes = 100
    elif platform.opt.dataset == 'tiny_imagenet':
        num_classes = 200
    elif platform.opt.dataset == 'imagenet':
        num_classes = 1000
    else:
        raise ValueError("Please check dataset")

    loss = nn.CrossEntropyLoss
    sparsity = 10 ** (math.log10(1-platform.opt.pruning_ratio))  # Pruning ratio
    prune_epochs = platform.opt.sf_epochs  # Iterative pruning.
    scope = 'global'  # 'global' or 'local' pruning
    schedule = 'exponential'  # Iterative pruning schedule 'exponential' or 'lineal
    prune_dataset_ratio = 10  # ratio of prune dataset size and number of classes (default: 10)
    prune_loader = get_dataloader(platform.opt.dataset, platform.opt.sf_batch_size, platform.opt.validation_size,
                                  loader_type="synflow", length=prune_dataset_ratio * num_classes)  # All 1

    # Pruning
    prune_loop(model, loss, pruner, prune_loader, platform.device, sparsity,
               prune_epochs, scope, schedule, reinitialize=False, train_mode=False)
