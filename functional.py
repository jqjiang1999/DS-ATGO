import os
import torch
from torch import nn

from base_config import args


def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'neuron_reset'):
            m.neuron_reset()


"From 'Advancing Spiking Neural Networks Toward Deep Residual Learning, TNNLS 2025'"
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        log_probs = self.logsoftmax(outputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


def criterion(outputs, labels, loss_function):
    """
    :param outputs: SNN outputLayer -> Tensor(batch_size, T, class_num)
    :param labels: true label -> Tensor(batch_size, )
    :param loss_function: CrossEntropyLoss or MSELoss or CrossEntropyLabelSmooth
    """
    if args.loss_function == 'ce':
        loss = loss_function(outputs.mean(dim=1), labels.long())
    elif args.loss_function == 'mse':
        labels_onehot = torch.nn.functional.one_hot(labels, outputs.size(dim=2))
        loss = loss_function(outputs.mean(dim=1), labels_onehot.float())
    elif args.loss_function == 'LabelSmooth':
        loss = loss_function(outputs.mean(dim=1), labels.long())
    else:
        raise ValueError("Loss {} not recognized".format(args.loss_function))
    return loss


def set_loss_function(var):
    return {
        'mse': torch.nn.MSELoss(),
        'ce': torch.nn.CrossEntropyLoss(),
        'LabelSmooth': CrossEntropyLabelSmooth(),
    }.get(var, 'error')


def set_optimizer(var, model, lr):
    return {
        'sgd': torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4),
        'adam': torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)),
    }.get(var, 'error')


def set_lr_scheduler(var, optimizer, step):
    return {
        'StepLR': torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1),
        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step, eta_min=0),
    }.get(var, 'error')


def setup_seed(seed=None):
    import random
    import numpy as np

    if seed is None:
        seed = np.random.randint(0, 1e4)
        print('randomSeed: ', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpu, if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True