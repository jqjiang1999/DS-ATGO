import os
import time
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Spiking Neural Networks')
parser.add_argument("--seed", type=int, default=2025, help='random seed')
parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers (default: 0)')

parser.add_argument("--data_set", type=str, default='***', help='dataset path')
parser.add_argument("--data_type", type=str, default='CIFAR10', help='dataset type')
parser.add_argument("--network_type", type=str, default='ResNet19', help='network architecture')
parser.add_argument('--data_augment', action='store_true', default=True, help='image augmentation')
parser.add_argument("--output_size", type=int, default=10, help='category')
parser.add_argument("--num_epoch", type=int, default=400, help='train epochs')
parser.add_argument("--batch_size", type=int, default=100, help='mini-batch')
parser.add_argument("--T", type=int, default=2, help='time steps')

parser.add_argument("--loss_function", type=str, default='LabelSmooth', choices=['ce', 'mse', 'LabelSmooth'], help='loss function')
parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer')
parser.add_argument("--learning_rate", type=float, default=1e-1, help='learning rate')
parser.add_argument("--lr_scheduler", type=str, default='CosineAnnealingLR', choices=['StepLR', 'CosineAnnealingLR'])
args = parser.parse_known_args()[0]