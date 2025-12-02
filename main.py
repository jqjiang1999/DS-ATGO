import os
import sys
import math
import time
import tqdm
import torch.utils.data
from datetime import datetime

home_dir = os.getcwd()
sys.path.insert(0, home_dir)

from functional import *
from base_config import args, device
from data_loader import loader
from network_model import ResNet19


def SNN():
    snn = ResNet19.ResNet19(args.T, args.output_size)
    snn.to(device)
    print('{}'.format(snn))

    loss_function = set_loss_function(args.loss_function)
    optimizer = set_optimizer(args.optimizer, snn, args.learning_rate)
    scheduler = set_lr_scheduler(args.lr_scheduler, optimizer, args.num_epoch)
    train_loader, test_loader = loader.DataLoader(args.data_type, args.data_set, args.batch_size, args.data_augment, args.num_workers)
    start_time = time.time()
    best_acc, best_epoch = 0, 0
    for epoch in range(args.num_epoch):
        snn.train()
        total, correct, train_loss = 0., 0., 0.
        for images, labels in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            reset_net(snn)
            images, labels = images.to(device), labels.to(device)
            outputs_T = snn(images)
            loss = criterion(outputs_T, labels, loss_function)
            train_loss += loss.cpu().detach().item()
            loss.backward()
            optimizer.step()

            total += labels.numel()
            correct += (outputs_T.mean(dim=1).argmax(dim=1) == labels).float().sum().item()
        time_point = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        train_acc = 100. * float(correct / total)
        print('%s | Epoch [%d/%d], Train_loss: %f, Train_acc: %.2f'
              % (time_point, epoch + 1, args.num_epoch, train_loss, train_acc))
        scheduler.step()

        total, correct, test_loss = 0., 0., 0.
        snn.eval()
        with torch.no_grad():
            for images, labels in tqdm.tqdm(test_loader):
                reset_net(snn)
                images, labels = images.to(device), labels.to(device)
                outputs_T = snn(images)
                loss = criterion(outputs_T, labels, loss_function)
                test_loss += loss.cpu().detach().item()

                total += labels.numel()
                correct += (outputs_T.mean(dim=1).argmax(dim=1) == labels).float().sum().item()
        time_elapsed = time.time() - start_time
        test_acc = 100. * float(correct / total)
        print('%s | Epoch [%d/%d], Test_loss: %f, Test_acc: %.2f, Time elapsed:%.fh %.0fm %.0fs'
              % (time_point, epoch + 1, args.num_epoch, test_loss, test_acc, time_elapsed // 3600,
                 (time_elapsed % 3600) // 60, time_elapsed % 60))

        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            print('Saving.....')
        print('best acc is %.2f in epoch %d\n' % (best_acc, best_epoch))


if __name__ == '__main__':
    print(' Arguments: ')
    for arg in vars(args):
        print('\n\t {:25} : {}'.format(arg, getattr(args, arg)))
    setup_seed(args.seed)
    SNN()