import os, sys, os.path as osp
this_path = osp.split(osp.abspath(__file__))[0]
sys.path += [osp.join(this_path, 'pytorch_utils')]

import torch
from dataset import get_dataloaders
from pytorch_utils import *
from models import emnist_net
import torch.nn as nn
from tqdm import tqdm

def train(model, train_loader, optimizer, loss_fn, epoch):
    print("training epoch #", epoch)

    loss_avgmeter = AverageMeter()
    total_correct_avgmeter = AverageMeter()

    if torch.cuda.is_available():
        model = model.cuda()

    model.train()

    for batch_index, (data, target) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = model(data)
        loss = loss_fn(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to determine accuracy
        current_correct = (out.softmax(dim=-1).argmax(dim=-1)[0] == target).sum().item()
        total_correct_avgmeter.update(current_correct, len(data))

        loss_avgmeter.update(loss.item(), 1)

        if batch_index % 50  == 0:
            print(f"epoch #{epoch}, ({batch_index}/{len(train_loader)}, loss={loss_avgmeter.avg}")

    print(f"epoch #{epoch} loss={loss_avgmeter.avg}, accuracy={total_correct_avgmeter.avg}")


def test(model, test_loader, loss_fn, epoch):
    print("test epoch #", epoch)

    loss_avgmeter = AverageMeter()
    total_correct_avgmeter = AverageMeter()

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    for batch_index, (data, target) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = model(data)
        loss = loss_fn(out, target)
        loss_avgmeter.update(loss.item(), 1)

        # to determine accuracy
        current_correct = (out.softmax(dim=-1).argmax(dim=-1)[0] == target).sum().item()
        total_correct_avgmeter.update(current_correct, len(data))

        if batch_index % 500 == 0:
            print(f"epoch #{epoch}, ({batch_index}/{len(test_loader)}, loss={loss_avgmeter.avg}")

    print(f"epoch #{epoch} loss={loss_avgmeter.avg}, accuracy={total_correct_avgmeter.avg}")


def main():
    # data
    train_loader, test_loader = get_dataloaders()

    # model
    model = emnist_net.EMNISTNet(len(train_loader.dataset.classes))

    # optimizer
    optimizer = build_optimizer(model)

    # lr scheduler
    lr_scheduler = build_lr_scheduler(optimizer=optimizer, lr_scheduler='multi_step', stepsize=[20,40,50])

    # objective functions
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(75):
        train(model, train_loader, optimizer, loss_fn, epoch)
        lr_scheduler.step(epoch)
        test(model, test_loader, loss_fn, epoch)

if __name__ == "__main__":
    main()