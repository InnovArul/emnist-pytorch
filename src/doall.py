import os, sys, os.path as osp
from datetime import datetime
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
        current_correct = (out.softmax(dim=-1).argmax(dim=-1) == target).float().mean().item()
        total_correct_avgmeter.update(current_correct, len(data))

        loss_avgmeter.update(loss.item(), 1)

        if batch_index % 50  == 0:
            print(f"epoch #{epoch}, ({batch_index}/{len(train_loader)}, loss={loss_avgmeter.avg}, accuracy={total_correct_avgmeter.avg}")
            #print(f"sum: {total_correct_avgmeter.sum}, count: {total_correct_avgmeter.count}")

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
        current_correct = (out.softmax(dim=-1).argmax(dim=-1) == target).float().mean().item()
        total_correct_avgmeter.update(current_correct, len(data))

        if batch_index % 500 == 0:
            print(f"epoch #{epoch}, ({batch_index}/{len(test_loader)}, loss={loss_avgmeter.avg}, accuracy={total_correct_avgmeter.avg}")

    print(f"epoch #{epoch} loss={loss_avgmeter.avg}, accuracy={total_correct_avgmeter.avg}")


def main():
    # determine experiment folder name
    expt_name = "emnist_" + datetime.now().strftime("%d%b%Y.%H%M%S")
    current_expt_path = osp.join("../scratch/", expt_name)
    mkdir_if_missing(current_expt_path)
    print("checkpoints are stored in: ", current_expt_path)

    # data
    train_loader, test_loader = get_dataloaders()

    # model
    model = emnist_net.EMNISTNet(len(train_loader.dataset.classes))

    # optimizer
    optimizer = build_optimizer(model)

    # lr scheduler
    lr_scheduler = build_lr_scheduler(optimizer=optimizer, lr_scheduler='multi_step', stepsize=[7,12,15,18])

    # objective functions
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        train(model, train_loader, optimizer, loss_fn, epoch)
        lr_scheduler.step(epoch)
        test(model, test_loader, loss_fn, epoch)
        save_checkpoint({'state_dict':model.state_dict(),
                         'epoch':epoch, 'optimizer':optimizer.state_dict()}, current_expt_path)

if __name__ == "__main__":
    main()
