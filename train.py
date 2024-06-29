import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
import torch.nn as nn
import os
import os.path as osp
import torchvision.transforms as T

import torchvision.transforms.functional as TF
import sys


from uavhuman import UAVHuman

from utils.AverageMeter import AverageMeter
from utils.logging import Logger

from models.multi_convnext2 import MultiConvnext2, MultiConvnext2Large

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

height = 224
width = 224

transforms_base = T.Compose([
        T.Resize(size=(width,height),interpolation=TF.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
        ])

transforms_train = T.Compose([
        T.Resize(size=(width,height),interpolation=TF.InterpolationMode.BICUBIC),
        # T.Pad(10),
        # T.RandomCrop((width,height)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=[1., 2.], saturation=[1., 2.]),
        T.ToTensor(),
        normalizer
        ])


class_weight=[
        torch.tensor([1.,1.]),
        torch.tensor([1.,1.,1.,1.,1.,1.]),
        torch.tensor([1.,1.,1.,1.,1.,1.]),
        torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]),
        torch.tensor([1.,1.,1.]),
        torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]),
        torch.tensor([1.,1.,1.,1.])
    ]




def adjust_learning_rate(base_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = base_lr #* (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = base_lr * 0.1
    elif epoch >= 20 and epoch < 30:
        lr = base_lr * 0.01
    elif epoch >= 30:
        lr = base_lr * 0.001
    elif epoch >= 40:
        lr = base_lr * 0.0001

    optimizer.param_groups[0]['lr'] = 1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    
    return lr

def test(model, epoch, given_zero_exit):
    model.eval()
    root = '/data0/ReIDData/uav-attr/uavhuman'
    test_set = UAVHuman(os.path.join(root, 'test'), transforms=transforms_base, split='test', verbose=False, zero_exit=given_zero_exit)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, 
                                num_workers=4, drop_last=False)
    pbar = tqdm(test_loader, total=len(test_loader))
    # f1_all_all = []
    acc_all_all = []
    pred_attr = [torch.tensor(()) for i in range(7)]
    label_attr = [torch.tensor(()) for i in range(7)]
    for i,(images,labels) in enumerate(pbar):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        for j,output in enumerate(outputs):
            pred = torch.argmax(output,1)
            pred_attr[j] = torch.cat((pred_attr[j],pred.cpu()),0)
            label_attr[j] = torch.cat((label_attr[j], labels[:,j].cpu()),0)
    for i in range(7):
        acc_all = accuracy_score(pred_attr[i], label_attr[i])
        acc_all_all.append(acc_all)
    print('Test Epoch', str(epoch + 1), 'acc_score:', np.mean(np.array(acc_all_all)), 'acc_score gender:',acc_all_all[0])
    print('Test Epoch', str(epoch + 1), 'acc_score_each:', acc_all_all)
    return np.mean(np.array(acc_all_all))
    

def train(model,num_epochs, batch_size=32):
    best_avg_acc = 0.0
    root = '/data0/ReIDData/uav-attr/uavhuman'
    train_set = UAVHuman(os.path.join(root, 'train'), transforms=transforms_train, split='train', verbose=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                num_workers=4, drop_last=False)
    lr_rate = 0.001 * 0.01
    # ignored_params = list(map(id, model.fc_list.parameters())) + list(map(id, model.projector_list.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    # 多卡
    ignored_params = list(map(id, model.module.fc_list.parameters())) + list(map(id, model.module.projector_list.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    # optimizer = torch.optim.Adam([
    #     {'params': base_params, 'lr': 1 * lr_rate},
    #     {'params': model.fc_list.parameters(), 'lr': lr_rate},
    #     {'params': model.projector_list.parameters(), 'lr': lr_rate},
    # ])

    # 多卡
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': 1 * lr_rate},
        {'params': model.module.fc_list.parameters(), 'lr': lr_rate},
        {'params': model.module.projector_list.parameters(), 'lr': lr_rate},
    ])

    ignored_list = [-1,-1,-1,-1,-1,-1,-1]
    # print(ignored_list)
    for epoch in range(num_epochs):
        current_lr = adjust_learning_rate(base_lr=lr_rate, optimizer=optimizer, epoch=epoch)
        ce_loss = AverageMeter()
        acc_avg = AverageMeter()
        acc_all_all = []
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for i,(images,labels) in enumerate(pbar):
            # print(images)
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            losses=0
            # f1_all=[]
            acc_all=[]

            for j,output in enumerate(outputs):
                loss_func = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=ignored_list[j])
                loss = loss_func(output,labels[:,j].long())
                pred = torch.argmax(output,1)
                losses+=loss
                acc_all.append(accuracy_score(pred.cpu(),labels[:,j].cpu()))
            
            losses/=len(labels)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            ce_loss.update(losses.item())
            acc_avg.update(np.mean(acc_all))
            acc_all_all.append(acc_all)

            n_total_steps = 50
            if (i+1) % (int(n_total_steps/1)) == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], \
                    LR: {current_lr:.6f}, Loss: {ce_loss.avg:.4f}, acc: {acc_avg.avg:.4f}' )

        if epoch % 1 == 0:
            acc = test(model, epoch, train_set.zero_exit)
            if acc > best_avg_acc:
                best_avg_acc = acc
                torch.save(model.state_dict(), log_path + '/model_best.pth')


log_path = '/datalog/attribute-log/'
if not os.path.isdir(log_path):
    os.makedirs(log_path)
sys.stdout = Logger(osp.join(log_path, 'log.txt'))

train_class_list = [2,6,6,11,3,10,4] # class num for each attribute in the training set

model_large = MultiConvnext2Large(class_list=train_class_list)
model_large.to("cuda")
# 多卡
model_large = nn.DataParallel(model_large)
train(model_large,50)
