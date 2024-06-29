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

from uavhuman import UAVHuman

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
    with torch.no_grad():
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
    print('Test Epoch', str(epoch + 1), 'acc_score:', np.mean(np.array(acc_all_all)))
    print('acc_score gender:',acc_all_all[0])
    print('acc_score back:',acc_all_all[1])
    print('acc_score hat:',acc_all_all[2])
    print('acc_score upper clothes color:',acc_all_all[3])
    print('acc_score upper clothes shape:',acc_all_all[4])
    print('acc_score lower clothes color:',acc_all_all[5])
    print('acc_score lower clothes shape:',acc_all_all[6])
    
    print('Test Epoch', str(epoch + 1), 'acc_score_each:', acc_all_all)
    return np.mean(np.array(acc_all_all))

class_list = [2,6,6,11,3,10,4]

zero_exit = {
    'g': 1,
    'b': 0,
    'h': 0,
    'ucc': 1,
    'ucs': 1,
    'lcc': 1,
    'lcs': 0
} # denotes if cls label `0`` is exited in the training set

model_large = MultiConvnext2Large(class_list=class_list,pre_train=False)
model_large.to("cuda")
model_large = nn.DataParallel(model_large)
model_path = '/datalog/attribute-log/model_best.pth'
model_stat = torch.load(model_path)
model_large.load_state_dict(model_stat)

test(model_large, epoch=0, given_zero_exit=zero_exit)