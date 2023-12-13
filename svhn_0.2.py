import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import gc
import sys
import copy
import math
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import linear_model, model_selection
from timm.utils import AverageMeter
from itertools import zip_longest
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

from metrics.forgetting import efficacy_upper_bound
from model.resnetc import resnet20
from utils.sam import SAM

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger,
)


# Choose GPU device and print status information
DEVICE, setup = system_startup()
seed = 42
set_random_seed(seed)
set_deterministic()
RNG = torch.Generator().manual_seed(seed)

unlearn_ratio = 0.2
unlearn_class = None
Path(f'./{unlearn_ratio}/{seed}').mkdir(parents=True, exist_ok=True)
sys.stdout = Logger(f'./{unlearn_ratio}/{seed}/log_{unlearn_ratio}.csv', sys.stdout)


# Dataset SVHN
def split_cls(data_set, num_cls, unlearn_class):
    y = np.array(data_set.targets)
    class_idx = [np.where(y==i)[0] for i in range(num_cls)]
    retcls_idx = []
    unlcls_idx = []
    for idx in range(num_cls):
        if idx in unlearn_class: # Df
            unlcls_idx += class_idx[idx].tolist()
        else: # Dr
            retcls_idx += class_idx[idx].tolist()
    forget_set = Subset(data_set, unlcls_idx)
    retain_set = Subset(data_set, retcls_idx)
    return forget_set, retain_set


def load_data(unlearn_ratio, unlearn_class=None):
    unlearn_ratio = unlearn_ratio
    num_cls = 10

    data_mean = (0.4914, 0.4822, 0.4465)
    data_std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)]
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ])
    train_set = torchvision.datasets.SVHN(root="/data/datasets/SVHN", split='train', download=True, transform=transform)
    test_set = torchvision.datasets.SVHN(root="/data/datasets/SVHN", split='test', download=True, transform=transform)

    # Split the train set into forget_set Df and a retain_set Dr
    if unlearn_class is None: # randomly select data to forget
        forget_length = int(len(train_set)*unlearn_ratio)
        retain_length = len(train_set) - forget_length
        forget_set_train, retain_set_train = random_split(train_set, [forget_length, retain_length], generator=RNG)
        forget_set_test, retain_set_test = None, None
    else:
        forget_set_train, retain_set_train = split_cls(train_set, num_cls, unlearn_class)
        forget_set_test, retain_set_test = split_cls(test_set, num_cls, unlearn_class)

    print(f"Train set: {len(train_set)}, Test set: {len(test_set)}, Forget set: {len(forget_set_train)}, Retain set: {len(retain_set_train)}")
    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=4, generator=RNG, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4, generator=RNG, drop_last=False)
    forget_loader = DataLoader(forget_set_train, batch_size=512, shuffle=True, num_workers=4, generator=RNG, drop_last=False)
    retain_loader = DataLoader(retain_set_train, batch_size=512, shuffle=True, num_workers=4, generator=RNG, drop_last=False)
    if unlearn_class is not None:
        forget_loader_test = DataLoader(forget_set_test, batch_size=512, shuffle=False, num_workers=4, generator=RNG, drop_last=False)
        retain_loader_test = DataLoader(retain_set_test, batch_size=512, shuffle=False, num_workers=4, generator=RNG, drop_last=False)
        return train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test
    else:
        return train_loader, forget_loader, retain_loader, test_loader


# Metrics
def compute_metrics(net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class):
    net.eval()
    retrain_net.eval()

    ## [1] Accuracy
    train_acc = accuracy(net, train_loader)
    test_acc = accuracy(net, test_loader)
    print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    forget_acc = accuracy(net, forget_loader)
    retain_acc = accuracy(net, retain_loader)
    if unlearn_class is not None:
        forget_acc_test = accuracy(net, forget_loader_test)
        retain_acc_test = accuracy(net, retain_loader_test)
        print(f"Forget Acc: {forget_acc:.4f}, Retain Acc: {retain_acc:.4f}, Forget Acc Test: {forget_acc_test:.4f}, Retain Acc Test: {retain_acc_test:.4f}")
    else:
        print(f"Forget Acc: {forget_acc:.4f}, Retain Acc: {retain_acc:.4f}")

    ## [2] MIA
    if unlearn_class is None:
        nonmember_loader = test_loader
    else:
        nonmember_loader = forget_loader_test
    mia = get_mia(net, forget_loader, nonmember_loader)
    print(f"The MIA has an accuracy of {mia:.3f} on forgotten vs unseen images")

    ## [3] Efficacy Score
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    efficacy = []
    for x, y in forget_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        efficacy.append(efficacy_upper_bound(net, x, y, ce_loss).cpu())
    ES = np.mean(efficacy)
    print(f"Efficacy Score (upper bound): {ES:.4f}")

    ## [4] Weight Distance
    wd = WeightDistance(net, retrain_net)
    print(f"Weight Distance: {wd:.4f}")

    ## [5] Wasserstein Distance
    wass_distance = WassersteinDistance(net, retrain_net, forget_loader)
    print(f"1-Wasserstein distance: {wass_distance:.4f}")

    print('------------------')
    Acc_t = test_acc if unlearn_class is None else retain_acc_test
    Acc_f = forget_acc
    Acc_r = retain_acc
    sample_data = [
        {
            'seed': seed,
            'Acc_t': Acc_t,
            'Acc_f': Acc_f,
            'Acc_r': Acc_r,
            'mia': mia,
            'ES': ES,
            'wd': wd,
            'wass_distance': wass_distance
        },
    ]
    return sample_data

def loss_val(net, loader):
    """Return loss on a dataset given by the data loader."""
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    with torch.no_grad():
        total = 0.
        test_loss = 0.
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            test_loss += criterion(outputs, targets).item()
            total += targets.size(0)
            torch.cuda.empty_cache()
            gc.collect()
    return test_loss / total

## [1] Accuracy
def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    net.eval()
    with torch.no_grad():
        total = 0.
        correct = 0.
        # acc = AverageMeter()
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            # acc.update(predicted.eq(targets).sum().item(), targets.size(0))
            torch.cuda.empty_cache()
            gc.collect()
    # return acc.avg
    return correct / total

## [2] MIA
def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets).detach().cpu().numpy()
        for l in losses:
            all_losses.append(l)
        torch.cuda.empty_cache()
        gc.collect()

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def get_mia(net, member_loader, nonmember_loader, n_splits=10, random_state=0):
    loss_mem = compute_losses(net, member_loader)
    loss_non = compute_losses(net, nonmember_loader)

    # make sure we have a balanced dataset for the MIA
    if len(loss_mem) > len(loss_non):
        np.random.shuffle(loss_mem)
        loss_mem = loss_mem[: len(loss_non)]
    else:
        np.random.shuffle(loss_non)
        loss_non = loss_non[: len(loss_mem)]

    samples_loss = np.concatenate((loss_mem, loss_non)).reshape((-1, 1))
    label_members = [1] * len(loss_mem) + [0] * len(loss_non)
    mia_scores = simple_mia(samples_loss, label_members, n_splits, random_state)
    return mia_scores.mean()

# [3] Efficacy Score
# efficacy_upper_bound

# [4] Weight Distance
def WeightDistance(model, model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        current_dist=(p.data-p0.data).pow(2).sum().item()
        current_norm=p.data.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    # print(f'Distance: {np.sqrt(distance)}')
    # print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)

# [5] 1-Wasserstein distance
def WassersteinDistance(model, model0, forget_loader):
    softmax = torch.nn.Softmax(dim=1)
    P1_normalized = []
    P2_normalized = []
    model.eval()
    model0.eval()
    with torch.no_grad():
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            P1 = model(inputs)
            P2 = model0(inputs)
            P1_normalized.append(softmax(P1))
            P2_normalized.append(softmax(P2))
            torch.cuda.empty_cache()
            gc.collect()
    P1_normalized = torch.cat(P1_normalized, dim=0).cpu().numpy()
    P2_normalized = torch.cat(P2_normalized, dim=0).cpu().numpy()
    distances = [wasserstein_distance(P1_normalized[i], P2_normalized[i]) for i in range(P1_normalized.shape[0])]
    wass_distance = np.mean(distances)
    # wass_distance = wasserstein_distance(P1_normalized.flatten(), P2_normalized.flatten())

    return wass_distance


# Load data
if unlearn_class is not None: # unlearn classes
    train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test = load_data(unlearn_ratio, unlearn_class)
else: # randomly unlearn
    train_loader, forget_loader, retain_loader, test_loader = load_data(unlearn_ratio, unlearn_class)
    forget_loader_test, retain_loader_test = None, None


# # Train the model
# def train(epochs, trainloader, testloader, net, optimizer, scheduler, criterion, retrain=False):
#     best_acc = 0.
#     for epoch in range(epochs):
#         net.train()
#         train_loss = 0.
#         train_loss = AverageMeter()
#         for inputs, targets in trainloader:
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             train_loss.update(loss.item(), targets.size(0))
#             torch.cuda.empty_cache()
#             gc.collect()
#         scheduler.step()

#         train_acc = accuracy(net, trainloader)
#         test_acc = accuracy(net, testloader)
#         if (epoch % 10 == 0) or (epoch == epochs-1):
#             print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
#         if test_acc > best_acc:
#             best_acc = test_acc
#             if retrain:
#                 torch.save(net.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_retrain.pth")
#             else:
#                 torch.save(net.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_origin.pth")

# net = resnet20(num_classes=10)
# net.to(DEVICE)
# epochs = 50
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
# criterion = nn.CrossEntropyLoss()
# st = time.time()
# train(epochs, train_loader, test_loader, net, optimizer, scheduler, criterion)
# train_time = time.time() - st
# print(f"Unscrubb Train time: {train_time:.4f}")
# del net, optimizer, scheduler

# net = resnet20(num_classes=10)
# net.to(DEVICE)
# epochs = 50
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
# criterion = nn.CrossEntropyLoss()
# st = time.time()
# train(epochs, retain_loader, test_loader, net, optimizer, scheduler, criterion, retrain=True)
# train_time = time.time() - st
# print(f"Retrain Train time: {train_time:.4f}")



# ###########
# Retrain model
print('------------------Retrain------------------')
retrain_net = resnet20(num_classes=10)
retrain_net.load_state_dict(torch.load(f"./{unlearn_ratio}/{seed}/svhn_retrain.pth"))
retrain_net.to(DEVICE)
retrain_net.eval()
# sample_data = compute_metrics(retrain_net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)


# Unscrubbed model
print('------------------Unscrubbed------------------')
origin_net = resnet20(num_classes=10)
origin_net.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_origin.pth'))
origin_net.to(DEVICE)
origin_net.eval()
# sample_data = compute_metrics(origin_net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)



# Finetune model
def unlearning_ft(net, retain_loader):
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    net.train()

    for _ in range(epochs):
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()

    net.eval()
    return net

print('------------------Finetune------------------')
finetune_net = resnet20(num_classes=10)
finetune_net.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_origin.pth'))
# finetune_net.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_ft.pth'))
finetune_net.to(DEVICE)
st = time.time()
finetune_net = unlearning_ft(finetune_net, retain_loader)
ft_time = time.time() - st
print(f"Finetune Train time: {ft_time:.4f}")
torch.save(finetune_net.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_ft.pth")
# sample_data = compute_metrics(finetune_net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)


# NegGrad model
def unlearning_ng(net, forget_loader):
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4, maximize=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()
    for _ in range(epochs):
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
        scheduler.step()

    net.eval()
    return net

print('------------------NegGrad------------------')
neggrad_net = resnet20(num_classes=10)
neggrad_net.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_origin.pth'))
# neggrad_net.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_ng.pth'))
neggrad_net.to(DEVICE)
st = time.time()
neggrad_net = unlearning_ng(neggrad_net, forget_loader) # 5, 1e-4
ng_time = time.time() - st
print(f"NegGrad Train time: {ng_time:.4f}")
torch.save(neggrad_net.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_ng.pth")
# sample_data = compute_metrics(neggrad_net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)


# FisherForget model
def hessian(dataset, model):
    model.eval()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    for p in model.parameters():
        p.grad_acc = 0
        p.grad2_acc = 0

    for data, orig_target in train_loader:
        data, orig_target = data.to(DEVICE), orig_target.to(DEVICE)
        output = model(data)
        prob = F.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    # p.grad_acc += (orig_target == target).float() * p.grad.data
                    # p.grad2_acc += prob[:, y] * p.grad.data.pow(2)
                    p.grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)

    for p in model.parameters():
        # p.grad_acc /= len(train_loader)
        p.grad2_acc /= len(train_loader)

def get_mean_var(p, is_base_dist=False, alpha=3e-6):
    num_classes = 10
    var = copy.deepcopy(1.0 / (p.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var

    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())
    if p.size(0) == num_classes and ((unlearn_ratio is None) and (unlearn_class is None)):
        mu[unlearn_class] = 0
        var[unlearn_class] = 0.0001
    if p.size(0) == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
    return mu, var

def fisherforget(retain_loader, net, alpha=3e-6):
    dataset = retain_loader.dataset
    for p in net.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    hessian(dataset, net)
    for i, p in enumerate(net.parameters()):
        mu, var = get_mean_var(p, False, alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
    return net

print('------------------FisherForget------------------')
ff_net = resnet20(num_classes=10)
ff_net.load_state_dict(torch.load(f"./{unlearn_ratio}/{seed}/svhn_origin.pth"))
# ff_net.load_state_dict(torch.load(f"./{unlearn_ratio}/{seed}/svhn_ff.pth"))
ff_net.to(DEVICE)
st = time.time()
ff_net = fisherforget(retain_loader, ff_net, alpha=1e-7) # 1e-7
ff_time = time.time() - st
print(f"FisherForget Train time: {ff_time:.4f}")
torch.save(ff_net.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_ff.pth")
# sample_data = compute_metrics(ff_net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)


# Influence unlearn model
def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr : curr + length].view(param.shape)
            curr += length

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = torch.autograd.grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)

def woodfisher(model, train_dl, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000
    o_vec = None
    criterion = nn.CrossEntropyLoss()
    for idx, (data, label) in enumerate(train_dl):
        model.zero_grad()
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

def Wfisher(retain_loader, forget_loader, model, alpha):
    criterion = nn.CrossEntropyLoss()

    retain_loader_1 = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False
    )
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(DEVICE)
    retain_grad = torch.zeros_like(torch.cat(params)).to(DEVICE)
    total = 0
    model.eval()
    for i, (data, label) in enumerate(forget_loader):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for i, (data, label) in enumerate(retain_loader):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        output = model(data)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num
    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2
    perturb = woodfisher(
        model,
        retain_loader_1,
        v=forget_grad - retain_grad,
    )
    apply_perturb(model, alpha * perturb)

    model.eval()
    return model

print('------------------Influence Unlearn------------------')
iu_net = resnet20(num_classes=10)
iu_net.load_state_dict(torch.load(f"./{unlearn_ratio}/{seed}/svhn_origin.pth"))
# iu_net.load_state_dict(torch.load(f"./{unlearn_ratio}/{seed}/svhn_iu.pth"))
iu_net.to(DEVICE)
st = time.time()
iu_net = Wfisher(retain_loader, forget_loader, iu_net, alpha=1.0) # 1.0
iu_time = time.time() - st
print(f"Influence Unlearn Train time: {iu_time:.4f}")
torch.save(iu_net.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_iu.pth")
# sample_data = compute_metrics(iu_net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)


# Finetune with sparse model
def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def unlearning_sparseft(net, retain_loader, test_loader, alpha=5e-4):
    sparse_epochs = 5
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
    net.train()

    for ep in range(epochs):
        total_loss = 0.
        net.train()
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if ep < sparse_epochs:
                current_alpha = alpha * (1 - ep / epochs)
            elif ep == sparse_epochs:
                current_alpha = alpha
            else:
                current_alpha = 0
            loss += current_alpha * l1_regularization(net)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()
            gc.collect()
        # print(f'current_alpha: {current_alpha:.6f}')
        # scheduler.step()

    net.eval()
    return net

print('------------------SparseFinetune------------------')
sparse_ft = resnet20(num_classes=10)
sparse_ft.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_origin.pth'))
# sparse_ft.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_sft.pth'))
sparse_ft.to(DEVICE)
st = time.time()
sparse_ft = unlearning_sparseft(sparse_ft, retain_loader, test_loader, alpha=0.01) # 0.01
sft_time = time.time() - st
print(f"SparseFinetune time: {sft_time:.4f}")
torch.save(sparse_ft.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_sft.pth")
# sample_data = compute_metrics(sparse_ft, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)


# Unlearning with Pruning on forget
def re_init_weights(shape, device):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
        mask = torch.squeeze(mask, 1)
    else:
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
    return mask

def create_dense_mask(net, device, value=1):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net

def snip(net, dataloader, sparsity, is_retain=False):
    criterion = nn.CrossEntropyLoss()
    grads = [torch.zeros_like(p) for p in net.parameters()]

    # compute grads
    cnt = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        net.zero_grad()
        loss.backward()
        for j, param in enumerate(net.parameters()):
            if param.grad is not None:
                # grads[j] += (param.grad.data).abs()
                grads[j] += ((param.grad.data).abs()) / len(dataloader)
        torch.cuda.empty_cache()
        gc.collect()
        cnt += 1
        if cnt >= 1:
            break

    # compute saliences to get the threshold
    weights = [p for p in net.parameters()]
    mask_ = create_dense_mask(copy.deepcopy(net), DEVICE, value=1)
    with torch.no_grad():
        abs_saliences = [(grad * weight).abs() for weight, grad in zip(weights, grads)]
        saliences = [saliences.view(-1).cpu() for saliences in abs_saliences]
        saliences = torch.cat(saliences)
        # threshold = np.percentile(saliences, sparsity)
        threshold = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0])
        if (threshold >= saliences.max() - 1e-12) or (threshold <= saliences.min() + 1e-12):
            threshold = (saliences.max() - saliences.min()) / 2.
        # print(f'threshold: {threshold}')

        # get mask to prune the weights (1 for retain, 0 for prune)
        for j, param in enumerate(mask_.parameters()):
            if is_retain:
                indx = ((grads[j]*weights[j]).abs() <= threshold) # prune for retain data
            else:
                indx = (abs_saliences[j] > threshold) # prune for forget data
            param.data[indx] = 0

        # update the weights of the original network with the mask
        for (name, param), (m_param) in zip(net.named_parameters(), mask_.parameters()):
            if ('bn' not in name) and ( ('layer2' in name) or ('layer3' in name) or ('layer4' in name) or ('fc' in name)):
                if ('weight' in name):
                    re_init_param = re_init_weights(param.data.shape, DEVICE)
                elif ('bias' in name):
                    re_init_param = torch.nn.init.zeros_(torch.empty(param.data.shape, device=DEVICE))
                param.data = param.data * m_param.data + re_init_param.data * (1 - m_param.data)

    return net

def unlearning_ft_snip(net, retain_loader, forget_loader, test_loader, sparsity=0.9, is_retain=False):
    # prune
    if is_retain:
        net = snip(net, retain_loader, sparsity=sparsity, is_retain=True)
    else:
        net = snip(net, forget_loader, sparsity=sparsity, is_retain=False)

    # funetune
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(filter(lambda p: p.requires_grad, net.parameters()), optim.SGD, rho=0.05, adaptive=False, lr=1e-2, momentum=0.9, weight_decay=5e-4) # rho=0.05, 5e-4
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    net.train()
    for ep in range(epochs):
        total_loss = 0.
        cnt = 0
        total = 0
        net.train()

        cnt_f = 0
        cnt_r = 0
        for data_r, data_u in zip_longest(retain_loader, forget_loader, fillvalue=None):
            if data_r is None and data_u is None:
                break

            elif (data_r is not None) and (data_u is not None):
                inputs_r, targets_r = data_r
                inputs_u, targets_u = data_u
                inputs_r, targets_r = inputs_r.to(DEVICE), targets_r.to(DEVICE)
                inputs_u, targets_u = inputs_u.to(DEVICE), targets_u.to(DEVICE)

                # first forward-backward step
                optimizer.zero_grad()
                loss1 = criterion(net(inputs_u), targets_u)
                loss1.backward()
                optimizer.reverse_step(zero_grad=True)
                # second forward-backward step
                loss = criterion(net(inputs_r), targets_r)
                loss.backward()
                optimizer.second_step(zero_grad=True)

                total_loss += loss.item()
                cnt += 1
                total += (inputs_u.shape[0] + inputs_r.shape[0])
                cnt_r += inputs_r.shape[0]
                cnt_f += inputs_u.shape[0]

            else:
                inputs_r, targets_r = data_r
                inputs, targets = inputs_r.to(DEVICE), targets_r.to(DEVICE)

                # first forward-backward step
                optimizer.zero_grad()
                outputs = net(inputs)
                loss1 = criterion(outputs, targets)
                loss1.backward()
                optimizer.first_step(zero_grad=True)
                # second forward-backward step
                loss = criterion(net(inputs), targets)
                loss.backward()
                optimizer.second_step(zero_grad=True)

                cnt_r += inputs_r.shape[0]

            torch.cuda.empty_cache()
            gc.collect()
            if cnt_r > 11721:
                break
        scheduler.step()

    net.eval()
    return net

print('------------------PruningFinetun_forget------------------')
pff_net = resnet20(num_classes=10)
pff_net.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_origin.pth'))
# pff_net.load_state_dict(torch.load(f'./{unlearn_ratio}/{seed}/svhn_pff.pth'))
pff_net.to(DEVICE)
st = time.time()
pff_net = unlearning_ft_snip(pff_net, retain_loader, forget_loader, test_loader, sparsity=0.9, is_retain=False) # 0.9 for partial, 0.8 for all
pff_time = time.time() - st
print(f"PruningFinetune_forget time: {pff_time:.4f}")
torch.save(pff_net.state_dict(), f"./{unlearn_ratio}/{seed}/svhn_pff.pth")
# sample_data = compute_metrics(pff_net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)


nets = [retrain_net, origin_net, finetune_net, neggrad_net, ff_net, iu_net, sparse_ft, pff_net]
name = ['retrain', 'origin', 'finetune', 'neggrad', 'ff', 'iu', 'sparseft', 'pff']
time_list = [0, 0, ft_time, ng_time, ff_time, iu_time, sft_time, pff_time]
for i, net in enumerate(nets):
    net.eval()
    sample_data = compute_metrics(net, retrain_net, train_loader, forget_loader, retain_loader, test_loader, forget_loader_test, retain_loader_test, unlearn_class)

    sample_data[0]['time'] = time_list[i]
    df = pd.DataFrame(sample_data)
    file_path = f"./{unlearn_ratio}/{seed}/results_{name[i]}.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

