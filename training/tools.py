import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from torch import nn
import math


def cal_performance(pred, gold, smoothing=False):

    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)

    percision = metrics.precision_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    recall = metrics.recall_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    f1_score = metrics.f1_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='weighted')

    n_correct = pred.eq(gold)
    n_correct = n_correct.sum().item()

    return loss, n_correct, percision, recall, f1_score

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = 2

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')
    return loss


import random
def train_epoch(model,training_data,train_y,optimizer, device, criterion,args): #smoothing: 标签平滑的参数，用于改进模型的泛化能力。
    ''' Epoch operation in training phase'''
    model.train()
    seq_len = len(training_data)
    train_seq = list(range(seq_len))[
                args.length:]
    random.shuffle(train_seq)

    n_count = 0
    total_loss = 0
    total_loss_count = 0


    batch_train = args.batch_size

    k=0
    for i in train_seq:
        X_train= training_data[i - args.length + 1: i + 1].to(device)
        mean = X_train.mean(dim=(0, 1), keepdim=True)
        std = X_train.std(dim=(0, 1), keepdim=True)

        X_train = (X_train - mean) / (std + 1e-6)

        if torch.isnan(X_train).any():
            print("X_train 中存在 NaN 值！")
        pred_1,s_n = model(X_train)

        loss_1 = criterion(pred_1, train_y[i])
        count = (pred_1[:, 0] > pred_1[:, 1]).sum().item()

        # State Probability Regularization Loss
        s_t = s_n.view(-1)  # Flatten to [N * K]
        mean_s = torch.mean(s_t)
        var_s = torch.var(s_t)
        omega = 1e-3
        sp_loss = omega * (var_s / (mean_s ** 2 + 1e-10))

        loss = loss_1+sp_loss
        k = k + 1


        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print_gradients(model, i)
                    #print(i)
                    print(f"梯度中存在 NaN 或 Inf 值: {name}")
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        #print_gradients(model,i)
        if total_loss_count % batch_train == batch_train-1 :
            #print((pred[:, 0] > pred[:, 1]).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train-1  :

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        #optimizer.step_and_update_lr()
    return total_loss / total_loss_count








def print_gradients(model,i):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - num: {i},Mean: {param.grad.mean()}, Std: {param.grad.std()}, Max: {param.grad.max()}")


def evaluate_epoch(model, x_eval, y_eval, optimizer, device, args):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[args.length:]
    preds = []
    trues = []
    y_eval = y_eval.to(device)  # 确保 y_eval 是 float32 类型




    for i in seq:
        X = x_eval[i - args.length + 1: i + 1].to(device)  # 确保 X 是 float32 类型
        mean = X.mean(dim=(0, 1), keepdim=True)
        std = X.std(dim=(0, 1), keepdim=True)
        X = (X - mean) / (std + 1e-6)
        output,s_n = model(X)

        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i].cpu().numpy())
        count = (output[:, 0] > output[:, 1]).sum().item()
        #print("test",count)  # 1 的总数

    acc, auc = metrics(trues, preds)
    #print(trues[0], preds[0])
    return acc, auc



def metrics(trues, preds):
    trues = np.concatenate(trues, -1)
    preds = np.concatenate(preds, 0)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    auc = roc_auc_score(trues, preds[:, 1])
    mcc = matthews_corrcoef(trues, preds.argmax(-1))
    return acc, auc




class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x