import pandas as pd
import csv
import numpy as np
import os
from torch import nn
import torch
import random
import pickle



def read_data(task, period, rnn_length):
    if period == "bull":
        data_path = r"../data2/bull_dataset.csv"

    elif period == "bear":
        data_path = r"../data2/bear_dataset.csv"

    elif period == "mixed":
        data_path = r"../data2/mixed_dataset.csv"

    else:
        print("period error")



    market_data = pd.read_csv(data_path)
    num_stock = len(market_data.STOCK_ID.unique())
    num_timestep = len(market_data.date.unique())

    x_col = ['x_earning_rate', 'x_BIDLO_rate', 'x_ASKHI_rate', 'x_turnover',
             'x_SMA5_rate', 'x_SMA15_rate', 'x_SMA30_rate', 'x_MIDPRICE_rate',
             'ADX', 'MACD', 'AROONOSC', 'PPO', 'ATR', 'NATR', 'AD', 'OBV',
             'x_earning_rate_rank', 'x_BIDLO_rate_rank', 'x_ASKHI_rate_rank', 'x_turnover_rank',
             'x_SMA5_rate_rank', 'x_SMA15_rate_rank', 'x_SMA30_rate_rank', 'x_MIDPRICE_rate_rank',
             'x_ADX_rank', 'x_MACD_rank',
             "x_AROONOSC_rank", "x_PPO_rank",
             'x_ATR_rank', 'x_NATR_rank',
             'x_AD_rank', 'x_OBV_rank']

    x_ = market_data[x_col].to_numpy().astype(np.float32).reshape(num_stock, num_timestep, -1).transpose(1, 0, 2)

    if task == "rank":
        y = market_data["y_earning_rate_tmr_rank"].to_numpy().astype(np.float32).reshape(num_stock, num_timestep).transpose(1, 0)
        y_ret = market_data["y_earning_rate_tmr"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)
    elif task == "reg":
        #     elif task == "reg":
        y = market_data["y_earning_rate_tmr"].to_numpy().astype(np.float32).reshape(num_stock, num_timestep).transpose(1, 0)
        y_ret = market_data["y_earning_rate_tmr"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)
    elif task == "cls":
        y = market_data["y_earning_class_tmr"].to_numpy().astype(np.int64).reshape(num_stock, num_timestep).transpose(1, 0)
        y_ret = market_data["y_earning_class_tmr"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)

    else:
        print("error")

    length = num_timestep
    train_index1 = int( 0.8*length)
    train_index = int(0.8 * length)
    vaild_index = int(0.9 * length)


    x_train, y_train = x_[:train_index1], y[:train_index1]
    x_eval, y_eval = x_[train_index - rnn_length + 1:vaild_index], y[train_index - rnn_length + 1:vaild_index]
    x_test, y_test = x_[vaild_index - rnn_length + 1:], y[vaild_index - rnn_length + 1:]

    return x_train,x_eval,x_test,y_train,y_eval,y_test


def read_data2(task, period, rnn_length):
    if period == "CAS_A":
        data_path = r"../CAS_data/data202006dao202506.csv"
        num_stock = 245
        num_timestep = 1213

    elif period == "CAS_B":
        data_path = r"../CAS_data/data201806dao202306.csv"
        num_stock = 198
        num_timestep = 1214

    elif period == "CAS_C":
        data_path = r"../CAS_data/data201606dao202106.csv"
        num_stock = 125
        num_timestep = 1217

    else:
        print("period error")

    market_data = pd.read_csv(data_path, header=None)


    x_col = market_data.iloc[:, 0:9]
    y_col = market_data.iloc[:, -1]

    x_ = x_col.to_numpy().astype(np.float32).reshape(num_stock, num_timestep, -1).transpose(1, 0, 2)
    #y = y_col.to_numpy().astype(np.float32).reshape(num_stock, num_timestep).transpose(1, 0)
    y = y_col.to_numpy().astype(np.int64).reshape(num_stock, num_timestep).transpose(1, 0)
    length = num_timestep

    train_index = int(0.8 * length)
    vaild_index = int(0.9 * length)

    x_train, y_train = x_[:train_index], y[:train_index]
    x_eval, y_eval = x_[train_index - rnn_length + 1:vaild_index], y[train_index - rnn_length + 1:vaild_index]
    x_test, y_test = x_[vaild_index - rnn_length + 1:], y[vaild_index - rnn_length + 1:]



    return x_train,x_eval,x_test,y_train,y_eval,y_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

