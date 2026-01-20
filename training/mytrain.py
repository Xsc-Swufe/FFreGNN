import argparse
import numpy as np
import random
import torch
from FFreGNN.models import FFreGNN
from training.load_data import *
from FFreGNN.Optim import ScheduledOptim
from training.tools import train_epoch,evaluate_epoch
import torch.optim as optim
import os
import pickle


import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
def main():
    torch.autograd.set_detect_anomaly(True)
    ''' Main function '''
    set_seed(223) #199
    parser = argparse.ArgumentParser()

    parser.add_argument('-length', default=60,
                        help='length of historical sequence for feature')
    parser.add_argument('-feature', default=9, help='input_size')
    parser.add_argument('-n_class', default=2, help='output_size')
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-d_model', type=int, default=64)


    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('-dropout', type=float, default=0.35
                        )
    parser.add_argument('-proj_share_weight', default='True')

    parser.add_argument('-log', default='../days/FFreGNN_valid1')
    parser.add_argument('-save_model', default='../days/FFreGNN_valid1')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('--weight-constraint', type=float, default='0',
                        help='L2 weight constraint')

    parser.add_argument('--clip', type=float, default='0.50',
                        help='rnn clip')

    parser.add_argument('--lr', type=float, default='5e-4',  #5e-4
                        help='Learning rate ')            

    parser.add_argument('-steps', default=1,
                        help='steps to make prediction')

    parser.add_argument('--save', type=bool, default=True,
                        help='save model')

    parser.add_argument('--soft-training', type=int, default='0',
                        help='0 False. 1 True')

    parser.add_argument('--scale_num', type=int, default='3',
                        help='number of time series scales')
    parser.add_argument('--path_num', type=int, default='6',
                        help='number of message passing paths')

    parser.add_argument('--mem_dim', type=int, default='96',
                        help='hidden of memory dim')


    args = parser.parse_args()

    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model #16
    device = torch.device('cuda' if args.cuda else 'cpu')
    #train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt = load_dataset2(device, args.length)

    #train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt = read_data('cls', 'mixed', args.length)  #bull bear mixed,美国数据用这个
    train_eod, valid_eod, test_eod, train_gt, valid_gt, test_gt = read_data2('cls', 'CAS_B',args.length)  # 中国数据用这个
    train_gt = torch.from_numpy(train_gt).to(device)
    valid_gt = torch.from_numpy(valid_gt).to(device)
    test_gt = torch.from_numpy(test_gt).to(device)
    train_eod = torch.from_numpy(train_eod).to(device)
    valid_eod = torch.from_numpy(valid_eod).to(device)
    test_eod = torch.from_numpy(test_eod).to(device)
    train_gt.to(torch.float32)
    valid_gt.to(torch.float32)
    test_gt.to(torch.float32)


    # ========= Preparing Model =========#
    print(args)


    model =  FFreGNN(
        num_stock =  train_eod.shape[1],
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        dropout=args.dropout,
        scale_num=args.scale_num,
        path_num= args.path_num,
        window_size = args.length,
        mem_dim = args.mem_dim

   ).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
    best_model_file = 0

    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0
    MAX_EPOCH=100
    criterion = torch.nn.NLLLoss()
    while epoch < MAX_EPOCH:
        start_time = time.time()

        train_loss = train_epoch(model, train_eod, train_gt, optimizer, device, criterion, args)
        eval_acc, eval_auc = evaluate_epoch(model, valid_eod, valid_gt, optimizer, device, args)
        test_acc, test_auc = evaluate_epoch(model, test_eod, test_gt, optimizer, device, args)

        epoch_time = time.time() - start_time
        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}, time{:.2f}s".format(
            epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc, epoch_time
        )
        print(eval_str)


        if eval_auc > eval_epoch_best and epoch > 30:
            eval_epoch_best = eval_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}".format(
                epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
            wait_epoch = 0

            if args.save:
                if best_model_file and os.path.exists(best_model_file):
                    os.remove(best_model_file)
                best_model_file = "./checkpoints/eval_auc{:.4f}_acc{:.4f}_test_auc{:.4f}_acc{:.4f}_epoch{}.pkl".format(
                    eval_auc, eval_acc, test_auc, test_acc, epoch)
                os.makedirs("./checkpoints", exist_ok=True)
                torch.save(model.state_dict(), best_model_file)
                print(f"New best model saved: {best_model_file}")
        else:
            wait_epoch += 1


        if wait_epoch > 200:
            print("Early stopping triggered!")
            print("Best result:", eval_best_str)
            break

        epoch += 1


    if best_model_file:
        print(f"Training finished. Best model: {best_model_file}")
        print("Best performance:", eval_best_str)
    else:
        print("No model was saved (possibly no improvement after epoch 20).")















if __name__ == '__main__':
    main()