import argparse
from utils import get_dgl_data, split_data, ava_split_data
from losses import edl_mse_loss, edl_log_loss, edl_digamma_loss
from train import train_model
import torch.nn as nn
import torch.optim as optim
import torch
from time import localtime, strftime
import os
from model import ETGNN
import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset choosen from [CrisisLexT26]",default="French_Twitter")
    parser.add_argument("--epoch", type=int, help="epoch num", default=100)
    parser.add_argument("--batch_size", type=int, default=1500)
    parser.add_argument("--neighbours_num", type=int, default=80)
    parser.add_argument("--GNN_h_dim", type=int, default=256)
    parser.add_argument("--GNN_out_dim", type=int, default=256)
    parser.add_argument("--E_h_dim", type=int, default=128)
    parser.add_argument("--use_uncertainty", action="store_true", help="whether or not to user uncertainty")
    parser.add_argument("--use_cuda", action="store_true", help="whether or not to user cuda")
    parser.add_argument("--gpuid", type=int, default=3)
    parser.add_argument("--mode",type=int,default=0)
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument(
        "--mse",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
    uncertainty_type_group.add_argument(
        "--digamma",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.",
    )
    uncertainty_type_group.add_argument(
        "--log",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )
    parser.add_argument("--save_path", type=str, help="dataset choosen from [CrisisLexT27]", default="../data/Eng_CrisisLexT26/evi1020191139")
            #/evi0915003850")#evi0915182943")#evi0915003850")#evi0915225100")#evi1018190901/")
    args = parser.parse_args()
    print("Using CUDA:", args.use_cuda)
    if args.use_cuda:
        torch.cuda.set_device(args.gpuid)

    views = ['h','e','u']
    g_dict, times, features, labels= get_dgl_data(args.dataset, views, args.noise)

    mask_path = "../data/%s/" % (args.dataset) + "masks/"
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
        train_indices, val_indices, test_indices = ava_split_data(len(labels), args.train_p, args.val_p, args.test_p, labels, len(set(labels)))
        torch.save(train_indices, mask_path + "train_indices.pt")
        torch.save(val_indices, mask_path + "val_indices.pt")
        torch.save(test_indices, mask_path + "test_indices.pt")

    if args.mode == 0:
        flag = ''
        if args.use_uncertainty:
            print("use_uncertainty")
            flag = "evi"
        save_path = "../data/%s/"%(args.dataset) + flag + strftime("%m%d%H%M%S",localtime()) + "/"
        print(save_path)
        os.mkdir(save_path)
    else:
        save_path = args.save_path

    if args.use_uncertainty:
        if args.digamma:
            criterion = edl_digamma_loss
        elif args.log:
            criterion = edl_log_loss
        elif args.mse:
            criterion = edl_mse_loss
        else:
            parser.error("--uncertainty requires --mse, --log or --digamma.")
    else:
        criterion = nn.CrossEntropyLoss()

    model = ETGNN(features.shape[1], args.GNN_h_dim, args.GNN_out_dim, args.E_h_dim, len(set(labels)), views)
    train_model(model, g_dict, views, features, times, labels, args.epoch, criterion, mask_path, save_path, args)



if __name__ == '__main__':
    main()
