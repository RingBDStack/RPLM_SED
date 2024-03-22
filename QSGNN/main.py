from train import initial_train,continue_train
import argparse
from utils import OnlineTripletLoss
from utils import HardestNegativeTripletSelector
from utils import RandomNegativeTripletSelector
from metric import AverageNonzeroTripletsMetric
import torch
from time import localtime, strftime
import os
import json
import numpy as np
from model import EDNN, GAT, simNN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('--finetune_epochs', default=3, type=int, #embeddings_0430063028
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--n_epochs', default=15, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--oldnum', default=20, type=int,
                        help="Number of sampling.")
    parser.add_argument('--novelnum', default=10, type=int,
                        help="Number of sampling.")
    parser.add_argument('--n_infer_epochs', default=0, type=int,
                        help="Number of inference epochs.")
    parser.add_argument('--window_size', default=3, type=int,
                        help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--margin', default=3., type=float,
                        help="Margin for computing triplet losses")
    parser.add_argument('--a', default=16., type=float,
                        help="Margin for computing pair-wise losses")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--batch_size', default=2100, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.")
    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', default=16, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=64, type=int,
                        help="Output dimension of tweet representations")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads in each GAT layer")
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")

    parser.add_argument('--validation_percent', default=0.1, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--test_percent', default=0.2, type=float,
                        help="Percentage of test nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False,
                        action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--metrics', type=str, default='nmi')
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=True,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--add_ort', dest='add_ort', default=True,
                        action='store_true',
                        help="Use orthorgonal constraint")
    parser.add_argument('--gpuid', type=int, default=3)
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")
    # offline or online situation
    parser.add_argument('--is_incremental', default=True, action='store_true')
    parser.add_argument('--data_path', default='../data/0414_ALL_French',
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--add_pair', action='store_true', default=True)

    args = parser.parse_args()
    use_cuda = True
    print("Using CUDA:", use_cuda)
    if use_cuda:
        torch.cuda.set_device(args.gpuid)

    embedding_save_path = args.data_path + '/embeddings_' + strftime("%m%d%H%M%S", localtime())
    os.mkdir(embedding_save_path)
    print("embedding_save_path: ", embedding_save_path)
    with open(embedding_save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    data_split = np.load(args.data_path + '/data_split.npy')

    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

    # Metrics
    metrics = [AverageNonzeroTripletsMetric()]

    if args.add_pair:
        #model, label_center_emb = initial_train(0, args, data_split, metrics,embedding_save_path, loss_fn, None)
        model = GAT(302, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        best_model_path = "../data/0413_ALL_English/embeddings_0507074357/block_0/models/best.pt"
        label_center_emb = torch.load("../data/0413_ALL_English/embeddings_0507074357/block_0/models/center.pth")
        # best_model_path = "../data/0414_ALL_French/embeddings_0507065144/block_0/models/best.pt"
        # label_center_emb = torch.load("../data/0414_ALL_French/embeddings_0507065144/block_0/models/center.pth")
        model.load_state_dict(torch.load(best_model_path))
        if args.use_cuda:
            model.cuda()

        if args.is_incremental:
            for i in range(1, data_split.shape[0]):
                print("incremental setting")
                print("enter i ",str(i))
                # Inference (prediction)
                _ = continue_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center_emb, args)

    else:
        model = initial_train(0, args, data_split, metrics, embedding_save_path, loss_fn, None)
