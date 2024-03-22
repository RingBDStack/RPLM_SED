
import numpy as np
import pandas as pd
import en_core_web_lg
from utils import *
import torch

import os


def detection(path_to_data):
    print(f"load data from '{path_to_data}'... ", end='')
    dataset = np.load(path_to_data, allow_pickle=True)
    print("\tDone")

    print(f"encode block samples, ")
    kmeans_scores,dbscan_scores=[],[]
    for i, blk in enumerate(dataset):
        if i ==0:
            continue
        print(f"Message Block{i}", flush=True)
        train, valid, test = (blk[n] for n in ('train', 'valid', 'test'))
        nlp = en_core_web_lg.load()
        features = []
        labels =[]
        for rowdata in test:
            feat = nlp(' '.join(rowdata.filtered_words)).vector
            features.append(feat)
            labels.append(rowdata.event_id)
        n_cluster = len(set(labels))
        print(f"test model on data block-{i} ...", flush=True)
        k_means_score = run_kmeans(features, n_cluster, labels)
        dbscan_score = run_hdbscan(features, labels)
        kmeans_scores.append(k_means_score)
        dbscan_scores.append(dbscan_score)

        print("KMeans:")
        print_scores(kmeans_scores)
        print("DBSCAN:")
        print_scores(dbscan_scores)


if __name__ == "__main__":
    data_path = "/home/lipu/HP_event/cache/cache_ori_rbert/twitter12.npy"
    detection(data_path)