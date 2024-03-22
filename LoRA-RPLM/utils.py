# -*- coding: utf8 -*-
import math
import os
import random
from collections import OrderedDict
from typing import Any
from collections import Counter
import numpy as np
import torch
from sklearn import metrics, manifold
from sklearn.cluster import KMeans, HDBSCAN, DBSCAN, OPTICS
from matplotlib import pyplot as plt




class CkptWrapper:
    def __init__(self, state: Any):
        self.state = state

    def state_dict(self):
        return self.state


def get_model_state(model, params, plm_tuning):
    if plm_tuning:
        return model
    else:
        model_state = model.state_dict()
        state = OrderedDict()
        for k, v in model_state.items():
            for p in params:
                if k.startswith(p):
                    state[k] = model_state[k]
                    break

        return CkptWrapper(state)


def width(text):
    return sum([2 if '\u4E00' <= c <= '\u9FA5' else 1 for c in text])


def print_table(tab):
    col_width = [max(width(x) for x in col) for col in zip(*tab)]
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")
    for line in tab:
        print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")


def data_generator(data, batch_size, shuffle=False, repeat=False):
    batch_num = math.ceil(len(data) / batch_size)
    return create_data_generator(data, batch_size, shuffle, repeat, batch_num), batch_num


def create_data_generator(data, batch_size, shuffle, repeat, batch_num):
    while True:
        if shuffle:
            shuffled_idx = [i for i in range(len(data))]
            random.shuffle(shuffled_idx)
            data = [data[i] for i in shuffled_idx]

        batch_id = 0
        while batch_id < batch_num:
            offset = batch_id * batch_size
            batch_data = data[offset:offset + batch_size]
            yield batch_data

            batch_id = batch_id + 1
        if repeat:
            continue
        else:
            break


def pad_seq(seq, max_len, pad=0, pad_left=False):
    """
    padding or truncate sequence to fixed length
    :param seq: input sequence
    :param max_len: max length
    :param pad: padding token id
    :param pad_left: pad on left
    :return: padded sequence
    """
    if max_len < len(seq):
        seq = seq[:max_len]
    elif max_len > len(seq):
        padding = [pad] * (max_len - len(seq))
        if pad_left:
            seq = padding + seq
        else:
            seq = seq + padding
    return seq


def run_kmeans(msg_feats, n_clust, msg_tags):
    # defalut:10
    k_means = KMeans(init="k-means++", n_clusters=n_clust, n_init=40,random_state=0)
    k_means.fit(msg_feats)

    msg_pred = k_means.labels_
    score_funcs = [
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
        ("ARI", metrics.adjusted_rand_score),
    ]

    scores = {m: fun(msg_tags, msg_pred) for m, fun in score_funcs}

    return scores


def run_hdbscan(msg_feats, msg_tags):
    hdb = HDBSCAN(min_cluster_size = 8)
    hdb.fit(msg_feats)

    msg_pred = hdb.labels_
    score_funcs = [
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
        ("ARI", metrics.adjusted_rand_score),
    ]

    scores = {m: fun(msg_tags, msg_pred) for m, fun in score_funcs}

    return scores


def run_dbscan(msg_feats, msg_tags):
    db = OPTICS(min_cluster_size=8, xi=0.01)
    db.fit(msg_feats)

    msg_pred = db.labels_
    score_funcs = [
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
        ("ARI", metrics.adjusted_rand_score),
    ]

    scores = {m: fun(msg_tags, msg_pred) for m, fun in score_funcs}

    return scores


def print_scores(scores):
    line = [' ' * 4] + [f'   M{i:02d} ' for i in range(1,len(scores)+1)]
    print("".join(line))

    score_names = ['NMI', 'AMI', 'ARI']
    for n in score_names:
        line = [f'{n} '] + [f'  {s[n]:1.3f}' for s in scores]
        print("".join(line))
    print('\n', flush=True)

    # line = [' ' * 3]+ [f'   M{i:02d}' for i in range(1, len(scores)+1)]
    # print("".join(line))
    #
    # score_names = ['NMI', 'AMI', 'ARI']
    # for n in score_names:
    #     line = [f'{n:3}'] + [f'  {s[n]:4.2f}' for s in scores]
    #     print("".join(line))
    # print('\n', flush=True)

def encode_samples(samples, raw_data, tokenizer,pmt_idx):
    data = []
    for tag, ev_idx, (tw_a, tw_b), pmt_feat in samples:
        tw_a_text = raw_data[tw_a].text
        tw_b_text = raw_data[tw_b].text
        tok = tokenizer(tw_a_text, tw_b_text, padding=True)

        # 只保留需要的关联特征
        # (entities, hashtags, user, words, time)
        pmt_feat = [pmt_feat[f] for f in pmt_idx]

        base = [2 * i for i in range(len(pmt_feat))]
        pmt_ids = [b + f for f, b in zip(pmt_feat, base)]

        if 'token_type_ids' not in tok:
            types = [0, 0, 1, 1]
            token_type_ids = tok.encodings[0].sequence_ids
            j = 0
            for i, t in enumerate(token_type_ids):
                if t is None:
                    token_type_ids[i] = types[j]
                    j += 1
        else:
            token_type_ids = tok['token_type_ids']

        data.append((tag, ev_idx, tw_a, tw_b, pmt_ids, tok['input_ids'], token_type_ids))



    # # not use relations
    #
    # data = []
    # for tag, ev_idx, (tw_a, tw_b) in samples:
    #     tw_a_text = raw_data[tw_a].text
    #     tw_b_text = raw_data[tw_b].text
    #
    #     tok = tokenizer(tw_a_text, tw_b_text)
    #
    #
    #
    #     if 'token_type_ids' not in tok:
    #         types = [0, 0, 1, 1]
    #         token_type_ids = tok.encodings[0].sequence_ids
    #         j = 0
    #         for i, t in enumerate(token_type_ids):
    #             if t is None:
    #                 token_type_ids[i] = types[j]
    #                 j += 1
    #     else:
    #         token_type_ids = tok['token_type_ids']
    #
    #     data.append((tag, ev_idx, tw_a, tw_b,  tok['input_ids'], token_type_ids))

    return data


def load_data_blocks(path_to_data, config, tokenizer):
    print(f"load data from '{path_to_data}'... ", end='')
    dataset = np.load(path_to_data, allow_pickle=True)
    print("\tDone")

    path_to_blocks = []
    print(f"encode block samples, ")

    for i, blk in enumerate(dataset):
        print(f"Message Block{i}", flush=True)
        train, valid, test = (blk[n] for n in ('train', 'valid', 'test'))


        path="/home/lipu/HP_event/cache/cache_bert12/"

        if not os.path.exists(path):
            os.makedirs(path)
        if config.offline:
            blk_path = os.path.join(path, f"{config.model_name}-{config.dataset_name}-offline.npy")
            # blk_path = os.path.join(path, f"{config.model_name}-{config.dataset_name}.npy")
        else:
            blk_path = os.path.join(path, f"{config.model_name}-{config.dataset_name}-M{i+1}.npy")

        if not os.path.exists(blk_path):
            print("train dateset processing",end=" ")
            train['samples'] = encode_samples(train['samples'], train['data'], tokenizer,config.pmt_feats)
            print("done")

            print("valid dateset processing",end=" ")
            valid['samples'] = encode_samples(valid['samples'], valid['data'], tokenizer,config.pmt_feats)
            print("done")

            print("test dateset processing",end=" ")
            test['samples'] = encode_samples(test['samples'], test['data'], tokenizer,config.pmt_feats)
            print("done")

            torch.save(
                {'train': train, 'valid': valid, 'test': test},
                blk_path
            )
        path_to_blocks.append(blk_path)

    del dataset
    print("Done")

    for blk_path in path_to_blocks:
        yield torch.load(blk_path)


def count_condition(data, key, threshold):
    return sum(entry[key] > threshold for entry in data), sum(entry[key] <= threshold for entry in data)




def calculate_average_min_score(newscore, min_score,max_score):
    for i, score in enumerate(newscore):
        for key, value in score.items():
            min_score[i][key] = min(min_score[i][key], value)
            max_score[i][key] = max(max_score[i][key], value)

    return  min_score,max_score



