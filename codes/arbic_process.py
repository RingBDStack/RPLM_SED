import datetime
import itertools
import os

import math

import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split

from datasets import  DataItem18, SAMPLE_NUM_TWEET

def to_sparse_matrix(feat_to_tw, tw_num, tao=0):
    tw_adj = sp.sparse.coo_matrix((tw_num, tw_num), dtype=np.int8)
    tw_adj = tw_adj.todok()  # convert to dok
    for f in feat_to_tw.keys():
        for i in feat_to_tw[f]:
            for j in feat_to_tw[f]:
                tw_adj[i, j] += 1

    tw_adj = tw_adj > tao
    tw_adj = tw_adj.tocsr().astype(np.int8)
    return tw_adj


def build_entity_adj(data):
    tw_num = len(data)
    feat_to_tw = {}
    for i, it in enumerate(data):
        feats = it.entities
        feats = [e for e in feats]

        for f in feats:
            f = f.lower()
            if f not in feat_to_tw:
                feat_to_tw[f] = set()
            feat_to_tw[f].add(i)

    return to_sparse_matrix(feat_to_tw, tw_num)


def build_hashtag_adj(data):
    tw_num = len(data)
    feat_to_tw = {}
    for i, it in enumerate(data):
        feats = it.hashtags

        for f in feats:
            f = f.lower()
            if f not in feat_to_tw:
                feat_to_tw[f] = set()
            feat_to_tw[f].add(i)

    return to_sparse_matrix(feat_to_tw, tw_num)


def build_words_adj(data):
    tw_num = len(data)
    feat_to_tw = {}
    for i, it in enumerate(data):
        feats = it.words

        for f in feats:
            f = f.lower()
            if f not in feat_to_tw:
                feat_to_tw[f] = set()
            feat_to_tw[f].add(i)

    return to_sparse_matrix(feat_to_tw, tw_num)


def build_user_adj(data):
    tw_num = len(data)
    feat_to_tw = {}
    for i, it in enumerate(data):
        feats = it.user_mentions
        feats.append(it.user_name)

        for f in feats:
            if f not in feat_to_tw:
                feat_to_tw[f] = set()
            feat_to_tw[f].add(i)

    return to_sparse_matrix(feat_to_tw, tw_num)


def build_creat_at_adj(data):
    tw_num = len(data)
    tw_feat_idx = []
    feat_to_idx = {}
    for i, it in enumerate(data):
        feats = it.created_at
        feats = [e for e, t in feats]

        for f in feats:
            if f not in feat_to_idx:
                feat_to_idx[f] = len(feat_to_idx)
            f_idx = feat_to_idx[f]

            tw_feat_idx.append([i, f_idx])

    tw_feat_val = np.ones((len(tw_feat_idx),), dtype=np.int32)
    tw_feat_idx = np.array(tw_feat_idx, dtype=np.int64).T

    feat_num = len(feat_to_idx)
    tw_feat_mat = sp.sparse.coo_matrix(
        (tw_feat_val, (tw_feat_idx[0, :], tw_feat_idx[1, :])),
        shape=(tw_num, feat_num),
        dtype=np.int8)

    tw_adj = tw_feat_mat @ tw_feat_mat.T
    return tw_adj


FEAT_COLS = [
    ("entities", build_entity_adj),
    ("hashtags", build_hashtag_adj),
    ("user", build_user_adj),  # user_mentions and user_id
    ("words", build_words_adj),

    # ("create_at", build_creat_at_adj)
]



def tweet_to_event(data):
    ev_ids = sorted(set(it.event_id for it in data))
    ev_to_idx = {eid: i for i, eid in enumerate(ev_ids)}

    tw_to_ev = [ev_to_idx[it.event_id] for it in data]
    return tw_to_ev, ev_to_idx


def build_feats_adj(data, feats):
    feats_adj = [func(data) for f, func in feats]
    return feats_adj


def build_feat_adj(data, cols):
    tw_num = len(data)
    tw_feat_idx = []
    feat_to_idx = {}
    cols = [DataItem18._fields.index(c) for c in cols] if isinstance(cols, list) else [DataItem18._fields.index(cols)]
    for i, it in enumerate(data):
        feats = [
            list(itertools.chain(*it[c])) if isinstance(it[c], list) or isinstance(it[c], tuple) else [it[c]]
            for c in cols  # 特征列
        ]
        feats = [f for cf in feats for f in cf]

        for f in feats:
            if f not in feat_to_idx:
                feat_to_idx[f] = len(feat_to_idx)
            f_idx = feat_to_idx[f]

            tw_feat_idx.append([i, f_idx])

    tw_feat_val = np.ones((len(tw_feat_idx),), dtype=np.int32)
    tw_feat_idx = np.array(tw_feat_idx, dtype=np.int64).T

    feat_num = len(feat_to_idx)
    tw_feat_mat = sp.sparse.coo_matrix(
        (tw_feat_val, (tw_feat_idx[0, :], tw_feat_idx[1, :])),
        shape=(tw_num, feat_num),
        dtype=np.int8)

    tw_adj = tw_feat_mat @ tw_feat_mat.T
    return tw_adj


def split_train_test_validation(data):
    dataset = []
    data_size = len(data)

    test_size = math.ceil(data_size * 0.2)
    valid_size = math.ceil(data_size * 0.1)
    train,  test = train_test_split(data, test_size=test_size, random_state=42, shuffle=True)
    train, valid = train_test_split(train, test_size=valid_size, random_state=42, shuffle=True)

    print("creat all dataset ...", end="\t")
    dataset.append(process_block({"train": train, "test": test, "valid": valid}))
    print("done")

    print(f"save data to '/home/lipu/HP_event/cache_arabic/arabic.npy' ... ", end='')
    np.save('/home/lipu/HP_event/cache_arabic/arabic.npy', dataset)
    print("\tDone")





def get_time_relation(tw_i, tw_j, delta: datetime.timedelta = datetime.timedelta(hours=4)):
    a, b = tw_i.created_at, tw_j.created_at
    return int(a - b < delta) if a > b else int(b - a < delta)


def make_train_samples(tw_adj, tw_to_ev, data):
    tw_adj_num = len(tw_adj)
    tw_num = len(tw_to_ev)
    ev_num = max(tw_to_ev) + 1

    tw_ev_mat = np.zeros(shape=(tw_num, ev_num), dtype=np.int8)
    for i, e in enumerate(tw_to_ev):
        tw_ev_mat[i, e] = 1

    eye = sp.sparse.eye(tw_num, tw_num, dtype=np.int8)
    adj = tw_adj[0] - eye
    for f in range(1, tw_adj_num):
        adj = adj + (tw_adj[f] - eye)

    adj = np.asarray(adj.todense())

    pairs = []
    for i in range(tw_num):
        ev_idx = tw_to_ev[i]
        ev_tw_vec = tw_ev_mat[:, ev_idx]
        ev_tw_num = ev_tw_vec.sum()
        if ev_tw_num < 5:
            # print(f"outlier or small events: {i} -- {tw_to_ev[i]}--{ev_tw_num[tw_to_ev[i]]}")
            continue

        adj_i_tw = adj[i, :]
        adj_i_tw_score = np.exp(adj_i_tw - (1. - ev_tw_vec) * 1e12)

        pos_idx, = np.nonzero(ev_tw_vec)
        p = sp.special.softmax(adj_i_tw_score.take(pos_idx))

        pos_idx = np.random.choice(pos_idx, size=SAMPLE_NUM_TWEET, p=p)
        # (tag, event, (tweet_a, tweet_b), [feats,])
        pos_pairs = [
            (
                int(tw_to_ev[i] == tw_to_ev[j]), tw_to_ev[i], (i, j),
                list(1 if tw_adj[f][i, j] > 0 else 0 for f in range(tw_adj_num)) + [get_time_relation(data[i], data[j])]
            )
            for j in pos_idx
        ]
        pairs.extend(pos_pairs)


        neg_idx, = np.nonzero(1 - ev_tw_vec)
        adj_i_tw_score = np.exp(adj_i_tw - ev_tw_vec * 1e12)

        p = sp.special.softmax(adj_i_tw_score.take(neg_idx))

        neg_idx = np.random.choice(neg_idx, size=SAMPLE_NUM_TWEET, p=p)

        # (tag, event, (tweet_a, tweet_b), [feats,])
        neg_pairs = [
            (
                int(tw_to_ev[i] == tw_to_ev[j]), tw_to_ev[i], (i, j),
                list(1 if tw_adj[f][i, j] > 0 else 0 for f in range(tw_adj_num)) + [get_time_relation(data[i], data[j])]
            )
            for j in neg_idx
        ]
        pairs.extend(neg_pairs)

    return pairs


def make_ref_samples(tw_adj, tw_to_ev, data):
    tw_adj_num = len(tw_adj)
    tw_num = len(tw_to_ev)

    pairs = []
    adj = tw_adj[0]
    for f in range(1, tw_adj_num):
        adj = adj + tw_adj[f]

    adj = np.asarray(adj.todense())
    eye = np.eye(tw_num, tw_num, dtype=np.int8)
    adj = adj * (1 - eye) + eye

    tw_idx = np.arange(tw_num)
    for i in range(tw_num):
        p = sp.special.softmax(np.exp(adj[i]))


        ref_idx = np.random.choice(tw_idx, size=SAMPLE_NUM_TWEET * 3, p=p)
        # (tag, event, (tweet_a, tweet_b), [feats,])
        ref_pairs = [
            (
                int(tw_to_ev[i] == tw_to_ev[j]),
                tw_to_ev[i], (i, j),
                list(1 if tw_adj[f][i, j] > 0 else 0 for f in range(tw_adj_num)) + [get_time_relation(data[i], data[j])]
            )
            for j in ref_idx
        ]
        pairs.extend(ref_pairs)



    return pairs


def process_block(block):
    blk = {}
    for name in ["train", "test","valid" ]:
        data = block[name]
        tw_to_ev, ev_to_idx = tweet_to_event(data)
        tw_adj = build_feats_adj(data, FEAT_COLS)

        blk[name] = {
            "data": data,
            "tw_to_ev": tw_to_ev,
            "ev_to_idx": ev_to_idx,
            "tw_adj": tw_adj
        }

        if name == "train" or name == "valid":
            if data:
                blk[name]["samples"] = make_train_samples(tw_adj, tw_to_ev, data)
            else:
                blk[name]["samples"] =[]

        if name == "test":
            if data:
                blk[name]["samples"] = make_ref_samples(tw_adj, tw_to_ev, data)
            else:
                blk[name]["samples"] =[]

    return blk







if __name__ == '__main__':
    if not os.path.exists('../../cache_arabic'):
        os.makedirs('../../cache_arabic')
    tw_adj = sp.sparse.coo_matrix((1000, 1000), dtype=np.int8)
    print(f"load data from /home/lipu/smed/datasets/Arabic_Twitter/All_Arabic.npy ... ", end='')
    df = np.load("/home/lipu/smed/datasets/Arabic_Twitter/All_Arabic.npy", allow_pickle=True)
    print("\tDone")
    data = [DataItem18(*it) for it in df]
    print("create train,valid and test sets... ")
    split_train_test_validation(data)
    print("\tDone")




    exit(0)
