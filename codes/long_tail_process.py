import datetime
import itertools
import os
from dateutil import parser
import math

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split

from datasets import  DataItem12, SAMPLE_NUM_TWEET,DataItem18,COLUMNS_12,COLUMNS_18

name = "event_2012"


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
        if name == "event_2012":
            feats = [e for e,f in feats]
        else:
            feats = [e for e in feats]

        for f in feats:
            f = f.lower()
            if f not in feat_to_tw:
                feat_to_tw[f] = set()
            feat_to_tw[f].add(i)
    print("create entity matrix")
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
    print("create hashtag matrix")
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

    print("create words matrix")
    return to_sparse_matrix(feat_to_tw, tw_num)


def build_user_adj(data):
    tw_num = len(data)
    feat_to_tw = {}
    for i, it in enumerate(data):
        feats = it.user_mentions
        if name=="event_2012":
            feats.append(it.user_id)
        else:
            feats.append(it.user_name)

        for f in feats:
            if f not in feat_to_tw:
                feat_to_tw[f] = set()
            feat_to_tw[f].add(i)
    print("create user matrix")
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
    cols = [DataItem._fields.index(c) for c in cols] if isinstance(cols, list) else [DataItem._fields.index(cols)]
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

def split_data(group, test_size=30, valid_size=20):
    # 乱序每个分组
    shuffled = group.sample(frac=1)
    # 抽取训练数据
    test = shuffled.iloc[:test_size]
    # 抽取验证数据
    valid = shuffled.iloc[test_size:test_size+valid_size]
    # 剩余的作为测试数据
    train = shuffled.iloc[test_size+valid_size:]
    return train, valid, test
def split_train_test_validation(data):
    dataset = []
    train_frames = []
    validation_frames = []
    test_frames = []
    groups = data.groupby('event_id')
    for _, group in groups:
        train, validation, test = split_data(group)
        train_frames.append(train)
        validation_frames.append(validation)
        test_frames.append(test)
    train = pd.concat(train_frames)
    train = [DataItem12(*it) for it in train.to_records(index=False)]
    valid = pd.concat(validation_frames)
    valid = [DataItem12(*it) for it in valid.to_records(index=False)]
    test = pd.concat(test_frames)
    test = [DataItem12(*it) for it in test.to_records(index=False)]
    print("creat all dataset ...", end="\t")
    dataset.append(process_block({"train": train, "test": test, "valid": valid}))
    print("done")

    return dataset





def get_time_relation(tw_i, tw_j, delta: datetime.timedelta = datetime.timedelta(hours=4)):
    a = pd.Timestamp(tw_i.created_at).to_pydatetime()
    b = pd.Timestamp(tw_j.created_at).to_pydatetime()
    return int(abs(a - b) < delta)


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
    # for name in ["valid", "train", "test"]:
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
    if not os.path.exists('/home/lipu/HP_event/cache/cache_long_tail/'):
        os.makedirs('/home/lipu/HP_event/cache/cache_long_tail/')
    if name == "event_2012":
        DataItem = DataItem12
        print(f"load data from /home/lipu/HP_event/cache/cache_long_tail/long_tail.npy ... ", end='')
        df = np.load("/home/lipu/HP_event/cache/cache_long_tail/long_tail.npy", allow_pickle=True)
        print("\tDone")
        data = [DataItem(*it) for it in df]
        data = pd.DataFrame(data, columns=COLUMNS_12)
        print("create train,valid and test sets... ")
        dataset = split_train_test_validation(data)
        print("\tDone")
        print(f"save data to '/home/lipu/HP_event/cache/cache_long_tail/long_tail_12.npy' ... ", end='')
        np.save('/home/lipu/HP_event/cache/cache_long_tail/long_tail_12.npy', dataset)
        print("\tDone")
    else:
        DataItem = DataItem18

        print(f"load data from /home/lipu/HP_event/cache/cache_long_tail/long_tail_french.npy ... ", end='')
        df = np.load("/home/lipu/HP_event/cache/cache_long_tail/long_tail_french.npy", allow_pickle=True)
        print("\tDone")
        data = [DataItem(*it) for it in df]
        data = pd.DataFrame(data, columns=COLUMNS_18)
        print("create train,valid and test sets... ")
        dataset = split_train_test_validation(data)
        print("\tDone")
        print(f"save data to '/home/lipu/HP_event/cache/cache_long_tail/long_tail_french_18.npy' ... ", end='')
        np.save('/home/lipu/HP_event/cache/cache_long_tail/long_tail_french_18.npy', dataset)
        print("\tDone")



    exit(0)
