import datetime
import itertools
import os

import math
from typing import List

import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split

from datasets import DATA_PATH_12, DataItem12, SAMPLE_NUM_TWEET,WINDOW_SIZE


def split_train_test_validation(data: List):
    block=[]
    off_dataset = []
    for i in range(len(data)):
        if i == 0:
            data_size = len(data[i])
            valid_size = math.ceil(data_size *0.2)
            train, valid = train_test_split(data[i], test_size=valid_size, random_state=42, shuffle=True)
            block.append({"train": train, "test": [], "valid": valid})

        elif i % WINDOW_SIZE == 0:

            sub_data = []
            for j in range(WINDOW_SIZE):
                sub_data += data[i-j]
            sub_data_size = len(sub_data)
            sub_valid_size = math.ceil( sub_data_size * 0.2)
            train, valid = train_test_split(sub_data, test_size=sub_valid_size, random_state=42, shuffle=True)
            block.append({"train": train, "test": data[i], "valid": valid})
        else:
            block.append({"train": [], "test": data[i], "valid": []})

    return block


def split_into_blocks(data):
    data = [DataItem12(*it) for it in data]
    data = sorted(data, key=lambda it: it.created_at)
    groups = itertools.groupby(data, key=lambda it: it.created_at.timetuple().tm_yday)
    groups = {k: list(g) for k, g in groups}

    days = sorted(groups.keys())
    blk0 = [groups[d] for d in days[:7]]
    blk0 = [it for b in blk0 for it in b]

    day_blk = [groups[d] for d in days[7:-1]]

    blocks = [blk0] + day_blk

    return split_train_test_validation(blocks)

def pre_process(data):
    print("split data into blocks... ")
    blocks = split_into_blocks(data)
    print("\tDone")

    return blocks


if __name__ == '__main__':
    p_part1 = f'{DATA_PATH_12}/68841_tweets_multiclasses_filtered_0722_part1.npy'
    p_part2 = f'{DATA_PATH_12}/68841_tweets_multiclasses_filtered_0722_part2.npy'
    if not os.path.exists('/home/lipu/HP_event/cache/cache_ori_rbert'):
        os.makedirs('/home/lipu/HP_event/cache/cache_ori_rbert')

    print(f"load data from {DATA_PATH_12} ... ", end='')
    np_part1 = np.load(p_part1, allow_pickle=True)
    np_part2 = np.load(p_part2, allow_pickle=True)
    np_data = np.concatenate((np_part1, np_part2), axis=0)
    print("\tDone")

    blk_data = pre_process(np_data)
    print(f"save data to '/home/lipu/HP_event/cache/cache_ori_rbert/twitter12.npy' ... ", end='')
    np.save('/home/lipu/HP_event/cache/cache_ori_rbert/twitter12.npy', blk_data)
    print("\tDone")


    exit(0)
