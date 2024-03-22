"""
This file generates the initial message features

To leverage the semantics in the data, we generate document feature for each message,
which is calculated as an average of the pre-trained word embeddings of all the words in the message.
We use the word embeddings pre-trained by en_core_web_lg for English data, fr_core_web_lg for French data, and aravec (see https://github.com/bakrianoo/aravec) for Arabic data.

To leverage the temporal information in the data, we generate temporal feature for each message,
which is calculated by encoding the times-tamps: we convert each timestamp to OLE date,
whose fractional and integral components form a 2-d vector.

The initial feature of a message is the concatenation of its document feature and temporal feature.
"""

import numpy as np
import pandas as pd
import en_core_web_lg
from datetime import datetime
import spacy
import fr_core_news_lg
from Arabic_preprocess import Preprocessor
import torch
import argparse
import os

# Calculate the embeddings of all the documents in the dataframe,
# the embedding of each document is an average of the pre-trained embeddings of all the words in it

# encode messages to get features
def documents_to_features(df,lang):
    if lang =="French":
        nlp = fr_core_news_lg.load()
    elif lang == "Arabic":
        nlp = spacy.load('spacy.arabic.model')
        nlp.tokenizer = Preprocessor(nlp.tokenizer)
    elif lang == "English":
        nlp = en_core_web_lg.load()
    else:
        print("not have that language!")

    features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector if len(x)!=0 else nlp(' ').vector).values
    print(features)
    return np.stack(features, axis=0)

# get word2id dict and corresponding embeddings of words
def get_word2id_emb(wordpath,embpath):
    word2id = {}
    with open(wordpath, 'r') as f:
        for i, w in enumerate(list(f.readlines()[0].split())):
            word2id[w] = i
    embeddings = np.load(embpath)
    return word2id,embeddings

# get transformed features in a nonlinear way
def nonlinear_transform_features(wordpath,embpath,df):
    word2id,embeddings = get_word2id_emb(wordpath,embpath)
    features = df.filtered_words.apply(lambda x: [embeddings[word2id[w]] for w in x])
    f_list = []
    for f in features:
        if len(f) != 0:
            f_list.append(np.mean(f, axis=0))
        else:
            f_list.append(np.zeros((300)))
    features = np.stack(f_list, axis=0)
    print(features.shape)
    return features

# get transformed features in a linear way
def getlinear_transform_features(features,src,tgt):
    W = torch.load("../datasets/dictrans/LinearTranWeight/spacy_{}_{}/best_mapping.pth".format(src,tgt))
    features = np.matmul(features,W)
    return features

# encode one times-tamp
# t_str: a string of format '2012-10-11 07:19:34'
def extract_time_feature(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

# encode the times-tamps of all the messages in the dataframe
def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features



def main():
    load_path = '../datasets/{}_Twitter/'.format(args.lang)
    save_path = '../datasets/features/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # load data
    if args.lang in ["French","Arabic"]:
        all = np.load(load_path + '/All_{}.npy'.format(args.lang), allow_pickle=True)
        df_np = all
        df = pd.DataFrame(data=df_np, columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
                                               "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
                                              "sampled_words"])
    elif args.lang == "English":
        p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
        p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
        df_np_part1 = np.load(p_part1, allow_pickle=True)
        df_np_part2 = np.load(p_part2, allow_pickle=True)
        df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
        df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc", \
                                               "place_type", "place_full_name", "place_country_code", "hashtags",
                                               "user_mentions", "image_urls", "entities",
                                               "words", "filtered_words", "sampled_words"])
        # np.save(load_path+'/All_English.npy',df_np)
    else:
        raise NotImplementedError("not contain that language")


    print("Loaded {} data  shape {}".format(args.lang, df.shape))
    print(df.head(10))

    t_features = df_to_t_features(df)
    print("Time features generated.")
    d_features = documents_to_features(df, args.lang)
    print("original dfeatures generated")

    combined_features = np.concatenate((d_features, t_features), axis=1)
    print("Concatenated document features and time features.")
    np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}.npy'.format(args.lang),
            combined_features)

    if args.TransLinear:
        dl_features = getlinear_transform_features(d_features,args.lang,args.tgt)
        lcombined_features = np.concatenate((dl_features, t_features), axis=1)
        print("linear trans features generated")
        np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}_{}.npy'.format(args.lang, args.tgt),
                lcombined_features)

    if args.TransNonlinear:
        dnl_features = nonlinear_transform_features(args.wordpath,args.embpath,df)
        dlcombined_features = np.concatenate((dnl_features, t_features), axis=1)
        print(dlcombined_features)
        print("Nonlinear trans features generated.")
        np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered_nonlinear-{}_{}.npy'.format(args.lang, args.tgt),
            dlcombined_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='French')
    parser.add_argument('--TransLinear',type=bool,default=True)
    parser.add_argument('--TransNonlinear',type=bool,default=True)
    parser.add_argument('--tgt',type=str,default='English')
    parser.add_argument('--embpath',type=str,default='../datasets/dictrans/fr-en-for.npy')
    parser.add_argument('--wordpath',type=str,default='../datasets/dictrans/wordsFrench.txt')
    args = parser.parse_args()
    main()