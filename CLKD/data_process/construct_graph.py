import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from scipy import sparse
import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import pickle
from collections import Counter
from time import time
import os
import argparse


# construct a graph using tweet ids, user ids, entities and hashtags


def construct_graph_from_df(df, G=None):
    if G is None:
        G = nx.Graph()
    for _, row in df.iterrows():
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)
        G.nodes[tid]['tweet_id'] = True

        user_ids = row['user_mentions']
        user_ids.append(row['user_id'])
        user_ids = ['u_' + str(each) for each in user_ids]
        G.add_nodes_from(user_ids)
        for each in user_ids:
            G.nodes[each]['user_id'] = True

        entities = row['entities']
        entities = ['e_' + str(each) for each in entities]
        G.add_nodes_from(entities)
        for each in entities:
            G.nodes[each]['entity'] = True

        hashtags = row['hashtags']
        hashtags = ['h_' + str(each) for each in hashtags]
        G.add_nodes_from(hashtags)
        for each in hashtags:
            G.nodes[each]['hashtag'] = True

        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities]
        edges += [(tid, each) for each in hashtags]
        G.add_edges_from(edges)

    return G


# convert networkx graph to dgl graph and store its sparse binary adjacency matrix
def to_dgl_graph_v3(G, save_path=None):
    message = ''
    print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
    message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
    all_start = time()

    print('\tGetting a list of all nodes ...')
    message += '\tGetting a list of all nodes ...\n'
    start = time()
    all_nodes = list(G.nodes)
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # print('All nodes: ', all_nodes)
    # print('Total number of nodes: ', len(all_nodes))

    print('\tGetting adjacency matrix ...')
    message += '\tGetting adjacency matrix ...\n'
    start = time()
    A = nx.to_numpy_matrix(G)  # Returns the graph adjacency matrix as a NumPy matrix.
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute commuting matrices
    print('\tGetting lists of nodes of various types ...')
    message += '\tGetting lists of nodes of various types ...\n'
    start = time()
    tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
    userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
    hash_nodes = list(nx.get_node_attributes(G, 'hashtag').keys())
    entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
    del G
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\tConverting node lists to index lists ...')
    message += '\tConverting node lists to index lists ...\n'
    start = time()
    indices_tid = [all_nodes.index(x) for x in tid_nodes]
    indices_userid = [all_nodes.index(x) for x in userid_nodes]
    indices_hashtag = [all_nodes.index(x) for x in hash_nodes]
    indices_entity = [all_nodes.index(x) for x in entity_nodes]
    del tid_nodes
    del userid_nodes
    del hash_nodes
    del entity_nodes
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-user-tweet
    print('\tStart constructing tweet-user-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-user matrix ...')
    message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user matrix ...\n'
    start = time()
    w_tid_userid = A[np.ix_(indices_tid, indices_userid)]
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_userid = sparse.csr_matrix(w_tid_userid)
    del w_tid_userid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_userid_tid = s_w_tid_userid.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-user * user-tweet ...')
    message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
    start = time()
    s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
        print("Sparse binary userid commuting matrix saved.")
        del s_m_tid_userid_tid
    del s_w_tid_userid
    del s_w_userid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-ent-tweet
    print('\tStart constructing tweet-ent-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-ent matrix ...')
    message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ...\n'
    start = time()
    w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
    del w_tid_entity
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_entity_tid = s_w_tid_entity.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-ent * ent-tweet ...')
    message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
    start = time()
    s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
        print("Sparse binary entity commuting matrix saved.")
        del s_m_tid_entity_tid
    del s_w_tid_entity
    del s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-hashtag-tweet
    print('\tStart constructing tweet-hashtag-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-hashtag matrix ...')
    message += '\tStart constructing tweet-hashtag-tweet commuting matrix ...\n\t\t\tStart constructing tweet-hashtag matrix ...\n'
    start = time()
    w_tid_hash = A[np.ix_(indices_tid, indices_hashtag)]
    del A
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_hash = sparse.csr_matrix(w_tid_hash)
    del w_tid_hash
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_hash_tid = s_w_tid_hash.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-hashtag * hashtag-tweet ...')
    message += '\t\t\tCalculating tweet-hashtag * hashtag-tweet ...\n'
    start = time()
    s_m_tid_hash_tid = s_w_tid_hash * s_w_hash_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_hash_tid.npz", s_m_tid_hash_tid)
        print("Sparse binary hashtag commuting matrix saved.")
        del s_m_tid_hash_tid
    del s_w_tid_hash
    del s_w_hash_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute tweet-tweet adjacency matrix
    print('\tComputing tweet-tweet adjacency matrix ...')
    message += '\tComputing tweet-tweet adjacency matrix ...\n'
    start = time()
    if save_path is not None:
        s_m_tid_userid_tid = sparse.load_npz(save_path + "s_m_tid_userid_tid.npz")
        print("Sparse binary userid commuting matrix loaded.")
        s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
        print("Sparse binary entity commuting matrix loaded.")
        s_m_tid_hash_tid = sparse.load_npz(save_path + "s_m_tid_hash_tid.npz")
        print("Sparse binary hashtag commuting matrix loaded.")

    s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
    del s_m_tid_userid_tid
    del s_m_tid_entity_tid
    s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_hash_tid).astype('bool')
    del s_m_tid_hash_tid
    del s_A_tid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'
    all_mins = (time() - all_start) / 60
    print('\tOver all time elapsed: ', all_mins, ' mins\n')
    message += '\tOver all time elapsed: '
    message += str(all_mins)
    message += ' mins\n'

    if save_path is not None:
        sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
        print("Sparse binary adjacency matrix saved.")
        s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
        print("Sparse binary adjacency matrix loaded.")

    # create corresponding dgl graph
    G = dgl.DGLGraph(s_bool_A_tid_tid)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    message += 'We have '
    message += str(G.number_of_nodes())
    message += ' nodes.'
    message += 'We have '
    message += str(G.number_of_edges())
    message += ' edges.\n'

    return all_mins, message


def construct_incremental_dataset_0922(args, df, save_path, features, nfeatures, test=False):
    data_split = []
    all_graph_mins = []
    message = ""
    # extract distinct dates
    distinct_dates = df.date.unique()
    # print("Distinct dates: ", distinct_dates)
    print("Number of distinct dates: ", len(distinct_dates))
    message += "Number of distinct dates: "
    message += str(len(distinct_dates))
    message += "\n"
    print("Start constructing initial graph ...")
    message += "\nStart constructing initial graph ...\n"


    if args.is_static:
        ini_df = df.loc[df['date'].isin(distinct_dates[:args.days])]
        days = args.days
    else:
        if args.lang != "Mix":
            ini_df = df.loc[df['date'].isin(distinct_dates[:1])]
            days = 1
        else:
            ini_df = df.loc[df['date'].isin(distinct_dates[:1])]
            days = 1

    print("Initial graph contains %d days"%(days))
    message += "Initial graph contains %d days\n"%(days)

    path = save_path + '0/'
    if not os.path.exists(path):
        os.mkdir(path)

    y = ini_df['event_id'].values
    y = [int(each) for each in y]
    np.save(path + 'labels.npy', np.asarray(y))

    G = construct_graph_from_df(ini_df)
    grap_mins, graph_message = to_dgl_graph_v3(G, save_path=path)
    message += graph_message
    print("Initial graph saved")
    message += "Initial graph saved\n"
    # record the total number of tweets
    data_split.append(ini_df.shape[0])
    # record the time spent for graph conversion
    all_graph_mins.append(grap_mins)
    # extract and save the labels of corresponding tweets
    y = ini_df['event_id'].values
    y = [int(each) for each in y]
    np.save(path + 'labels.npy', np.asarray(y))
    np.save(path + 'df.npy', ini_df)
    # ini_df['created_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    np.save(path + "time.npy", ini_df['created_at'].values)
    print("Labels and times saved.")
    message += "Labels and times saved.\n"
    # extract and save the features of corresponding tweets
    indices = ini_df['index'].values.tolist()
    x = features[indices, :]
    np.save(path + 'features.npy', x)
    print("Features saved.")
    message += "Features saved."
    if nfeatures is not None:
        # save trans nonlinear features
        nx = nfeatures[indices, :]
        np.save(path + '{}-{}-features.npy'.format(args.lang, args.tgtlang), nx)
        print("trans features saved")
        message += "Nonlinear Trans Features saved.\n\n"

    if not args.is_static:
        if args.lang == "Mix":
            inidays = 1
            j = 0
        else:
            inidays = 1
            j = 0
        for i in range(inidays, len(distinct_dates)):
            print("Start constructing graph ", str(i - j), " ...")
            message += "\nStart constructing graph "
            message += str(i - j)
            message += " ...\n"
            incr_df = df.loc[df['date'] == distinct_dates[i]]
            path = save_path + str(i - j) + '/'
            if not os.path.exists(path):
                os.mkdir(path)
            np.save(path + "/" + "dataframe.npy", incr_df)

            G = construct_graph_from_df(
                incr_df)  # remove obsolete, version 2: construct graph using only the data of the day
            grap_mins, graph_message = to_dgl_graph_v3(G, save_path=path)
            message += graph_message
            print("Graph ", str(i - j), " saved")
            message += "Graph "
            message += str(i - j)
            message += " saved\n"
            # record the total number of tweets
            data_split.append(incr_df.shape[0])
            # record the time spent for graph conversion
            all_graph_mins.append(grap_mins)
            # extract and save the labels of corresponding tweets
            # y = np.concatenate([y, incr_df['event_id'].values], axis = 0)
            y = [int(each) for each in incr_df['event_id'].values]
            np.save(path + 'labels.npy', y)
            # incr_df['created_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            np.save(path + "time.npy", incr_df['created_at'].values)
            print("Labels saved.")
            message += "Labels saved.\n"
            # extract and save the features of corresponding tweets
            indices = incr_df['index'].values.tolist()
            x = features[indices, :]
            # x = np.concatenate([x, x_incr], axis = 0)
            np.save(path + 'features.npy', x)
            np.save(path + 'df.npy', incr_df)
            print("Features saved.")
            message += "Features saved."
            if nfeatures is not None:
                # save trans nonlinear features
                nx = nfeatures[indices, :]
                np.save(path + '{}-{}-features.npy'.format(args.lang, args.tgtlang), nx)
                print("trans features saved")
                message += "trans features saved.\n"
    return message, data_split, all_graph_mins


def main(args):
    # create save path
    if args.is_static:
        save_path = "../datasets/318_hash_static-{}-{}/".format(str(args.days), args.lang)
    else:
        save_path = "../datasets/318_ALL_{}/".format(args.lang)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

     # load df data
    if args.lang == "Mix":
        filename = "../datasets/Mix_Twitter/mix20000_nonlinear_fea.npy"
        # filename = './baselines/mbert_{}_features.npy'.format(lang)
        dfpath = "../datasets/Mix_Twitter/mix20000df.npy"
        df = np.load(dfpath, allow_pickle=True)
        df = pd.DataFrame(columns=['tweet_id', 'user_id', 'user_mentions', 'event_id', 'created_at', 'text', 'entities',
                                   'hashtags', 'image_urls', 'filtered_words', 'lang'], data=df)

    if args.lang == "French" or args.lang == "Arabic":
        df_np = np.load('../datasets/{}_Twitter/All_{}.npy'.format(args.lang, args.lang), allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["tweet_id", "user_id", "text", "time", "event_id", "user_mentions",
                                               "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
                                               "sampled_words"])
        if args.lang == "Arabic":
            name2id = {}
            for id, name in enumerate(df['event_id'].unique()):
                name2id[name] = id
            print(name2id)
            df['event_id'] = df['event_id'].apply(lambda x: name2id[x])
            df.drop_duplicates(['tweet_id'], inplace=True, keep='first')

    elif args.lang == "English":
        p_part1 = '../datasets/English_Twitter/68841_tweets_multiclasses_filtered_0722_part1.npy'
        p_part2 = '../datasets/English_Twitter/68841_tweets_multiclasses_filtered_0722_part2.npy'
        df_np_part1 = np.load(p_part1, allow_pickle=True)
        df_np_part2 = np.load(p_part2, allow_pickle=True)
        df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
        df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc", \
                                               "place_type", "place_full_name", "place_country_code", "hashtags",
                                               "user_mentions", "image_urls", "entities",
                                               "words", "filtered_words", "sampled_words"])
    print("{} Data converted to dataframe.".format(args.lang))


    # sort data by time
    df = df.sort_values(by='created_at').reset_index()
    # append date
    df['date'] = [d.date() for d in df['created_at']]

    nf = None
    # load features
    if args.lang == "Mix":
        f = np.load(filename)
    else:
        f = np.load('../datasets/features/features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}.npy'.format(args.lang))
        nonleafilename = "../datasets/features/features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}_{}.npy".format(args.lang,args.tgtlang)
        nf = np.load(nonleafilename)

    # construct graph
    message, data_split, all_graph_mins = construct_incremental_dataset_0922(args, df, save_path, f, nf, False)
    with open(save_path + "node_edge_statistics.txt", "w") as text_file:
        text_file.write(message)
    np.save(save_path + 'data_split.npy', np.asarray(data_split))
    print("Data split: ", data_split)
    np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
    print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_static', type=bool, default=False)
    parser.add_argument('--lang', type=str, default='English')
    parser.add_argument('--tgtlang', type=str, default='French')
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    main(args)