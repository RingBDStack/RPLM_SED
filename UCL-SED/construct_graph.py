import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse
import datetime
import dgl
import en_core_web_lg
from dataprocess import load_data
import spacy
import fr_core_news_lg

def get_nlp(lang):
    if lang == "English" or lang == "Arabic":
        nlp = en_core_web_lg.load()
    elif lang == "French":
        nlp = fr_core_news_lg.load()
    return nlp

def construct_graph_base_eles(view_dict, df, path, lang = 'English'):
    '''
    nlp = get_nlp(lang)
    df = df.drop_duplicates(subset=['tweet_id'])
    df.reset_index()
    df.drop_duplicates(["tweet_id"],keep='first',inplace=True)
    print("generate text features---------")
    features = np.stack(df['filtered_words'].apply(lambda x: nlp(' '.join(x)).vector).values, axis=0)
    print(features.shape)
    np.save(path + "features.npy", features)
    print("text features are saved in {}features.npy".format(path))
    np.save(path + "time.npy", df['timestamp'].values)
    print("time features are saved in {}time.npy".format(path))
    df["event_id"] = df["event_id"].apply(lambda x:int(x))
    np.save(path + "label.npy", df['event_id'].values)
    print("labels are saved in {}label.npy".format(path))
    '''    
    true_matrix = np.eye(df.shape[0])
    for i in range(df.shape[0]):
        label_i = df["event_id"].values[i]
        indices = df[df["event_id"]==label_i].index
        true_matrix[i,indices] = 1
    print(true_matrix)
    
   # for view in ['h','e','u']:
    view_matrix = sparse.load_npz("../data/French_Twitter_kpgnn/" + "s_tweet_tweet_matrix_all.npz").todense().astype(int)
    print(view_matrix)
    print(true_matrix.shape,view_matrix.shape)
    all_num = view_matrix.sum()
    true_num = true_matrix[np.where(view_matrix>0)].sum()
    print(true_num/all_num)
        #print(np.where(true_matrix==view_matrix)[0].shape)
    '''
    print("construct graph---------------")
    G = nx.Graph()
    for _, row in df.iterrows():
        tid = str(row['tweet_id'])
        G.add_node(tid)
        G.nodes[tid]['tweet_id'] = True  # right-hand side value is irrelevant for the lookup
        edges = []
        for view in view_dict.values():
            for ele in view:
                if len(row[ele]) > 0 :
                    ele_values = row[ele]
                    G.add_nodes_from(ele_values)
                    for each in ele_values:
                        G.nodes[each][ele] = True
                    edges += [(tid, each) for each in row[ele]]

        G.add_edges_from(edges)

    # co_hashtags = np.load("../data/%s/co_hashtags.npy" % dataset)
    # G.add_edges_from(co_hashtags)

    all_nodes = list(G.nodes)
    matrix = nx.to_numpy_array(G)
    tweet_nodes = list(nx.get_node_attributes(G, "tweet_id").keys())
    print(tweet_nodes)
    print(len(tweet_nodes))
    tweet_index = [all_nodes.index(t_node)  for t_node in tweet_nodes]
    for v, view in zip(view_dict.keys(), view_dict.values()):
        s_tweet_tweet_matrix = sparse.csr_matrix(np.identity(len(tweet_nodes)))
        for ele in view:
            ele_nodes = list(nx.get_node_attributes(G, ele).keys())
            ele_index = [all_nodes.index(e_node) for e_node in ele_nodes]
            tweet_ele_matrix = matrix[np.ix_(tweet_index, ele_index)]
            # if ele == "hashtags":
            #     print("hashtags co_matrix-------")
            #     tweet_hashtags_matrix = matrix[np.ix_(ele_index,ele_index)]
            #     print(tweet_hashtags_matrix)
            #     tweet_ele_matrix = tweet_ele_matrix + np.matmul(tweet_ele_matrix, tweet_hashtags_matrix)
            s_ele_tweet_tweet_matrix = sparse.csr_matrix(np.matmul(tweet_ele_matrix, tweet_ele_matrix.transpose()))
            s_tweet_tweet_matrix += s_ele_tweet_tweet_matrix
        s_tweet_tweet_matrix = s_tweet_tweet_matrix.astype('bool')
        sparse.save_npz(path + "s_tweet_tweet_matrix_{}.npz".format(v), s_tweet_tweet_matrix)
        print("Sparse binary {} commuting matrix is saved in {}s_tweet_tweet_matrix.npz".format(v,path))

    # edges = np.where(s_tweet_tweet_matrix.todense()==True)
    # edges = np.array(edges).reshape([-1,2])
    # for i in range(edges.shape[0]):
    #     src_id = tweet_nodes[edges[i,0]]
    #     tgt_id = tweet_nodes[edges[i,1]]
    #     src_time = df[df['tweet_id'] == int(src_id)]['timestamp'].values[0]
    #     tgt_time = df[df['tweet_id'] == int(tgt_id)]['timestamp'].values[0]
    #     delta = abs((src_time - tgt_time).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
        #G.add_weighted_edges_from([(tweet_nodes[edges[i,0]],tweet_nodes[edges[i,1]],weight)])
    '''

dataset = "French_Twitter"
event_df = load_data(dataset)
view_dict = {"h":["hashtags","urls"],"u":["mention_user"], "e":["entities"]}
construct_graph_base_eles(view_dict,event_df,"../data/{}/".format(dataset),"English")
