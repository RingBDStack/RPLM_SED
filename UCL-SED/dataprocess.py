import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import datetime

def str2list(str_ele):
    if str_ele == "[]":
        value = []
    else:
        value = [e.replace('\'','').lstrip().replace(":",'') for e in str(str_ele)[1:-1].split(',') if len(e.replace('\'','').lstrip().replace(":",''))>0]
    return value

def load_data(dataset):
    # assert (dataset in ["CrisisLexT26_1"]), "dataset not found"
    data_path = "../data/" + dataset
    if "Twitter" in dataset:
        lang = dataset.split("_")[0]
        if lang == "English":
            p_part1 = '68841_tweets_multiclasses_filtered_0722_part1.npy'
            p_part2 = '68841_tweets_multiclasses_filtered_0722_part2.npy'
            df_np_part1 = np.load(data_path + "/" +p_part1, allow_pickle=True)
            df_np_part2 = np.load(data_path + "/" +p_part2, allow_pickle=True)
            df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
            ori_df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc", \
                                                   "place_type", "place_full_name", "place_country_code", "hashtags",
                                                   "user_mentions", "urls", "entities",
                                                   "words", "filtered_words", "sampled_words"])


        else:
            all = np.load(data_path + '/All_{}.npy'.format(lang), allow_pickle=True)
            ori_df = pd.DataFrame(data=all, columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
                                                   "hashtags", "urls", "words", "created_at", "filtered_words",
                                                   "entities",
                                                   "sampled_words"])
            if "French" in dataset:
                ori_df['text'] = ori_df['words'].apply(lambda x:" ".join(x))
        ori_df.drop_duplicates(["tweet_id"], keep='first', inplace=True)
        event_id_num_dict = {}
        select_index_list = []

        for id in set(ori_df["event_id"]):
            num = len(ori_df.loc[ori_df["event_id"] == id])
            if int(num/3)>=25:
                event_id_num_dict[id] = int(num/3+50)
                select_index_list += list(ori_df.loc[ori_df["event_id"]==id].index)[0:int(num/3+50)]
        select_df = ori_df.loc[select_index_list]
        select_df = select_df.reset_index(drop=True)
        id_num = sorted(event_id_num_dict.items(),key=lambda x:x[1],reverse=True)
        for (i,j) in id_num[0:100]:
            print(j,end=",")
        sorted_id_dict = dict(zip(np.array(id_num)[:,0],range(0,len(set(ori_df["event_id"])))))
        sorted_df = select_df
        sorted_df["event_id"] = sorted_df["event_id"].apply(lambda x:sorted_id_dict[x])



    elif "CrisisLexT26" in dataset:
        ori_df = pd.DataFrame(pd.read_csv(data_path + "/" + data_path.split("/")[-1] + ".csv"))
        for col in ['user', 'hashtag', 'Entity', 'url', 'filter_word']:
            ori_df[col] = ori_df[col].apply(lambda x: str2list(x))
        for i, time in enumerate(ori_df['Timestamp'].values):
            if type(time) != str or len(time) != 19:
                print(i, time)
        ori_df['Timestamp'] = ori_df['Timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data_value = ori_df[['Tweet ID', 'user', ' Tweet Text', 'hashtag', 'Entity', 'url', 'filter_word', 'Timestamp',
                             'Event_id']].values
        ori_df = pd.DataFrame(data=data_value,
                                columns=["tweet_id", "user_mentions", "text", "hashtags", "entities", "urls",
                                         "filtered_words", "created_at", "event_id"])
        id2num = {}
        id2id = {}
        for id in set(ori_df["event_id"]):
            num = int(len(ori_df.loc[ori_df["event_id"] == id]))
            id2num[id] = num
        sorted_id2num = sorted(id2num.items(), key=lambda x: x[1], reverse=True)
        print(sorted_id2num)
        event_id_num_dict = {}
        select_index_list = []
        for i, (id, num) in enumerate(sorted_id2num):
            num = min(int((sorted_id2num[0][1] - 50) * pow(0.5, i)+50), num)
            event_id_num_dict[id] = num
            select_index_list += list(ori_df.loc[ori_df["event_id"] == id].index)[0:int(num)]
            id2id[id] = i
        print(event_id_num_dict.values())
        sorted_df = ori_df.loc[select_index_list]
        sorted_df = sorted_df.reset_index(drop=True)
        sorted_df['event_id'] = sorted_df['event_id'].apply(lambda x: id2id[x])

    print(sorted_df.shape)
    data_value = sorted_df[["tweet_id","user_mentions","text","hashtags","entities","urls","filtered_words","created_at", "event_id"]].values
    event_df = pd.DataFrame(data = data_value, columns=["tweet_id", "mention_user", "text", "hashtags", "entities", "urls", "filtered_words", "timestamp", "event_id"])
    event_df['hashtags'] = event_df['hashtags'].apply(lambda x: ["h_"+ i for i in x])
    event_df['entities'] = event_df['entities'].apply(lambda x: ["e_" + str(i) for i in x])
    event_df['mention_user'] = event_df['mention_user'].apply(lambda x: ["u_" + str(i) for i in x])
    event_df = event_df.loc[event_df['event_id']<100]
    event_df = event_df.reset_index(drop=True)

    print(event_df.shape)
    return event_df
if __name__ == "__main__":
    event_df = load_data("French_Twitter")
