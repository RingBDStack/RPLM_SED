# Codes of all methods
file "RPLM-SED" is the code of "Relational Prompt-based Pre-trained Language Models for Social Event Detection" (Ours).

file "LoRA-RPLM" is the code of the variants of RPLM_SED which are fine-tuned by using Low-Rank Adaptation (LoRA).

file "ori_PLM" is the code of fine-tuning or no fine-tuning PLMS to achieve SED.

file "CLKD" is the code of "Towards Cross-lingual Social Event Detection with Hybrid Knowledge Distillation" [1].

file "EventX" is the code of "Story Forest: Extracting Events and Telling Stories from Breaking News" [2].

file "FinEvent" is the code of "Reinforced, Incremental and Cross-Lingual Event Detection From Social Messages" [3].

file "KPGNN" is the code of "Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs" [4].

file "QSGNN" is the code of "From Known to Unknown: Quality-aware Self-improving Graph Neural Network for Open Set Social Event Detection" [5].

file "PPGCN" is the code of "Fine-grained Event Categorization with Heterogeneous Graph Convolutional Networks" [6].

file "UCL-SED" is the code of "Uncertainty-guided Boundary Learning for Imbalanced Social Event Detection" [7].


# Twitter Datasets
The Events2012 [8] dataset contains 68,841 annotated English tweets covering 503 different event categories, encompassing tweets over a consecutive 29-day period. The Event2012_100 contains 100 events with a total of 15,019 tweets, where the maximal event comprises 2,377 tweets, and the minimally has 55 tweets, with an imbalance ratio of approximately 43.

The Events2018 [9] includes 64,516 annotated French tweets covering 257 different event
categories, with data spanning over a consecutive 23-day period. The Event2018_100 contains 100 events
with a total of 19,944 tweets, where the maximal event comprises 4,189 tweets and the minimally has 27
tweets, an imbalance ratio of approximately 155.

The Arabic-Twitter [10] dataset comprises 9,070 annotated Arabic tweets, covering seven
catastrophic-class events from various periods.

# Function Mode
offline + online  + Low-Resource + Long-tail Recognition

## To run RPLM_SED offline and online
Step 1. run RPLM-SED/twitter_process.py for data preprocessing.

step 2. run RPLM-SED/train_long_tail.py. To change related paths, including the PLM used, model storage path, obtained message representation storage path, etc. Modify the parameter "offline" to False (online) or True (offline).

## To run RPLM_SED Low-Resource
Step 1. run RPLM-SED/arbic_process.py.py  for data preprocessing.

step 2. run RPLM-SED/train_tail.py. To change related paths, including the PLM used, model storage path, obtained message representation storage path, etc. Modify the parameter "offline" to True.

## To run RPLM_SED  Long-tail Recognition
Step 1. run RPLM-SED/long_tail_process.py  for data preprocessing.

step 2. run RPLM-SED/train_long_tail.py. To change related paths, including the PLM used, model storage path, obtained message representation storage path, etc.





# Citation
If you find this repository helpful, please consider citing the following paper.

# Reference
[1] Jiaqian Ren, Hao Peng, Lei Jiang, Jia Wu, Yongxin Tong, Lihong Wang, Xu Bai, Bo Wang, and Qiang Yang. 2021. Transferring knowledge
distillation for multilingual social event detection. arXiv preprint arXiv:2108.03084 (2021), 1–31.

[2] Bang Liu, Fred X Han, Di Niu, Linglong Kong, Kunfeng Lai, and Yu Xu. 2020. Story forest: Extracting events and telling stories from
breaking news. ACMTransactions on Knowledge Discovery from Data (TKDD) 14, 3 (2020), 1–28.

[3] Hao Peng, Ruitong Zhang, Shaoning Li, Yuwei Cao, Shirui Pan, and Philip S. Yu. 2022. Reinforced, incremental and cross-lingual event
detection from social messages. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 1 (2022), 980–998.

[4] Yuwei Cao, Hao Peng, Jia Wu, Yingtong Dou, Jianxin Li, and Philip S. Yu. 2021. Knowledge-preserving incremental social event detection
via heterogeneous gnns. In Proceedings ofthe Web Conference 2021. 3383–3395.

[5] Jiaqian Ren, Lei Jiang, Hao Peng, Yuwei Cao, Jia Wu, Philip S. Yu, and Lifang He. 2022. From known to unknown: quality-aware
self-improving graph neural network for open set social event detection. In Proceedings ofthe 31st ACM International Conference on
Information & Knowledge Management. 1696–1705.

[6] Hao Peng, Jianxin Li, Qiran Gong, Yangqiu Song, Yuanxing Ning, Kunfeng Lai, and Philip S. Yu. 2019. Fine-grained event categorization
with heterogeneous graph convolutional networks. In Proceedings ofthe 28th International Joint Conference on Artificial Intelligence.
3238–3245.

[7] Jiaqian Ren, Hao Peng, Lei Jiang, Zhiwei Liu, Jia Wu, Zhengtao Yu, and Philip S. Yu. 2023. Uncertainty-guided Boundary Learning for
Imbalanced Social Event Detection. IEEE Transactions on Knowledge and Data Engineering (2023), 1–15.

[8] Andrew J McMinn, Yashar Moshfeghi, and Joemon M Jose. 2013. Building a large-scale corpus for evaluating event detection on twitter.
In Proceedings ofthe 22nd ACM international conference on Information & Knowledge Management. ACM, 409–418.

[9] Béatrice Mazoyer, Julia Cagé, Nicolas Hervé, and Céline Hudelot. 2020. A French Corpus for Event Detection on Twitter. In Proceedings
ofthe Twelfth Language Resources and Evaluation Conference. 6220–6227.

[10] Alaa Alharbi and Mark Lee. 2021. Kawarith: an Arabic Twitter corpus for crisis events. In Proceedings ofthe Sixth Arabic Natural
Language ProcessingWorkshop. Association for Computational Linguistics, 42–52.
