# Codes of all methods
file "RPLM_SED" is the code of "Relational Prompt-based Pre-trained Language Models for Social Event Detection" (Ours).

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

## To run FinEvent Offline
step 1-3 ditto (change the file path)

step 4. run offline.py

## To run FinEvent Incremental
step 1. run utils/generate_initial_features.py to generate the initial features for the messages

step 2. run utils/custom_message_graph.py to construct incremental message graphs. To construct small message graphs for test purpose, set test=True when calling construct_incremental_dataset_0922(). To use all the messages (see Appendix of the paper for a statistic of the number of messages in the graphs), set test=False.

step 3. run utils/save_edge_index.py in advance to acclerate the training process.

step 4. run main.py



## To run FinEvent Cross-lingual
step 1-3 ditto (change the file path)

step 4. run main.py (Train a model from high-source dataset)

step 5. run resume.py



# Citation
If you find this repository helpful, please consider citing the following paper.

# Reference
[1] Andrew J McMinn, Yashar Moshfeghi, and Joemon M Jose. 2013. Building a large-scale corpus for evaluating event detection on twitter. In Proceedings of the CIKM.ACM, 409–418.

[2] Cao, Yuwei, et al. "Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs." Proceedings of the Web Conference 2021. 2021.

[3] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. In Proceedings of ICLR.

[4] David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. JMLR 3, Jan (2003), 993–1022.

[5] Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. 2015. From word embeddings to document distances. In Proceedings of the ICML. 957–966.

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

[7] Hao Peng, Jianxin Li, Qiran Gong, Yangqiu Song, Yuanxing Ning, Kunfeng Lai, and Philip S. Yu. 2019. Fine-grained event categorization with heterogeneous graph convolutional networks. In Proceedings of the IJCAI. 3238–3245.

[8] Bang Liu, Fred X Han, Di Niu, Linglong Kong, Kunfeng Lai, and Yu Xu. 2020. Story Forest: Extracting Events and Telling Stories from Breaking News. TKDD 14, 3 (2020), 1–28.

[9] Alex Graves and Jürgen Schmidhuber. 2005. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural networks 18, 5-6 (2005), 602–610.
