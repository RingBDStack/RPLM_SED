#!/usr/bin/env python
# encoding: utf-8
import logging
from argparse import ArgumentParser
import gc
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import RunningAverage, Average
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch import nn, softmax
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer

from longtail_model import PairPfxTuningEncoder
from utils import *

logging.basicConfig(level=logging.WARN,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')

# twhin-bert-large
def get_args():
    parser = ArgumentParser()

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    parser.add_argument('--dataset', type=str, default='/home/lipu/HP_event/cache/cache_long_tail/long_tail_12.npy')
    parser.add_argument('--plm_path', type=str, default='/home/lipu/HP_event/base_plm_model/roberta-large/')
    # parser.add_argument('--plm_tuning', action='store_true')
    parser.add_argument('--plm_tuning', type=bool,default=True)
    # parser.add_argument('--use_ctx_att', action='store_true')
    parser.add_argument('--use_ctx_att', type=bool,default=True)
    # parser.add_argument('--offline', action='store_true')
    parser.add_argument('--offline', type=bool,default=True)
    parser.add_argument('--ctx_att_head_num', type=int, default=2)
    parser.add_argument("--pmt_feats", type=list_of_ints, default=(0, 1, 2, 4),
                        help="(entities, hashtags, user, words, time)")

    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--lmda1", type=float, default=0.010)
    parser.add_argument("--lmda2", type=float, default=0.005)
    parser.add_argument("--tao", type=float, default=0.90)

    parser.add_argument("--optimizer", type=str, default='Adam', help="Optimizer, Adam, AdamW or SGD")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument("--step_lr_gamma", type=float, default=0.98, help="gamma for step learning rate schedule")

    parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs")
    #ablation_ckpt  ablation_Eva_datas
    parser.add_argument('--ckpt_path', type=str, default='/home/lipu/HP_event/ckpt/lotail_2012', help='path to checkpoint files')
    parser.add_argument("--eva_data", type=str, default="/home/lipu/HP_event/Eva_data/lotail_2012",help="path to Evaluate_datas")

    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_monitor", type=str, default='loss', help='loss')

    parser.add_argument("--device", type=str,
                        default="cuda:1" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    cfg = parser.parse_args()

    return cfg


def get_model(cfg):
    return PairPfxTuningEncoder(
        len(cfg.pmt_feats), cfg.plm_path, cfg.plm_tuning,
        use_ctx_att=cfg.use_ctx_att, ctx_att_head_num=cfg.ctx_att_head_num)


def initialize(model, cfg, num_train_batch):
    # parameters = model.parameters()  # 优化器的初始化
    parameters = [
        {
            'name': 'pair_cls',
            'params': model.pair_cls.parameters(),
            'lr': cfg.lr
        }, {
            'name': 'pfx_embedding',
            'params': model.pfx_embedding.parameters(),
            'lr': cfg.lr
        }
    ]

    if cfg.plm_tuning:
        parameters.append(
            {
                'name': 'encoder',
                'params': model.plm.parameters(),
                'lr': cfg.lr / 100.
            }
        )

    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RAdam':
        optimizer = optim.RAdam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    else:
        raise Exception("unsupported optimizer %s" % cfg.optimizer)

    lr_scheduler = None
    if cfg.step_lr_gamma > 0:
        lr_scheduler = StepLR(optimizer, step_size=num_train_batch, gamma=cfg.step_lr_gamma)

    return optimizer, lr_scheduler


def batch_to_tensor(batch, cfg):
    tags = [tag for tag, evt, a, b, fix, tok, _ in batch]
    events = [evt for tag, evt, a, b, fix, tok, _ in batch]
    prefix = [fix for tag, evt, a, b, fix, tok, _ in batch]
    toks = [tok for tag, evt, a, b, fix, tok, _ in batch]
    typs = [typ for tag, evt, a, b, fix, tok, typ in batch]

    max_len = min(max([len(it) for it in toks]), 200)
    toks = [pad_seq(it, pad=cfg.pad_tok_id, max_len=max_len) for it in toks]
    toks = torch.tensor(toks, dtype=torch.long)
    typs = [pad_seq(it, pad=cfg.pad_tok_id, max_len=max_len) for it in typs]
    typs = torch.tensor(typs, dtype=torch.long)
    tags = torch.tensor(tags, dtype=torch.long)
    events = torch.tensor(events, dtype=torch.long)
    prefix = torch.tensor(prefix, dtype=torch.long)

    return toks, typs, prefix, tags, events


# loss functions
cls_loss = torch.nn.BCEWithLogitsLoss()
dist_loss = torch.nn.PairwiseDistance()
label_loss = torch.nn.CrossEntropyLoss()
# y = torch.tensor([...])

# # 计算每个类别的样本数
# n_samples = torch.tensor([(y == class_id).sum() for class_id in range(100)])
#
# # 计算权重：类别出现频率的倒数
# weights = 1.0 / n_samples.float()
#
# # 归一化权重，使得最小权重为1.0（可选）
# weights = weights / weights.min()
#
# # 确保权重和类别标签的数据类型和设备匹配
# weights = weights.to(y.device)
def create_trainer(model, optimizer, lr_scheduler, cfg):
    evt_proto = torch.zeros((cfg.train_evt_num, model.feat_size()), device=cfg.device, requires_grad=False)

    # update event cluster center prototype by training batch
    def update_evt_proto(feats, events, alpha):
        proto = torch.zeros_like(evt_proto)
        proto.index_add_(0, events, feats)

        ev_idx, ev_idx_inv, ev_count = torch.unique(events, return_inverse=True, return_counts=True)
        proto_a = torch.index_select(proto, dim=0, index=ev_idx) / ev_count.unsqueeze(-1)
        proto_b = torch.index_select(evt_proto, dim=0, index=ev_idx)

        source = alpha * proto_a + (1 - alpha) * proto_b
        # source = proto_a
        source.detach_()
        source.requires_grad = False

        evt_proto.index_copy_(0, ev_idx, source)


        return proto_a

    # training logic for iteration
    def _train_step(_, batch):
        data = batch_to_tensor(batch, cfg)

        toks, typs, prefix, tags, events = [x.to(cfg.device) for x in data]
        mask = torch.not_equal(toks, cfg.pad_tok_id).to(cfg.device)



        model.train()
        logit, left_feat,label_feat = model(toks, typs, prefix, mask)

        loss_label = label_loss(label_feat,events)
        pre_label = torch.argmax(softmax(label_feat,dim=1),dim=1)
        loss = cls_loss(logit, tags.float())
        pred = torch.gt(logit, 0.)


        feats =  left_feat
        evt_feats = update_evt_proto(feats, events, 0.8)
        protos = evt_proto.index_select(0, events)

        intra_dist = dist_loss(feats, protos)
        intra_dist_loss = intra_dist.mean()

        rand_idx = torch.randperm(evt_feats.size(0), device=cfg.device)
        rand_evt_feats = evt_feats.index_select(0, rand_idx)
        inter_dist_loss = torch.nn.functional.pairwise_distance(evt_feats, rand_evt_feats)

        inter_dist_loss = torch.maximum(10 - inter_dist_loss, torch.zeros_like(inter_dist_loss)).mean()


        if cfg.lmda1 > 0.:
            loss = loss+ loss_label + cfg.lmda1 * inter_dist_loss + cfg.lmda2 * intra_dist_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        del toks, prefix,mask
        acc = accuracy_score(tags.cpu(), pred.cpu())
        acc_label = accuracy_score(events.cpu(), pre_label.cpu())

        return loss, acc, inter_dist_loss, intra_dist_loss,acc_label

        # Define trainer engine

    trainer = Engine(_train_step)

    # metrics for trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'inter')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'intra')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'acc_label')

    # Add progress bar showing trainer metrics
    mtcs = ['loss', 'acc', 'inter', 'intra','acc_label']
    ProgressBar(persist=True).attach(trainer, mtcs)

    return trainer


def create_evaluator(model, cfg):
    def _validation_step(_, batch):
        model.eval()
        with torch.no_grad():
            data = batch_to_tensor(batch, cfg)

            toks, typs, prefix, tags, events = [x.to(cfg.device) for x in data]
            mask = torch.not_equal(toks, cfg.pad_tok_id).to(cfg.device)

            logit, left_feat, label_feat = model(toks, typs, prefix, mask)

            loss_label = label_loss(label_feat, events)
            pre_label = torch.argmax(softmax(label_feat, dim=1), dim=1)


            loss = cls_loss(logit, tags.float())
            pred = torch.gt(logit, 0.)


            acc = accuracy_score(tags.cpu(), pred.cpu())
            acc_label = accuracy_score(events.cpu(), pre_label.cpu())


            return loss, acc,loss_label,acc_label

    evaluator = Engine(_validation_step)

    Average(lambda x: x[0]).attach(evaluator, 'loss')
    Average(lambda x: x[1]).attach(evaluator, 'acc')
    Average(lambda x: x[2]).attach(evaluator, 'loss_label')
    Average(lambda x: x[3]).attach(evaluator, 'acc_label')

    ProgressBar(persist=True).attach(evaluator)
    return evaluator


def create_tester(model, cfg, msg_feats, ref_num,all_pre_labels):
    def _test_step(_, batch):
        model.eval()
        with torch.no_grad():
            data = batch_to_tensor(batch, cfg)

            toks, typs, prefix, tags, events = [x.to(cfg.device) for x in data]
            mask = torch.not_equal(toks, cfg.pad_tok_id).to(cfg.device)

            idx = [a for tag, evt, a, b, fix, tok, _ in batch]
            idx = torch.tensor(idx, dtype=torch.long).to(cfg.device)

            me = [True if a == b else False for tag, evt, a, b, fix, tok, _ in batch]
            me = torch.tensor(me, dtype=torch.long).to(cfg.device)



            logit, left_feat, label_feat = model(toks, typs, prefix, mask)

            loss_label = label_loss(label_feat, events)
            pre_label = torch.argmax(softmax(label_feat, dim=1), dim=1)



            loss = cls_loss(logit, tags.float())
            pred = torch.gt(logit, 0.)

            # top_k_values, top_k_indices = torch.topk(torch.sigmoid(logit), k=90, largest=True)#

            msk = torch.gt(torch.sigmoid(logit), cfg.tao)
            # msk = torch.gt(torch.sigmoid(logit), 0)

            acc = accuracy_score(tags.cpu(), pred.cpu())
            acc_label = accuracy_score(events.cpu(), pre_label.cpu())


            msk = torch.logical_or(msk, me)

            msk_idx, = torch.nonzero(msk, as_tuple=True)
            idx = torch.index_select(idx, dim=0, index=msk_idx)
            pre_label = torch.index_select(pre_label, dim=0, index=msk_idx)

            for label, index in zip(pre_label, idx):
                row = all_pre_labels[index]  # 获取当前索引对应的行
                for i in range(len(row)):
                    if row[i] == -1:
                        all_pre_labels[index, i] = label  # 更新该位置的值为当前的标签
            # idx = torch.index_select(idx, dim=0, index=top_k_indices)#
            ## feats = (pfx_feat + left_feat) / 2.
            feats = left_feat
            feat = torch.index_select(feats, dim=0, index=msk_idx)
            evt = torch.index_select(events, dim=0, index=msk_idx)

            # feat = torch.index_select(feats, dim=0, index=top_k_indices)#
            # evt = torch.index_select(events, dim=0, index=top_k_indices)#

            msg_feats.index_add_(0, idx.cpu(), feat.cpu())
            ref_num.index_add_(0, idx.cpu(), torch.ones_like(evt, device='cpu'))

            return loss, acc,loss_label,acc_label

    tester = Engine(_test_step)

    Average(lambda x: x[0]).attach(tester, 'loss')
    Average(lambda x: x[1]).attach(tester, 'acc')
    Average(lambda x: x[2]).attach(tester, 'loss_label')
    Average(lambda x: x[3]).attach(tester, 'acc_label')

    ProgressBar(persist=True).attach(tester)
    return tester


def test_on_block(model, cfg, blk,b=0):
    test = blk['test']

    ###
    # test['samples'] = test['samples'][:1000]

    msg_tags = np.array(test['tw_to_ev'], dtype=np.int32)

    tst_num = msg_tags.shape[0]
    msg_feats = torch.zeros((tst_num, model.feat_size()), device='cpu')  # cfg.feat_dim
    ref_num = torch.zeros((tst_num,), dtype=torch.long, device='cpu')
    all_pre_labels = torch.full((tst_num, 180), -1)
    test_gen, test_batch_num = data_generator(test['samples'], cfg.batch_size)
    tester = create_tester(model, cfg, msg_feats, ref_num,all_pre_labels)

    print("Evaluate model on test data ...")
    test_state = tester.run(test_gen, epoch_length=test_batch_num)

    test_metrics = [(m, test_state.metrics[m]) for m in ['loss', 'acc','loss_label','acc_label']]
    test_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in test_metrics])
    print(f"Test: {test_metrics}\n", flush=True)

    # clustering & report
    msg_feats = msg_feats / (ref_num + torch.eq(ref_num, 0).float()).unsqueeze(-1)
    msg_feats = msg_feats.numpy()
    n_clust = len(test['ev_to_idx'])

    all_pre_labels= all_pre_labels.numpy()

    np.save(f'/home/lipu/HP_event/Eva_data/pre_label/all_pre_labels.npy', all_pre_labels)
    most_frequent_elements = []

    for row in all_pre_labels:
        # 移除-1
        filtered_row = row[row != -1]
        if filtered_row.size > 0:  # 如果过滤后的行不是空的
            unique, counts = np.unique(filtered_row, return_counts=True)
            most_frequent = unique[np.argmax(counts)]
            most_frequent_elements.append(most_frequent)
        else:
            most_frequent_elements.append(0)
    if not os.path.exists(cfg.eva_data):
        os.makedirs(cfg.eva_data)
    recall = recall_score(msg_tags,most_frequent_elements, average='macro')  # 'macro'表示未加权平均，'micro'表示加权平均
    print(f'Recall: {recall}')

    # 计算F1分数
    f1 = f1_score(msg_tags,most_frequent_elements, average='macro')
    print(f'F1 Score: {f1}')

    # 计算准确率
    accuracy = accuracy_score(msg_tags,most_frequent_elements)
    print(f'Accuracy: {accuracy}')

    Evaluate_datas = {'msg_feats': msg_feats, 'msg_tags': msg_tags, 'n_clust': n_clust}
    if cfg.offline :
        print(f"save Evaluate_datas_offline to '{cfg.eva_data}/evaluate_data_long_tail.npy'", end='\t')
        np.save(f'{cfg.eva_data}/evaluate_data_long_tail.npy', Evaluate_datas)
    else:

        print(f"save Evaluate_datas{b} to '{cfg.eva_data}/evaluate_data_M{b}.npy'", end='\t')
        e_path = cfg.eva_data +f'/evaluate_data_M{b}.npy'

        np.save( e_path, Evaluate_datas)

    print('done')



    k_means_score = run_kmeans(msg_feats, n_clust, msg_tags)
    dbscan_score = run_hdbscan(msg_feats, msg_tags)

    # del Evaluate_datas
    del  msg_feats

    return k_means_score, dbscan_score
def load_ckpt(model, cfg, ckpt, b):
    print(f"Load best ckpt for block {b} from '{ckpt}'")

    ckpt = torch.load(ckpt, map_location=cfg.device)
    model.load_state_dict(ckpt['model'], strict=False)

    ckpt_cfg = ckpt['cfg']
    ckpt_cfg.dataset = cfg.dataset
    ckpt_cfg.plm_path = cfg.plm_path
    ckpt_cfg.batch_size = cfg.batch_size
    ckpt_cfg.device = cfg.device
    ckpt_cfg.tao = cfg.tao

    return model, ckpt_cfg
def start_run(cfg, blocks):
    cfg.pad_tok_id = tokenizer.pad_token_id

    model = get_model(cfg).to(cfg.device)
    # print settings
    print_table([(k, str(v)[0:60]) for k, v in vars(cfg).items()])


    kmeans_scores, dbscan_scores = [], []
    ckpt_list = []
    for b, blk in enumerate(blocks):
        # if b > 0:
        #     print(f"test model on data block-{b} ...", flush=True)
        #     kms_score, dbs_score = test_on_block(model, cfg, blk, b)
        #     kmeans_scores.append(kms_score)
        #     dbscan_scores.append(dbs_score)
        #
        #     print("KMeans:")
        #     print_scores(kmeans_scores)
        #     print("DBSCAN:")
        #     print_scores(dbscan_scores)
        #
        # if b % 3 ==0  :
        #     gc.collect()
        #     print(f"train on data block-{b} ...", flush=True)
        #     model, ckpt = train_on_block(model, cfg, blk, b)
        #     ckpt_list.append(ckpt)
        inl_model = '/home/lipu/HP_event/ckpt/lotail_2012/roberta-large-tuning-long_tail_12-offline_checkpoint_-0.0340.pt'
        model, cfg = load_ckpt(model, cfg, inl_model, b)
        if b == 0 and cfg.offline:
            print(f"close test on data block-{b} ...", flush=True)
            kms_score, dbs_score = test_on_block(model, cfg, blk,b)
            kmeans_scores.append(kms_score)
            dbscan_scores.append(dbs_score)

            print("KMeans:")
            print_scores(kmeans_scores)
            print("DBSCAN:")
            print_scores(dbscan_scores)

    if cfg.offline:
        ckpt_list_file = os.path.join(cfg.ckpt_path, 'best_off_model.txt')
    else:
        ckpt_list_file = os.path.join(cfg.ckpt_path, 'ckpt_list.txt')

    with open(ckpt_list_file, 'w', encoding='utf8') as f:
        ckpt_list = [str(p) for p in ckpt_list]
        f.write("\n".join(ckpt_list))


def train_on_block(model, cfg, blk, blk_id=0):
    # reload plm in tuning mode
    if blk_id > 0 and cfg.plm_tuning:
        print("accumulate reload PLM parameters", flush=True)
        model.accumulate_reload_plm(cfg.device)
    train, valid = blk['train'], blk['valid']
    # fewer data for code test
    ###
    # train['samples'] = train['samples'][:500]
    # valid['samples'] = valid['samples'][:200]


    cfg.train_evt_num = len(train['ev_to_idx'])

    train_gen, train_batch_num = data_generator(train['samples'], cfg.batch_size, True, True)
    valid_gen, valid_batch_num = data_generator(valid['samples'], cfg.batch_size, False, True)

    # create model, optimizer and learning rate scheduler
    optimizer, lr_scheduler = initialize(model, cfg, train_batch_num)

    # print model parameters
    # summary(model, input_size=((cfg.batch_size, 50), (cfg.batch_size, 50)))

    # Setup model trainer and evaluator
    trainer = create_trainer(model, optimizer, lr_scheduler, cfg)
    evaluator = create_evaluator(model, cfg)

    @trainer.on(Events.EPOCH_STARTED)
    def log_learning_rates(_):
        for param_group in optimizer.param_groups:
            print("{} lr: {:>1.2e}".format(param_group.get('name', ''), param_group["lr"]))

    # Run model evaluation every epoch and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model():  # eng
        print("Evaluate model on eval data ...")
        val_state = evaluator.run(valid_gen, epoch_length=valid_batch_num)

        eval_metrics = [(m, val_state.metrics[m]) for m in ['loss', 'acc']]
        eval_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in eval_metrics])

        print(f"Eval: {eval_metrics}\n", flush=True)

    def score_function(_):
        if cfg.early_stop_monitor == 'loss':
            return - evaluator.state.metrics['loss']
        elif cfg.early_stop_monitor in evaluator.state.metrics:
            return evaluator.state.metrics[cfg.early_stop_monitor]
        else:
            raise Exception('unsupported metric %s' % cfg.early_stop_monitor)

    if cfg.offline:
        filename_prefix = f"{cfg.model_name}-{'tuning' if config.plm_tuning else 'fixed'}-{cfg.dataset_name}-offline"
    else:
        filename_prefix = f"{cfg.model_name}-{'tuning' if config.plm_tuning else 'fixed'}-{cfg.dataset_name}-{blk_id}"
    ckpt_handler = ModelCheckpoint(cfg.ckpt_path, score_function=score_function,
                                   filename_prefix=filename_prefix, n_saved=3,
                                   create_dir=True, require_empty=False)

    # if not tuning plm,
    model_state = get_model_state(model, ['pair_cls', 'pfx_embedding'], cfg.plm_tuning)
    ckpt = {'model': model_state, 'cfg': CkptWrapper(cfg)}
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler, ckpt)

    hdl_early_stop = EarlyStopping(patience=cfg.early_stop_patience, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, hdl_early_stop)

    # start training loop
    trainer.run(train_gen, max_epochs=cfg.max_epochs, epoch_length=train_batch_num)

    # load best checkpoint
    best_ckpt = ckpt_handler.last_checkpoint
    print(f"Load best model checkpoint from '{best_ckpt}'")
    ckpt = torch.load(best_ckpt)
    model.load_state_dict(ckpt['model'], strict=False)
    del ckpt
    return model, best_ckpt



if __name__ == "__main__":
    logger = logging.getLogger(os.path.basename(__file__))

    torch.manual_seed(2357)

    config = get_args()

    config.model_name = os.path.basename(os.path.normpath(config.plm_path))
    dataset_name = os.path.basename(config.dataset)
    config.dataset_name = dataset_name.replace(".npy", "")


    # if config.offline:
    #     config.dataset = "/home/lipu/HP_event/cache/offline.npy"

    if 'cuda' in config.device:
        torch.cuda.manual_seed(2357)

    tokenizer = AutoTokenizer.from_pretrained(config.plm_path)

    data_blocks = load_data_blocks(config.dataset, config, tokenizer)
    start_run(config, data_blocks)

    exit(0)
