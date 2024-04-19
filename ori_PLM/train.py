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
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer

from smed_model import PairPfxTuningEncoder
from utils import *

logging.basicConfig(level=logging.WARN,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')


def get_args():
    parser = ArgumentParser()

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    parser.add_argument('--dataset', type=str, default='/home/lipu/HP_event/cache/cache_ori_rbert/twitter12.npy')
    parser.add_argument('--plm_path', type=str, default='/home/lipu/HP_event/base_plm_model/roberta-base/')
    parser.add_argument('--offline', type=bool,default=False)

    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--lmda1", type=float, default=0.01)
    parser.add_argument("--lmda2", type=float, default=0.005)
    parser.add_argument("--tao", type=float, default=0.90)

    parser.add_argument("--optimizer", type=str, default='Adam', help="Optimizer, Adam, AdamW or SGD")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument("--step_lr_gamma", type=float, default=0.98, help="gamma for step learning rate schedule")

    parser.add_argument("--max_epochs", type=int, default=10, help="Number of training epochs")
    #ablation_ckpt  ablation_Eva_datas
    parser.add_argument('--ckpt_path', type=str, default='/home/lipu/HP_event/ckpt/ori_plm', help='path to checkpoint files')
    parser.add_argument("--eva_data", type=str, default="/home/lipu/HP_event/Eva_data/ori_plm_no_tuning",help="path to Evaluate_datas")

    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_monitor", type=str, default='loss', help='loss')

    parser.add_argument("--device", type=str,
                        default="cuda:1" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    cfg = parser.parse_args()

    return cfg


def get_model(cfg):
    return PairPfxTuningEncoder(cfg.plm_path)


def initialize(model, cfg, num_train_batch):
    parameters = model.parameters()  # 优化器的初始化


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
    toks = [tok for evt, tok, typs in batch]
    events = [evt for evt, tok, typs in batch]
    typs = [typs for evt, tok, typs in batch]


    max_len = min(max([len(it) for it in toks]), 200)
    toks = [pad_seq(it, pad=cfg.pad_tok_id, max_len=max_len) for it in toks]
    toks = torch.tensor(toks, dtype=torch.long)
    typs = [pad_seq(it, pad=cfg.pad_tok_id, max_len=max_len) for it in typs]
    typs = torch.tensor(typs, dtype=torch.long)
    events = torch.tensor(events, dtype=torch.long)

    return toks, typs, events


# loss functions
cls_loss = OnlineTripletLoss(10, RandomNegativeTripletSelector(10))
dist_loss = torch.nn.PairwiseDistance()


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

        toks, typs, events = [x.to(cfg.device) for x in data]
        mask = torch.not_equal(toks, cfg.pad_tok_id).to(cfg.device)

        model.train()
        feats = model(toks, typs, mask)

        loss = cls_loss(feats, events)

        evt_feats = update_evt_proto(feats, events, 0.8)
        protos = evt_proto.index_select(0, events)

        intra_dist = dist_loss(feats, protos)
        intra_dist_loss = intra_dist.mean()

        rand_idx = torch.randperm(evt_feats.size(0), device=cfg.device)
        rand_evt_feats = evt_feats.index_select(0, rand_idx)
        inter_dist_loss = torch.nn.functional.pairwise_distance(evt_feats, rand_evt_feats)

        inter_dist_loss = torch.maximum(10 - inter_dist_loss, torch.zeros_like(inter_dist_loss)).mean()

        loss = loss[0]
        if loss.item() ==0:
            loss = loss + 0.00000001 * inter_dist_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        del toks, mask

        return loss, inter_dist_loss, intra_dist_loss

        # Define trainer engine

    trainer = Engine(_train_step)

    # metrics for trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'inter')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'intra')

    # Add progress bar showing trainer metrics
    mtcs = ['loss', 'inter', 'intra']
    ProgressBar(persist=True).attach(trainer, mtcs)

    return trainer


def create_evaluator(model, cfg):
    def _validation_step(_, batch):
        model.eval()
        with torch.no_grad():
            data = batch_to_tensor(batch, cfg)

            toks, typs,  events = [x.to(cfg.device) for x in data]
            mask = torch.not_equal(toks, cfg.pad_tok_id).to(cfg.device)

            feats = model(toks, typs,  mask)
            loss = cls_loss(feats, events)
            return loss

    evaluator = Engine(_validation_step)

    Average(lambda x: x[0]).attach(evaluator, 'loss')

    ProgressBar(persist=True).attach(evaluator)
    return evaluator


def create_tester(model, cfg, feature_container, label_container):
    def _test_step(_, batch):
        model.eval()
        with torch.no_grad():
            data = batch_to_tensor(batch, cfg)

            toks, typs, events = [x.to(cfg.device) for x in data]
            mask = torch.not_equal(toks, cfg.pad_tok_id).to(cfg.device)
            feats = model(toks, typs, mask)
            loss = cls_loss(feats, events)
            feature_container['features'] = torch.cat((feature_container['features'], feats.detach().cpu()), dim=0)
            label_container['labels'] = torch.cat((label_container['labels'], events.detach().cpu()), dim=0)

            return loss

    tester = Engine(_test_step)

    Average(lambda x: x[0]).attach(tester, 'loss')

    ProgressBar(persist=True).attach(tester)
    return tester


def test_on_block(model, cfg, blk,b=0):
    test = blk['test']

    ###
    # test = test[:1000]

    feature_container = {'features': torch.FloatTensor([])}
    label_container = {'labels': torch.Tensor([])}
    tst_num = len(test)

    test_gen, test_batch_num = data_generator(test, cfg.batch_size)
    tester = create_tester(model, cfg, feature_container,label_container)

    print("Evaluate model on test data ...")
    test_state = tester.run(test_gen, epoch_length=test_batch_num)

    test_metrics = [(m, test_state.metrics[m]) for m in ['loss']]
    test_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in test_metrics])
    print(f"Test: {test_metrics}\n", flush=True)

    # clustering & report
    msg_feats = feature_container['features'].numpy()
    msg_tags = label_container['labels'].numpy().astype(int)
    n_clust = len(set(msg_tags))

    if not os.path.exists(cfg.eva_data):
        os.makedirs(cfg.eva_data)

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
        if b > 0:
            print(f"test model on data block-{b} ...", flush=True)
            kms_score, dbs_score = test_on_block(model, cfg, blk, b)
            kmeans_scores.append(kms_score)
            dbscan_scores.append(dbs_score)

            print("KMeans:")
            print_scores(kmeans_scores)
            print("DBSCAN:")
            print_scores(dbscan_scores)
    #
        # if b % 3 ==0  :
        #     gc.collect()
        #     print(f"train on data block-{b} ...", flush=True)
        #     model, ckpt = train_on_block(model, cfg, blk, b)
        #     ckpt_list.append(ckpt)
        #
        # if b == 0 and cfg.offline:
        #     print(f"close test on data block-{b} ...", flush=True)
        #     kms_score, dbs_score = test_on_block(model, cfg, blk,b)
        #     kmeans_scores.append(kms_score)
        #     dbscan_scores.append(dbs_score)
        #
        #     print("KMeans:")
        #     print_scores(kmeans_scores)
        #     print("DBSCAN:")
        #     print_scores(dbscan_scores)
    #
    # if cfg.offline:
    #     ckpt_list_file = os.path.join(cfg.ckpt_path, 'best_off_model.txt')
    # else:
    #     ckpt_list_file = os.path.join(cfg.ckpt_path, 'ckpt_list.txt')
    #
    # with open(ckpt_list_file, 'w', encoding='utf8') as f:
    #     ckpt_list = [str(p) for p in ckpt_list]
    #     f.write("\n".join(ckpt_list))


def train_on_block(model, cfg, blk, blk_id=0):
    train, valid = blk['train'], blk['valid']
    # fewer data for code test
    ###
    # train = train[:500]
    # valid = valid[:200]


    cfg.train_evt_num = len(set([item[0] for item in train]))
    train_gen, train_batch_num = data_generator(train, cfg.batch_size, True, True)
    valid_gen, valid_batch_num = data_generator(valid, cfg.batch_size, False, True)

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

        eval_metrics = [(m, val_state.metrics[m]) for m in ['loss']]
        eval_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in eval_metrics])

        print(f"Eval: {eval_metrics}\n", flush=True)

    def score_function(_):
        if cfg.early_stop_monitor == 'loss':
            return  - evaluator.state.metrics['loss']
        elif cfg.early_stop_monitor in evaluator.state.metrics:
            return evaluator.state.metrics[cfg.early_stop_monitor]
        else:
            raise Exception('unsupported metric %s' % cfg.early_stop_monitor)

    if cfg.offline:
        filename_prefix = f"{cfg.model_name}-'tuning'-{cfg.dataset_name}-offline"
    else:
        filename_prefix = f"{cfg.model_name}-'tuning'-{cfg.dataset_name}-{blk_id}"
    ckpt_handler = ModelCheckpoint(cfg.ckpt_path, score_function=score_function,
                                   filename_prefix=filename_prefix, n_saved=3,
                                   create_dir=True, require_empty=False)

    # if not tuning plm,
    model_state = get_model_state(model, ['pair_cls', 'pfx_embedding'])
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