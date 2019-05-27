import os
import sys
import pickle
import argparse
import time
import distance
import numpy as np
import networkx as nx
from datetime import datetime
# from torchnlp.word_to_vector import CharNGram

import torch
import torch.nn as nn
import torch.optim as optim

import loader
import utils
import train_utils
import rank_models as Ranker


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


parser = argparse.ArgumentParser(description='process user given parameters')
parser.register('type', 'bool', str2bool)
parser.add_argument('-p', "--per", type=str, default='Bin', help='Pat or Bin')
parser.add_argument('-d', '--days', type=str, default='1', help='1 7 30 90 180 365 all')
#
parser.add_argument("--random_seed", type=float, default=42)
parser.add_argument('--num_oov', type=int, default=2000)
parser.add_argument('--re_sample_test', type='bool', default=False)
parser.add_argument('--train_neg_num', type=int, default=50)
parser.add_argument('--test_neg_num', type=int, default=100)
parser.add_argument("--num_contexts", type=int, default=200, help="# contexts for interaction")
parser.add_argument('--max_contexts', type=int, default=1000, help='max contexts to look at')
parser.add_argument('--context_gamma', type=float, default=0.3)
# model parameters
parser.add_argument('--ngram_embed_dim', type=int, default=100)
parser.add_argument('--n_grams', type=str, default='2, 3, 4')
parser.add_argument("--word_embed_dim", type=int, default=100, help="embedding dimention for word")
parser.add_argument('--node_embed_dim', type=int, default=128)
parser.add_argument("--dropout", type=float, default=0, help="size of testing set")
parser.add_argument('--bi_out_dim', type=int, default=50, help='dim for the last bilinear layer for output')
# model selection
parser.add_argument('--use_context', type='bool', default=True)
parser.add_argument('--do_ctx_interact', type='bool', default=True)

parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs for training")
parser.add_argument("--log_interval", type=int, default=2000, help='step interval for log')
parser.add_argument("--test_interval", type=int, default=1, help='epoch interval for testing')
parser.add_argument("--early_stop_epochs", type=int, default=10)
parser.add_argument("--metric", type=str, default='map', help='mrr or map')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--min_epochs', type=int, default=30, help='minimum number of epochs')
parser.add_argument('--clip_grad', type=float, default=5.0)
parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')

# path to external files
parser.add_argument("--embed_filename", type=str, default='../data/embeddings/glove.6B.100d.txt')
parser.add_argument('--node_embed_path', type=str, default='../data/embeddings/line2nd_ttcooc_embedding.txt')
parser.add_argument('--ngram_embed_path', type=str, default='../data/embeddings/charNgram.txt')
# parser.add_argument('--restore_para_file', type=str, default='./final_pretrain_cnn_model_parameters.pkl')
parser.add_argument('--restore_model_path', type=str, required=True, default='')
parser.add_argument('--restore_idx_data', type=str, default='')
parser.add_argument("--logging", type='bool', default=False)
parser.add_argument("--log_name", type=str, default='empty.txt')
parser.add_argument('--restore_model_epoch', type=int, default=600)
parser.add_argument("--save_best", type='bool', default=True, help='save model in the best epoch or not')
parser.add_argument("--save_dir", type=str, default='./saved_models', help='save model in the best epoch or not')
parser.add_argument("--save_interval", type=int, default=5, help='intervals for saving models')

parser.add_argument('--random_test', type='bool', default=True)
parser.add_argument('--neg_sampling', type='bool', default=False)
parser.add_argument('--num_negs', type=int, default=5)

args = parser.parse_args()
print('args: ', args)

print('********Key parameters:******')
print('Use GPU? {0}'.format(torch.cuda.is_available()))
print('Model Parameters: ')

print('Dataset: {0} {1}'.format(args.per, args.days))
print('Train # negative samples: {0}'.format(args.train_neg_num))
print('Test # negative samples: {0}'.format(args.test_neg_num))
print('# contexts to aggregate: {0}'.format(args.num_contexts))

print('*****************************')

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

args.restore_para_file = './saved_models/{0}_{1}_pretrain_model_dict.pkl'.format(args.per, args.days)
# path to store
args.save_dir = './saved_models/rank_model_per{0}_{1}'.format(args.per, args.days)
# print(args.save_dir)

# global parameters
args.term_strings = pickle.load(open('../data/mappings/term_string_mapping.pkl', 'rb'))
args.term_concept_dict = pickle.load(open('../data/mappings/term_concept_mapping.pkl', 'rb'))
args.concept_term_dict = pickle.load(open('../data/mappings/concept_term_mapping.pkl', 'rb'))
args.all_iv_terms = pickle.load(open('../data/sym_data/all_iv_terms_per{0}_{1}.pkl'.format(args.per, args.days), 'rb'))
args.neighbors = []

args.cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loading data
print('Begin loading data ...')
train_graph = nx.read_gpickle('../data/sym_data/train_graph_nx_per' + args.per + '_' + args.days + '.gpickle')
test_graph = nx.read_gpickle('../data/sym_data/test_graph_nx_per' + args.per + '_' + args.days + '.gpickle')
train_multi = [(x, list(train_graph.adj[x])) for x in train_graph.nodes]
test_multi = [(x, list(test_graph.adj[x])) for x in test_graph.nodes]

np.random.shuffle(train_multi)
np.random.shuffle(test_multi)

train_pos = train_multi
dev_pos = test_multi[: int(len(test_multi) / 2)]
iv_test_pos = test_multi[int(len(test_multi) / 2)::]
dis_iv_pos = utils.filter_sim_terms(iv_test_pos, args.term_strings)
print('Data: # train: {0}, # val: {1}, # iv test: {2}, # dis iv test {3}'.format(len(train_pos),
                                                                                 len(dev_pos),
                                                                                 len(iv_test_pos),
                                                                                 len(dis_iv_pos)))
# print('Sum # pos: train: {0}, val: {1}, iv test: {2}, '
#       'dis iv test {3}'.format(np.sum([len(x[1]) for x in train_pos]),
#                                np.sum([len(x[1]) for x in dev_pos]),
#                                np.sum([len(x[1]) for x in iv_test_pos]),
#                                np.sum([len(x[1]) for x in dis_iv_pos])))
#
# print('Average # pos: train: {0:.3}, val: {1:.3}, iv test: {2:.3}, '
#       'dis iv test {3:.3}'.format(np.mean([len(x[1]) for x in train_pos]),
#                                   np.mean([len(x[1]) for x in dev_pos]),
#                                   np.mean([len(x[1]) for x in iv_test_pos]),
#                                   np.mean([len(x[1]) for x in dis_iv_pos])))

all_oov_test = pickle.load(open('../data/sym_data/oov_test_dict_per' + args.per + '_' + args.days + '.pkl', 'rb'))
all_oov_test = list(all_oov_test.items())
np.random.shuffle(all_oov_test)

oov_test_pos = all_oov_test[:args.num_oov]
dis_oov_pos = utils.filter_sim_terms(oov_test_pos, args.term_strings)

# print('# oov test: {0}, average # pos: {1:3}, sum # pos: {2}'.format(
#     len(oov_test_pos), np.mean([len(x[1]) for x in oov_test_pos]), np.sum([len(x[1]) for x in oov_test_pos])))
#
# print('# dis oov test: {0}, average # pos: {1:.3}, sum # pos: {2}'.format(
#     len(dis_oov_pos), np.mean([len(x[1]) for x in dis_oov_pos]), np.sum([len(x[1]) for x in dis_oov_pos])))

reuse_oov_path = '../data/sym_data/dev_test_data_per{0}_{1}_{2}.pkl'.format(args.per, args.days, args.num_oov)
if os.path.exists(reuse_oov_path):
    dev, iv_test, dis_iv_test, oov_test, dis_oov_test = pickle.load(open(reuse_oov_path, 'rb'))
else:
    print('Re-sample negative samples for testing!')
    dev = utils.random_neg_sampling(dev_pos, args.all_iv_terms, args.term_concept_dict, args.test_neg_num)

    iv_test = utils.random_neg_sampling(iv_test_pos, args.all_iv_terms, args.term_concept_dict, args.test_neg_num)
    dis_iv_test = utils.random_neg_sampling(dis_iv_pos, args.all_iv_terms, args.term_concept_dict, args.test_neg_num)

    oov_test = utils.random_neg_sampling(oov_test_pos, args.all_iv_terms, args.term_concept_dict, args.test_neg_num)
    dis_oov_test = utils.random_neg_sampling(dis_oov_pos, args.all_iv_terms, args.term_concept_dict, args.test_neg_num)

    pickle.dump([dev, iv_test, dis_iv_test, oov_test, dis_oov_test], open(reuse_oov_path, 'wb'), protocol=-1)

print('Data loaded!')

# testing
# train_pos = train_pos[:100]
# dev = dev[:10]
# iv_test = iv_test[:10]
# oov_test = oov_test[:10]

# data pre-processing
restore_para_file = './saved_models/{0}_{1}_pretrain_model_dict.pkl'.format(args.per, args.days)
if os.path.exists(restore_para_file):
    model_dict = pickle.load(open(args.restore_para_file, 'rb'))
    # print('Model Parameters: ', model_dict.keys())
    # output
    args.node_to_id = model_dict['node_to_id']
    args.id_to_node = model_dict['id_to_node']
    args.node_vocab_size = model_dict['node_vocab_size']
    args.node_embed_dim = model_dict['node_embed_dim']
    args.n_grams = model_dict['n_grams']
    # ngram
    args.pre_train_ngram = model_dict['pre_train_ngram']
    args.ngram_to_id = model_dict['ngram_to_id']
    args.ngram_vocab_size = model_dict['ngram_vocab_size']
    args.ngram_embed_dim = model_dict['ngram_embed_dim']
    # word
    args.word_hidden_dim = model_dict['word_hidden_dim']
    args.word_vocab_size = model_dict['word_vocab_size']
    args.word_to_id = model_dict['word_to_id']
    args.pre_train_words = model_dict['pre_train_words']

    args.pre_train_nodes = loader.node_mapping(args, args.node_to_id)
    print(args.pre_train_nodes.shape)
    print('-- Pre-stored parameters loaded! -- ')

else:
    raise ValueError('Need Pre-stored Parameters!')

print('Begin digitalizing ...')
# prepare digital samples
dev_idx = utils.make_idx_data(dev, args)
iv_test_idx = utils.make_idx_data(iv_test, args)
dis_iv_test_idx = utils.make_idx_data(dis_iv_test, args)
oov_test_idx = utils.make_idx_data(oov_test, args)
dis_oov_test_idx = utils.make_idx_data(dis_oov_test, args)

print(len(dev_idx), len(iv_test_idx), len(dis_iv_test), len(oov_test_idx), len(dis_oov_test_idx))

model = Ranker.DeepTermRankingListNet(args)
if args.cuda:
    model = model.cuda()
print(model)
print([name for name, p in model.named_parameters()])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

# training
last_epoch = 0
best_on_dev = 0.0
train_loss = 0
train_logits = []
train_labels = []
print('Begin trainning...')
for epoch in range(args.num_epochs):
    model.train()
    steps = 0
    np.random.shuffle(train_pos)
    for cur_sample in train_pos:
        # the batch size is the number of documents
        cur_raw_data = utils.random_neg_sampling([cur_sample], args.all_iv_terms, args.term_concept_dict, args.train_neg_num)
        cur_idx_data = utils.make_idx_data(cur_raw_data, args)
        t1_list, t2_list, onehot_labels = utils.preprocess(cur_idx_data[0], args)
        # print([x.shape for x in t2_list])
        onehot_labels = torch.FloatTensor(onehot_labels).reshape(1, -1)

        t1_list = [torch.tensor(x) for x in t1_list[:-1]] + [t1_list[-1]]
        t2_list = [torch.tensor(x) for x in t2_list[:-1]] + [t2_list[-1]]
        if args.cuda:
            onehot_labels = onehot_labels.cuda()
            t1_list = [x.cuda() for x in t1_list[:-1]] + [t1_list[-1]]
            t2_list = [x.cuda() for x in t2_list[:-1]] + [t2_list[-1]]

        optimizer.zero_grad()
        logits = model(t1_list, t2_list)
        rank_loss = model.listwise_cost(logits, onehot_labels)
        # rank_loss = criterion(logits, ce_labels)
        loss = rank_loss

        train_loss += loss
        if args.cuda:
            train_logits.append(logits.cpu().detach().numpy()[0])
            train_labels.append(onehot_labels.cpu().numpy()[0])
        else:
            train_logits.append(logits.detach().numpy()[0])
            train_labels.append(onehot_labels.numpy()[0])

        # bp
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        steps += 1
        if steps % args.log_interval == 0:
            print("Epoch-{0}, steps-{1}: Train {2} - {3:.5}, Train Loss - {4:.5}".
                  format(epoch, steps, args.metric.upper(), utils.eval_metric(train_logits, train_labels, args.metric),
                         train_loss / len(train_labels)))
            train_loss = 0
            train_logits = []
            train_labels = []

    utils.adjust_learning_rate(optimizer, args.learning_rate / (1 + (epoch + 1) * args.lr_decay))

    if epoch % args.test_interval == 0:
        all_dev = train_utils.evaluation(dev_idx, model, criterion, args)
        print("Epoch-{0}: All Dev {1}: {2:.5}".format(epoch, args.metric.upper(), all_dev))

        if all_dev > best_on_dev:
            print(datetime.now().strftime("%m/%d/%Y %X"))
            best_on_dev = all_dev
            last_epoch = epoch

            all_iv = train_utils.evaluation(iv_test_idx, model, criterion, args)
            print("--- Testing: All IV Test {0}: {1:.5}".format(args.metric.upper(), all_iv))

            all_iv = train_utils.evaluation(dis_iv_test_idx, model, criterion, args)
            print("--- Testing: All Dis IV Test {0}: {1:.5}".format(args.metric.upper(), all_iv))

            all_oov = train_utils.evaluation(oov_test_idx, model, criterion, args)
            print("--- Testing: All OOV Test {0}: {1:.5}".format(args.metric.upper(), all_oov))

            all_oov = train_utils.evaluation(dis_oov_test_idx, model, criterion, args)
            print("--- Testing: All Dis OOV Test {0}: {1:.5}".format(args.metric.upper(), all_oov))

            if args.save_best:
                utils.save(model, args.save_dir, 'best', epoch)

        else:
            if epoch - last_epoch > args.early_stop_epochs and epoch > args.min_epochs:
                print('Early stop at {0} epoch.'.format(epoch))
                break

    elif epoch % args.save_interval == 0:
        utils.save(model, args.save_dir, 'snapshot', epoch)
