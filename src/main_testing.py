import os
import sys
import pickle
import argparse
import time
import distance
import numpy as np
import networkx as nx
from datetime import datetime
from torchnlp.word_to_vector import CharNGram

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
parser.add_argument("--num_contexts", type=int, default=100, help="# contexts for interaction")
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
parser.add_argument('--restore_para_file', type=str, default='')
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

parser.add_argument('--rank_model_path', type=str, required=True, default='')
parser.add_argument('--num_results', type=int, default=10, help='number of results to show')

args = parser.parse_args()
print('args: ', args)

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

# global parameters
args.term_strings = pickle.load(open('../data/mappings/term_string_mapping.pkl', 'rb'))
args.term_concept_dict = pickle.load(open('../data/mappings/term_concept_mapping.pkl', 'rb'))
args.concept_term_dict = pickle.load(open('../data/mappings/concept_term_mapping.pkl', 'rb'))
args.all_iv_terms = pickle.load(open('../data/sym_data/all_iv_terms_per{0}_{1}.pkl'.format(args.per, args.days), 'rb'))
args.neighbors = []

args.cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data pre-processing
args.restore_para_file = './saved_models/{0}_{1}_pretrain_model_dict.pkl'.format(args.per, args.days)
if os.path.exists(args.restore_para_file):
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
    # print(args.pre_train_nodes.shape)
    print('-- Pre-stored parameters loaded! -- ')

else:
    raise ValueError('Need Pre-stored Parameters!')

model = Ranker.DeepTermRankingListNet(args)
if args.cuda:
    model = model.cuda()
print(model)
# print([name for name, p in model.named_parameters()])

model.load_state_dict(torch.load(args.rank_model_path), strict=True)
model.eval()
print('Pretrained model loaded!')

while True:
    query = str(input('Input your query (Press \'exit\' to exit): '))
    if query == 'exit':
        exit()
    scores = utils.get_ranking_score(model, query, args.all_iv_terms, args)
    sorted_rank_idx = np.argsort(scores)[::-1]
    sorted_ids = np.array(args.all_iv_terms)[sorted_rank_idx]
    print('Top ranking:')
    print('SurfCon: {0}'.format([args.term_strings[args.all_iv_terms[x]] for x in sorted_rank_idx[:args.num_results]]))



















