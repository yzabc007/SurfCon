import os
import re
import sys
import pickle
import argparse
import time
import numpy as np
import networkx as nx
import sklearn.metrics
from datetime import datetime
from collections import Counter
# from torchnlp.word_to_vector import CharNGram
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import encoder_models
import utils
import loader


def make_idx_data_term(dataset, args):
    data = []  # [(node_id, [(node_id1, ppmi?count?), (node_id2, ppmi?count?), ...]), (), ...]
    for cur_sample in dataset:
        cur_dict = {}
        cur_dict['id'] = cur_sample[0]
        cur_dict['str'] = args.term_strings[cur_sample[0]]

        ngram_ids = []
        for g in utils.get_single_ngrams(args.term_strings[cur_sample[0]], args.n_grams):
            if g in args.ngram_to_id:
                ngram_ids.append(args.ngram_to_id[g])
        if ngram_ids is []:
            ngram_ids.append(args.ngram_to_id['<UNK>'])
        cur_dict['ngram_ids'] = ngram_ids

        word_list = args.term_strings[cur_sample[0]].split()
        cur_dict['word_ids'] = [args.word_to_id[w if w in args.word_to_id else '<UNK>'] for w in word_list]
        cur_dict['word_len'] = len(word_list)

        # one-hot labels
        label_vec = np.zeros(args.node_vocab_size)
        for y in cur_sample[1]:
            if y[0] in args.node_to_id:
                label_vec[args.node_to_id[y[0]]] = y[1]
                # label_vec[args.node_to_id[y[0]]] = 1
        cur_dict['y'] = label_vec / np.sum(label_vec)

        data.append(cur_dict)
    return data


def make_idx_term_ns(dataset, args):
    data = []  # [(node_id, [(node_id1, ppmi?count?), (node_id2, ppmi?count?), ...]), (), ...]
    for cur_sample in dataset:
        cur_dict = {}
        cur_dict['id'] = cur_sample[0]
        cur_dict['str'] = args.term_strings[cur_sample[0]]

        ngram_ids = []
        for g in utils.get_single_ngrams(args.term_strings[cur_sample[0]], args.n_grams):
            if g in args.ngram_to_id:
                ngram_ids.append(args.ngram_to_id[g])
        if ngram_ids is []:
            ngram_ids.append(args.ngram_to_id['<UNK>'])
        cur_dict['ngram_ids'] = ngram_ids

        word_list = args.term_strings[cur_sample[0]].split()
        cur_dict['word_ids'] = [args.word_to_id[w if w in args.word_to_id else '<UNK>'] for w in word_list]
        cur_dict['word_len'] = len(word_list)

        # context terms
        for context in cur_sample[1]:
            new_dict = dict(cur_dict)
            new_dict['context'] = args.node_to_id[context[0]]
            data.append(new_dict)

    return data


def batch_process_term(batch_data, args):
    y = []
    word_ids = []
    word_len = []
    ngram_ids = []
    ngram_length = []

    for sample in batch_data:
        y.append(sample['y'])
        word_ids.append(sample['word_ids'])
        word_len.append(sample['word_len'])
        ngram_ids.append(sample['ngram_ids'])
        ngram_length.append(len(sample['ngram_ids']))

    y = np.array(y)
    word_ids = utils.pad_sequence(word_ids).astype(int)
    word_lengths = np.array(word_len).astype(np.float32)
    ngram_ids = utils.pad_sequence(ngram_ids).astype(int)
    ngram_length = np.array(ngram_length).astype(np.float32)

    return [word_ids, word_lengths, ngram_ids, ngram_length], y


def batch_process_ns(batch_data, args):
    word_ids = []
    word_len = []
    ngram_ids = []
    ngram_length = []
    contexts = []

    for sample in batch_data:
        word_ids.append(sample['word_ids'])
        word_len.append(sample['word_len'])
        ngram_ids.append(sample['ngram_ids'])
        ngram_length.append(len(sample['ngram_ids']))
        contexts.append(sample['context'])

    word_ids = utils.pad_sequence(word_ids).astype(int)
    word_lengths = np.array(word_len).astype(np.float32)
    ngram_ids = utils.pad_sequence(ngram_ids).astype(int)
    ngram_length = np.array(ngram_length).astype(np.float32)
    contexts = np.array(contexts)

    return [word_ids, word_lengths, ngram_ids, ngram_length, contexts]


def main():

    def str2bool(string):
        return string.lower() in ['yes', 'true', 't', 1]

    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    # path path path
    parser.add_argument('--embed_filename', type=str, default='../data/embeddings/glove.6B.100d.txt')
    parser.add_argument('--ngram_embed_path', type=str, default='../data/embeddings/charNgram.txt')

    parser.add_argument('-p', "--per", type=str, default='Bin', help='Pat or Bin')
    parser.add_argument('-d', '--days', type=str, default='1', help='1 7 30 90 180 365 all')

    parser.add_argument('--ngram_embed_dim', type=int, default=100)
    parser.add_argument('--n_grams', type=str, default='2, 3, 4')
    parser.add_argument('--node_embed_dim', type=int, default=128)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--word_embed_dim', type=int, default=100)

    parser.add_argument("--num_epochs", type=int, default=10000, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument('--random_seed', type=int, default=43)
    parser.add_argument("--dropout", type=float, default=0.5, help="size of testing set")
    parser.add_argument("--log_interval", type=int, default=100, help='step interval for log')
    parser.add_argument("--test_interval", type=int, default=1, help='epoch interval for testing')
    parser.add_argument("--early_stop_epochs", type=int, default=1000, help='epoch interval for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    parser.add_argument("--save_best", type='bool', default=True, help='save model in the best epoch or not')
    parser.add_argument("--save_dir", type=str, default='./saved_models/saved_pretrained')
    parser.add_argument("--save_interval", type=int, default=1, help='intervals for saving models')

    parser.add_argument('--neg_sampling', type='bool', default=False)
    parser.add_argument('--num_negs', type=int, default=5)

    args = parser.parse_args()
    print('args: ', args)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # global variables
    args.cuda = torch.cuda.is_available()

    args.term_strings = pickle.load(open('../data/mappings/term_string_mapping.pkl', 'rb'))
    dataset = pickle.load(open('../data/sym_data/sub_neighbors_dict_ppmi_per' + args.per + '_' + args.days + '.pkl', 'rb'))

    # prepare context labels
    context_terms = list(dataset.keys())
    print('Total number of candidates: ', len(context_terms))

    reuse_stored_path = './saved_models/{0}_{1}_pretrain_model_dict.pkl'.format(args.per, args.days)
    if os.path.exists(reuse_stored_path):
        model_dict = pickle.load(open(reuse_stored_path, 'rb'))
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
        print('Pre-stored parameters loaded!')
    else:
        # data pre-processing
        args.node_to_id = {node: idx for idx, node in enumerate(context_terms)}
        args.id_to_node = {idx: node for node, idx in args.node_to_id.items()}
        args.node_vocab_size = len(args.node_to_id)
        args.node_embed_dim = args.node_embed_dim

        args.n_grams = [int(x) for x in args.n_grams.split(',')]
        list_words = list(set([x for x in args.term_strings.values()]))
        ngram_vocab, ngram_to_id, id_to_gram, pre_train_ngram = loader.g2v_mapping_pretrain(list_words, args)
        args.pre_train_ngram = pre_train_ngram
        args.ngram_to_id = ngram_to_id
        args.ngram_vocab_size = len(ngram_to_id)

        list_phrases = args.term_strings.values()
        word_vocab, word_to_id, id_to_word, pre_train_words = loader.w2v_mapping_pretrain(list_phrases, args)
        args.pre_train_words = pre_train_words
        args.word_to_id = word_to_id
        args.word_vocab_size = len(word_to_id)

        model_dict = {}
        # output
        model_dict['node_to_id'] = args.node_to_id
        model_dict['id_to_node'] = args.id_to_node
        model_dict['node_vocab_size'] = args.node_vocab_size
        model_dict['node_embed_dim'] = args.node_embed_dim
        # ngram
        model_dict['pre_train_ngram'] = args.pre_train_ngram
        model_dict['ngram_to_id'] = args.ngram_to_id
        model_dict['ngram_vocab_size'] = args.ngram_vocab_size
        model_dict['ngram_embed_dim'] = args.ngram_embed_dim
        model_dict['n_grams'] = args.n_grams
        # word level
        model_dict['word_hidden_dim'] = args.word_hidden_dim
        model_dict['word_vocab_size'] = args.word_vocab_size
        model_dict['word_to_id'] = args.word_to_id
        model_dict['pre_train_words'] = args.pre_train_words

        pickle.dump(model_dict, open(reuse_stored_path, 'wb'), protocol=-1)
        print('Model Parameters Stored!')

    # optimizer and loss function
    '''
    model = encoder_models.ContextPredictionWordNGram(args)
    if args.cuda:
        model = model.cuda()
    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total # parameter: {0} w/ embeddings'.format(pytorch_total_params))

    pytorch_total_params = sum(p.numel() for name, p in model.named_parameters()
                               if p.requires_grad and name.count('embeddings') == 0)
    print('Total # parameter: {0} w/o embeddings'.format(pytorch_total_params))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)    
    '''
    if args.neg_sampling:
        degree_list = pickle.load(open('../data/sym_data/degree_list_perBin_1.pkl', 'rb'))
        weights = np.zeros(args.node_vocab_size)
        for y in degree_list:
            if y[0] in args.node_to_id:
                weights[args.node_to_id[y[0]]] = y[1]

        model = encoder_models.ContextPredictionWordNGram(args, weights)
        if args.cuda:
            model = model.cuda()
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        train = [(x, dataset[x]) for x in dataset.keys()]
        np.random.shuffle(train)
        print('Training terms: {0}'.format(len(train)))
        # digitalization
        train_idx = make_idx_term_ns(train, args)
        print('Total number of pairs:', len(train_idx))

        args.log_interval = len(train_idx) // 10
        # training
        last_epoch = 0
        best_loss = np.inf
        num_batches = len(train_idx) // args.batch_size
        print('Begin trainning...')
        model.train()
        for epoch in range(args.num_epochs):
            steps = 0
            train_loss = []
            np.random.shuffle(train_idx)
            for i in range(num_batches):
                train_batch = train_idx[i * args.batch_size: (i + 1) * args.batch_size]
                if i == num_batches - 1:
                    train_batch = train_idx[i * args.batch_size::]

                local_xs = batch_process_ns(train_batch, args)
                local_xs = [torch.tensor(x) for x in local_xs]
                if args.cuda:
                    local_xs = [x.cuda() for x in local_xs]

                optimizer.zero_grad()
                loss = model.forward_ns_loss(local_xs)  # (64, 1)
                train_loss.append(loss.item())

                # bp
                loss.backward()
                optimizer.step()

                steps += 1

                if steps % args.log_interval == 0:
                    print('Epoch {0} step {1} - Train loss\u2193:{2:.10}'.format(epoch, steps, np.mean(train_loss)))

            print('Epoch {0} - Train loss\u2193:{1:.10}'.format(epoch, np.mean(train_loss)))

            # if epoch % args.save_interval == 0:
            # if steps % 10000 == 0:
            #     print(datetime.now().strftime("%m/%d/%Y %X"))
            #     utils.save(model, args.save_dir, 'ns_snapshot_' + str(steps), epoch)

            if epoch % args.save_interval == 0:
                print(datetime.now().strftime("%m/%d/%Y %X"))
                utils.save(model, args.save_dir, 'ns_snapshot', epoch)

    else:
        model = encoder_models.ContextPredictionWordNGram(args)
        if args.cuda:
            model = model.cuda()
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # dataset splitting
        train = [(x, dataset[x]) for x in dataset.keys()]
        np.random.shuffle(train)
        print('Training terms: {0}'.format(len(train)))
        # digitalization
        train_idx = make_idx_data_term(train, args)

        # training
        last_epoch = 0
        best_loss = np.inf
        num_batches = len(train) // args.batch_size
        print('Begin trainning...')
        model.train()
        for epoch in range(args.num_epochs):
            steps = 0
            train_loss = 0.0
            np.random.shuffle(train_idx)
            for i in range(num_batches):
                train_batch = train_idx[i * args.batch_size: (i + 1) * args.batch_size]
                if i == num_batches - 1:
                    train_batch = train_idx[i * args.batch_size::]

                local_xs, local_y = batch_process_term(train_batch, args)

                local_y = torch.FloatTensor(local_y)
                local_xs = [torch.tensor(x) for x in local_xs]
                if args.cuda:
                    local_y = local_y.cuda()
                    local_xs = [x.cuda() for x in local_xs]

                optimizer.zero_grad()
                logits = model(local_xs)  # (64, 1)
                loss = model.line2_ce_loss(logits, local_y)
                train_loss += loss.item()
                # bp
                loss.backward()
                optimizer.step()

                steps += 1

            print('Epoch {0} - Train loss\u2193:{1:.10}'.format(epoch, train_loss / len(train)))

            if epoch % args.save_interval == 0:
                print(datetime.now().strftime("%m/%d/%Y %X"))
                utils.save(model, args.save_dir, 'snapshot', epoch)

    return


if __name__ == '__main__':
    main()

