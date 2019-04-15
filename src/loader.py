import os
import sys
import re
import pickle
import numpy as np
import networkx as nx
import distance
import jellyfish
import re
from collections import Counter
from torchnlp.word_to_vector import CharNGram

import torch

import utils


def loading_unified_data(per, days, test_neg_num, num_oov,
                         re_sample_test,
                         term_strings, term_concept_dict, all_iv_terms,
                         random_sample=True,
                         use_train_single=False):

    train_graph = nx.read_gpickle('../data/train_graph_nx_per' + per + '_' + days + '.gpickle')
    test_graph = nx.read_gpickle('../data/test_graph_nx_per' + per + '_' + days + '.gpickle')

    train_multi = [(x, list(train_graph.adj[x])) for x in train_graph.nodes]
    test_multi = [(x, list(test_graph.adj[x])) for x in test_graph.nodes]
    train_single = list(train_graph.edges())
    test_single = list(test_graph.edges())

    np.random.shuffle(train_multi)
    np.random.shuffle(test_multi)

    if use_train_single:
        train_pos = train_single
    else:
        train_pos = train_multi

    dev_pos = test_multi[: int(len(test_multi) / 2)]
    iv_test_pos = test_multi[int(len(test_multi) / 2)::]
    dis_iv_pos = utils.filter_sim_terms(iv_test_pos, term_strings)
    print('Data: # train: {0}, # val: {1}, # iv test: {2}, # dis iv test {3}'.format(len(train_pos),
                                                                                     len(dev_pos),
                                                                                     len(iv_test_pos),
                                                                                     len(dis_iv_pos)))
    print('Sum # pos: train: {0}, val: {1}, iv test: {2}, '
          'dis iv test {3}'.format(np.sum([len(x[1]) for x in train_pos]),
                                      np.sum([len(x[1]) for x in dev_pos]),
                                      np.sum([len(x[1]) for x in iv_test_pos]),
                                      np.sum([len(x[1]) for x in dis_iv_pos])))

    print('Average # pos: train: {0:.3}, val: {1:.3}, iv test: {2:.3}, '
          'dis iv test {3:.3}'.format(np.mean([len(x[1]) for x in train_pos]),
                                      np.mean([len(x[1]) for x in dev_pos]),
                                      np.mean([len(x[1]) for x in iv_test_pos]),
                                      np.mean([len(x[1]) for x in dis_iv_pos])))

    all_oov_test = pickle.load(open('../data/oov_test_dict_per' + per + '_' + days + '.pkl', 'rb'))
    all_oov_test = list(all_oov_test.items())
    np.random.shuffle(all_oov_test)
    oov_test_pos = all_oov_test[:num_oov]
    dis_oov_pos = utils.filter_sim_terms(oov_test_pos, term_strings)
    print('# oov test: {0}, average # pos: {1:3}, sum # pos: {2}'.format(
        len(oov_test_pos), np.mean([len(x[1]) for x in oov_test_pos]), np.sum([len(x[1]) for x in oov_test_pos])))

    print('# dis oov test: {0}, average # pos: {1:.3}, sum # pos: {2}'.format(
        len(dis_oov_pos), np.mean([len(x[1]) for x in dis_oov_pos]), np.sum([len(x[1]) for x in dis_oov_pos])))

    # pickle.dump([dev_pos, iv_test_pos, dis_iv_pos, oov_test_pos, dis_oov_pos], open('../data/dev_test_pos.pkl', 'wb'),
    #             protocol=-1)

    if random_sample:
        if re_sample_test:
            print('Re-sample negative samples for testing!')
            dev = utils.random_neg_sampling(dev_pos, all_iv_terms, term_concept_dict, test_neg_num)

            iv_test = utils.random_neg_sampling(iv_test_pos, all_iv_terms, term_concept_dict, test_neg_num)
            dis_iv_test = utils.random_neg_sampling(dis_iv_pos, all_iv_terms, term_concept_dict, test_neg_num)

            oov_test = utils.random_neg_sampling(oov_test_pos, all_iv_terms, term_concept_dict, test_neg_num)
            dis_oov_test = utils.random_neg_sampling(dis_oov_pos, all_iv_terms, term_concept_dict, test_neg_num)

            pickle.dump([dev, iv_test, dis_iv_test, oov_test, dis_oov_test],
                        open('../data/dev_test_data_per{0}_{1}_{2}_{3}.pkl'.
                             format(per, days, num_oov, len(dis_oov_pos)), 'wb'), protocol=-1)
        else:
            dev, iv_test, dis_iv_test, oov_test, dis_oov_test = pickle.load(
                open('../data/dev_test_data_per{0}_{1}_{2}_{3}.pkl'.
                     format(per, days, num_oov, len(dis_oov_pos)), 'rb'))
    else:
        dev, iv_test, dis_iv_test, oov_test, dis_oov_test = pickle.load(
            open('../data/new_dev_test_data_per{0}_{1}_{2}.pkl'.
                 format(per, days, num_oov), 'rb'))

    print('Data loaded!')
    return train_pos, dev, iv_test, dis_iv_test, oov_test, dis_oov_test


def load_pretrain_vec(embed_filename, dim=None):
    word_dict = {}
    with open(embed_filename) as f:
        for idx, line in enumerate(f):
            L = line.strip().split()
            word, vec = L[0], L[1::]
            # word = L[0].lower()
            if dim is None and len(vec) > 1:
                dim = len(vec)
            elif len(vec) == 1:
                print('header? ', L)
                continue
            elif dim != len(vec):
                raise RuntimeError('Wrong dimension!')

            word_dict[word] = np.array(vec, dtype=np.float32)
            # assert(len(word_dict[word]) == input_dim)
    return word_dict


def load_pretrain_graph_embed(file_name):
    f = open(file_name).readlines()
    # print('Node embeddings: ', f[0].strip())
    node_num = int(f[0].strip().split()[0])
    node_embed = int(f[0].strip().split()[1])
    node_dict = {int(x.strip().split()[0]): np.array(x.strip().split()[1::], dtype=np.float32)
                 for x in f[1::]}

    return node_dict, node_embed, node_num


def w2v_mapping(list_phrases):
    phrases_list = [x.split() for x in list_phrases]
    words_list = [x.lower() for y in phrases_list for x in y]
    word_vocab = Counter(words_list)
    word_vocab = [x[0] for x in word_vocab.items() if x[1] > 10]
    word_to_id = {x: idx+1 for idx, x in enumerate(word_vocab)}
    word_to_id['<UNK>'] = len(word_to_id) + 1
    id_to_word = {v: k for k, v in word_to_id.items()}
    print('Find {0} words.'.format(len(word_to_id)))
    return word_vocab, word_to_id, id_to_word


def pre_train_word_mapping(word_to_id, args):

    def load_glove_vec(embed_filename, vocab, input_dim):
        word_dict = {}
        with open(embed_filename) as f:
            for idx, line in enumerate(f):
                L = line.split()
                word = L[0].lower()
                if word in vocab:
                    word_dict[word] = np.array(L[-input_dim::], dtype=np.float32)
                    # assert(len(word_dict[word]) == input_dim)
        return word_dict

    word_dict = load_glove_vec(args.embed_filename, word_to_id.keys(), args.word_embed_dim)

    initrange = 0.5 / args.word_embed_dim
    W = np.random.uniform(-initrange, initrange, (len(word_to_id) + 1, args.word_embed_dim))
    i = 0
    for cur_word in word_to_id.keys():
        if cur_word in word_dict:
            i += 1
            W[word_to_id[cur_word]] = word_dict[cur_word]

    print(i / float(len(word_to_id)))
    return W


def w2v_mapping_pretrain(list_phrases, args):
    pretrain_dict = load_pretrain_vec(args.embed_filename, args.word_embed_dim)
    pretrain_word_vocab = pretrain_dict.keys()

    phrases_list = [x.split() for x in list_phrases]
    words_list = [x.lower() for y in phrases_list for x in y]
    full_word_vocab = Counter(words_list)

    word_vocab = [x[0] for x in full_word_vocab.items() if x[0] in pretrain_word_vocab or x[1] > 20]
    # word_vocab = [x[0] for x in full_word_vocab.items() if x[1] > 10]

    word_to_id = {x: idx + 1 for idx, x in enumerate(word_vocab)}  # skip 0 id
    word_to_id['<UNK>'] = len(word_to_id) + 1
    id_to_word = {v: k for k, v in word_to_id.items()}

    initrange = 0.5 / args.word_embed_dim
    W = np.random.uniform(-initrange, initrange, (len(word_to_id) + 1, args.word_embed_dim))
    W[0] = np.zeros(args.word_embed_dim)
    i = 0
    for cur_word in word_to_id.keys():
        if cur_word in pretrain_dict:
            i += 1
            W[word_to_id[cur_word]] = pretrain_dict[cur_word]

    print('Find {0} words with pretrain ratio: {1}'.format(len(word_to_id), i / float(len(word_to_id))))

    return word_vocab, word_to_id, id_to_word, W


def c2v_mapping(list_words):
    # [str1, str2, str3, ...]
    chars_list = list(' '.join(list_words))
    char_vocab = Counter(chars_list)
    char_to_id = {x[0]: idx+1 for idx, x in enumerate(char_vocab.items())}
    char_to_id['<UNK>'] = len(char_to_id) + 1
    char_to_id['<s>'] = len(char_to_id) + 1
    char_to_id['</s>'] = len(char_to_id) + 1
    # counting frequency of characters
    id_to_char = {v: k for k, v in char_to_id.items()}
    assert((' ' in char_to_id) and (0 not in id_to_char))
    print('Find %i character.' % (len(char_to_id)))
    return char_vocab, char_to_id, id_to_char


def g2v_mapping_pretrain(list_words, args):
    gram_dict = load_pretrain_vec(args.ngram_embed_path, args.ngram_embed_dim)
    pretrain_gram_vocab = gram_dict.keys()  # 874474

    ngram_list = []
    list_words = list(set(list_words))
    for w in list_words:
        ngram_list += get_single_ngrams(w, args.n_grams)

    # for n in args.n_grams:
    #     cur_list = [ngrams_pretrain(w, n) for w in list_words]
    #     ngram_list += [g for x in cur_list for g in x]

    full_ngrams_vocab = Counter(ngram_list)
    ngrams_vocab = [x[0] for x in full_ngrams_vocab.items() if x[0] in pretrain_gram_vocab or x[1] > 100]
    # ngrams_vocab = [x[0] for x in full_ngrams_vocab.items() if x[0] in pretrain_gram_vocab]

    ngrams_to_id = {x: idx + 1 for idx, x in enumerate(ngrams_vocab)}
    ngrams_to_id['<UNK>'] = len(ngrams_to_id) + 1
    id_to_ngrams = {v: k for k, v in ngrams_to_id.items()}

    initrange = 0.5 / args.ngram_embed_dim
    W = np.random.uniform(-initrange, initrange, (len(ngrams_to_id) + 1, args.ngram_embed_dim))
    W[0] = np.zeros(args.ngram_embed_dim)
    W[-1] = np.zeros(args.ngram_embed_dim)

    i = 0
    for cur_gram in ngrams_to_id.keys():
        if cur_gram in pretrain_gram_vocab:
            i += 1
            W[ngrams_to_id[cur_gram], :] = gram_dict[cur_gram]

    print('Find {0} grams with pretrain ratio: {1}'.format(len(ngrams_to_id), i / float(len(ngrams_to_id))))

    return ngrams_vocab, ngrams_to_id, id_to_ngrams, W


def node_mapping(args, node_to_id=None):
    f = open(args.node_embed_path).readlines()
    # print('Node embeddings: ', f[0].strip())
    node_dict = {int(x.strip().split()[0]): np.array(x.strip().split()[1::], dtype=np.float32)
                 for x in f[1::]}
    # print(len(node_dict))
    if not node_to_id:
        terms_mat = np.zeros((len(node_dict) + 1, int(f[0].strip().split()[1])))
        terms_to_idx = {-1: 0}
        idx_to_terms = {0: -1}
        for idx, term_id in enumerate(node_dict.keys()):
            terms_mat[idx + 1, :] = node_dict[term_id].reshape(1, -1)
            terms_to_idx[term_id] = idx + 1
            idx_to_terms[idx + 1] = term_id

        return terms_to_idx, idx_to_terms, terms_mat
    else:
        terms_mat = np.zeros((len(node_to_id), int(f[0].strip().split()[1])))
        for term_id, idx in node_to_id.items():
            terms_mat[idx, :] = node_dict[term_id].reshape(-1)

        return terms_mat
