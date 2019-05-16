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
from sklearn.metrics.pairwise import cosine_similarity

import torch

import loader


def get_ranking_score(model, query, candidates, args):
    all_logits = []
    batch_size = 10000
    num_batches = len(candidates) // batch_size

    for i in range(num_batches):
        cur_batch = candidates[i * batch_size: (i + 1) * batch_size]
        if i == num_batches - 1:
            cur_batch = candidates[i * batch_size::]

        samples = [query, [(x, 0) for x in cur_batch]]
        cur_idx_data = make_idx_data([samples], args)
        t1_list, t2_list, onehot_labels = preprocess(cur_idx_data[0], args)

        t1_list = [torch.tensor(x) for x in t1_list[:-1]] + [t1_list[-1]]
        t2_list = [torch.tensor(x) for x in t2_list[:-1]] + [t2_list[-1]]
        if args.cuda:
            t1_list = [x.cuda() for x in t1_list[:-1]] + [t1_list[-1]]
            t2_list = [x.cuda() for x in t2_list[:-1]] + [t2_list[-1]]

        logits = model(t1_list, t2_list)

        if args.cuda:
            logits = logits.cpu().detach().numpy()
        else:
            logits = logits.detach().numpy()

        all_logits += list(logits.reshape(-1))

    return all_logits


def unsupervised_pair_ranking(idx_data, score_func, metric):
    all_logits = []
    all_labels = []
    for sample in idx_data:
        labels = [x[1] for x in sample[1]]
        logits = [score_func(sample[0], x[0]) for x in sample[1]]

        all_logits.append(logits)
        all_labels.append(labels)

    return eval_metric(all_logits, all_labels, metric)


def unsupervised_ranking(idx_data, query_func, cand_func, args):
    all_logits = []
    all_labels = []
    if args.cal_all_neighbors:
        querys = [x[0] for x in idx_data]
        all_neighbors = cal_distance_for_all_query(querys, args.all_iv_terms,
                                                   args.get_charngram, args.get_charngram, args.num_contexts)

    for sample in idx_data:
        labels = [x[1] for x in sample[1]]

        if args.batch_embed:
            query, _ = query_func(sample[0])
            candidates, labels = cand_func([x[0] for x in sample[1]], labels)
        else:
            if args.cal_all_neighbors:
                query = query_func(sample[0], all_neighbors)
            else:
                query = query_func(sample[0])

            candidates = np.concatenate([cand_func(x[0]) for x in sample[1]], 0)
        logits = cosine_similarity(query, candidates).reshape(-1)

        all_logits.append(logits)
        all_labels.append(labels)

    return eval_metric(all_logits, all_labels, args.metric)


def share_concept_cooc(t1, t2, term_concept_dict, threshold=0):
    if (t1 == t2) or (t1 not in term_concept_dict) or (t2 not in term_concept_dict):
        return False
    c1 = set(term_concept_dict[t1])
    c2 = set(term_concept_dict[t2])
    return True if len(c1 & c2) > threshold else False


def cal_distance_for_one_query(query, candidates, query_func, cand_func, num_nearest_k):
    query_vec = query_func(query)
    embed_dim = query_vec.shape[1]

    terms_mat = np.zeros((len(candidates), embed_dim))
    terms_to_idx = {}
    idx_to_terms = {}
    for idx, term_id in enumerate(candidates):
        terms_mat[idx, :] = cand_func(term_id)
        terms_to_idx[term_id] = idx
        idx_to_terms[idx] = term_id

    pw_cos_mat = cosine_similarity(query_vec, terms_mat)  # (1, N)
    argsort_mat = np.argsort(pw_cos_mat)  # (1, N)

    neighbors = [idx_to_terms[x] for x in argsort_mat[0][::-1][:num_nearest_k]]
    return neighbors


def cal_distance_for_all_query(querys, candidates, query_func, cand_func, num_nearest_k):
    embed_dim = query_func(querys[0]).shape[1]
    query_mat = np.zeros((len(querys), embed_dim))  # (M, D)
    query_to_idx = {}
    idx_to_query = {}
    for idx, term_id in enumerate(querys):
        query_mat[idx, :] = query_func(term_id)
        query_to_idx[term_id] = idx
        idx_to_query[idx] = term_id

    terms_mat = np.zeros((len(candidates), embed_dim))  # (N, D)
    terms_to_idx = {}
    idx_to_terms = {}
    for idx, term_id in enumerate(candidates):
        terms_mat[idx, :] = cand_func(term_id)
        terms_to_idx[term_id] = idx
        idx_to_terms[idx] = term_id

    pw_cos_mat = cosine_similarity(query_mat, terms_mat)  # (M, N)
    argsort_mat = np.argsort(pw_cos_mat)  # (M, N)

    neighbors = {}
    for i in range(len(querys)):
        neighbors[idx_to_query[i]] = [idx_to_terms[x] for x in argsort_mat[i][::-1][:num_nearest_k]]
    return neighbors


def make_idx_one_term(dataset, args):
    # dataset: a list of term ids
    data_dict_list = []
    for cur_term in dataset:
        if type(cur_term ) is not str:
            cur_dict = {}
            cur_dict['id'] = cur_term
            cur_dict['str'] = args.term_strings[cur_term]
            # default values - reduce the memory load
            cur_dict['ngram_ids'] = [0]
            cur_dict['char_list_ids'] = [[0]]
            cur_dict['char_list_len'] = [0]
            cur_dict['ngram_list_ids'] = [[0]]
            cur_dict['ngram_list_len'] = [0]
            cur_dict['word_ids'] = [0]
            cur_dict['word_len'] = len(args.term_strings[cur_term].split())
            cur_dict['gold_ctx'] = [0]

            '''keep the following few lines to be the same as those in main_pretrain'''
            ngram_ids = []
            for g in get_single_ngrams(args.term_strings[cur_term], args.n_grams):
                if g in args.ngram_to_id:
                    ngram_ids.append(args.ngram_to_id[g])
            if ngram_ids is []:
                ngram_ids.append(args.ngram_to_id['<UNK>'])
            cur_dict['ngram_ids'] = ngram_ids

            word_list = args.term_strings[cur_term].split()
            cur_dict['word_ids'] = [args.word_to_id[w if w in args.word_to_id else '<UNK>'] for w in word_list]
            cur_dict['word_len'] = len(word_list)

            # collect golden contexts
            if args.use_context:
                # cur_dict['gold_ctx'] = [args.node_to_id[cur_term]]
                if cur_term in args.node_to_id:
                    cur_dict['gold_ctx'] = [args.node_to_id[cur_term]]
                else:
                    cur_dict['gold_ctx'] = [0]
        else:
            cur_dict = {}
            cur_dict['str'] = cur_term
            # default values - reduce the memory load
            cur_dict['ngram_ids'] = [0]
            cur_dict['char_list_ids'] = [[0]]
            cur_dict['char_list_len'] = [0]
            cur_dict['ngram_list_ids'] = [[0]]
            cur_dict['ngram_list_len'] = [0]
            cur_dict['word_ids'] = [0]
            cur_dict['word_len'] = len(cur_term.split())
            cur_dict['gold_ctx'] = [0]

            '''keep the following few lines to be the same as those in main_pretrain'''
            ngram_ids = []
            for g in get_single_ngrams(cur_term, args.n_grams):
                if g in args.ngram_to_id:
                    ngram_ids.append(args.ngram_to_id[g])
            if ngram_ids is []:
                ngram_ids.append(args.ngram_to_id['<UNK>'])
            cur_dict['ngram_ids'] = ngram_ids

            word_list = cur_term.split()
            cur_dict['word_ids'] = [args.word_to_id[w if w in args.word_to_id else '<UNK>'] for w in word_list]
            cur_dict['word_len'] = len(word_list)

        data_dict_list.append(cur_dict)
    return data_dict_list


def make_idx_data(dataset, args, testing=False):
    data_list = []  # [(t1, [(t2, 1), (t2, 0), ...]), (), (), ...]
    for cur_sample in dataset:
        cur_dict = {}
        if testing:
            t1_terms = [cur_sample[0]] * len(cur_sample[1])
            cur_dict['t1_dicts'] = make_idx_one_term(t1_terms, args)
        else:
            t1_terms = [cur_sample[0]]
            cur_dict['t1_dicts'] = make_idx_one_term(t1_terms, args)

        t2_terms = [x[0] for x in cur_sample[1]]
        cur_dict['t2_dicts'] = make_idx_one_term(t2_terms, args)

        labels = [x[1] for x in cur_sample[1]]
        cur_dict['labels'] = labels
        # add context

        data_list.append(cur_dict)
    return data_list


def batch_process_terms(batch_data, args, labels=None):
    word_ids = []
    word_len = []
    ngram_ids = []
    ngram_length = []

    for sample in batch_data:
        word_ids.append(sample['word_ids'])
        word_len.append(sample['word_len'])
        ngram_ids.append(sample['ngram_ids'])
        ngram_length.append(len(sample['ngram_ids']))

    word_ids = pad_sequence(word_ids).astype(int)
    word_lengths = np.array(word_len).astype(np.float32)
    ngram_ids = pad_sequence(ngram_ids).astype(int)
    ngram_length = np.array(ngram_length).astype(np.float32)

    if labels:
        labels = np.array(labels)

    return [word_ids, word_lengths, ngram_ids, ngram_length], labels


def preprocess(sample, args, testing=False):
    # sample: ['t1_dict': {}, 't2_dict': {}, 'labels': array]

    if testing:
        x1_list, _ = batch_process_terms(sample['t1_dicts'], args)
        x1_gold_ctx = [x['gold_ctx'] for x in sample['t1_dicts']]
    else:
        x1_list, _ = batch_process_terms(sample['t1_dicts'], args)
        x1_gold_ctx = sample['t1_dicts'][0]['gold_ctx']

    x2_list, onehot_labels = batch_process_terms(sample['t2_dicts'], args, sample['labels'])
    x2_gold_ctx = [x['gold_ctx'] for x in sample['t2_dicts']]

    x1_list.append(x1_gold_ctx)
    x2_list.append(x2_gold_ctx)

    return x1_list, x2_list, onehot_labels


def pad_sequence(list_ids, min_length=0, max_length=None, padder=0):
    if not max_length:
        max_length = max([len(x) for x in list_ids] + [min_length])
    # print(max_length)
    new_list_ids = []
    for ids in list_ids:
        new_list_ids.append(ids + [padder] * (max_length - len(ids)))
    # return torch.tensor(new_list_ids)
    return np.array(new_list_ids)


def pad_pad_sequence(list_list_ids, min_length=0, list_padder=0):
    max_char_length = max([len(x) for w in list_list_ids for x in w] + [min_length])
    max_list_length = max(len(x) for x in list_list_ids)
    new_list_list = []
    for cur_list in list_list_ids:
        cur_array = pad_sequence(cur_list, min_length, max_char_length)
        pad_zeros = np.ones((max_list_length - cur_array.shape[0], max_char_length)) * list_padder
        cur_array = np.concatenate([cur_array, pad_zeros], axis=0)
#         if cur_array.shape[0] < max_list_length:
#             pad_zeros = np.zeros((max_list_length - cur_array.shape[0], max_char_length))
#             cur_array = np.concatenate([cur_array, pad_zeros], axis=0)
        new_list_list.append(cur_array)
    return np.array(new_list_list)


def preprocess_baseline(sample, args):

    labels = [x[1] for x in sample[1]]

    if args.pair_score:
        onehot_labels = np.array(labels)
        features = np.concatenate([args.score_func(sample[0], x[0]) for x in sample[1]], 0)
        return [features], onehot_labels

    else:
        if args.batch_embed:
            query, _ = args.query_func(sample[0])
            candidates, labels = args.cand_func([x[0] for x in sample[1]], labels)
        else:
            query = args.query_func(sample[0])
            candidates = np.concatenate([args.cand_func(x[0]) for x in sample[1]], 0)

        t1_vecs = np.repeat(query, len(sample[1]), axis=0)
        t2_vecs = candidates

        onehot_labels = np.array(labels)
        return [t1_vecs, t2_vecs], onehot_labels


def preprocess_relation(batch_data, args):

    labels = np.array([args.labels_mapping[x[-1]] for x in batch_data])

    if args.pair_score:
        features = np.concatenate([args.score_func(x[0], x[1]) for x in batch_data], 0)
        return [features], labels
    else:
        if args.batch_embed:
            querys, _ = args.query_func([x[0] for x in batch_data])
            candidates, _ = args.cand_func([x[1] for x in batch_data])
        else:
            querys = np.concatenate([args.query_func(x[0]) for x in batch_data], 0)
            candidates = np.concatenate([args.cand_func(x[1]) for x in batch_data], 0)

    return [querys, candidates], labels


# def share_concept_cooc(t1, t2, term_concept_dict, threshold=0):
#     # if term_id_1 not in term_concept_mapping or term_id_2 not in term_concept_mapping:
#     #     return False
#     if t2 not in term_concept_dict:
#         return False
#     c1 = set(term_concept_dict[t1])
#     c2 = set(term_concept_dict[t2])
#     return True if len(c1 & c2) > threshold else False

def random_neg_sampling(pos_pairs, all_candidates, term_concept_mapping, neg_number=50, list_wise=True):
    # pos_pairs: [(1, (2,3,4)), (4,(5, 6, 7)), (6, (7, 8, 9)), ...]
    dataset = []

    if list_wise:
        # return a list of (t1, [t2, t2, t2, ...])
        for pos_pair in pos_pairs:
            t1 = pos_pair[0]
            cur_list = [(x, 1) for x in pos_pair[1]]
            exists = [t1] + pos_pair[1]
            count = 0
            while count < neg_number:
                # t2 = np.random.choice(all_candidates)
                t2 = all_candidates[np.random.randint(len(all_candidates))]
                if (t2 not in exists) and (not share_concept_cooc(t1, t2, term_concept_mapping)):
                    exists.append(t2)
                    cur_list.append((t2, 0))
                    count += 1

            np.random.shuffle(cur_list)
            dataset.append((t1, cur_list))
    else:
        # return a list of tuple, (t1, pos, neg), ...
        pos_pairs = pos_pairs + [(x[1], x[0]) for x in pos_pairs]
        for pos_pair in pos_pairs:
            t1 = pos_pair[0]
            exists = [t1, pos_pair[1]]
            count = 0
            while count < neg_number:
                t2 = all_candidates[np.random.randint(len(all_candidates))]
                if (t2 not in exists) and (not share_concept_cooc(t1, t2, term_concept_mapping)):
                    exists.append(t2)
                    count += 1
                    dataset.append((t1, pos_pair[1], t2))

    return dataset


def filter_sim_terms(pairs, term_strings):
    new_pairs = []
    for pair in pairs:
        query = pair[0]
        cands = pair[1]
        new_cands = [x for x in cands if not is_string_similar(term_strings[query], term_strings[x])]
        if new_cands:
            new_pairs.append((query, new_cands))

    return new_pairs


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def eval_metric(logits, labels, metric='mrr'):
    # print(logits[0], labels[0])
    # input: (list of) list of scores, numpy arrays
    assert(type(labels) == list)
    metric = metric.lower()
    rs = []
    score = 0
    for logit, label in zip(logits, labels):
        ordered = np.argsort(logit)[::-1]
        rs.append(np.array(label)[ordered])

    if metric == 'mrr':
        score = mean_reciprocal_rank(rs)
    elif metric == 'map':
        score = mean_average_precision(rs)
    else:
        raise ValueError('Wrong metric selection!')

    return score


def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{0}_epoch_{1}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)


def is_string_similar(t1, t2):
    # edit distance
    if distance.levenshtein(t1, t2, normalized=True) < 0.8:
        return True

    # LCS >= 5
    # if max([len(x) for x in list(distance.lcsubstrings(t1, t2)) + ['']]) >= 5:
    #     return True

    # a simple pattern: reverse
    if set(re.findall(r'[\w]+', t1)) == set(re.findall(r'[\w]+', t2)):
        return True

    return False


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ngrams_pretrain(string, n):
    assert(n > 0)
    # add START and END symbol
    string = re.sub(r'[,-./]|\sBD', r'', string)
    if n > 1:
        char_list = ['#BEGIN#'] + list(string) + ['#END#']
    else:
        char_list = list(string)
    ngrams = zip(*[char_list[i:] for i in range(n)])
    return ['{0}gram-{1}'.format(n, ''.join(ngram)) for ngram in ngrams]


def get_single_ngrams(string, gram_num_list):
    ngrams_list = []
    for n in gram_num_list:
        ngrams_list += ngrams_pretrain(string, n)
    return ngrams_list
