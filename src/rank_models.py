import os
import sys
import numpy as np
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import encoder_models as Encoder


class DeepTermRankingListNet(nn.Module):
    def __init__(self, args):
        super(DeepTermRankingListNet, self).__init__()
        self.args = args
        self.context_gamma = self.args.context_gamma
        self.context_dim = args.node_embed_dim
        self.dropout = nn.Dropout(args.dropout)

        self.TermEncoder = Encoder.ContextPredictionWordNGram(args)

        self.str_dim = self.TermEncoder.char_output_dim
        self.con_dim = self.context_dim

        self.context_features = nn.Embedding.from_pretrained(torch.FloatTensor(args.pre_train_nodes), freeze=True)

        self.bi_c = nn.Bilinear(self.con_dim, self.con_dim, 1, bias=True)

        self.att_mat = nn.Parameter(torch.Tensor(self.context_dim, self.context_dim))
        initrange = 0.5 / self.context_dim
        nn.init.uniform_(self.att_mat, -initrange, initrange)

        self.out = nn.LogSoftmax(dim=1)
        self.cand_length = 0

    def _pairwise_interaction(self, mat_A, mat_B):
        # mat_A: (W, D), mat_B: (V, D)
        dim_A = mat_A.shape[0]
        dim_B = mat_B.shape[0]

        mat_sim = torch.matmul(torch.matmul(mat_A, self.att_mat), torch.t(mat_B))
        mat_sim = torch.tanh(mat_sim)

        rows = F.softmax(torch.mean(mat_sim, dim=1), dim=0).reshape(-1, 1)
        cols = F.softmax(torch.mean(mat_sim, dim=0), dim=0).reshape(-1, 1)

        new_A = torch.sum(mat_A * rows, dim=0, keepdim=True)
        new_B = torch.sum(mat_B * cols, dim=0, keepdim=True)
        return new_A, new_B

    def _list_context_vector(self, t1_context, t2s_contexts):
        if type(t1_context) != torch.Tensor:
            t1_context = torch.tensor(t1_context)
            if self.args.cuda:
                t1_context = t1_context.cuda()

        t1_context = self.context_features(t1_context)  # (W, dim for node embedding - D2)

        t1_context_vecs = []
        t2_context_vecs = []
        for i in range(self.cand_length):
            t2_context = t2s_contexts[i]
            if type(t2_context) != torch.Tensor:
                t2_context = torch.tensor(t2_context)  # (context size for t2 - V, M2)
                if self.args.cuda:
                    t2_context = t2_context.cuda()

            t2_context = self.context_features(t2_context)  # (V, D2)

            # pairwise interaction
            if self.args.do_ctx_interact:
                t1_context_vec, t2_context_vec = self._pairwise_interaction(t1_context, t2_context)
            else:
                t1_context_vec = torch.mean(t1_context, dim=0, keepdim=True)  # (1, 128)
                t2_context_vec = torch.mean(t2_context, dim=0, keepdim=True)  # (1, 128)
                # print(t1_context_vec.shape, t2_context_vec.shape)

            t1_context_vecs.append(t1_context_vec)
            t2_context_vecs.append(t2_context_vec)

        t1_context_vecs = torch.cat(t1_context_vecs, dim=0)
        t2_context_vecs = torch.cat(t2_context_vecs, dim=0)  # (N, word dim)
        return t1_context_vecs, t2_context_vecs

    def listwise_cost(self, list_logits, list_labels):
        return -torch.sum(F.softmax(list_labels, 1) * self.out(list_logits))
        # return -torch.sum(list_labels * self.out(list_logits))

    def forward(self, t1_list, t2_list):
        # t1s: (string information, context information (if exists))
        # t2s: (string information, context information)
        # t1s[0]: char_ids, char_lengths, char_list_ids, word_ids, word_lengths, ngram_ids -> a list of vectors
        # t1s[1]: context ids -> one vector
        # t2s[0]: same as t1s[0] -> a list of matrix
        # t2s[1]: same as t1s[1] -> a list of vectors
        self.cand_length = t2_list[0].shape[0]

        # string encoding - the last element is the golden contexts
        t1_intputs = t1_list[:-1]
        t2_intputs = t2_list[:-1]

        t1_embed = self.TermEncoder.get_sub_embed_cell(t1_intputs)  # (1, dim for CNN encoder - D1)
        str_t1s = t1_embed.repeat(self.cand_length, 1)  # (cand_length, char_output_dim)
        str_t2s = self.TermEncoder.get_sub_embed_cell(t2_intputs)  # (N, D1)

        # dropout
        if self.args.dropout:
            str_t1s = self.dropout(str_t1s)
            str_t2s = self.dropout(str_t2s)

        # scoring function
        str_score = F.cosine_similarity(str_t1s, str_t2s)  # (N, D1) (N, D1),-> (N, 1)

        if self.args.use_context:
            # context prediction
            t1_pred_context = self.ContextPredictor(t1_intputs)  # (1, context_vocab)
            t1_contexts = torch.topk(t1_pred_context, self.args.num_contexts)[1].reshape(-1)  # (1, max_contexts)
            t2_pred_context = self.ContextPredictor(t2_intputs)  # (cand_length, context_vocab)
            t2_contexts = torch.topk(t2_pred_context, self.args.num_contexts)[1]  # (cand_length, max_contexts)

            # context interaction
            ctx_t1s, ctx_t2s = self._list_context_vector(t1_contexts, t2_contexts)

            # scoring function
            if self.args.dropout:
                ctx_t1s = self.dropout(ctx_t1s)
                ctx_t2s = self.dropout(ctx_t2s)

            con_score = self.bi_c(ctx_t1s, ctx_t2s)

            str_score = str_score.reshape(-1)
            con_score = con_score.reshape(-1)
            # print('str', str_score)
            # print('con', con_score)
            y = ((1 - self.context_gamma) * str_score + self.context_gamma * con_score).reshape(1, -1)
        else:
            y = str_score.reshape(1, -1)

        return y


