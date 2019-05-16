import os
import sys
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextPredictionWordNGram(nn.Module):
    def __init__(self, args, weights=None):
        super(ContextPredictionWordNGram, self).__init__()
        self.args = args
        # self.dropout = args.dropout
        self.ngram_D = args.ngram_embed_dim
        self.ngram_V = args.ngram_vocab_size
        self.word_D = args.word_embed_dim
        self.word_V = args.word_vocab_size
        self.output_V = args.node_vocab_size
        self.char_output_dim = args.node_embed_dim

        # embeddings
        self.ngrams_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(args.pre_train_ngram), freeze=False)
        self.w2v_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(args.pre_train_words), freeze=False)

        # nets
        self.fc_out = nn.Linear(self.ngram_D + self.word_D, self.char_output_dim)
        self.context_out = nn.Linear(self.char_output_dim, self.output_V, bias=False)
        self.out = nn.LogSoftmax(dim=1)

        # negative sampling
        if args.neg_sampling:
            self.context_embeddings = nn.Embedding.from_pretrained(self.context_out.weight, freeze=False)
            self.num_negs = args.num_negs
            if weights is not None:
                wf = np.power(weights, 0.75)
                wf = wf / wf.sum()
                self.weights = torch.FloatTensor(wf)
            else:
                self.weights = None

    def line2_ce_loss(self, logits, labels):
        '''
        :param logits: [N, C] - un-normalized predicted distribution
        :param labels: [N, C] - un-normalized empirical distribution
        :return:
        '''
        return -torch.sum(labels * self.out(logits))

    def get_sub_embed_cell(self, x_list):
        # x1: [[[1,3,2], [1,2,0], [3,0,0]], [[4,5,6], [dummy], [dummy]], [[7,8,0], [8,0,0], [dummy]]
        # x1: [B, W, C] - batch size, word length, character length
        # x2: [B, W]
        x1, length = x_list[2], x_list[3]
        x1 = self.ngrams_embeddings(x1)  # (B, C_D)
        x1 = torch.sum(x1, dim=1) / length.reshape(-1, 1)  # (B, C_D)
        x2, word_len = x_list[0], x_list[1]
        x2 = self.w2v_embeddings(x2)  # (B, W_D)
        x2 = torch.sum(x2, dim=1) / word_len.reshape(-1, 1)

        x = torch.cat((x1, x2), dim=-1)  # (B, W_D + C_Out)
        x_out = self.fc_out(torch.tanh(x))
        return x_out

    def forward_ns_loss(self, x_list):
        contexts = x_list[-1]
        inputs_embed = self.get_sub_embed_cell(x_list)  # [B, D]
        outputs_embed = self.context_embeddings(contexts)  # [B, D]

        batch_size = contexts.shape[0]
        if self.weights is not None:
            neg_terms = torch.multinomial(self.weights, batch_size * self.num_negs, replacement=True).view(batch_size, -1)
        else:
            neg_terms = torch.FloatTensor(batch_size, self.num_negs).uniform_(0, self.output_V - 1).long()

        if self.args.cuda:
            neg_terms = neg_terms.cuda()

        neg_embed = self.context_embeddings(neg_terms).neg()  # [B, S, D]
        inputs_embed = inputs_embed.unsqueeze(2)  # [B, D]
        outputs_embed = outputs_embed.unsqueeze(1)  # [B, D]
        pos_loss = torch.bmm(outputs_embed, inputs_embed).squeeze().sigmoid().log().mean(0)
        neg_loss = torch.bmm(neg_embed, inputs_embed).squeeze().sigmoid().log().sum(1).mean(0)

        return -(pos_loss + neg_loss).mean()

    def forward(self, x_list):
        x = self.get_sub_embed_cell(x_list)  # [B, D]

        if self.args.neg_sampling:
            weight_mat = self.context_embeddings.weight  # [V, D]
            y = torch.mm(x, weight_mat.t())  # [B, V]
        else:
            y = self.context_out(x)
        return y





























