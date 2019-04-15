import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics

import utils


def evaluation(data, model, criterion, args):
    with torch.no_grad():
        logit_list = []
        label_list = []
        loss = 0
        f = open(args.log_name, 'w')
        i = 0
        model.eval()
        for cur_sample in data:
            # cur_idx_data = make_idx_data([cur_sample], args)
            # t1_list, t2_list, onehot_labels = preprocess(cur_idx_data[0], args)
            t1_list, t2_list, onehot_labels = utils.preprocess(cur_sample, args)

            # print([x.shape for x in t2_list])
            onehot_labels = torch.FloatTensor(onehot_labels).reshape(1, -1)

            t1_list = [torch.tensor(x) for x in t1_list[:-1]] + [t1_list[-1]]
            t2_list = [torch.tensor(x) for x in t2_list[:-1]] + [t2_list[-1]]
            if args.cuda:
                onehot_labels = onehot_labels.cuda()
                t1_list = [x.cuda() for x in t1_list[:-1]] + [t1_list[-1]]
                t2_list = [x.cuda() for x in t2_list[:-1]] + [t2_list[-1]]

            logits = model(t1_list, t2_list)

            if args.cuda:
                logit_list.append(logits.cpu().detach().numpy()[0])
                label_list.append(onehot_labels.cpu().numpy()[0])
            else:
                logit_list.append(logits.detach().numpy()[0])
                label_list.append(onehot_labels.numpy()[0])

            i += 1
            if i < 1000 and args.logging:
                t1 = cur_sample['t1_string']
                cur_logits = F.softmax(torch.FloatTensor(cur_logits)).numpy()
                for t2, logit in zip(cur_sample['t2s_string'], cur_logits):
                    # print(t1, t2, logit)
                    f.write("{0}\t{1}\t{2:.5}\n".format(t1, t2, logit))
                f.write('\n')
        f.close()
        # print('Finish storing!')

    return utils.eval_metric(logit_list, label_list, args.metric)
