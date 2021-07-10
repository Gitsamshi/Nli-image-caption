from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

# def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
#     batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
#     seq_per_img = batch_size // len(data_gts)
#
#     res = OrderedDict()
#
#     gen_result = gen_result.data.cpu().numpy()
#     greedy_res = greedy_res.data.cpu().numpy()
#     for i in range(batch_size):
#         res[i] = [array_to_str(gen_result[i])]
#     for i in range(batch_size):
#         res[batch_size + i] = [array_to_str(greedy_res[i])]
#     print ('data_gts', data_gts)
#     gts = OrderedDict()
#     for i in range(len(data_gts)):
#         gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
#
#     res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
#     res__ = {i: res[i] for i in range(2 * batch_size)}
#     gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
#     print ('gts', gts)
#     if opt.cider_reward_weight > 0:
#         _, cider_scores = CiderD_scorer.compute_score(gts, res_)
#         print('Cider scores:', _)
#     else:
#         cider_scores = 0
#     if opt.bleu_reward_weight > 0:
#         _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
#         bleu_scores = np.array(bleu_scores[3])
#         print('Bleu scores:', _[3])
#     else:
#         bleu_scores = 0
#     scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
#
#     scores = scores[:batch_size] - scores[batch_size:]
#
#     rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
#
#     return rewards


def get_self_critical_reward(greedy_res, data_gts_tuple, gen_result, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    data_gts, prob_gts = data_gts_tuple  # prob_gts = num_img * ref_num
    seq_per_img = batch_size // len(data_gts)
    # print ('len of data_gts', len(data_gts))
    # print (data_gts[0], data_gts[1])
    # print ('len of prob_gts', len(prob_gts))
    # print (prob_gts[0], prob_gts[1])

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    gts_p = {i: prob_gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    gts_f = (gts, gts_p)
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_f, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def get_self_constrained_critical_reward(greedy_res, data_gts_triple, gen_result, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    data_gts, prob_gts, substractor = data_gts_triple  # prob_gts = num_img * ref_num
    # print ('data_gts, prob_gts, substractor', data_gts, prob_gts, substractor)
    seq_per_img = batch_size // len(data_gts)
    # print ('len of data_gts', len(data_gts))
    # print (data_gts[0], data_gts[1])
    # print ('len of prob_gts', len(prob_gts))
    # print (prob_gts[0], prob_gts[1])

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    gts_p = {i: prob_gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    gts_g = {i: substractor[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    gts_f = (gts, gts_p, gts_g)
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score_con(gts_f, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
