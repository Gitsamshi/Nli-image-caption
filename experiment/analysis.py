import json
import pickle as pkl
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import h5py
import json
from nltk.tree import Tree
import os
# os.environ['STANFORD_MODELS'] = '/home/18zs11/AoANet/experiment/stanford-parser-full-2020-11-17'
# os.environ['STANFORD_PARSER'] = '/home/18zs11/AoANet/experiment/stanford-parser-full-2020-11-17'
# from nltk.parse import stanford
# sp = stanford.StanfordParser()
# from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
import ast
import networkx
import math
import sys
from tqdm import tqdm
# sys.path.append('/home/18zs11/AoANet/experiment')
# import complexity as cmp
import matplotlib.pyplot as plt
def coco_nli(coco_snli_path, label_path):

    label_file = h5py.File(label_path, 'r', driver='core')
    label_start_ix = label_file['label_start_ix'][:]
    label_end_ix = label_file['label_end_ix'][:]

    snli_pairs = []

    for idx, (start, end) in tqdm(enumerate(zip(label_start_ix, label_end_ix))):
        ref_num = end - start + 1
        for i in range(ref_num - 1):
            for j in range(i+1, ref_num):
                snli_pairs.append((idx, i, j))
                snli_pairs.append((idx, j, i))

    with open(coco_snli_path, 'r') as f:
        data = f.readlines()
    assert len(snli_pairs) == len(data), 'snli pairs number should equal to data number'

    ret = defaultdict(list)
    ret_prob = defaultdict(list)
    for line, snli_numbers in tqdm(zip(data, snli_pairs), total=len(data)):
        line = line.strip().split(maxsplit=1)
        rela, prob = line[0], ast.literal_eval(line[1])
        ret_prob[snli_numbers[0]].append((snli_numbers[2], snli_numbers[1], prob[0]))

        if rela == 'entailment':
            ret[snli_numbers[0]].append((snli_numbers[2], snli_numbers[1], 1.0))
        else:
            ret[snli_numbers[0]].append((snli_numbers[2], snli_numbers[1], 0.0))

    with open('coco_nli_relation.json', 'w') as f:
        json.dump({'graph': ret, 'prob_graph': ret_prob}, f)

    return {'graph': ret, 'prob_graph': ret_prob}


def run_graph_weights_new(graph_file, output_name, alpha_para):
    with open(graph_file ,'r') as f:
        data = json.load(f)['prob_graph']

    weights = {}
    for k, v in tqdm(data.items(), total=len(data)):
        cur_graph = networkx.DiGraph()
        nodes = list(range(math.ceil(math.sqrt(len(v)))))
        cur_graph.add_nodes_from(nodes)
        cur_edges_dic = defaultdict(float)
        cur_edges = []
        for e in v:
            cur_edges_dic[(e[0], e[1])] = e[2]
        for i in range(len(nodes)-1):
            for j in range(i+1, len(nodes)):
                if cur_edges_dic[(nodes[i], nodes[j])] > 0.5 and cur_edges_dic[(nodes[j], nodes[i])] < 0.5:
                    cur_edges.append([nodes[i], nodes[j], cur_edges_dic[(nodes[i], nodes[j])]])
                elif cur_edges_dic[(nodes[i], nodes[j])] < 0.5 and cur_edges_dic[(nodes[j], nodes[i])] > 0.5:
                    cur_edges.append([nodes[j], nodes[i], cur_edges_dic[(nodes[j], nodes[i])]])

        cur_graph.add_weighted_edges_from(cur_edges)
        PAGE_RANK = networkx.pagerank(cur_graph, alpha=alpha_para, max_iter=500)
        cur_weight = [0.0] * len(PAGE_RANK)
        for kk, vv in PAGE_RANK.items():
            cur_weight[kk] = vv
        weights[int(k)] = cur_weight

    keys = sorted(weights.keys())
    result = []
    cnt = 0
    for key in keys:
        tmp = []
        for idx, weight in enumerate(weights[key]):
            tmp.append([idx + cnt, weight])
        cnt += len(weights[key])
        result.append(tmp)

    with open(os.path.join('../data', output_name), 'w') as f:
        json.dump(result, f)

    return


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    if 0 in s1:
        s1.remove(0)
    s2 = set(list2)
    if 0 in s2:
        s2.remove(0)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def run_weights_undirected(label_path):
    # use jaccard similarity
    label_file = h5py.File(label_path, 'r', driver='core')
    label_start_ix = label_file['label_start_ix'][:] - 1
    label_end_ix = label_file['label_end_ix'][:] - 1
    label = label_file['labels'][:]

    snli_pairs = defaultdict(list)

    for idx, (start, end) in tqdm(enumerate(zip(label_start_ix, label_end_ix))):
        ref_num = end - start + 1

        for i in range(ref_num - 1):
            for j in range(i + 1, ref_num):
                # print ('calcu', label[start+i], label[start+j])
                i_j_similar = jaccard_similarity(label[start+i], label[start+j])
                # print ('similar', i_j_similar)
                snli_pairs[idx].append([i, j, i_j_similar])
                snli_pairs[idx].append([j, i, i_j_similar])
                # snli_pairs.append((idx, j, i))

    with open('coco_jaccard_similar.json', 'w') as f:
        json.dump({'graph': snli_pairs}, f)

    return {'graph': snli_pairs}

if __name__ == '__main__':
    graph_file = '../experiment/coco_nli_new.json'
    run_graph_weights_new(graph_file, 'nli_dist_mle', 0.95)
    run_graph_weights_new(graph_file, 'nli_dist_rl', 0.1)





