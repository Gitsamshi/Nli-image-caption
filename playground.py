import ast
import networkx as nx
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm

# nodes = [0, 1, 2, 3, 4]
# graph = [[4, 3, 0.75], [4, 1, 0.81], [4, 2, 0.97], [4, 0, 0.52]]
# page_rank_probs = defaultdict(float)
# DG = nx.DiGraph()
# DG.add_nodes_from(nodes)
# DG.add_weighted_edges_from(graph)
# PAGE_RANK = nx.pagerank(DG, alpha=0.95)
# # for sub_graph in nx.weakly_connected_components(DG):
# #     sub_graph_size = len(sub_graph)
# #     PAGE_RANK = nx.pagerank(DG.subgraph(list(sub_graph)))
# #
# #     normalized_PAGERANK = {k: v * (sub_graph_size) / 5 for k, v in PAGE_RANK.items()}
# #     page_rank_probs.update(normalized_PAGERANK)
# #     # print ('normalized_PAGERANK', normalized_PAGERANK)
# #
# print(page_rank_probs)
# print(PAGE_RANK)
#
# # def find_ngrams(input_list, n):
# #     print ([input_list[i:] for i in range(n)])
# #     return zip(*[input_list[i:] for i in range(n)])
# #
# # # for ng in find_ngrams(['I', 'live', 'in', 'kingston'], 2):
# # #     print ('ng', ng)
# # print (range(5))
#
# # a = np.array([[1, 4, 2], [3, 5, 6]])
# # b = np.array([[1,1], [2,2], [3, 3], [4, 4], [5, 5], [6, 6]])
# # print (b[a[0]])
# a = np.array([1, 2, 4, 5, 56])
# print ( int(np.sum(a > 5)))

# def argsort(seq):
#     # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
#     return sorted(range(len(seq)), key=seq.__getitem__)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) * 1)
    return e_x / e_x.sum()


def artificial_nli(cider_data, nli_data, nli_origin, thres):
    d = json.load(open(cider_data, 'r'))
    dn = json.load(open(nli_data, 'r'))
    do = json.load(open(nli_origin, 'r'))['prob_graph']
    prob_array = []
    cnt = 0
    for e in do.values():
        if cnt >= 500:
            break
        for ee in e:
            # print ('ee', ee, 'e', e)
            if ee[2] > 0.5:
                prob_array.append(ee[2])
                cnt += 1
    prob_array = np.array(prob_array)

    prob_edges = []
    node_nums = []
    has_edges = 0
    xxx = 0
    for e, en in tqdm(zip(d, dn)):
        tmp_edges = []
        num = len(en)
        new_e = softmax(np.array([ele[1] for ele in e[:num]]))
        # print (new_e)
        arg_new_e = np.argsort(new_e)
        low = 1.0 / len(en) * (thres)
        high = 1.0 / len(en) * (1.0 / thres)
        small_num = int(np.sum(new_e < low))
        # print ('small_num', small_num)
        large_num = int(np.sum(new_e > high))
        # print ('large_num', large_num)
        if small_num > 0:
            # print (np.random.choice(prob_array, size=(num - small_num,)))
            # print (np.random.rand(num - small_num,) * 0.1)
            sampl = np.random.choice(prob_array, size=(num - small_num,)) + (np.random.rand(num - small_num,) - 0.5) * 0.05
            sampl = np.sort(sampl)
            sampl = np.clip(sampl, 0.501, 0.999)
            # sampl = np.sort(np.random.uniform(low=0.5, high=1.0, size=(num - small_num,)))
            for i in range(num - small_num):
                tmp_edges.append([arg_new_e[small_num-1], arg_new_e[small_num + i], sampl[i]])

        if large_num > 0:
            sampl = np.random.choice(prob_array, size=(large_num,)) + (np.random.rand(large_num,) - 0.5) * 0.1
            sampl = np.sort(sampl)
            sampl = np.clip(sampl, 0.501, 0.999)
            for i in range(1, large_num+1):
                if num - i > small_num:
                    tmp_edges.append([arg_new_e[small_num], arg_new_e[-i], sampl[-i]])
                else:
                    break

        if small_num > 0 or large_num > 0:
            has_edges += 1


        ext_edges = []
        for i in range(len(tmp_edges)):
            cur = tmp_edges[i][1]
            for j in range(i+1, len(tmp_edges)):
                if cur == tmp_edges[j][0]:
                    sampl = np.random.choice(prob_array, size=(1)) + (np.random.rand(1)) * 0.1
                    ext_edges.append([tmp_edges[i][0], tmp_edges[j][1], sampl[0]])
                    xxx += 1
        tmp_edges.extend(ext_edges)

        prob_edges.append(tmp_edges)
        node_nums.append(num)
    print('xxx', xxx)
    json.dump({'edges': prob_edges, 'nodes': node_nums}, open('experiment/coco_nli_graph_pg1.json', 'w'))
    print ('has_edges', has_edges)

    return prob_edges


def minorchanges_nli(cider_data, nli_data, nli_origin, change=True):
    d = json.load(open(cider_data, 'r')) # test and val data excluded
    dn = json.load(open(nli_data, 'r'))
    do = json.load(open(nli_origin, 'r'))

    h_adj, l_adj, no_change, no_change_hi = 0, 0, 0, 0
    for idx, (e, en) in tqdm(enumerate(zip(d, dn))):
        num = len(en)
        new_e = softmax(np.array([ele[1] for ele in e[:num]]))
        # print (new_e)
        if np.array_equal(new_e, np.array([0.2, 0.2, 0.2, 0.2, 0.2])):
            # print ('skip')
            continue
        arg_new_e = np.argsort(new_e)
        lo, hi = arg_new_e[0], arg_new_e[-1]

        nli_e = np.array([ele[1] for ele in en])
        arg_nli_e = np.argsort(nli_e)
        nlo, nhi = arg_nli_e[0], arg_nli_e[-1]
        if lo == nlo and hi == nhi:
            no_change += 1
            continue
        if hi == nhi:
            no_change_hi += 1
        if change:
            current_prob = do['prob_graph'][str(idx)]
            current_edge = do['graph'][str(idx)]

            forward_ix_list, backward_ix_list = [], []
            for ix in range(len(current_prob)):
                if current_prob[ix][0] == lo:
                    forward_ix_list.append(ix)
                if current_prob[ix][0] == hi:
                    backward_ix_list.append(ix)

            assert len(forward_ix_list) > 0 and len(backward_ix_list) > 0, 'has to find the index'

            def samp(a, b):
                return np.random.uniform(low=a, high=b, size=1)[0]
            # sampl2 = np.random.uniform(low=0.01, high=0.05, size=1)[0]

            for f_ix in forward_ix_list:
                if current_prob[f_ix][2] < 0.5:
                    if current_prob[f_ix][1] == hi:
                        current_prob[f_ix][2] = samp(0.89, 0.99)
                        current_edge[f_ix][2] = 1.0
                        h_adj += 1
                    elif np.random.uniform() > 0.85:
                        current_prob[f_ix][2] = samp(0.51, 0.95)
                        current_edge[f_ix][2] = 1.0
                        h_adj += 1

            for b_ix in backward_ix_list:
                if current_prob[b_ix][2] > 0.5:
                    current_prob[b_ix][2] = samp(0.01, 0.05)
                    current_edge[b_ix][2] = 0.0
                    l_adj += 1

            do['prob_graph'][str(idx)] = current_prob
            do['graph'][str(idx)] = current_edge

    print('no need to change', no_change)
    print('h_adj', h_adj, 'l_adj', l_adj, 'nochangehi', no_change_hi)
    if change:
        json.dump(do, open('experiment/coco_nli_new.json', 'w'))

    return


def show_lo_and_high(cider_data, nli_data, data_json):
    d = json.load(open(cider_data, 'r'))  # test and val data excluded
    dn = json.load(open(nli_data, 'r'))
    da = json.load(open(data_json, 'r'))['images']
    cnt = 0
    for idx, (e, en, enn) in tqdm(enumerate(zip(d, dn, da))):
        num = len(en)
        new_e = softmax(np.array([ele[1] for ele in e[:num]]))
        if np.array_equal(new_e, np.array([0.2, 0.2, 0.2, 0.2, 0.2])):
            continue
        arg_new_e = np.argsort(new_e)
        lo, hi = arg_new_e[0], arg_new_e[-1]
        cnt += 1
        if cnt > 100:
            break
        print ("*"*20)
        print ('low:', enn['sentences'][lo]['raw'])
        print ('high', enn['sentences'][hi]['raw'])
        print ("*"*20)

    return


minorchanges_nli('data/prob_cand_inst_v3', 'data/nli_dist_rl', 'experiment/coco_nli_relation.json')
# show_lo_and_high('data/prob_cand_inst_v3', 'data/nli_weights_v2', 'data/dataset_karparthy.json')
# print (np.array([1.0/5]*5))

