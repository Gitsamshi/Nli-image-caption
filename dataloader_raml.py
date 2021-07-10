from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
# from models.ass_fun import *

import torch
import torch.utils.data as data
import json

import multiprocessing


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        # self.sentence_path = 'data/sents_dist'
        self.sentence_path = opt.sample_dist
        self.varied_gts = opt.varied_gts
        self.guess_gts = opt.guess_gts
        self.mixed_gts = opt.mixed_gts
        self.gap_gts = opt.gap_gts
        self.restrained_gts = opt.restrained_gts
        self.soft_neg_gts = opt.soft
        self.hard_neg_gts = opt.hard
        if self.soft_neg_gts is not None:
            self.sneg_sent = json.load(open(self.soft_neg_gts, 'r'))
        if self.hard_neg_gts is not None:
            self.hneg_sent = json.load(open(self.hard_neg_gts, 'r'))

        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        # whether use relationship info or not
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        if self.sentence_path is None:
            self.sent_dist = None
        else:
            self.sent_dist = json.load(open(self.sentence_path, 'r'))
        # self.ix_pair, self.ix_id_pair = json.load(open(self.opt.dict_json))
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir,
              opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        print("seq_size:{0}".format(seq_size))
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        self.label = self.h5_label_file['labels'][:]
        # ----------retrieved---------
        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        # the ixs are here #
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
                # self.split_ix['train'].append(ix) ## wired setting
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))
        # print('assigned %d images to split experiment' % len(self.split_ix['experiment']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        gts_idx = list(range(self.label_start_ix[ix] - 1, self.label_end_ix[ix]))

        prob_list = [float(ele[1]) for ele in self.sent_dist[ix] if int(ele[0]) in gts_idx]
        idx_list = [int(ele[0]) for ele in self.sent_dist[ix] if int(ele[0]) in gts_idx]

        normalized_prob = np.array(prob_list)
        select_list = np.random.choice(idx_list, seq_per_img, p=normalized_prob)
        seq = np.zeros([seq_per_img, self.seq_length], dtype='int')

        for q in range(seq_per_img):
            seq[q, :] = self.h5_label_file['labels'][select_list[q % select_list.shape[0]], :self.seq_length]
        return seq

    def get_captions2(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        # print ('ix', ix1, ix2)
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        # newly added: data['protos'], data['proto_masks']
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = []  # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')

        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')

        wrapped = False

        infos = []
        gts = []
        gts_p = []
        gts_g = []
        sneg_gts = []
        hneg_gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, \
            ix, tmp_wrapped = self._prefetch_process[split].get()

            # -----------------------------------just for log loss use --------------
            ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            # print ('ix', ix1, ix2)
            ncap = ix2 - ix1 + 1  # number of captions available for this image
            # if ncap != seq_per_img:
            #     continue
            # ------------------------------------------------------------------------

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            # seq = self.get_captions(ix, seq_per_img)
            if self.sentence_path is None:
                seq = self.get_captions2(ix, seq_per_img)
            else:
                seq = self.get_captions(ix, seq_per_img)
            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1] = seq

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            if self.mixed_gts:
                assert self.sentence_path is not None
                prob_list = [float(ele[1]) for ele in self.sent_dist[ix]]
                idx_list = [int(ele[0]) for ele in self.sent_dist[ix]]

                gts_idx = list(range(self.label_start_ix[ix] - 1, self.label_end_ix[ix]))
                tmp_p = np.array([prob for prob, idx in zip(prob_list, idx_list) if idx in gts_idx])
                gts_p.append(tmp_p)
                gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
                num = self.label_end_ix[ix] - self.label_start_ix[ix] + 1
                gts_p.append(np.array([1.0/num]*num))

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # print('fc_batch_0', len(fc_batch))
        fc_batch, att_batch, label_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: 0,
                        reverse=True))
        # print ('fc_batch', len(fc_batch))

        data = {}
        data['fc_feats'] = np.stack(sum([[_]*seq_per_img for _ in fc_batch], []))
        max_att_len = max([_.shape[0] for _ in att_batch])

        # merge att_feats
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        if self.restrained_gts:
            data['gts'] = [gts, gts_p, gts_g]
        else:
            data['gts'] = (gts, gts_p) if len(gts_p) != 0 else gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos
        # print ('soft_neg_gts', self.soft_neg_gts)

        data['sneg_gts'] = sneg_gts
        data['hneg_gts'] = hneg_gts

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in data.items()}

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = np.load(os.path.join(self.input_box_dir, str(self.info['images'][ix]['id']) + '.npy'))
                # devided by image width and height
                x1, y1, x2, y2 = np.hsplit(box_feat, 4)
                h, w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack(
                    (x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x: x[-1], reverse=True))
        else:
            att_feat = np.zeros((1, 1, 1))
        return (np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy')),
                att_feat,
                ix)

    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=4,  # 4 is usually enough
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]
