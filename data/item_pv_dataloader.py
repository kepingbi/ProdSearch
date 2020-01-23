import torch
from torch.utils.data import DataLoader
import others.util as util
import numpy as np
import random
from data.batch_data import ProdSearchTrainBatch, ProdSearchTestBatch, ItemPVBatch


class ItemPVDataloader(DataLoader):
    def __init__(self, args, dataset, prepare_pv=True, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(ItemPVDataloader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)
        self.args = args
        self.prod_pad_idx = self.dataset.prod_pad_idx
        self.word_pad_idx = self.dataset.word_pad_idx
        self.seg_pad_idx = self.dataset.seg_pad_idx
        self.global_data = self.dataset.global_data
        self.prod_data = self.dataset.prod_data

    def _collate_fn(self, batch):
        if self.prod_data.set_name == 'train':
            return self.get_train_batch(batch)
        else: #validation or test
            return self.get_test_batch(batch)

    def get_test_batch(self, batch):
        query_idxs = [entry[0] for entry in batch]
        query_word_idxs = [self.global_data.query_words[x] for x in query_idxs]
        user_idxs = [entry[1] for entry in batch]
        target_prod_idxs = [entry[2] for entry in batch]
        candi_prod_idxs = [entry[4] for entry in batch]
        candi_u_item_idxs = []
        for _, user_idx, prod_idx, review_idx, candidate_items in batch:
            do_seq = self.args.do_seq_review_test and not self.args.train_review_only
            u_prev_review_idxs = self.get_user_review_idxs(user_idx, review_idx, do_seq, fix=True)
            u_item_idxs = [self.global_data.review_u_p[x][1] for x in u_prev_review_idxs]
            candi_u_item_idxs.append(u_item_idxs)

        candi_prod_idxs = util.pad(candi_prod_idxs, pad_id = self.prod_pad_idx)
        candi_u_item_idxs = util.pad(candi_u_item_idxs, pad_id = self.prod_pad_idx)

        batch = ItemPVBatch(query_word_idxs, target_prod_idxs, candi_u_item_idxs,
                query_idxs=query_idxs, user_idxs=user_idxs, candi_prod_idxs=candi_prod_idxs)
        return batch

    def get_test_batch_seq(self, batch):
        query_idxs = [entry[0] for entry in batch]
        query_word_idxs = [self.global_data.query_words[x] for x in query_idxs]
        user_idxs = [entry[1] for entry in batch]
        target_prod_idxs = [entry[2] for entry in batch]
        candi_prod_idxs = [entry[4] for entry in batch]
        candi_seg_idxs = []
        candi_seq_item_idxs = []
        for _, user_idx, prod_idx, review_idx, candidate_items in batch:
            do_seq = self.args.do_seq_review_test and not self.args.train_review_only
            u_prev_review_idxs = self.get_user_review_idxs(user_idx, review_idx, do_seq, fix=True)
            u_item_idxs = [self.global_data.review_u_p[x][1] for x in u_prev_review_idxs]

            candi_batch_item_idxs = []
            candi_batch_seg_idxs = []
            for candi_i in candidate_items:
                cur_candi_i_item_idxs = u_item_idxs + [candi_i]
                cur_candi_i_masks = [0] + [1] * len(u_prev_review_idxs) + [2]
                candi_batch_seg_idxs.append(cur_candi_i_masks)
                candi_batch_item_idxs.append(cur_candi_i_item_idxs)
            candi_seg_idxs.append(candi_batch_seg_idxs)
            candi_seq_item_idxs.append(candi_batch_item_idxs)

        candi_prod_idxs = util.pad(candi_prod_idxs, pad_id = -1)
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=1)
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        candi_seq_item_idxs = util.pad_3d(candi_seq_item_idxs, pad_id = self.prod_pad_idx, dim=1)
        candi_seq_item_idxs = util.pad_3d(candi_seq_item_idxs, pad_id = self.prod_pad_idx, dim=2)

        batch = ItemPVBatch(query_word_idxs, target_prod_idxs, candi_prod_idxs, None,
                candi_seg_idxs, None, candi_seq_item_idxs,
                query_idxs=query_idxs, user_idxs=user_idxs)
        return batch

    def get_user_review_idxs(self, user_idx, review_idx, do_seq, fix=True):
        if do_seq:
            loc_in_u = self.global_data.review_loc_time[review_idx][0]
            u_prev_review_idxs = self.global_data.u_r_seq[user_idx][:loc_in_u]
            u_prev_review_idxs = self.global_data.u_r_seq[user_idx][-self.args.uprev_review_limit:]
            #u_prev_review_idxs = self.global_data.u_r_seq[user_idx][max(0,loc_in_u-self.uprev_review_limit):loc_in_u]
        else:
            u_prev_review_idxs = self.prod_data.u_reviews[user_idx]
            if len(u_prev_review_idxs) > self.args.uprev_review_limit:
                if fix:
                    u_prev_review_idxs = u_prev_review_idxs[:self.args.uprev_review_limit]
                    #u_prev_review_idxs = u_prev_review_idxs[-self.args.uprev_review_limit:]
                else:
                    u_prev_review_idxs = random.sample(u_prev_review_idxs, self.args.uprev_review_limit)
        return u_prev_review_idxs

    def get_train_batch(self, batch):
        batch_query_word_idxs, batch_word_idxs = [],[]
        batch_u_item_idxs, batch_target_prod_idxs = [],[]
        #batch_neg_prod_idxs = np.random.choice(
        #        self.prod_data.product_size,
        #        size=(len(batch), self.args.neg_per_pos), p=self.prod_data.product_dists)
        cur_no = 0
        for word_idxs, review_idx in batch:
            batch_word_idxs.append(word_idxs)
            user_idx, prod_idx = self.global_data.review_u_p[review_idx]
            query_idx = random.choice(self.prod_data.product_query_idx[prod_idx])
            query_word_idxs = self.global_data.query_words[query_idx]

            u_prev_review_idxs = self.get_user_review_idxs(user_idx, review_idx, self.args.do_seq_review_train, fix=False)
            u_item_idxs = [self.global_data.review_u_p[x][1] for x in u_prev_review_idxs]
            batch_query_word_idxs.append(query_word_idxs)
            batch_target_prod_idxs.append(prod_idx)
            batch_u_item_idxs.append(u_item_idxs)

        batch_u_item_idxs = util.pad(batch_u_item_idxs, pad_id = self.prod_pad_idx)
        batch = ItemPVBatch(batch_query_word_idxs, batch_target_prod_idxs, batch_u_item_idxs, batch_word_idxs)
        return batch

    def prepare_train_batch_pad_ui_seq(self, batch):
        batch_query_word_idxs, batch_word_idxs = [],[]
        batch_pos_seg_idxs, batch_pos_item_idxs = [],[]
        batch_neg_seg_idxs, batch_neg_item_idxs = [],[]
        batch_neg_prod_idxs = np.random.choice(
                self.prod_data.product_size,
                size=(len(batch), self.args.neg_per_pos), p=self.prod_data.product_dists)
        cur_no = 0
        for word_idxs, review_idx in batch:
            batch_word_idxs.append(word_idxs)
            user_idx, prod_idx = self.global_data.review_u_p[review_idx]
            query_idx = random.choice(self.prod_data.product_query_idx[prod_idx])
            query_word_idxs = self.global_data.query_words[query_idx]

            u_prev_review_idxs = self.get_user_review_idxs(user_idx, review_idx, self.args.do_seq_review_train, fix=False)
            u_item_idxs = [self.global_data.review_u_p[x][1] for x in u_prev_review_idxs]
            pos_seq_item_idxs =  u_item_idxs + [prod_idx]
            pos_seg_idxs = [0] + [1] * len(u_prev_review_idxs) + [2]
            neg_seg_idxs = []
            neg_seq_item_idxs = []
            for neg_i in batch_neg_prod_idxs[cur_no]:
                cur_neg_i_item_idxs = u_item_idxs + [neg_i]
                cur_neg_i_masks = [0] + [1] * len(u_prev_review_idxs) + [2]
                neg_seq_item_idxs.append(cur_neg_i_item_idxs)
                neg_seg_idxs.append(cur_neg_i_masks)
            batch_query_word_idxs.append(query_word_idxs)
            batch_pos_seg_idxs.append(pos_seg_idxs)
            batch_pos_item_idxs.append(pos_seq_item_idxs)
            batch_neg_seg_idxs.append(neg_seg_idxs)
            batch_neg_item_idxs.append(neg_seq_item_idxs)

        data_batch = [batch_query_word_idxs, batch_word_idxs, batch_pos_seg_idxs,
                batch_neg_seg_idxs, batch_pos_item_idxs, batch_neg_item_idxs]
        return data_batch

    def get_train_batch_ui_seq(self, batch):
        query_word_idxs, pos_iword_idxs, pos_seg_idxs, neg_seg_idxs, \
                pos_seq_item_idxs, neg_seq_item_idxs = self.prepare_train_batch(batch)
        target_prod_idxs = [x[-1] for x in pos_seq_item_idxs]
        pos_seg_idxs = util.pad(pos_seg_idxs, pad_id = self.seg_pad_idx)
        pos_seq_item_idxs = util.pad(pos_seq_item_idxs, pad_id = self.prod_pad_idx)
        batch_size, prev_item_count = np.asarray(pos_seq_item_idxs).shape
        #batch, neg_k, item_count
        neg_seg_idxs = util.pad_3d(neg_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        neg_seq_item_idxs = util.pad_3d(neg_seq_item_idxs, pad_id = self.prod_pad_idx, dim=2)

        batch = ItemPVBatch(query_word_idxs, target_prod_idxs, [], pos_seg_idxs,
                neg_seg_idxs, pos_seq_item_idxs, neg_seq_item_idxs, pos_iword_idxs=pos_iword_idxs)
        return batch

