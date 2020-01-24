import torch
from torch.utils.data import DataLoader
import others.util as util
import numpy as np
import random
from data.batch_data import ProdSearchTrainBatch, ProdSearchTestBatch


class ProdSearchDataLoader(DataLoader):
    def __init__(self, args, dataset, prepare_pv=True, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(ProdSearchDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)
        self.args = args
        self.prepare_pv = prepare_pv
        self.shuffle = shuffle
        self.prod_pad_idx = self.dataset.prod_pad_idx
        self.user_pad_idx = self.dataset.user_pad_idx
        self.review_pad_idx = self.dataset.review_pad_idx
        self.word_pad_idx = self.dataset.word_pad_idx
        self.seg_pad_idx = self.dataset.seg_pad_idx
        self.global_data = self.dataset.global_data
        self.prod_data = self.dataset.prod_data
        self.shuffle_review_words = self.dataset.shuffle_review_words
        self.total_review_limit = self.args.uprev_review_limit + self.args.iprev_review_limit
        if self.args.do_subsample_mask:
            self.review_words = self.global_data.review_words
            self.sub_sampling_rate = self.prod_data.sub_sampling_rate
        else:
            self.review_words = self.global_data.padded_review_words
            self.sub_sampling_rate = None
        #if subsampling_rate is 0 then sub_sampling_rate is [1,1,1], all the words are kept

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
        candi_prod_ridxs = []
        candi_seg_idxs = []
        candi_seq_user_idxs = []
        candi_seq_item_idxs = []
        #candi_prod_ridxs = [entry[4] for entry in batch]
        #candi_seg_idxs = [entry[5] for entry in batch]
        for _, user_idx, prod_idx, review_idx, candidate_items in batch:
            #if self.args.train_review_only:
            #    u_prev_review_idxs = self.prod_data.u_reviews[user_idx][:self.args.uprev_review_limit]
            #else:
            do_seq = self.args.do_seq_review_test and not self.args.train_review_only
            u_prev_review_idxs = self.get_user_review_idxs(user_idx, review_idx, do_seq, fix=True)
            i_prev_review_idxs = self.get_item_review_idxs(prod_idx, review_idx, do_seq, fix=True)
            review_time_stamp = None
            if self.args.do_seq_review_test:
                review_time_stamp = self.global_data.review_loc_time[review_idx][2]
            u_item_idxs = [self.global_data.review_u_p[x][1] for x in u_prev_review_idxs]

            candi_batch_item_idxs = []
            candi_batch_user_idxs = []
            candi_batch_seg_idxs = []
            candi_batch_prod_ridxs = []
            for candi_i in candidate_items:
                #if self.args.train_review_only:
                #    candi_i_prev_review_idxs = self.prod_data.p_reviews[candi_i][:self.args.iprev_review_limit]
                #else:
                candi_i_prev_review_idxs = self.get_item_review_idxs(
                        candi_i, None, do_seq, review_time_stamp, fix=True)
                candi_i_user_idxs = [self.global_data.review_u_p[x][0] for x in candi_i_prev_review_idxs]
                cur_candi_i_user_idxs =  [self.user_pad_idx] + [user_idx] * len(u_prev_review_idxs) + candi_i_user_idxs
                cur_candi_i_user_idxs = cur_candi_i_user_idxs[:self.total_review_limit+1]
                cur_candi_i_item_idxs =  [self.prod_pad_idx] + u_item_idxs + [candi_i] * len(candi_i_prev_review_idxs)
                cur_candi_i_item_idxs = cur_candi_i_item_idxs[:self.total_review_limit+1]
                cur_candi_i_masks = [0] + [1] * len(u_prev_review_idxs) + [2] * len(candi_i_prev_review_idxs) #might be 0
                cur_candi_i_masks = cur_candi_i_masks[:self.total_review_limit+1]
                cur_candi_i_review_idxs = u_prev_review_idxs + candi_i_prev_review_idxs
                cur_candi_i_review_idxs = cur_candi_i_review_idxs[:self.total_review_limit]
                candi_batch_seg_idxs.append(cur_candi_i_masks)
                candi_batch_prod_ridxs.append(cur_candi_i_review_idxs)
                candi_batch_item_idxs.append(cur_candi_i_item_idxs)
                candi_batch_user_idxs.append(cur_candi_i_user_idxs)
            candi_prod_ridxs.append(candi_batch_prod_ridxs)
            candi_seg_idxs.append(candi_batch_seg_idxs)
            candi_seq_item_idxs.append(candi_batch_item_idxs)
            candi_seq_user_idxs.append(candi_batch_user_idxs)

        candi_prod_idxs = util.pad(candi_prod_idxs, pad_id = -1) #pad reviews
        candi_prod_ridxs = util.pad_3d(candi_prod_ridxs, pad_id = self.review_pad_idx, dim=1) #pad candi products
        candi_prod_ridxs = util.pad_3d(candi_prod_ridxs, pad_id = self.review_pad_idx, dim=2) #pad reviews of each candi
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=1)
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        candi_seq_user_idxs = util.pad_3d(candi_seq_user_idxs, pad_id = self.user_pad_idx, dim=1)
        candi_seq_user_idxs = util.pad_3d(candi_seq_user_idxs, pad_id = self.user_pad_idx, dim=2)
        candi_seq_item_idxs = util.pad_3d(candi_seq_item_idxs, pad_id = self.prod_pad_idx, dim=1)
        candi_seq_item_idxs = util.pad_3d(candi_seq_item_idxs, pad_id = self.prod_pad_idx, dim=2)

        batch = ProdSearchTestBatch(query_idxs, user_idxs, target_prod_idxs, candi_prod_idxs,
                query_word_idxs, candi_prod_ridxs, candi_seg_idxs,
                candi_seq_user_idxs, candi_seq_item_idxs)
        return batch

    def get_item_review_idxs_prev(self, prod_idx, review_idx, do_seq, review_time_stamp=None,fix=True):
        if do_seq:
            if review_idx is None:
                loc_in_i = self.dataset.bisect_right(
                        self.global_data.i_r_seq[prod_idx], self.global_data.review_loc_time, review_time_stamp)
            else:
                loc_in_i = self.global_data.review_loc_time[review_idx][1]
            if loc_in_i == 0:
                return []
            i_prev_review_idxs = self.global_data.i_r_seq[prod_idx][:loc_in_i]
            i_prev_review_idxs = i_prev_review_idxs[-self.args.iprev_review_limit:]
            #i_prev_review_idxs = self.global_data.i_r_seq[prod_idx][max(0,loc_in_i-self.args.iprev_review_limit):loc_in_i]

        else:
            i_prev_review_idxs = self.prod_data.p_reviews[prod_idx]
            if len(i_prev_review_idxs) > self.args.iprev_review_limit:
                if fix:
                    i_prev_review_idxs = i_prev_review_idxs[:self.args.iprev_review_limit]
                    #i_prev_review_idxs = i_prev_review_idxs[-self.args.iprev_review_limit:]
                else:
                    i_prev_review_idxs = random.sample(i_prev_review_idxs, self.args.iprev_review_limit)

        return i_prev_review_idxs

    def get_item_review_idxs(self, prod_idx, review_idx, do_seq, review_time_stamp=None,fix=True):
        i_seq_review_idxs = self.global_data.i_r_seq[prod_idx]
        i_train_review_set = self.prod_data.p_reviews[prod_idx]
        if do_seq:
            if review_idx is None:
                loc_in_i = self.dataset.bisect_right(
                        self.global_data.i_r_seq[prod_idx], self.global_data.review_loc_time, review_time_stamp)
            else:
                loc_in_i = self.global_data.review_loc_time[review_idx][1]
            if loc_in_i == 0:
                return []
            i_prev_review_idxs = self.global_data.i_r_seq[prod_idx][:loc_in_i]
            i_prev_review_idxs = i_prev_review_idxs[-self.args.iprev_review_limit:]
        else:
            i_seq_train_review_idxs = [x for x in i_seq_review_idxs if x in i_train_review_set and x!= review_idx]
            i_prev_review_idxs = i_seq_train_review_idxs
            if len(i_prev_review_idxs) > self.args.iprev_review_limit:
                if fix:
                    #i_prev_review_idxs = i_prev_review_idxs[:self.args.iprev_review_limit]
                    i_prev_review_idxs = i_prev_review_idxs[-self.args.iprev_review_limit:]
                else:
                    rand_review_set = random.sample(i_seq_train_review_idxs, self.args.iprev_review_limit)
                    rand_review_set = set(rand_review_set)
                    i_prev_review_idxs = [x for x in i_seq_train_review_idxs if x in rand_review_set]

        return i_prev_review_idxs

    def get_user_review_idxs_prev(self, user_idx, review_idx, do_seq, fix=True):
        if do_seq:
            loc_in_u = self.global_data.review_loc_time[review_idx][0]
            u_prev_review_idxs = self.global_data.u_r_seq[user_idx][:loc_in_u]
            u_prev_review_idxs = u_prev_review_idxs[-self.args.uprev_review_limit:]
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

    def get_user_review_idxs(self, user_idx, review_idx, do_seq, fix=True):
        u_seq_review_idxs = self.global_data.u_r_seq[user_idx]
        u_train_review_set = self.prod_data.u_reviews[user_idx] #set
        if do_seq:
            loc_in_u = self.global_data.review_loc_time[review_idx][0]
            u_prev_review_idxs = self.global_data.u_r_seq[user_idx][:loc_in_u]
            u_prev_review_idxs = u_prev_review_idxs[-self.args.uprev_review_limit:]
        else:
            u_seq_train_review_idxs = [x for x in u_seq_review_idxs if x in u_train_review_set and x!= review_idx]
            u_prev_review_idxs = u_seq_train_review_idxs
            if len(u_seq_train_review_idxs) > self.args.uprev_review_limit:
                if fix:
                    u_prev_review_idxs = u_seq_train_review_idxs[-self.args.uprev_review_limit:]
                else:
                    rand_review_set = random.sample(u_seq_train_review_idxs, self.args.uprev_review_limit)
                    rand_review_set = set(rand_review_set)
                    u_prev_review_idxs = [x for x in u_seq_train_review_idxs if x in rand_review_set]
        return u_prev_review_idxs

    def prepare_train_batch(self, batch):
        batch_query_word_idxs = []
        batch_pos_prod_ridxs, batch_pos_seg_idxs, batch_pos_user_idxs, batch_pos_item_idxs = [],[],[],[]
        batch_neg_prod_ridxs, batch_neg_seg_idxs, batch_neg_user_idxs, batch_neg_item_idxs = [],[],[],[]
        for line_id, user_idx, prod_idx, review_idx in batch:
            query_idx = random.choice(self.prod_data.product_query_idx[prod_idx])
            query_word_idxs = self.global_data.query_words[query_idx]
            u_prev_review_idxs = self.get_user_review_idxs(user_idx, review_idx, self.args.do_seq_review_train, fix=False)
            i_prev_review_idxs = self.get_item_review_idxs(prod_idx, review_idx, self.args.do_seq_review_train, fix=False)
            review_time_stamp = None
            if self.args.do_seq_review_train:
                review_time_stamp = self.global_data.review_loc_time[review_idx][2]
            if len(i_prev_review_idxs) == 0:
                continue
            i_user_idxs = [self.global_data.review_u_p[x][0] for x in i_prev_review_idxs]
            u_item_idxs = [self.global_data.review_u_p[x][1] for x in u_prev_review_idxs]
            pos_user_idxs =  [self.user_pad_idx] + [user_idx] * len(u_prev_review_idxs) + i_user_idxs
            pos_user_idxs = pos_user_idxs[:self.total_review_limit + 1]
            pos_item_idxs =  [self.prod_pad_idx] + u_item_idxs + [prod_idx] * len(i_prev_review_idxs)
            pos_item_idxs = pos_item_idxs[:self.total_review_limit + 1]
            pos_seg_idxs = [0] + [1] * len(u_prev_review_idxs) + [2] * len(i_prev_review_idxs)
            pos_seg_idxs = pos_seg_idxs[:self.total_review_limit + 1]
            pos_prod_ridxs = u_prev_review_idxs + i_prev_review_idxs
            pos_prod_ridxs = pos_prod_ridxs[:self.total_review_limit] # or select reviews with the most words

            neg_prod_idxs = self.prod_data.neg_sample_products[line_id] #neg_per_pos
            neg_prod_ridxs = []
            neg_seg_idxs = []
            neg_user_idxs = []
            neg_item_idxs = []
            for neg_i in neg_prod_idxs:
                neg_i_prev_review_idxs = self.get_item_review_idxs(
                        neg_i, None, self.args.do_seq_review_train, review_time_stamp, fix=False)
                if len(neg_i_prev_review_idxs) == 0:
                    continue
                neg_i_user_idxs = [self.global_data.review_u_p[x][0] for x in neg_i_prev_review_idxs]
                cur_neg_i_user_idxs =  [self.user_pad_idx] + [user_idx] * len(u_prev_review_idxs) + neg_i_user_idxs
                cur_neg_i_user_idxs = cur_neg_i_user_idxs[:self.total_review_limit+1]
                cur_neg_i_item_idxs =  [self.prod_pad_idx] + u_item_idxs + [neg_i] * len(neg_i_prev_review_idxs)
                cur_neg_i_item_idxs = cur_neg_i_item_idxs[:self.total_review_limit+1]
                cur_neg_i_masks = [0] + [1] * len(u_prev_review_idxs) + [2] * len(neg_i_prev_review_idxs)
                cur_neg_i_masks = cur_neg_i_masks[:self.total_review_limit+1]
                cur_neg_i_review_idxs = u_prev_review_idxs + neg_i_prev_review_idxs
                cur_neg_i_review_idxs = cur_neg_i_review_idxs[:self.total_review_limit]
                neg_user_idxs.append(cur_neg_i_user_idxs)
                neg_item_idxs.append(cur_neg_i_item_idxs)
                neg_seg_idxs.append(cur_neg_i_masks)
                neg_prod_ridxs.append(cur_neg_i_review_idxs)
                #neg_prod_rword_idxs.append([self.global_data.review_words[x] for x in cur_neg_i_review_idxs])
            if len(neg_prod_ridxs) == 0:
                #all the neg prod do not have available reviews
                continue
            batch_query_word_idxs.append(query_word_idxs)
            batch_pos_prod_ridxs.append(pos_prod_ridxs)
            batch_pos_seg_idxs.append(pos_seg_idxs)
            batch_pos_user_idxs.append(pos_user_idxs)
            batch_pos_item_idxs.append(pos_item_idxs)
            batch_neg_prod_ridxs.append(neg_prod_ridxs)
            batch_neg_seg_idxs.append(neg_seg_idxs)
            batch_neg_user_idxs.append(neg_user_idxs)
            batch_neg_item_idxs.append(neg_item_idxs)

        data_batch = [batch_query_word_idxs, batch_pos_prod_ridxs, batch_pos_seg_idxs,
                batch_neg_prod_ridxs, batch_neg_seg_idxs, batch_pos_user_idxs,
                batch_neg_user_idxs, batch_pos_item_idxs, batch_neg_item_idxs]
        return data_batch
    '''
    u, Q, i (positive, negative)
    Q; ru1,ru2,ri1,ri2 and k negative (ru1,ru2,rn1i1,rn1i2; ru1,ru2,rnji1,rnji2)
    segs 0; 1,1;pos 2,2, -1,-1 neg_1, neg_2
    r: word_id1, word_id2, ...
    pos_seg_idxs:0,1,1,2,2,-1
    word_count can be computed with words that are not padding
    review of u concat with review of i
    review of u concat with review of each negative i
    batch_size, review_count (u+i), max_word_count_per_review
    batch_size, neg_k, review_count (u+i), max_word_count_per_review
    '''
    def get_train_batch(self, batch):
        query_word_idxs, pos_prod_ridxs, pos_seg_idxs, \
                neg_prod_ridxs, neg_seg_idxs, pos_user_idxs, \
                neg_user_idxs, pos_item_idxs, neg_item_idxs = self.prepare_train_batch(batch)
        if len(query_word_idxs) == 0:
            print("0 available instance in the batch")
            return None
        pos_prod_ridxs = util.pad(pos_prod_ridxs, pad_id = self.review_pad_idx) #pad reviews
        pos_seg_idxs = util.pad(pos_seg_idxs, pad_id = self.seg_pad_idx)
        pos_user_idxs = util.pad(pos_user_idxs, pad_id = self.user_pad_idx)
        pos_item_idxs = util.pad(pos_item_idxs, pad_id = self.prod_pad_idx)
        pos_prod_ridxs = np.asarray(pos_prod_ridxs)
        batch_size, pos_rcount = pos_prod_ridxs.shape
        pos_prod_rword_idxs = [self.review_words[x] for x in pos_prod_ridxs.reshape(-1)]
        #pos_prod_rword_idxs = util.pad(pos_prod_rword_idxs, pad_id = self.word_pad_idx)
        pos_prod_rword_idxs = np.asarray(pos_prod_rword_idxs).reshape(batch_size, pos_rcount, -1)
        pos_prod_rword_masks = self.dataset.get_pv_word_masks(
                #pos_prod_rword_idxs, self.prod_data.sub_sampling_rate, pad_id=self.word_pad_idx)
                pos_prod_rword_idxs, self.sub_sampling_rate, pad_id=self.word_pad_idx)
        neg_prod_ridxs = util.pad_3d(neg_prod_ridxs, pad_id = self.review_pad_idx, dim=1) #pad neg products
        neg_prod_ridxs = util.pad_3d(neg_prod_ridxs, pad_id = self.review_pad_idx, dim=2) #pad reviews of each neg
        neg_seg_idxs = util.pad_3d(neg_seg_idxs, pad_id = self.seg_pad_idx, dim=1)
        neg_seg_idxs = util.pad_3d(neg_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        neg_user_idxs = util.pad_3d(neg_user_idxs, pad_id = self.user_pad_idx, dim=1)
        neg_user_idxs = util.pad_3d(neg_user_idxs, pad_id = self.user_pad_idx, dim=2)
        neg_item_idxs = util.pad_3d(neg_item_idxs, pad_id = self.prod_pad_idx, dim=1)
        neg_item_idxs = util.pad_3d(neg_item_idxs, pad_id = self.prod_pad_idx, dim=2)
        neg_prod_ridxs = np.asarray(neg_prod_ridxs)
        _, neg_k, nr_count = neg_prod_ridxs.shape
        neg_prod_rword_idxs = [self.review_words[x] for x in neg_prod_ridxs.reshape(-1)]
        #neg_prod_rword_idxs = util.pad(neg_prod_rword_idxs, pad_id = self.word_pad_idx)
        neg_prod_rword_idxs = np.asarray(neg_prod_rword_idxs).reshape(batch_size, neg_k, nr_count, -1)

        if "pv" in self.dataset.review_encoder_name and self.prepare_pv:
            pos_prod_rword_idxs_pvc = pos_prod_rword_idxs
            neg_prod_rword_idxs_pvc = neg_prod_rword_idxs
            batch_size, pos_rcount, word_limit = pos_prod_rword_idxs.shape
            pv_window_size = self.dataset.pv_window_size
            if self.shuffle_review_words:
                self.dataset.shuffle_words_in_reviews(pos_prod_rword_idxs)
            slide_pos_prod_rword_idxs = self.dataset.slide_padded_matrices_for_pv(
                    pos_prod_rword_idxs.reshape(-1, word_limit),
                    pv_window_size, self.word_pad_idx)
            slide_pos_prod_rword_masks = self.dataset.slide_padded_matrices_for_pv(
                    pos_prod_rword_masks.reshape(-1, word_limit),
                    pv_window_size, pad_id = 0)
            #seg_count, batch_size * pos_rcount, pv_window_size
            seg_count = slide_pos_prod_rword_idxs.shape[0]
            slide_pos_prod_rword_idxs = slide_pos_prod_rword_idxs.reshape(
                    seg_count, batch_size, pos_rcount, pv_window_size).reshape(
                            -1, pos_rcount, pv_window_size) #seg_count, batch_size
            slide_pos_prod_rword_masks = slide_pos_prod_rword_masks.reshape(
                    batch_size, pos_rcount, -1, pv_window_size).reshape(
                            -1, pos_rcount, pv_window_size) #seg_count, batch_size
            batch_indices = np.repeat(np.expand_dims(np.arange(batch_size),0), seg_count, axis=0)
            if self.shuffle:
                I = np.random.permutation(batch_size * seg_count)
                batch_indices = batch_indices.reshape(-1)[I].reshape(seg_count, batch_size)
                slide_pos_prod_rword_idxs = slide_pos_prod_rword_idxs[I]
                slide_pos_prod_rword_masks = slide_pos_prod_rword_masks[I]
            slide_pos_prod_rword_idxs = slide_pos_prod_rword_idxs.reshape(seg_count, batch_size, pos_rcount, -1)
            slide_pos_prod_rword_masks = slide_pos_prod_rword_masks.reshape(seg_count, batch_size, pos_rcount, -1)
            query_word_idxs, pos_prod_ridxs, pos_seg_idxs, neg_prod_ridxs, neg_seg_idxs \
                    = map(np.asarray, [query_word_idxs, pos_prod_ridxs, pos_seg_idxs, neg_prod_ridxs, neg_seg_idxs])
            batch = [ProdSearchTrainBatch(query_word_idxs[batch_indices[i]],
                pos_prod_ridxs[batch_indices[i]], pos_seg_idxs[batch_indices[i]],
                slide_pos_prod_rword_idxs[i], slide_pos_prod_rword_masks[i],
                neg_prod_ridxs[batch_indices[i]], neg_seg_idxs[batch_indices[i]],
                pos_user_idxs[batch_indices[i]], neg_user_idxs[batch_indices[i]],
                pos_item_idxs[batch_indices[i]], neg_item_idxs[batch_indices[i]],
                pos_prod_rword_idxs_pvc = pos_prod_rword_idxs_pvc[batch_indices[i]],
                neg_prod_rword_idxs_pvc = neg_prod_rword_idxs_pvc[batch_indices[i]]) for i in range(seg_count)]
        else:
            neg_prod_rword_masks = self.dataset.get_pv_word_masks(
                    #neg_prod_rword_idxs, self.prod_data.sub_sampling_rate, pad_id=self.word_pad_idx)
                    neg_prod_rword_idxs, self.sub_sampling_rate, pad_id=self.word_pad_idx)
            batch = ProdSearchTrainBatch(query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                    pos_prod_rword_idxs, pos_prod_rword_masks,
                    neg_prod_ridxs, neg_seg_idxs,
                    pos_user_idxs, neg_user_idxs,
                    pos_item_idxs, neg_item_idxs,
                    neg_prod_rword_idxs = neg_prod_rword_idxs,
                    neg_prod_rword_masks = neg_prod_rword_masks)
        return batch

