import torch
from torch.utils.data import DataLoader
import others.util as util
import numpy as np
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
        self.review_pad_idx = self.dataset.review_pad_idx
        self.word_pad_idx = self.dataset.word_pad_idx
        self.seg_pad_idx = self.dataset.seg_pad_idx
        self.global_data = self.dataset.global_data
        self.prod_data = self.dataset.prod_data
        self.shuffle_review_words = self.dataset.shuffle_review_words
        self.total_review_limit = self.args.uprev_review_limit + self.args.iprev_review_limit


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
        #candi_prod_ridxs = [entry[4] for entry in batch]
        #candi_seg_idxs = [entry[5] for entry in batch]
        for _, user_idx, _, review_idx, candidate_items in batch:
            if self.args.train_review_only:
                u_prev_review_idxs = self.prod_data.u_reviews[user_idx][:self.args.uprev_review_limit]
            else:
                review_time_stamp = self.global_data.review_loc_time[review_idx][2]
                loc_in_u = self.global_data.review_loc_time[review_idx][0]
                u_prev_review_idxs = self.global_data.u_r_seq[user_idx][max(0,loc_in_u-self.args.uprev_review_limit):loc_in_u]

            candi_batch_seg_idxs = []
            candi_batch_prod_ridxs = []
            for candi_i in candidate_items:
                if self.args.train_review_only:
                    candi_i_prev_review_idxs = self.prod_data.p_reviews[candi_i][:self.args.iprev_review_limit]
                else:
                    loc_in_candi_i = self.bisect_right(
                            self.global_data.i_r_seq[candi_i], self.global_data.review_loc_time, review_time_stamp)
                    candi_i_prev_review_idxs = self.global_data.i_r_seq[candi_i][max(0,loc_in_candi_i-self.args.iprev_review_limit):loc_in_candi_i]
                cur_candi_i_masks = [0] + [1] * len(u_prev_review_idxs) + [2] * len(candi_i_prev_review_idxs) #might be 0
                cur_candi_i_masks = cur_candi_i_masks[:self.total_review_limit+1]
                cur_candi_i_review_idxs = u_prev_review_idxs + candi_i_prev_review_idxs
                cur_candi_i_review_idxs = cur_candi_i_review_idxs[:self.total_review_limit]
                candi_batch_seg_idxs.append(cur_candi_i_masks)
                candi_batch_prod_ridxs.append(cur_candi_i_review_idxs)
            candi_prod_ridxs.append(candi_batch_prod_ridxs)
            candi_seg_idxs.append(candi_batch_seg_idxs)

        candi_prod_idxs = util.pad(candi_prod_idxs, pad_id = -1) #pad reviews
        candi_prod_ridxs = util.pad_3d(candi_prod_ridxs, pad_id = self.review_pad_idx, dim=1) #pad candi products
        candi_prod_ridxs = util.pad_3d(candi_prod_ridxs, pad_id = self.review_pad_idx, dim=2) #pad reviews of each candi
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=1)
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        candi_prod_ridxs = np.asarray(candi_prod_ridxs)
        batch_size, candi_k, nr_count = candi_prod_ridxs.shape

        batch = ProdSearchTestBatch(query_idxs, user_idxs, target_prod_idxs, candi_prod_idxs,
                query_word_idxs, candi_prod_ridxs, candi_seg_idxs)
        return batch

    def get_test_batch_memcost(self, batch):
        query_idxs = [entry[0] for entry in batch]
        query_word_idxs = [self.global_data.query_words[x] for x in query_idxs]
        user_idxs = [entry[1] for entry in batch]
        target_prod_idxs = [entry[2] for entry in batch]
        candi_prod_idxs = [entry[3] for entry in batch]
        candi_prod_ridxs = [entry[4] for entry in batch]
        candi_seg_idxs = [entry[5] for entry in batch]
        candi_prod_idxs = util.pad(candi_prod_idxs, pad_id = -1) #pad reviews
        candi_prod_ridxs = util.pad_3d(candi_prod_ridxs, pad_id = self.review_pad_idx, dim=1) #pad candi products
        candi_prod_ridxs = util.pad_3d(candi_prod_ridxs, pad_id = self.review_pad_idx, dim=2) #pad reviews of each candi
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=1)
        candi_seg_idxs = util.pad_3d(candi_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        candi_prod_ridxs = np.asarray(candi_prod_ridxs)
        batch_size, candi_k, nr_count = candi_prod_ridxs.shape
        #candi_prod_rword_idxs = [self.global_data.review_words[x] for x in candi_prod_ridxs.reshape(-1)]
        #candi_prod_rword_idxs = util.pad(candi_prod_rword_idxs, pad_id = self.word_pad_idx)
        #candi_prod_rword_idxs = np.asarray(candi_prod_rword_idxs).reshape(batch_size, candi_k, nr_count, -1)

        batch = ProdSearchTestBatch(query_idxs, user_idxs, target_prod_idxs, candi_prod_idxs,
                query_word_idxs, candi_prod_ridxs, candi_seg_idxs)
        return batch

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
        query_idxs = [entry[0] for entry in batch]
        query_word_idxs = [self.global_data.query_words[x] for x in query_idxs]
        pos_prod_ridxs = [entry[1] for entry in batch]
        pos_seg_idxs = [entry[2] for entry in batch]
        #batch_size, review_count, word_limit
        neg_prod_ridxs = [entry[3] for entry in batch]
        #batch_size, neg_k, review_count
        neg_seg_idxs = [entry[4] for entry in batch]
        pos_prod_ridxs = util.pad(pos_prod_ridxs, pad_id = self.review_pad_idx) #pad reviews
        pos_seg_idxs = util.pad(pos_seg_idxs, pad_id = self.seg_pad_idx)
        pos_prod_ridxs = np.asarray(pos_prod_ridxs)
        batch_size, pos_rcount = pos_prod_ridxs.shape
        pos_prod_rword_idxs = [self.global_data.review_words[x] for x in pos_prod_ridxs.reshape(-1)]
        #pos_prod_rword_idxs = util.pad(pos_prod_rword_idxs, pad_id = self.word_pad_idx)
        pos_prod_rword_idxs = np.asarray(pos_prod_rword_idxs).reshape(batch_size, pos_rcount, -1)
        pos_prod_rword_masks = self.dataset.get_pv_word_masks(
                pos_prod_rword_idxs, self.prod_data.sub_sampling_rate, pad_id=self.word_pad_idx)
        neg_prod_ridxs = util.pad_3d(neg_prod_ridxs, pad_id = self.review_pad_idx, dim=1) #pad neg products
        neg_prod_ridxs = util.pad_3d(neg_prod_ridxs, pad_id = self.review_pad_idx, dim=2) #pad reviews of each neg
        neg_seg_idxs = util.pad_3d(neg_seg_idxs, pad_id = self.seg_pad_idx, dim=1)
        neg_seg_idxs = util.pad_3d(neg_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        neg_prod_ridxs = np.asarray(neg_prod_ridxs)
        _, neg_k, nr_count = neg_prod_ridxs.shape
        neg_prod_rword_idxs = [self.global_data.review_words[x] for x in neg_prod_ridxs.reshape(-1)]
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
                pos_prod_rword_idxs_pvc = pos_prod_rword_idxs_pvc[batch_indices[i]],
                neg_prod_rword_idxs_pvc = neg_prod_rword_idxs_pvc[batch_indices[i]]) for i in range(seg_count)]
        else:
            neg_prod_rword_masks = self.dataset.get_pv_word_masks(
                    neg_prod_rword_idxs, self.prod_data.sub_sampling_rate, pad_id=self.word_pad_idx)
            batch = [ProdSearchTrainBatch(query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                    pos_prod_rword_idxs, pos_prod_rword_masks,
                    neg_prod_ridxs, neg_seg_idxs,
                    neg_prod_rword_idxs = neg_prod_rword_idxs,
                    neg_prod_rword_masks = neg_prod_rword_masks)]
        return batch

