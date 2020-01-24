import torch
from torch.utils.data import Dataset
import numpy as np
import random
import others.util as util

from collections import defaultdict

""" load training, validation and test data
u,Q,i for a purchase i given u, Q
        negative samples i- for u, Q
read reviews of u, Q before i (previous m reviews)
reviews of others before r_i (review of i from u)
load test data
for each review, collect random words in reviews or words in a sliding window in reviews.
                 or all the words in the review
"""

class ProdSearchDataset(Dataset):
    def __init__(self, args, global_data, prod_data):
        self.args = args
        self.valid_candi_size = args.valid_candi_size
        self.user_pad_idx = global_data.user_size
        self.prod_pad_idx = global_data.product_size
        self.word_pad_idx = global_data.vocab_size - 1
        self.review_pad_idx = global_data.review_count - 1
        self.seg_pad_idx = 3 # 0, 1, 2
        self.shuffle_review_words = args.shuffle_review_words
        self.review_encoder_name = args.review_encoder_name
        self.pv_window_size = args.pv_window_size
        self.corrupt_rate = args.corrupt_rate
        self.train_review_only = args.train_review_only
        self.uprev_review_limit = args.uprev_review_limit
        self.iprev_review_limit = args.iprev_review_limit #can be a really large number, can random select
        self.total_review_limit = self.uprev_review_limit + self.iprev_review_limit
        self.global_data = global_data
        self.prod_data = prod_data
        if prod_data.set_name == "train":
            self._data = self.collect_train_samples(self.global_data, self.prod_data)
        else:
            self._data = self.collect_test_samples(self.global_data, self.prod_data, args.candi_batch_size)

    def collect_test_samples(self, global_data, prod_data, candi_batch_size=1000):
        #Q, review of u + review of pos i, review of u + review of neg i;
        #words of pos reviews; words of neg reviews, all if encoder is not pv
        test_data = []
        uq_set = set()
        for line_id, user_idx, prod_idx, review_idx in prod_data.review_info:
            if (line_id+1) % 10000 == 0:
                progress = (line_id+1.) / len(prod_data.review_info) * 100
                print("{}% data processed".format(progress))
            #user_idx, prod_idx, review_idx = review
            #query_idx = prod_data.review_query_idx[line_id]
            query_idxs = prod_data.product_query_idx[prod_idx]
            for query_idx in query_idxs:
                if (user_idx, query_idx) in uq_set:
                    continue
                uq_set.add((user_idx, query_idx))

                #candidate item list according to user_idx and query_idx, or by default all the items
                #candidate_items = list(range(global_data.product_size))[:1000]
                if prod_data.uq_pids is None:
                    if self.prod_data.set_name == "valid" and self.valid_candi_size > 1:
                        candidate_items = np.random.choice(global_data.product_size,
                                size=self.valid_candi_size-1, replace=False, p=prod_data.product_dists).tolist()
                        #candidate_items = np.random.randint(0, global_data.product_size, size =self.valid_candi_size-1).tolist()
                        candidate_items.append(prod_idx)
                        random.shuffle(candidate_items)
                    else:
                        candidate_items = list(range(global_data.product_size))
                else:
                    #print(global_data.user_ids[user_idx], query_idx)
                    candidate_items = prod_data.uq_pids[(global_data.user_ids[user_idx], query_idx)]
                    random.shuffle(candidate_items)
                    #candidate_items = [global_data.product_asin2ids[x] for x in asin_list]
                    #print(len(candidate_items))
                seg_count = int((len(candidate_items) - 1) / candi_batch_size) + 1
                for i in range(seg_count):
                    test_data.append([query_idx, user_idx, prod_idx, review_idx,
                        candidate_items[i*candi_batch_size:(i+1)*candi_batch_size]])
        print(len(uq_set))

        return test_data


    def collect_train_samples(self, global_data, prod_data):
        #Q, review of u + review of pos i, review of u + review of neg i;
        #words of pos reviews; words of neg reviews, all if encoder is not pv
        return prod_data.review_info

    def get_pv_word_masks(self, prod_rword_idxs, subsampling_rate, pad_id):
        if subsampling_rate is not None:
            rand_numbers = np.random.random(prod_rword_idxs.shape)
            #subsampling_rate_arr = np.asarray([[subsampling_rate[prod_rword_idxs[i][j]] \
            #        for j in range(prod_rword_idxs.shape[1])] for i in range(prod_rword_idxs.shape[0])])
            subsampling_rate_arr = subsampling_rate[prod_rword_idxs]
            masks = np.logical_and(prod_rword_idxs !=pad_id, rand_numbers < subsampling_rate_arr)
        else:
            masks = (prod_rword_idxs !=pad_id)
        return masks

    def shuffle_words_in_reviews(self, prod_rword_idxs):
        #consider random shuffle words
        for row in prod_rword_idxs:
            np.random.shuffle(row)

    def slide_matrices_for_pv(self, prod_rword_idxs, pv_window_size):
        #review_count * review_word_limit
        seg_prod_rword_idxs = []
        cur_length = 0
        while cur_length < prod_rword_idxs.shape[1]: # review_word_limit
            seg_prod_rword_idxs.append(prod_rword_idxs[:,cur_length:cur_length+pv_window_size]) #).tolist())
            cur_length += pv_window_size
        return seg_prod_rword_idxs

    def slide_padded_matrices_for_pv(self, prod_rword_idxs, pv_window_size, pad_id):
        '''
        word_limit = prod_rword_idxs.shape[1]
        seg_count = word_limit / pv_window_size
        mod = word_limit % pv_window_size
        if mod > 0:
            seg_count += 1
        new_length = pv_window_size * seg_count
        prod_rword_idxs = util.pad_3d(
                prod_rword_idxs.tolist(), pad_id=pad_id, dim=2, width=new_length) #pad words
        #seg_count = (prod_rword_idxs.shape[1]-1)/pv_window_size + 1
        '''
        pad_size = pv_window_size - (prod_rword_idxs.shape[1] % pv_window_size)
        if pad_size < pv_window_size:
            prod_rword_idxs = np.pad(prod_rword_idxs, ((0,0),(0,pad_size)),mode='constant', constant_values=pad_id)

        seg_count = int(prod_rword_idxs.shape[1]/pv_window_size)
        return np.asarray([prod_rword_idxs[:,i*pv_window_size:(i+1)*pv_window_size] for i in range(seg_count)])

    def bisect_right(self, review_arr, review_loc_time_arr, timestamp, lo=0, hi=None):
        """Return the index where timestamp is larger than the review in review_arr (sorted)
        The return value i is such that all e in a[:i] have e <= x, and all e in
        a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
        insert just after the rightmost x already there.
        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """

        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(review_arr)
        while lo < hi:
            mid = (lo+hi)//2
            if timestamp < review_loc_time_arr[review_arr[mid]][2]: hi = mid
            else: lo = mid+1
        return lo

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
