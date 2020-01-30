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

class ItemPVDataset(Dataset):
    def __init__(self, args, global_data, prod_data):
        self.args = args
        self.valid_candi_size = args.valid_candi_size
        self.prod_pad_idx = global_data.product_size
        self.word_pad_idx = global_data.vocab_size - 1
        self.seg_pad_idx = 3 # 0, 1, 2
        self.pv_window_size = args.pv_window_size
        self.train_review_only = args.train_review_only
        self.uprev_review_limit = args.uprev_review_limit
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
            #query_idx = prod_data.review_query_idx[line_id]
            query_idxs = prod_data.product_query_idx[prod_idx]
            for query_idx in query_idxs:
                if (user_idx, query_idx) in uq_set:
                    continue
                uq_set.add((user_idx, query_idx))

                #candidate item list according to user_idx and query_idx, or by default all the items
                if prod_data.uq_pids is None:
                    if self.prod_data.set_name == "valid" and self.valid_candi_size > 1:
                        candidate_items = np.random.choice(global_data.product_size,
                                size=self.valid_candi_size-1, replace=False, p=prod_data.product_dists).tolist()
                        candidate_items.append(prod_idx)
                        random.shuffle(candidate_items)
                    else:
                        candidate_items = list(range(global_data.product_size))
                else:
                    candidate_items = prod_data.uq_pids[(global_data.user_ids[user_idx], query_idx)]
                    random.shuffle(candidate_items)
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
        train_data = []
        rand_numbers = np.random.random(sum(global_data.review_length))
        entry_id = 0
        word_idxs = []
        for line_no, user_idx, prod_idx, review_idx in prod_data.review_info:
            cur_review_word_idxs = self.global_data.review_words[review_idx]
            random.shuffle(cur_review_word_idxs)
            for word_idx in cur_review_word_idxs:
                if rand_numbers[entry_id] > prod_data.sub_sampling_rate[word_idx]:
                    continue
                word_idxs.append(word_idx)
                if len(word_idxs) == self.pv_window_size:
                    train_data.append([word_idxs, review_idx])
                    word_idxs = []
                entry_id += 1
        if len(word_idxs) > 0:
            train_data.append([word_idxs+[self.word_pad_idx]*(self.pv_window_size-len(word_idxs)), review_idx])
        return train_data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
