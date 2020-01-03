import torch
import numpy as np

from collections import defaultdict
import others.util as util
import gzip

class ProdSearchData():
    def __init__(self, args, vocab_size, input_train_dir, set_name, product_size=None):
        self.args = args
        self.neg_per_pos = args.neg_per_pos
        self.set_name = set_name
        self.product_size = product_size
        self.vocab_size = vocab_size
        self.vocab_distribute, self.review_info = self.read_reviews("{}/{}.txt.gz".format(input_train_dir, set_name))
        self.vocab_distribute = self.vocab_distribute.tolist()
        self.sub_sampling_rate = None
        self.neg_sample_products = None
        self.word_dists = None

        if set_name == "train":
            self.sub_sampling(args.subsampling_rate)
            self.word_dists = self.neg_distributes(self.vocab_distribute)
            self.train_review_size = len(self.review_info)

        #self.product_query_idx = GlobalProdSearchData.read_arr_from_lines("{}/{}_query_idx.txt.gz".format(input_train_dir, set_name))
        self.review_ids, self.review_query_idx = self.read_review_id("{}/{}_id.txt.gz".format(input_train_dir, set_name))

    def initialize_epoch(self):
        self.neg_sample_products = np.random.randint(0, self.product_size, size = (self.train_review_size, self.neg_per_pos))

    def read_review_id(self, fname):
        review_ids = []
        query_ids = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                review_ids.append(int(arr[-2].split('_')[-1]))
                query_ids.append(int(arr[-1]))
        return review_ids, query_ids

    def read_reviews(self, fname):
        vocab_distribute = np.zeros(self.vocab_size)
        review_info = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
                review_text = [int(i) for i in arr[2].split(' ')]
                for idx in review_text:
                    vocab_distribute[idx] += 1
        return vocab_distribute, review_info
    def sub_sampling(self, subsample_threshold):
        self.sub_sampling_rate = [1.0 for _ in range(self.vocab_size)]
        if subsample_threshold == 0.0:
            return
        threshold = sum(self.vocab_distribute) * subsample_threshold
        for i in range(self.vocab_size):
            #vocab_distribute[i] could be zero if the word does not appear in the training set
            if self.vocab_distribute[i] == 0:
                self.sub_sampling_rate[i] = 0
                #if this word does not appear in training set, set the rate to 0.
                continue
            self.sub_sampling_rate[i] = min(1.0, (np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]))

        self.sample_count = sum([self.sub_sampling_rate[i] * self.vocab_distribute[i] for i in range(self.vocab_size)])
        print("sample_count", self.sample_count)

    def neg_distributes(self, weights, distortion = 0.75):
        #print weights
        weights = np.asarray(weights)
        #print weights.sum()
        wf = weights / weights.sum()
        wf = np.power(wf, distortion)
        wf = wf / wf.sum()
        return wf


class GlobalProdSearchData():
    def __init__(self, args, data_path, input_train_dir):
        self.review_word_limit = args.review_word_limit

        self.product_ids = self.read_lines("{}/product.txt.gz".format(data_path))
        self.product_size = len(self.product_ids)
        self.user_ids = self.read_lines("{}/users.txt.gz".format(data_path))
        self.user_size = len(self.user_ids)
        self.words = self.read_lines("{}/vocab.txt.gz".format(data_path))
        self.vocab_size = len(self.words) + 1
        self.query_words = self.read_words_in_lines("{}/query.txt.gz".format(input_train_dir))
        self.query_words = util.pad(self.query_words, pad_id=self.vocab_size-1)

        self.review_words = self.read_words_in_lines("{}/review_text.txt.gz".format(data_path))
        self.review_length = [len(x) for x in self.review_words]
        self.review_words = util.pad(self.review_words, pad_id=self.vocab_size-1, width=args.review_word_limit)
        self.review_count = len(self.review_words) + 1
        self.review_words.append([-1] * args.review_word_limit)
        #so that review_words[-1] = -1, ..., -1
        self.u_r_seq = self.read_arr_from_lines("{}/u_r_seq.txt.gz".format(data_path)) #list of review ids
        self.i_r_seq = self.read_arr_from_lines("{}/p_r_seq.txt.gz".format(data_path)) #list of review ids
        self.review_loc_time = self.read_arr_from_lines("{}/review_uloc_ploc_and_time.txt.gz".format(data_path)) #(loc_in_u, loc_in_i, time) of each review

        print("Data statistic: vocab %d, review %d, user %d, product %d\n" % (self.vocab_size,
                    self.review_count, self.user_size, self.product_size))


    '''
    def read_review_loc_time(self, fname):
        line_arr = []
        line_no = 0
        with gzip.open(fname, 'r') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                arr = [int(x) for x in arr]
                line_arr.append([line_no] + arr)
                #review_id, location in user's review, timestamp
                line_no += 1
        line_arr.sort(lambda x:x[-1])
        rtn_line_arr = [[] for i in range(len(line_arr))]
        for rank, review_info in enumerate(line_arr):
            review_id, loc, time = review_info
            rtn_line_arr[review_id] += [loc, time, rank]
        return rtn_line_arr
    '''

    @staticmethod
    def read_arr_from_lines(fname):
        line_arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                filter_arr = []
                for idx in arr:
                    if len(idx) < 1:
                        continue
                    filter_arr.append(int(idx))
                line_arr.append(filter_arr)
        return line_arr

    @staticmethod
    def read_lines(fname):
        arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr.append(line.strip())
        return arr

    @staticmethod
    def read_words_in_lines(fname):
        line_arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                words = [int(i) for i in line.strip().split(' ')]
                line_arr.append(words)
        return line_arr

