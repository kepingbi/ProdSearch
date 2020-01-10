import torch
import numpy as np

from collections import defaultdict
import others.util as util
import gzip

class ProdSearchData():
    def __init__(self, args, input_train_dir, set_name,
            vocab_size, review_count,
            user_size, product_size, line_review_id_map):
        self.args = args
        self.neg_per_pos = args.neg_per_pos
        self.set_name = set_name
        self.review_count = review_count
        self.product_size = product_size
        self.user_size = user_size
        self.vocab_size = vocab_size
        self.sub_sampling_rate = None
        self.neg_sample_products = None
        self.word_dists = None
        self.product_dists = None

        if set_name == "train":
            self.vocab_distribute, self.product_distribute = self.read_reviews("{}/{}.txt.gz".format(input_train_dir, set_name))
            self.vocab_distribute = self.vocab_distribute.tolist()
            self.sub_sampling(args.subsampling_rate)
            self.word_dists = self.neg_distributes(self.vocab_distribute)
            self.product_dists = self.neg_distributes(self.product_distribute)

        #self.product_query_idx = GlobalProdSearchData.read_arr_from_lines("{}/{}_query_idx.txt.gz".format(input_train_dir, set_name))
        self.review_info, self.review_query_idx = self.read_review_id(
                "{}/{}_id.txt.gz".format(input_train_dir, set_name),
                line_review_id_map)
        self.set_review_size = len(self.review_info)
        if args.train_review_only and set_name != "train":
            #u:reviews i:reviews
            self.train_review_info, _ = self.read_review_id(
                    "{}/train_id.txt.gz".format(input_train_dir),
                    line_review_id_map)
            self.u_reviews, self.p_reviews = self.get_u_i_reviews(user_size, product_size, self.train_review_info)

    def get_u_i_reviews(self, user_size, product_size, review_info):
        u_reviews = [[] for i in range(self.user_size)]
        p_reviews = [[] for i in range(self.product_size)]
        for u_idx, p_idx, r_idx in review_info:
            u_reviews[u_idx].append(r_idx)
            p_reviews[p_idx].append(r_idx)
        return u_reviews, p_reviews

    def initialize_epoch(self):
        #self.neg_sample_products = np.random.randint(0, self.product_size, size = (self.set_review_size, self.neg_per_pos))
        self.neg_sample_products = np.random.choice(self.product_size,
                size = (self.set_review_size, self.neg_per_pos), replace=True, p=self.product_dists)


    def read_review_id(self, fname, line_review_id_map):
        query_ids = []
        review_info = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                review_id = line_review_id_map[int(arr[-2].split('_')[-1])]
                review_info.append((int(arr[0]), int(arr[1]), review_id))#(user_idx, product_idx)
                query_ids.append(int(arr[-1]))
        return review_info, query_ids

    def read_reviews(self, fname):
        vocab_distribute = np.zeros(self.vocab_size)
        product_distribute = np.zeros(self.product_size)
        #review_info = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                #review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
                product_distribute[int(arr[1])] += 1
                review_text = [int(i) for i in arr[2].split(' ')]
                for idx in review_text:
                    vocab_distribute[idx] += 1
        #return vocab_distribute, review_info
        return vocab_distribute, product_distribute
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
        self.sub_sampling_rate = np.asarray(self.sub_sampling_rate)
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
        self.word_pad_idx = self.vocab_size-1
        self.query_words = util.pad(self.query_words, pad_id=self.word_pad_idx)

        self.review_words = self.read_words_in_lines(
                "{}/review_text.txt.gz".format(data_path), cutoff=args.review_word_limit)
        self.review_length = [len(x) for x in self.review_words]
        #self.review_words = util.pad(self.review_words, pad_id=self.vocab_size-1, width=args.review_word_limit)
        self.review_count = len(self.review_words) + 1
        self.review_words.append([self.word_pad_idx]) # * args.review_word_limit)
        #so that review_words[-1] = -1, ..., -1
        self.u_r_seq = self.read_arr_from_lines("{}/u_r_seq.txt.gz".format(data_path)) #list of review ids
        self.i_r_seq = self.read_arr_from_lines("{}/p_r_seq.txt.gz".format(data_path)) #list of review ids
        self.review_loc_time = self.read_arr_from_lines("{}/review_uloc_ploc_and_time.txt.gz".format(data_path)) #(loc_in_u, loc_in_i, time) of each review
        self.line_review_id_map = self.read_review_id_line_map("{}/review_id.txt.gz".format(data_path))

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
    def read_review_id_line_map(fname):
        line_review_id_map = dict()
        with gzip.open(fname, 'rt') as fin:
            idx = 0
            for line in fin:
                ori_line_id = int(line.strip().split('_')[-1])
                line_review_id_map[ori_line_id] = idx
                idx += 1
        return line_review_id_map

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
    def read_words_in_lines(fname, cutoff=-1):
        line_arr = []
        with gzip.open(fname, 'rt') as fin:
            for line in fin:
                words = [int(i) for i in line.strip().split(' ')]
                if cutoff < 0:
                    line_arr.append(words)
                else:
                    line_arr.append(words[:cutoff])
        return line_arr

