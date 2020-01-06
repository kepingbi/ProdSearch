import torch
from torch.utils.data import DataLoader
import others.util as util
import numpy as np


class ProdSearchDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(ProdSearchDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)
        self.shuffle = shuffle
        self.review_pad_idx = self.dataset.review_pad_idx
        self.word_pad_idx = self.dataset.word_pad_idx
        self.seg_pad_idx = self.dataset.seg_pad_idx
        self.global_data = self.dataset.global_data
        self.prod_data = self.dataset.prod_data

    class ProdSearchBatch(object):
        def __init__(self, query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                        pos_prod_rword_idxs, pos_prod_rword_masks,
                        neg_prod_ridxs, neg_seg_idxs, neg_prod_rword_idxs=None,
                        neg_prod_rword_masks=None,
                        pos_prod_rword_idxs_pvc=None,
                        neg_prod_rword_idxs_pvc=None,
                        to_tensor=True): #"cpu" or "cuda"
            self.query_word_idxs = query_word_idxs
            self.pos_prod_ridxs = pos_prod_ridxs
            self.pos_seg_idxs = pos_seg_idxs
            self.pos_prod_rword_idxs = pos_prod_rword_idxs
            self.pos_prod_rword_masks = pos_prod_rword_masks
            self.neg_prod_ridxs = neg_prod_ridxs
            self.neg_seg_idxs = neg_seg_idxs
            self.neg_prod_rword_idxs = neg_prod_rword_idxs
            self.neg_prod_rword_masks = neg_prod_rword_masks
            #for pvc
            self.neg_prod_rword_idxs_pvc = neg_prod_rword_idxs_pvc
            self.pos_prod_rword_idxs_pvc = pos_prod_rword_idxs_pvc
            if to_tensor:
                self.to_tensor()

        def to_tensor(self):
            self.query_word_idxs = torch.tensor(self.query_word_idxs)
            self.pos_prod_ridxs = torch.tensor(self.pos_prod_ridxs)
            self.pos_seg_idxs = torch.tensor(self.pos_seg_idxs)
            self.pos_prod_rword_idxs = torch.tensor(self.pos_prod_rword_idxs)
            self.neg_prod_ridxs = torch.tensor(self.neg_prod_ridxs)
            self.neg_seg_idxs = torch.tensor(self.neg_seg_idxs)
            self.pos_prod_rword_masks = torch.ByteTensor(self.pos_prod_rword_masks)
            if self.neg_prod_rword_idxs is not None:
                self.neg_prod_rword_idxs = torch.tensor(self.neg_prod_rword_idxs)
            if self.neg_prod_rword_masks is not None:
                self.neg_prod_rword_masks = torch.ByteTensor(self.neg_prod_rword_masks)
            #for pvc
            if self.neg_prod_rword_idxs_pvc is not None:
                    self.neg_prod_rword_idxs_pvc = torch.tensor(self.neg_prod_rword_idxs_pvc)
            if self.pos_prod_rword_idxs_pvc is not None:
                    self.pos_prod_rword_idxs_pvc = torch.tensor(self.pos_prod_rword_idxs_pvc)

        def to(self, device):
            if device == "cpu":
                return self
            else:
                query_word_idxs = self.query_word_idxs.to(device)
                pos_prod_ridxs = self.pos_prod_ridxs.to(device)
                pos_seg_idxs = self.pos_seg_idxs.to(device)
                pos_prod_rword_idxs = self.pos_prod_rword_idxs.to(device)
                pos_prod_rword_masks = self.pos_prod_rword_masks.to(device)
                neg_prod_ridxs = self.neg_prod_ridxs.to(device)
                neg_seg_idxs = self.neg_seg_idxs.to(device)
                neg_prod_rword_idxs = None if self.neg_prod_rword_idxs is None \
                        else self.neg_prod_rword_idxs.to(device)
                neg_prod_rword_masks = None if self.neg_prod_rword_masks is None \
                        else self.neg_prod_rword_masks.to(device)
                #for pvc
                neg_prod_rword_idxs_pvc = None if self.neg_prod_rword_idxs_pvc is None \
                        else self.neg_prod_rword_idxs_pvc.to(device)
                pos_prod_rword_idxs_pvc = None if self.pos_prod_rword_idxs_pvc is None \
                        else self.pos_prod_rword_idxs_pvc.to(device)
                return self.__class__(
                        query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                        pos_prod_rword_idxs, pos_prod_rword_masks,
                        neg_prod_ridxs, neg_seg_idxs, neg_prod_rword_idxs,
                        neg_prod_rword_masks,
                        pos_prod_rword_idxs_pvc,
                        neg_prod_rword_idxs_pvc, to_tensor=False)


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
    def _collate_fn(self, batch):
        query_word_idxs = [entry[0] for entry in batch]
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
        pos_prod_rword_idxs = util.pad(pos_prod_rword_idxs, pad_id = self.word_pad_idx)
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
        neg_prod_rword_idxs = util.pad(neg_prod_rword_idxs, pad_id = self.word_pad_idx)
        neg_prod_rword_idxs = np.asarray(neg_prod_rword_idxs).reshape(batch_size, neg_k, nr_count, -1)

        if "pv" in self.dataset.review_encoder_name:
            pos_prod_rword_idxs_pvc = pos_prod_rword_idxs
            neg_prod_rword_idxs_pvc = neg_prod_rword_idxs
            batch_size, pos_rcount, word_limit = pos_prod_rword_idxs.shape
            pv_window_size = self.dataset.pv_window_size
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
            batch = [self.ProdSearchBatch(query_word_idxs[batch_indices[i]],
                pos_prod_ridxs[batch_indices[i]], pos_seg_idxs[batch_indices[i]],
                slide_pos_prod_rword_idxs[i], slide_pos_prod_rword_masks[i],
                neg_prod_ridxs[batch_indices[i]], neg_seg_idxs[batch_indices[i]],
                pos_prod_rword_idxs_pvc = pos_prod_rword_idxs_pvc[batch_indices[i]],
                neg_prod_rword_idxs_pvc = neg_prod_rword_idxs_pvc[batch_indices[i]]) for i in range(seg_count)]
        else:
            neg_prod_rword_masks = self.dataset.get_pv_word_masks(
                    neg_prod_rword_idxs, self.prod_data.sub_sampling_rate, pad_id=self.word_pad_idx)
            batch = [self.ProdSearchBatch(query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                    pos_prod_rword_idxs, pos_prod_rword_masks,
                    neg_prod_ridxs, neg_seg_idxs,
                    neg_prod_rword_idxs = neg_prod_rword_idxs,
                    neg_prod_rword_masks = neg_prod_rword_masks)]
        return batch

