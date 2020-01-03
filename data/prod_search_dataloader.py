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
        self.review_pad_idx = self.dataset.review_pad_idx
        self.word_pad_idx = self.dataset.word_pad_idx
        self.seg_pad_idx = self.dataset.seg_pad_idx

    class ProdSearchBatch(object):
        def __init__(self, query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                        pos_prod_rword_idxs, pos_prod_rword_masks,
                        neg_prod_ridxs, neg_seg_idxs, neg_prod_rword_idxs,
                        neg_prod_rword_masks,
                        #rand_pos_prod_rword_idxs,
                        #rand_neg_prod_rword_idxs,
                        pos_prod_rword_idxs_pvc,
                        neg_prod_rword_idxs_pvc,
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
            #self.rand_neg_prod_rword_idxs = rand_neg_prod_rword_idxs
            #self.rand_pos_prod_rword_idxs = rand_pos_prod_rword_idxs
            self.neg_prod_rword_idxs_pvc = neg_prod_rword_idxs_pvc
            self.pos_prod_rword_idxs_pvc = pos_prod_rword_idxs_pvc
            if to_tensor:
                self.to_tensor()

        def to_tensor(self):
            self.query_word_idxs = torch.tensor(self.query_word_idxs)
            self.pos_prod_ridxs = torch.tensor(self.pos_prod_ridxs)
            self.pos_seg_idxs = torch.tensor(self.pos_seg_idxs)
            self.pos_prod_rword_idxs = torch.tensor(self.pos_prod_rword_idxs)
            if self.pos_prod_rword_masks is None:
                self.pos_prod_rword_masks = self.pos_prod_rword_idxs.ne(-1)
            else:
                self.pos_prod_rword_masks = torch.tensor(self.pos_prod_rword_masks)
            self.neg_prod_ridxs = torch.tensor(self.neg_prod_ridxs)
            self.neg_seg_idxs = torch.tensor(self.neg_seg_idxs)
            self.neg_prod_rword_idxs = torch.tensor(self.neg_prod_rword_idxs)
            if self.neg_prod_rword_masks is None:
                self.neg_prod_rword_masks = self.neg_prod_rword_idxs.ne(-1)
            else:
                self.neg_prod_rword_masks = torch.tensor(self.neg_prod_rword_masks)

            #for pvc
            '''
            self.rand_neg_prod_rword_idxs = None if self.rand_neg_prod_rword_idxs is None \
                    else torch.tensor(self.rand_neg_prod_rword_idxs)
            self.rand_pos_prod_rword_idxs = None if self.rand_pos_prod_rword_idxs is None \
                    else torch.tensor(self.rand_pos_prod_rword_idxs)
            '''
            self.neg_prod_rword_idxs_pvc = None if self.neg_prod_rword_idxs_pvc is None \
                    else torch.tensor(self.neg_prod_rword_idxs_pvc)
            self.pos_prod_rword_idxs_pvc = None if self.pos_prod_rword_idxs_pvc is None \
                    else torch.tensor(self.pos_prod_rword_idxs_pvc)

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
                neg_prod_rword_idxs = self.neg_prod_rword_idxs.to(device)
                neg_prod_rword_masks = self.neg_prod_rword_masks.to(device)
                #for pvc
                '''
                rand_neg_prod_rword_idxs = None if self.rand_neg_prod_rword_idxs is None \
                        else self.rand_neg_prod_rword_idxs.to(device)
                rand_pos_prod_rword_idxs = None if self.rand_pos_prod_rword_idxs is None \
                        else self.rand_pos_prod_rword_idxs.to(device)
                '''
                neg_prod_rword_idxs_pvc = None if self.neg_prod_rword_idxs_pvc is None \
                        else self.neg_prod_rword_idxs_pvc.to(device)
                pos_prod_rword_idxs_pvc = None if self.pos_prod_rword_idxs_pvc is None \
                        else self.pos_prod_rword_idxs_pvc.to(device)
                return self.__class__(
                        query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                        pos_prod_rword_idxs, pos_prod_rword_masks,
                        neg_prod_ridxs, neg_seg_idxs, neg_prod_rword_idxs,
                        neg_prod_rword_masks,
                        #rand_pos_prod_rword_idxs,
                        #rand_neg_prod_rword_idxs,
                        pos_prod_rword_idxs_pvc,
                        neg_prod_rword_idxs_pvc, to_tensor=False)


    def _collate_fn(self, batch):
        query_word_idxs = [entry[0] for entry in batch]
        pos_prod_ridxs = [entry[1] for entry in batch]
        pos_seg_idxs = [entry[2] for entry in batch]
        pos_prod_rword_idxs = [entry[3] for entry in batch]
        #batch_size, review_count, word_limit
        pos_prod_rword_masks = [entry[4] for entry in batch]
        neg_prod_ridxs = [entry[5] for entry in batch]
        #batch_size, neg_k, review_count
        neg_seg_idxs = [entry[6] for entry in batch]
        neg_prod_rword_idxs = [entry[7] for entry in batch]
        #batch_size, neg_k, review_count, word_limit
        neg_prod_rword_masks = [entry[8] for entry in batch]
        #rand_pos_prod_rword_idxs = [entry[9] for entry in batch]
        #rand_neg_prod_rword_idxs = [entry[10] for entry in batch]
        pos_prod_rword_idxs_pvc = [entry[9] for entry in batch]
        neg_prod_rword_idxs_pvc = [entry[10] for entry in batch]
        pos_prod_ridxs = util.pad(pos_prod_ridxs, pad_id = self.review_pad_idx)
        pos_seg_idxs = util.pad(pos_seg_idxs, pad_id = self.seg_pad_idx)
        pos_prod_rword_idxs = util.pad_3d(pos_prod_rword_idxs, pad_id = self.word_pad_idx, dim=1)
        if pos_prod_rword_masks[0] is None:
            pos_prod_rword_masks = None
        else:
            pos_prod_rword_masks = util.pad_3d(pos_prod_rword_masks, pad_id = 0, dim=1)
        neg_prod_ridxs = util.pad_3d(neg_prod_ridxs, pad_id = self.review_pad_idx, dim=1)
        neg_prod_ridxs = util.pad_3d(neg_prod_ridxs, pad_id = self.review_pad_idx, dim=2)
        neg_seg_idxs = util.pad_3d(neg_seg_idxs, pad_id = self.seg_pad_idx, dim=1)
        neg_seg_idxs = util.pad_3d(neg_seg_idxs, pad_id = self.seg_pad_idx, dim=2)
        neg_prod_rword_idxs = util.pad_4d_dim1(neg_prod_rword_idxs, pad_id = self.word_pad_idx)
        neg_prod_rword_idxs = util.pad_4d_dim2(neg_prod_rword_idxs, pad_id = self.word_pad_idx)
        batch_size, neg_k, nr_count, word_limit = np.asarray(neg_prod_rword_idxs).shape
        if neg_prod_rword_masks[0] is None:
            neg_prod_rword_masks = None
        else:
            neg_prod_rword_masks = util.pad_4d_dim1(neg_prod_rword_masks, pad_id = 0)
            neg_prod_rword_masks = util.pad_4d_dim2(neg_prod_rword_masks, pad_id = 0)
        if pos_prod_rword_idxs_pvc[0] is None or neg_prod_rword_idxs_pvc[0] is None:
            pos_prod_rword_idxs_pvc = None
            neg_prod_rword_idxs_pvc = None
        else:
            pos_prod_rword_idxs_pvc = util.pad_3d(pos_prod_rword_idxs_pvc, pad_id = self.word_pad_idx, dim=1)
            neg_prod_rword_idxs_pvc = util.pad_4d_dim1(neg_prod_rword_idxs_pvc, pad_id = self.word_pad_idx)
            neg_prod_rword_idxs_pvc = util.pad_4d_dim2(neg_prod_rword_idxs_pvc, pad_id = self.word_pad_idx)
        '''
        if rand_pos_prod_rword_idxs[0] is None or rand_neg_prod_rword_idxs[0] is None:
            rand_pos_prod_rword_idxs = None
            rand_neg_prod_rword_idxs = None
        else:
            rand_pos_prod_rword_idxs = util.pad_3d(rand_pos_prod_rword_idxs, pad_id = self.word_pad_idx, dim=1)
            rand_pos_prod_rword_idxs = util.pad_3d(rand_pos_prod_rword_idxs, pad_id = self.word_pad_idx, dim=2)
            #pad to review_count, pad to random selected_word_count
            rand_neg_prod_rword_idxs = util.pad_3d(rand_neg_prod_rword_idxs, pad_id = self.word_pad_idx, dim=1)
            rand_neg_prod_rword_idxs = util.pad_3d(rand_neg_prod_rword_idxs, pad_id = self.word_pad_idx, dim=2)

            #rand_neg_prod_rword_idxs = np.asarray(rand_neg_prod_rword_idxs).reshape(batch_size, neg_k, nr_count, -1)
            #rand_neg_prod_rword_idxs = util.pad_4d_dim2(rand_neg_prod_rword_idxs, pad_id = -1) #only process the case of dim=2
        '''

        return self.ProdSearchBatch(query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                pos_prod_rword_idxs, pos_prod_rword_masks,
                neg_prod_ridxs, neg_seg_idxs, neg_prod_rword_idxs,
                neg_prod_rword_masks,
                #rand_pos_prod_rword_idxs,
                #rand_neg_prod_rword_idxs,
                pos_prod_rword_idxs_pvc,
                neg_prod_rword_idxs_pvc) #"cpu" or "cuda"
