import torch

class ProdSearchTestBatch(object):
    def __init__(self, query_idxs, user_idxs, target_prod_idxs, candi_prod_idxs, query_word_idxs,
            candi_prod_ridxs, candi_seg_idxs, candi_seq_user_idxs, candi_seq_item_idxs, to_tensor=True): #"cpu" or "cuda"
        self.query_idxs = query_idxs
        self.user_idxs = user_idxs
        self.target_prod_idxs = target_prod_idxs
        self.candi_prod_idxs = candi_prod_idxs
        self.candi_seq_user_idxs = candi_seq_user_idxs
        self.candi_seq_item_idxs = candi_seq_item_idxs

        self.query_word_idxs = query_word_idxs
        self.candi_prod_ridxs = candi_prod_ridxs
        self.candi_seg_idxs = candi_seg_idxs
        if to_tensor:
            self.to_tensor()

    def to_tensor(self):
        #self.query_idxs = torch.tensor(query_idxs)
        #self.user_idxs = torch.tensor(user_idxs)
        #self.target_prod_idxs = torch.tensor(target_prod_idxs)
        #self.candi_prod_idxs = torch.tensor(candi_prod_idxs)
        self.query_word_idxs = torch.tensor(self.query_word_idxs)
        self.candi_prod_ridxs = torch.tensor(self.candi_prod_ridxs)
        self.candi_seg_idxs = torch.tensor(self.candi_seg_idxs)
        self.candi_seq_user_idxs = torch.tensor(self.candi_seq_user_idxs)
        self.candi_seq_item_idxs = torch.tensor(self.candi_seq_item_idxs)

    def to(self, device):
        if device == "cpu":
            return self
        else:
            query_word_idxs = self.query_word_idxs.to(device)
            candi_prod_ridxs = self.candi_prod_ridxs.to(device)
            candi_seg_idxs = self.candi_seg_idxs.to(device)
            candi_seq_user_idxs = self.candi_seq_user_idxs.to(device)
            candi_seq_item_idxs = self.candi_seq_item_idxs.to(device)

            return self.__class__(self.query_idxs, self.user_idxs,
                    self.target_prod_idxs, self.candi_prod_idxs, query_word_idxs,
                    candi_prod_ridxs, candi_seg_idxs, candi_seq_user_idxs,
                    candi_seq_item_idxs, to_tensor=False)

class ProdSearchTrainBatch(object):
    def __init__(self, query_word_idxs, pos_prod_ridxs, pos_seg_idxs,
                    pos_prod_rword_idxs, pos_prod_rword_masks,
                    neg_prod_ridxs, neg_seg_idxs,
                    pos_user_idxs, neg_user_idxs,
                    pos_item_idxs, neg_item_idxs,
                    neg_prod_rword_idxs=None,
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
        self.pos_user_idxs = pos_user_idxs
        self.neg_user_idxs = neg_user_idxs
        self.pos_item_idxs = pos_item_idxs
        self.neg_item_idxs = neg_item_idxs
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
        self.pos_user_idxs = torch.tensor(self.pos_user_idxs)
        self.neg_user_idxs = torch.tensor(self.neg_user_idxs)
        self.pos_item_idxs = torch.tensor(self.pos_item_idxs)
        self.neg_item_idxs = torch.tensor(self.neg_item_idxs)
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
            pos_user_idxs = self.pos_user_idxs.to(device)
            neg_user_idxs = self.neg_user_idxs.to(device)
            pos_item_idxs = self.pos_item_idxs.to(device)
            neg_item_idxs = self.neg_item_idxs.to(device)

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
                    neg_prod_ridxs, neg_seg_idxs,
                    pos_user_idxs, neg_user_idxs,
                    pos_item_idxs, neg_item_idxs,
                    neg_prod_rword_idxs,
                    neg_prod_rword_masks,
                    pos_prod_rword_idxs_pvc,
                    neg_prod_rword_idxs_pvc, to_tensor=False)
