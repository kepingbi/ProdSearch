"""Encode reviews. It can be:
    1) Read from previously trained paragraph vectors.
    2) From word embeddings [avg, projected weight avg, or CNN, RNN]
    3) Train embedding jointly with the loss of purchases
        review_id, a group of words in the review (random -> PV with corruption; in order -> PV)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder import get_vector_mean

import argparse

class ParagraphVector(nn.Module):
    def __init__(self, word_embeddings, word_dists, review_count, dropout=0.0):
        super(ParagraphVector, self).__init__()
        self.word_embeddings = word_embeddings
        self.dropout_ = dropout
        self.word_dists = word_dists
        self._embedding_size = self.word_embeddings.weight.size()[-1]
        self.review_count = review_count
        self.review_pad_idx = review_count-1
        self.review_embeddings = nn.Embedding(self.review_count, self._embedding_size, padding_idx=self.review_pad_idx)
            #, scale_grad_by_freq = scale_grad, sparse=self.is_emb_sparse
        self.drop_layer = nn.Dropout(p=self.dropout_)
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean

    @property
    def embedding_size(self):
        return self._embedding_size

    def forward(self, review_ids, review_word_emb, review_word_mask, n_negs):
        batch_size, pv_window_size, embedding_size = review_word_emb.size()
        #for each target word, there is k words negative sampling
        review_emb = self.review_embeddings(review_ids)
        self.drop_layer = nn.Dropout(p=self.dropout_)
        #vocab_size = self.word_embeddings.weight.size() - 1
        #compute the loss of review generating positive and negative words
        neg_sample_idxs = torch.multinomial(self.word_dists, batch_size * pv_window_size * n_negs, replacement=True)
        neg_sample_emb = self.word_embeddings(neg_sample_idxs.view(batch_size,-1))
        output_pos = torch.bmm(review_word_emb, review_emb.unsqueeze(2)) # batch_size, pv_window_size, 1
        output_neg = torch.bmm(neg_sample_emb, review_emb.unsqueeze(2)).view(batch_size, pv_window_size, -1)
        scores = torch.cat((output_pos, output_neg), dim=-1) #batch_size, pv_window_size, 1+n_negs
        target = torch.cat((torch.ones(output_pos.size(), device=scores.device),
            torch.zeros(output_neg.size(), device=scores.device)), dim=-1)
        loss = self.bce_logits_loss(scores, target).sum(-1) #batch_size, pv_window_size
        loss = get_vector_mean(loss.unsqueeze(-1), review_word_mask)
        #cuda.longtensor
        #negative sampling according to x^0.75
        #each word has n_neg corresponding samples
        '''
        oloss = torch.bmm(review_word_emb, review_emb.unsqueeze(2)).squeeze(-1)
        nloss = torch.bmm(neg_sample_emb.neg(), review_emb.unsqueeze(2)).squeeze(-1)
        nloss = nloss.view(batch_size, pv_window_size, -1)
        oloss = oloss.sigmoid().log() #batch_size, pv_window_size
        nloss = nloss.sigmoid().log().sum(2)# batch_size, pv_window_size#(n_negs->1)
        loss = -(nloss + oloss) # * review_word_mask.float()
        loss = get_vector_mean(loss.unsqueeze(-1), review_word_mask)
        #(batch_size, )
        #loss = loss.sum() / review_ids.ne(self.review_pad_idx).float().sum()
        '''

        return review_emb, loss

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" ReviewEncoder initialization started.")
        #otherwise, load pretrained embeddings
        nn.init.normal_(self.review_embeddings.weight)
        if logger:
            logger.info(" ReviewEncoder initialization finished.")

