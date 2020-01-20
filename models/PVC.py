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
from others.util import load_pretrain_embeddings

import argparse

class ParagraphVectorCorruption(nn.Module):
    def __init__(self, word_embeddings, word_dists, corrupt_rate,
            dropout=0.0, pretrain_emb_path=None, vocab_words=None, fix_emb=False):
        super(ParagraphVectorCorruption, self).__init__()
        self.word_embeddings = word_embeddings
        self.word_dists = word_dists
        self._embedding_size = self.word_embeddings.weight.size()[-1]
        vocab_size = self.word_embeddings.weight.size()[0]
        self.word_pad_idx = vocab_size - 1
        if pretrain_emb_path is not None and vocab_words is not None:
            word_index_dic, pretrained_weights = load_pretrain_embeddings(pretrain_emb_path)
            word_indices = torch.tensor([0] + [word_index_dic[x] for x in vocab_words[1:]] + [self.word_pad_idx])
            pretrained_weights = torch.FloatTensor(pretrained_weights)

            self.context_embeddings = nn.Embedding.from_pretrained(pretrained_weights[word_indices], padding_idx=self.word_pad_idx)
        else:
            self.context_embeddings = nn.Embedding(
                vocab_size, self._embedding_size, padding_idx=self.word_pad_idx)
        if fix_emb:
            self.context_embeddings.weight.requires_grad = False
            self.dropout_ = 0
        self.corrupt_rate = corrupt_rate
        self.train_corrupt_rate = corrupt_rate
        self.dropout_ = dropout
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean
        #vocab_size - 1

    @property
    def embedding_size(self):
        return self._embedding_size

    def apply_token_dropout(self, inputs, drop_prob):
        #randomly dropout some token in the review
        #batch_size, review_word_count, embedding_size
        probs = inputs.data.new().resize_(inputs.size()[:-1]).fill_(drop_prob)
        #batch_size, review_word_count
        mask = torch.bernoulli(probs).byte().unsqueeze(-1) #the probability of drawing 1
        #batch_size, review_word_count, 1
        inputs.data.masked_fill_(mask, 0).mul_(1./(1-drop_prob)) #drop_prob to fill data to 0
        return inputs

    def get_para_vector(self, prod_rword_idxs_pvc):
        pvc_word_emb = self.context_embeddings(prod_rword_idxs_pvc)
        if self.corrupt_rate > 0.:
            self.apply_token_dropout(pvc_word_emb, self.corrupt_rate)
        review_emb = get_vector_mean(pvc_word_emb, prod_rword_idxs_pvc.ne(self.word_pad_idx))
        return review_emb

    def set_to_evaluation_mode(self):
        self.corrupt_rate = 0.

    def set_to_train_mode(self):
        self.corrupt_rate = self.train_corrupt_rate

    def forward(self, review_word_emb, review_word_mask, prod_rword_idxs_pvc, n_negs):
        '''
            prod_rword_idxs_pvc: batch_size (real_batch_size * review_count), review_word_limit
            review_word_emb: batch_size * reivew_count, embedding_size
            review_word_mask: indicate which target is valid
        '''
        batch_size, pv_window_size, embedding_size = review_word_emb.size()
        pvc_word_emb = self.context_embeddings(prod_rword_idxs_pvc)
        review_emb = get_vector_mean(pvc_word_emb, prod_rword_idxs_pvc.ne(self.word_pad_idx))
        self.apply_token_dropout(pvc_word_emb, self.corrupt_rate)
        corr_review_emb = get_vector_mean(pvc_word_emb, prod_rword_idxs_pvc.ne(self.word_pad_idx))

        #for each target word, there is k words negative sampling
        #compute the loss of review generating positive and negative words
        neg_sample_idxs = torch.multinomial(self.word_dists, batch_size * pv_window_size * n_negs, replacement=True)
        neg_sample_emb = self.word_embeddings(neg_sample_idxs.view(batch_size, -1))
        output_pos = torch.bmm(review_word_emb, corr_review_emb.unsqueeze(2)) # batch_size, pv_window_size, 1
        output_neg = torch.bmm(neg_sample_emb, corr_review_emb.unsqueeze(2)).view(batch_size, pv_window_size, -1)
        scores = torch.cat((output_pos, output_neg), dim=-1) #batch_size, pv_window_size, 1+n_negs
        target = torch.cat((torch.ones(output_pos.size(), device=scores.device),
            torch.zeros(output_neg.size(), device=scores.device)), dim=-1)
        loss = self.bce_logits_loss(scores, target).sum(-1) #batch_size, pv_window_size
        loss = get_vector_mean(loss.unsqueeze(-1), review_word_mask)

        #cuda.longtensor
        #negative sampling according to x^0.75
        return review_emb, loss

    def forward_deprecated(self, review_word_emb, review_word_mask, prod_rword_idxs_pvc, rand_prod_rword_idxs, n_negs):
        ''' rand_prod_rword_idxs: batch_size, review_count, pv_window_size * pvc_word_count
            prod_rword_idxs_pvc: batch_size, review_count, review_word_limit
            review_word_mask: indicate which target is valid
        '''
        batch_size, pv_window_size, embedding_size = review_word_emb.size()
        _,_,word_count = prod_rword_idxs_pvc.size()
        pvc_word_count = word_count / pv_window_size

        rand_word_emb = self.word_embeddings(rand_prod_rword_idxs.view(-1, pvc_word_count))
        corr_review_vector = get_vector_mean(rand_word_emb, rand_prod_rword_idxs.ne(self.word_pad_idx))
        corr_review_vector = corr_review_vector.view(-1, embedding_size, 1)

        #for each target word, there is k words negative sampling
        vocab_size = word_embeddings.weight.size() - 1
        #compute the loss of review generating positive and negative words
        neg_sample_idxs = torch.multinomial(self.word_dists, batch_size * pv_window_size * n_negs, replacement=True)
        neg_sample_emb = self.word_embeddings(neg_sample_idxs)
        #cuda.longtensor
        #negative sampling according to x^0.75
        #each word has n_neg corresponding samples
        target_emb = review_word_emb.view(batch_size*pv_window_size, embedding_size).unsqueeze(1)
        oloss = torch.bmm(target_emb, corr_review_vector).squeeze(-1).squeeze(-1).view(batch_size, -1)
        nloss = torch.bmm(neg_sample_emb.unsqueeze(1).neg(), corr_review_vector).squeeze(-1).squeeze(-1)
        nloss = nloss.view(batch_size, pv_window_size, -1)
        oloss = oloss.sigmoid().log() #batch_size, pv_window_size
        nloss = nloss.sigmoid().log().sum(2)# batch_size, pv_window_size#(n_negs->1)
        loss = -(nloss + oloss) #* review_word_mask.float()
        loss = get_vector_mean(loss.unsqueeze(-1), review_word_mask)
        #(batch_size, )
        #loss = get_vector_mean(loss.unsqueeze(-1), review_ids.ne(self.review_pad_idx))
        #loss = loss.mean()
        _,rcount, review_word_limit = prod_rword_idxs_pvc.size()
        pvc_word_emb = self.word_embeddings(prod_rword_idxs_pvc.view(-1, review_word_limit))
        review_emb = get_vector_mean(pvc_word_emb, prod_rword_idxs_pvc.ne(self.word_pad_idx))

        return review_emb.view(-1, rcount, embedding_size), loss

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" Another group of embeddings initialization started.")
        #otherwise, load pretrained embeddings
        #nn.init.normal_(self.review_embeddings.weight)
        if logger:
            logger.info(" Another group of embeddings initialization finished.")

