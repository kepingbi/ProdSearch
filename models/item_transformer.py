""" transformer based on reviews
    Q+r_{u1}+r_{u2} <> r_1, r_2 (of a target i)
"""
"""
review_encoder
query_encoder
transformer
"""
import os
import torch
import torch.nn as nn
from models.PV import ParagraphVector
from models.PVC import ParagraphVectorCorruption
from models.text_encoder import AVGEncoder, FSEncoder, get_vector_mean
from models.transformer import TransformerEncoder
from models.neural import MultiHeadedAttention
from models.optimizers import Optimizer
from others.logging import logger
from others.util import pad, load_pretrain_embeddings, load_user_item_embeddings


class ItemTransformerRanker(nn.Module):
    def __init__(self, args, device, vocab_size, product_size, vocab_words, word_dists=None):
        super(ItemTransformerRanker, self).__init__()
        self.args = args
        self.device = device
        self.train_review_only = args.train_review_only
        self.embedding_size = args.embedding_size
        self.vocab_words = vocab_words
        self.word_dists = None
        if word_dists is not None:
            self.word_dists = torch.tensor(word_dists, device=device)
        self.prod_dists = torch.ones(product_size, device=device)
        self.prod_pad_idx = product_size
        self.word_pad_idx = vocab_size - 1
        self.seg_pad_idx = 3
        self.emb_dropout = args.dropout
        self.pretrain_emb_dir = None
        if os.path.exists(args.pretrain_emb_dir):
            self.pretrain_emb_dir = args.pretrain_emb_dir
        self.pretrain_up_emb_dir = None
        if os.path.exists(args.pretrain_up_emb_dir):
            self.pretrain_up_emb_dir = args.pretrain_up_emb_dir
        self.dropout_layer = nn.Dropout(p=args.dropout)

        self.product_emb = nn.Embedding(product_size+1, self.embedding_size, padding_idx=self.prod_pad_idx)
        if args.sep_prod_emb:
            self.hist_product_emb = nn.Embedding(product_size+1, self.embedding_size, padding_idx=self.prod_pad_idx)
        '''
        else:
            pretrain_product_emb_path = os.path.join(self.pretrain_up_emb_dir, "product_emb.txt")
            pretrained_weights = load_user_item_embeddings(pretrain_product_emb_path)
            pretrained_weights.append([0.] * len(pretrained_weights[0]))
            self.product_emb = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_weights), padding_idx=self.prod_pad_idx)
        '''
        self.product_bias = nn.Parameter(torch.zeros(product_size+1), requires_grad=True)
        self.word_bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)

        if self.pretrain_emb_dir is not None:
            word_emb_fname = "word_emb.txt.gz" #for query and target words in pv and pvc
            pretrain_word_emb_path = os.path.join(self.pretrain_emb_dir, word_emb_fname)
            word_index_dic, pretrained_weights = load_pretrain_embeddings(pretrain_word_emb_path)
            word_indices = torch.tensor([0] + [word_index_dic[x] for x in self.vocab_words[1:]] + [self.word_pad_idx])
            #print(len(word_indices))
            #print(word_indices.cpu().tolist())
            pretrained_weights = torch.FloatTensor(pretrained_weights)
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_weights[word_indices], padding_idx=self.word_pad_idx)
            #vectors of padding idx will not be updated
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, self.embedding_size, padding_idx=self.word_pad_idx)
        if self.args.model_name == "item_transformer":
            self.transformer_encoder = TransformerEncoder(
                    self.embedding_size, args.ff_size, args.heads,
                    args.dropout, args.inter_layers)
        #if self.args.model_name == "ZAM" or self.args.model_name == "AEM":
        else:
            self.attention_encoder = MultiHeadedAttention(args.heads, self.embedding_size, args.dropout)

        if args.query_encoder_name == "fs":
            self.query_encoder = FSEncoder(self.embedding_size, self.emb_dropout)
        else:
            self.query_encoder = AVGEncoder(self.embedding_size, self.emb_dropout)
        self.seg_embeddings = nn.Embedding(4, self.embedding_size, padding_idx=self.seg_pad_idx)
        #for each q,u,i
        #Q, previous purchases of u, current available reviews for i, padding value
        #self.logsoftmax = torch.nn.LogSoftmax(dim = -1)
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean

        self.initialize_parameters(logger) #logger
        self.to(device) #change model in place
        self.item_loss = 0
        self.ps_loss = 0

    def clear_loss(self):
        self.item_loss = 0
        self.ps_loss = 0

    def load_cp(self, pt, strict=True):
        self.load_state_dict(pt['model'], strict=strict)

    def test(self, batch_data):
        if self.args.model_name == "item_transformer":
            if self.args.use_dot_prod:
                return self.test_dotproduct(batch_data)
            else:
                return self.test_trans(batch_data)
        else:
            return self.test_attn(batch_data)

    def test_dotproduct(self, batch_data):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        u_item_idxs = batch_data.u_item_idxs
        candi_prod_idxs = batch_data.candi_prod_idxs
        batch_size, prev_item_count = u_item_idxs.size()
        _, candi_k = candi_prod_idxs.size()
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)
        u_item_mask = u_item_mask.unsqueeze(1).expand(-1,candi_k,-1)
        column_mask = column_mask.unsqueeze(1).expand(-1,candi_k,-1)
        candi_item_seq_mask = torch.cat([column_mask, u_item_mask], dim=2)
        candi_item_emb = self.product_emb(candi_prod_idxs) #batch_size, candi_k, embedding_size
        if self.args.sep_prod_emb:
            u_item_emb = self.hist_product_emb(u_item_idxs)
        else:
            u_item_emb = self.product_emb(u_item_idxs)
        candi_sequence_emb = torch.cat(
                [query_emb.unsqueeze(1).expand(-1, candi_k, -1).unsqueeze(2),
                    u_item_emb.unsqueeze(1).expand(-1, candi_k, -1, -1)],
                    dim=2)

        out_pos = -1 if self.args.use_item_pos else 0
        top_vecs = self.transformer_encoder.encode(
                candi_sequence_emb.view(batch_size*candi_k, prev_item_count+1, -1),
                candi_item_seq_mask.view(batch_size*candi_k, prev_item_count+1),
                use_pos=self.args.use_pos_emb)
        candi_out_emb = top_vecs[:,out_pos,:]
        candi_scores = torch.bmm(candi_out_emb.unsqueeze(1), candi_item_emb.view(batch_size*candi_k, -1).unsqueeze(2))
        candi_scores = candi_scores.view(batch_size, candi_k)
        if self.args.sim_func == "bias_product":
            candi_bias = self.product_bias[candi_prod_idxs.view(-1)].view(batch_size, candi_k)
            candi_scores += candi_bias
        return candi_scores

    def test_attn(self, batch_data):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        u_item_idxs = batch_data.u_item_idxs
        candi_prod_idxs = batch_data.candi_prod_idxs
        batch_size, prev_item_count = u_item_idxs.size()
        _, candi_k = candi_prod_idxs.size()
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        embed_size = query_emb.size()[-1]
        query_emb = query_emb.unsqueeze(1).expand(-1, candi_k, -1).unsqueeze(2).contiguous()
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)
        candi_item_seq_mask = u_item_mask.unsqueeze(1).expand(-1,candi_k,-1)
        candi_item_emb = self.product_emb(candi_prod_idxs) #batch_size, candi_k, embedding_size
        if self.args.sep_prod_emb:
            u_item_emb = self.hist_product_emb(u_item_idxs)
        else:
            u_item_emb = self.product_emb(u_item_idxs)

        candi_sequence_emb = u_item_emb.unsqueeze(1).expand(-1, candi_k, -1, -1)

        if self.args.model_name == "ZAM":
            zero_column = torch.zeros(batch_size, 1, embed_size, device=query_word_idxs.device)
            column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
            column_mask = column_mask.unsqueeze(1).expand(-1,candi_k,-1)
            candi_item_seq_mask = torch.cat([column_mask, candi_item_seq_mask], dim=2)
            pos_sequence_emb = torch.cat([zero_column, u_item_emb], dim=1)
            candi_sequence_emb = torch.cat([zero_column.expand(-1, candi_k, -1).unsqueeze(2),
                        candi_sequence_emb], dim=2)

        candi_item_seq_mask = candi_item_seq_mask.contiguous().view(batch_size*candi_k, 1, -1)
        out_pos = 0
        candi_sequence_emb = candi_sequence_emb.contiguous().view(batch_size*candi_k, -1, embed_size)
        top_vecs = self.attention_encoder(
                candi_sequence_emb, candi_sequence_emb, query_emb,
                mask = 1-candi_item_seq_mask)
        candi_out_emb = top_vecs[:,out_pos,:]
        candi_scores = torch.bmm(candi_out_emb.unsqueeze(1), candi_item_emb.view(batch_size*candi_k, -1).unsqueeze(2))
        candi_scores = candi_scores.view(batch_size, candi_k)

        if self.args.sim_func == "bias_product":
            candi_bias = self.product_bias[candi_prod_idxs.view(-1)].view(batch_size, candi_k)
            candi_scores += candi_bias
        return candi_scores


    def test_trans(self, batch_data):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        u_item_idxs = batch_data.u_item_idxs
        candi_prod_idxs = batch_data.candi_prod_idxs
        batch_size, prev_item_count = u_item_idxs.size()
        _, candi_k = candi_prod_idxs.size()
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)
        u_item_mask = u_item_mask.unsqueeze(1).expand(-1,candi_k,-1)
        column_mask = column_mask.unsqueeze(1).expand(-1,candi_k,-1)
        candi_item_seq_mask = torch.cat([column_mask, u_item_mask, column_mask], dim=2)
        candi_seg_idxs = torch.cat([column_mask*0,
            column_mask.expand(-1, -1, prev_item_count),
            column_mask*2], dim = 2)
        candi_item_emb = self.product_emb(candi_prod_idxs) #batch_size, candi_k, embedding_size
        if self.args.sep_prod_emb:
            u_item_emb = self.hist_product_emb(u_item_idxs)
        else:
            u_item_emb = self.product_emb(u_item_idxs)
        candi_sequence_emb = torch.cat(
                [query_emb.unsqueeze(1).expand(-1, candi_k, -1).unsqueeze(2),
                    u_item_emb.unsqueeze(1).expand(-1, candi_k, -1, -1),
                    candi_item_emb.unsqueeze(2)], dim=2)
        candi_seg_emb = self.seg_embeddings(candi_seg_idxs.long()) #batch_size, candi_k, max_prev_item_count+1, embedding_size
        candi_sequence_emb += candi_seg_emb

        out_pos = -1 if self.args.use_item_pos else 0
        candi_scores = self.transformer_encoder(
                candi_sequence_emb.view(batch_size*candi_k, prev_item_count+2, -1),
                candi_item_seq_mask.view(batch_size*candi_k, prev_item_count+2),
                use_pos=self.args.use_pos_emb, out_pos=out_pos)
        candi_scores = candi_scores.view(batch_size, candi_k)
        return candi_scores

    def test_seq(self, batch_data):
        query_word_idxs = batch_data.query_word_idxs
        candi_seg_idxs = batch_data.neg_seg_idxs
        candi_seq_item_idxs = batch_data.neg_seq_item_idxs
        batch_size, candi_k, prev_item_count = batch_data.neg_seq_item_idxs.size()
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        candi_seq_item_emb = self.product_emb(candi_seq_item_idxs)#batch_size, candi_k, max_prev_item_count, embedding_size
        #concat query_emb with pos_review_emb and candi_review_emb
        query_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        candi_prod_idx_mask = candi_seq_item_idxs.ne(self.prod_pad_idx)
        candi_seq_item_mask = torch.cat([query_mask.unsqueeze(1).expand(-1,candi_k,-1), candi_prod_idx_mask], dim=2)
        #batch_size, 1, embedding_size
        candi_sequence_emb = torch.cat(
                (query_emb.unsqueeze(1).expand(-1, candi_k, -1).unsqueeze(2), candi_seq_item_emb), dim=2)
        #batch_size, candi_k, max_review_count+1, embedding_size
        candi_seg_emb = self.seg_embeddings(candi_seg_idxs) #batch_size, candi_k, max_review_count+1, embedding_size
        candi_sequence_emb += candi_seg_emb

        candi_scores = self.transformer_encoder(
                candi_sequence_emb.view(batch_size*candi_k, prev_item_count+1, -1),
                candi_seq_item_mask.view(batch_size*candi_k, prev_item_count+1))
        candi_scores = candi_scores.view(batch_size, candi_k)
        return candi_scores

    def item_to_words(self, target_prod_idxs, target_word_idxs, n_negs):
        batch_size, pv_window_size = target_word_idxs.size()
        prod_emb = self.product_emb(target_prod_idxs)
        target_word_emb = self.word_embeddings(target_word_idxs)

        #for each target word, there is k words negative sampling
        #vocab_size = self.word_embeddings.weight.size() - 1
        #compute the loss of review generating positive and negative words
        neg_sample_idxs = torch.multinomial(self.word_dists, batch_size * pv_window_size * n_negs, replacement=True)
        neg_sample_emb = self.word_embeddings(neg_sample_idxs.view(batch_size,-1))
        output_pos = torch.bmm(target_word_emb, prod_emb.unsqueeze(2)) # batch_size, pv_window_size, 1
        output_neg = torch.bmm(neg_sample_emb, prod_emb.unsqueeze(2)).view(batch_size, pv_window_size, -1)
        pos_bias = self.word_bias[target_word_idxs.view(-1)].view(batch_size, pv_window_size, 1)
        neg_bias = self.word_bias[neg_sample_idxs].view(batch_size, pv_window_size, -1)
        output_pos += pos_bias
        output_neg += neg_bias

        scores = torch.cat((output_pos, output_neg), dim=-1) #batch_size, pv_window_size, 1+n_negs
        target = torch.cat((torch.ones(output_pos.size(), device=scores.device),
            torch.zeros(output_neg.size(), device=scores.device)), dim=-1)
        loss = self.bce_logits_loss(scores, target).sum(-1) #batch_size, pv_window_size
        loss = get_vector_mean(loss.unsqueeze(-1), target_word_idxs.ne(self.word_pad_idx))
        loss = loss.mean()
        return loss

    def forward_trans(self, batch_data, train_pv=False):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        u_item_idxs = batch_data.u_item_idxs
        batch_size, prev_item_count = u_item_idxs.size()
        neg_k = self.args.neg_per_pos
        pos_iword_idxs = batch_data.pos_iword_idxs
        neg_item_idxs = torch.multinomial(self.prod_dists, batch_size * neg_k, replacement=True)
        neg_item_idxs = neg_item_idxs.view(batch_size, -1)
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)
        pos_item_seq_mask = torch.cat([column_mask, u_item_mask, column_mask], dim=1) #batch_size, 1+max_review_count

        pos_seg_idxs = torch.cat(
                [column_mask*0, column_mask.expand(-1, prev_item_count), column_mask*2], dim=1)
        column_mask = column_mask.unsqueeze(1).expand(-1,neg_k,-1)
        neg_item_seq_mask = torch.cat([column_mask, u_item_mask.unsqueeze(1).expand(-1,neg_k,-1), column_mask], dim=2)
        neg_seg_idxs = torch.cat([column_mask*0,
            column_mask.expand(-1, -1, prev_item_count),
            column_mask*2], dim = 2)
        target_item_emb = self.product_emb(target_prod_idxs)
        neg_item_emb = self.product_emb(neg_item_idxs) #batch_size, neg_k, embedding_size
        if self.args.sep_prod_emb:
            u_item_emb = self.hist_product_emb(u_item_idxs)
        else:
            u_item_emb = self.product_emb(u_item_idxs)
        pos_sequence_emb = torch.cat([query_emb.unsqueeze(1), u_item_emb, target_item_emb.unsqueeze(1)], dim=1)
        pos_seg_emb = self.seg_embeddings(pos_seg_idxs.long())
        neg_sequence_emb = torch.cat(
                [query_emb.unsqueeze(1).expand(-1, neg_k, -1).unsqueeze(2),
                    u_item_emb.unsqueeze(1).expand(-1, neg_k, -1, -1),
                    neg_item_emb.unsqueeze(2)], dim=2)
        neg_seg_emb = self.seg_embeddings(neg_seg_idxs.long()) #batch_size, neg_k, max_prev_item_count+1, embedding_size
        pos_sequence_emb += pos_seg_emb
        neg_sequence_emb += neg_seg_emb

        out_pos = -1 if self.args.use_item_pos else 0
        pos_scores = self.transformer_encoder(pos_sequence_emb, pos_item_seq_mask, use_pos=self.args.use_pos_emb, out_pos=out_pos)

        neg_scores = self.transformer_encoder(
                neg_sequence_emb.view(batch_size*neg_k, prev_item_count+2, -1),
                neg_item_seq_mask.view(batch_size*neg_k, prev_item_count+2), use_pos=self.args.use_pos_emb, out_pos=out_pos)
        neg_scores = neg_scores.view(batch_size, neg_k)
        pos_weight = 1
        if self.args.pos_weight:
            pos_weight = self.args.neg_per_pos
        prod_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device) * pos_weight,
                        torch.ones(batch_size, neg_k, dtype=torch.uint8, device=query_word_idxs.device)], dim=-1)
        prod_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        target = torch.cat([torch.ones(batch_size, 1, device=query_word_idxs.device),
            torch.zeros(batch_size, neg_k, device=query_word_idxs.device)], dim=-1)
        #ps_loss = self.bce_logits_loss(prod_scores, target, weight=prod_mask.float())
        #for all positive items, there are neg_k negative items
        ps_loss = nn.functional.binary_cross_entropy_with_logits(
                prod_scores, target,
                weight=prod_mask.float(),
                reduction='none')
        ps_loss = ps_loss.sum(-1).mean()
        item_loss = self.item_to_words(target_prod_idxs, pos_iword_idxs, self.args.neg_per_pos)
        self.ps_loss += ps_loss.item()
        self.item_loss += item_loss.item()
        #logger.info("ps_loss:{} item_loss:{}".format(, item_loss.item()))

        return ps_loss + item_loss

    def forward(self, batch_data, train_pv=False):
        if self.args.model_name == "item_transformer":
            if self.args.use_dot_prod:
                return self.forward_dotproduct(batch_data)
            else:
                return self.forward_trans(batch_data)
        else:
            return self.forward_attn(batch_data)

    def forward_attn(self, batch_data, train_pv=False):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        u_item_idxs = batch_data.u_item_idxs
        batch_size, prev_item_count = u_item_idxs.size()
        neg_k = self.args.neg_per_pos
        pos_iword_idxs = batch_data.pos_iword_idxs
        neg_item_idxs = torch.multinomial(self.prod_dists, batch_size * neg_k, replacement=True)
        neg_item_idxs = neg_item_idxs.view(batch_size, -1)
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        embed_size = query_emb.size()[-1]
        query_emb = query_emb.unsqueeze(1)
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)
        target_item_emb = self.product_emb(target_prod_idxs)
        neg_item_emb = self.product_emb(neg_item_idxs) #batch_size, neg_k, embedding_size
        if self.args.sep_prod_emb:
            u_item_emb = self.hist_product_emb(u_item_idxs)
        else:
            u_item_emb = self.product_emb(u_item_idxs)
        pos_sequence_emb = u_item_emb
        neg_sequence_emb = u_item_emb.unsqueeze(1).expand(-1, neg_k, -1, -1)
        pos_item_seq_mask = u_item_mask
        neg_item_seq_mask = u_item_mask.unsqueeze(1).expand(-1,neg_k,-1)
        if self.args.model_name == "ZAM":
            zero_column = torch.zeros(batch_size, 1, embed_size, device=query_word_idxs.device)
            column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
            pos_item_seq_mask = torch.cat([column_mask, u_item_mask], dim=1) #batch_size, 1+max_review_count
            column_mask = column_mask.unsqueeze(1).expand(-1,neg_k,-1)
            neg_item_seq_mask = torch.cat([column_mask, u_item_mask.unsqueeze(1).expand(-1,neg_k,-1)], dim=2)
            pos_sequence_emb = torch.cat([zero_column, u_item_emb], dim=1)
            neg_sequence_emb = torch.cat([zero_column.expand(-1, neg_k, -1).unsqueeze(2),
                        neg_sequence_emb], dim=2)

        pos_item_seq_mask = pos_item_seq_mask.unsqueeze(1)
        neg_item_seq_mask = neg_item_seq_mask.contiguous().view(batch_size*neg_k, 1, -1)
        out_pos = 0
        top_vecs = self.attention_encoder(pos_sequence_emb, pos_sequence_emb, query_emb, mask=1-pos_item_seq_mask)
        pos_out_emb = top_vecs[:,out_pos,:] #batch_size, embedding_size
        pos_scores = torch.bmm(pos_out_emb.unsqueeze(1), target_item_emb.unsqueeze(2)).squeeze()
        neg_sequence_emb = neg_sequence_emb.contiguous().view(batch_size*neg_k, -1, embed_size)
        query_emb = query_emb.expand(-1, neg_k, -1).unsqueeze(2).contiguous()
        top_vecs = self.attention_encoder(
                neg_sequence_emb, neg_sequence_emb, query_emb,
                mask = 1-neg_item_seq_mask)
        neg_out_emb = top_vecs[:,out_pos,:]
        neg_scores = torch.bmm(neg_out_emb.unsqueeze(1), neg_item_emb.view(batch_size*neg_k, -1).unsqueeze(2))
        neg_scores = neg_scores.view(batch_size, neg_k)
        if self.args.sim_func == "bias_product":
            pos_bias = self.product_bias[target_prod_idxs.view(-1)].view(batch_size)
            neg_bias = self.product_bias[neg_item_idxs.view(-1)].view(batch_size, neg_k)
            pos_scores += pos_bias
            neg_scores += neg_bias
        pos_weight = 1
        if self.args.pos_weight:
            pos_weight = self.args.neg_per_pos
        prod_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device) * pos_weight,
                        torch.ones(batch_size, neg_k, dtype=torch.uint8, device=query_word_idxs.device)], dim=-1)
        prod_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        target = torch.cat([torch.ones(batch_size, 1, device=query_word_idxs.device),
            torch.zeros(batch_size, neg_k, device=query_word_idxs.device)], dim=-1)
        #ps_loss = self.bce_logits_loss(prod_scores, target, weight=prod_mask.float())
        #for all positive items, there are neg_k negative items
        ps_loss = nn.functional.binary_cross_entropy_with_logits(
                prod_scores, target,
                weight=prod_mask.float(),
                reduction='none')
        ps_loss = ps_loss.sum(-1).mean()
        item_loss = self.item_to_words(target_prod_idxs, pos_iword_idxs, self.args.neg_per_pos)
        self.ps_loss += ps_loss.item()
        self.item_loss += item_loss.item()
        #logger.info("ps_loss:{} item_loss:{}".format(, item_loss.item()))

        return ps_loss + item_loss

    def forward_dotproduct(self, batch_data, train_pv=False):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        u_item_idxs = batch_data.u_item_idxs
        batch_size, prev_item_count = u_item_idxs.size()
        neg_k = self.args.neg_per_pos
        pos_iword_idxs = batch_data.pos_iword_idxs
        neg_item_idxs = torch.multinomial(self.prod_dists, batch_size * neg_k, replacement=True)
        neg_item_idxs = neg_item_idxs.view(batch_size, -1)
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        column_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        u_item_mask = u_item_idxs.ne(self.prod_pad_idx)
        #pos_item_seq_mask = torch.cat([column_mask, u_item_mask, column_mask], dim=1) #batch_size, 1+max_review_count
        pos_item_seq_mask = torch.cat([column_mask, u_item_mask], dim=1) #batch_size, 1+max_review_count

        #pos_seg_idxs = torch.cat(
        #        [column_mask*0, column_mask.expand(-1, prev_item_count), column_mask*2], dim=1)
        column_mask = column_mask.unsqueeze(1).expand(-1,neg_k,-1)
        #neg_item_seq_mask = torch.cat([column_mask, u_item_mask.unsqueeze(1).expand(-1,neg_k,-1), column_mask], dim=2)
        neg_item_seq_mask = torch.cat([column_mask, u_item_mask.unsqueeze(1).expand(-1,neg_k,-1)], dim=2)
        #neg_seg_idxs = torch.cat([column_mask*0,
        #    column_mask.expand(-1, -1, prev_item_count),
        #    column_mask*2], dim = 2)
        target_item_emb = self.product_emb(target_prod_idxs)
        neg_item_emb = self.product_emb(neg_item_idxs) #batch_size, neg_k, embedding_size
        if self.args.sep_prod_emb:
            u_item_emb = self.hist_product_emb(u_item_idxs)
        else:
            u_item_emb = self.product_emb(u_item_idxs)
        #pos_sequence_emb = torch.cat([query_emb.unsqueeze(1), u_item_emb, target_item_emb.unsqueeze(1)], dim=1)
        pos_sequence_emb = torch.cat([query_emb.unsqueeze(1), u_item_emb], dim=1)
        #pos_seg_emb = self.seg_embeddings(pos_seg_idxs.long())
        neg_sequence_emb = torch.cat(
                [query_emb.unsqueeze(1).expand(-1, neg_k, -1).unsqueeze(2),
                    u_item_emb.unsqueeze(1).expand(-1, neg_k, -1, -1),
                    ], dim=2)
                    #neg_item_emb.unsqueeze(2)], dim=2)
        #neg_seg_emb = self.seg_embeddings(neg_seg_idxs.long()) #batch_size, neg_k, max_prev_item_count+1, embedding_size
        #pos_sequence_emb += pos_seg_emb
        #neg_sequence_emb += neg_seg_emb

        out_pos = -1 if self.args.use_item_pos else 0
        top_vecs = self.transformer_encoder.encode(pos_sequence_emb, pos_item_seq_mask, use_pos=self.args.use_pos_emb)
        pos_out_emb = top_vecs[:,out_pos,:] #batch_size, embedding_size
        pos_scores = torch.bmm(pos_out_emb.unsqueeze(1), target_item_emb.unsqueeze(2)).squeeze()
        top_vecs = self.transformer_encoder.encode(
                #neg_sequence_emb.view(batch_size*neg_k, prev_item_count+2, -1),
                #neg_item_seq_mask.view(batch_size*neg_k, prev_item_count+2),
                neg_sequence_emb.view(batch_size*neg_k, prev_item_count+1, -1),
                neg_item_seq_mask.view(batch_size*neg_k, prev_item_count+1),
                use_pos=self.args.use_pos_emb)
        neg_out_emb = top_vecs[:,out_pos,:]
        neg_scores = torch.bmm(neg_out_emb.unsqueeze(1), neg_item_emb.view(batch_size*neg_k, -1).unsqueeze(2))
        neg_scores = neg_scores.view(batch_size, neg_k)
        if self.args.sim_func == "bias_product":
            pos_bias = self.product_bias[target_prod_idxs.view(-1)].view(batch_size)
            neg_bias = self.product_bias[neg_item_idxs.view(-1)].view(batch_size, neg_k)
            pos_scores += pos_bias
            neg_scores += neg_bias
        pos_weight = 1
        if self.args.pos_weight:
            pos_weight = self.args.neg_per_pos
        prod_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device) * pos_weight,
                        torch.ones(batch_size, neg_k, dtype=torch.uint8, device=query_word_idxs.device)], dim=-1)
        prod_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        target = torch.cat([torch.ones(batch_size, 1, device=query_word_idxs.device),
            torch.zeros(batch_size, neg_k, device=query_word_idxs.device)], dim=-1)
        #ps_loss = self.bce_logits_loss(prod_scores, target, weight=prod_mask.float())
        #for all positive items, there are neg_k negative items
        ps_loss = nn.functional.binary_cross_entropy_with_logits(
                prod_scores, target,
                weight=prod_mask.float(),
                reduction='none')
        ps_loss = ps_loss.sum(-1).mean()
        item_loss = self.item_to_words(target_prod_idxs, pos_iword_idxs, self.args.neg_per_pos)
        self.ps_loss += ps_loss.item()
        self.item_loss += item_loss.item()
        #logger.info("ps_loss:{} item_loss:{}".format(, item_loss.item()))

        return ps_loss + item_loss

    def forward_seq(self, batch_data, train_pv=False):
        query_word_idxs = batch_data.query_word_idxs
        target_prod_idxs = batch_data.target_prod_idxs
        pos_seg_idxs = batch_data.pos_seg_idxs
        neg_seg_idxs = batch_data.neg_seg_idxs
        pos_seq_item_idxs = batch_data.pos_seq_item_idxs
        neg_seq_item_idxs = batch_data.neg_seq_item_idxs
        pos_iword_idxs = batch_data.pos_iword_idxs
        batch_size, neg_k, prev_item_count = neg_seq_item_idxs.size()
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        query_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        pos_seq_item_mask = torch.cat([query_mask, pos_seq_item_idxs.ne(self.prod_pad_idx)], dim=1) #batch_size, 1+max_review_count
        neg_prod_idx_mask = neg_seq_item_idxs.ne(self.prod_pad_idx)
        neg_seq_item_mask = torch.cat([query_mask.unsqueeze(1).expand(-1,neg_k,-1), neg_prod_idx_mask], dim=2)
        #batch_size, 1, embedding_size
        pos_seq_item_emb = self.product_emb(pos_seq_item_idxs)
        pos_sequence_emb = torch.cat((query_emb.unsqueeze(1), pos_seq_item_emb), dim=1)
        pos_seg_emb = self.seg_embeddings(pos_seg_idxs)
        neg_seq_item_emb = self.product_emb(neg_seq_item_idxs)#batch_size, neg_k, max_prev_item_count, embedding_size
        neg_sequence_emb = torch.cat(
                (query_emb.unsqueeze(1).expand(-1, neg_k, -1).unsqueeze(2), neg_seq_item_emb), dim=2)
        #batch_size, neg_k, max_review_count+1, embedding_size
        neg_seg_emb = self.seg_embeddings(neg_seg_idxs) #batch_size, neg_k, max_prev_item_count+1, embedding_size
        pos_sequence_emb += pos_seg_emb
        neg_sequence_emb += neg_seg_emb

        pos_scores = self.transformer_encoder(pos_sequence_emb, pos_seq_item_mask, use_pos=self.args.use_pos_emb)
        neg_scores = self.transformer_encoder(
                neg_sequence_emb.view(batch_size*neg_k, prev_item_count+1, -1),
                neg_seq_item_mask.view(batch_size*neg_k, prev_item_count+1), use_pos=self.args.use_pos_emb)
        neg_scores = neg_scores.view(batch_size, neg_k)
        pos_weight = 1
        if self.args.pos_weight:
            pos_weight = self.args.neg_per_pos
        prod_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device) * pos_weight,
                        torch.ones(batch_size, neg_k, dtype=torch.uint8, device=query_word_idxs.device)], dim=-1)
        prod_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        target = torch.cat([torch.ones(batch_size, 1, device=query_word_idxs.device),
            torch.zeros(batch_size, neg_k, device=query_word_idxs.device)], dim=-1)
        #ps_loss = self.bce_logits_loss(prod_scores, target, weight=prod_mask.float())
        #for all positive items, there are neg_k negative items
        ps_loss = nn.functional.binary_cross_entropy_with_logits(
                prod_scores, target,
                weight=prod_mask.float(),
                reduction='none')
        ps_loss = ps_loss.sum(-1).mean()
        item_loss = self.item_to_words(target_prod_idxs, pos_iword_idxs, self.args.neg_per_pos)
        self.ps_loss += ps_loss.item()
        self.item_loss += item_loss.item()
        #logger.info("ps_loss:{} item_loss:{}".format(, item_loss.item()))

        return ps_loss + item_loss

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" ItemTransformerRanker initialization started.")
        if self.pretrain_emb_dir is None:
            nn.init.normal_(self.word_embeddings.weight)
        nn.init.normal_(self.seg_embeddings.weight)
        self.query_encoder.initialize_parameters(logger)
        if self.args.model_name == "item_transformer":
            self.transformer_encoder.initialize_parameters(logger)
        if logger:
            logger.info(" ItemTransformerRanker initialization finished.")

