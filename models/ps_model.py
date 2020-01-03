""" transformer based on reviews
    Q+r_{u1}+r_{u2} <> r_1, r_2 (of a target i)
"""
"""
review_encoder
query_encoder
transformer
"""
import torch
import torch.nn as nn
from models.PV import ParagraphVector
from models.PVC import ParagraphVectorCorruption
from models.text_encoder import AVGEncoder, FSEncoder
from models.transformer import TransformerEncoder
from models.optimizers import Optimizer
from others.logging import logger

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '' and checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '' and checkpoint is not None:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.device == "cuda":
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim

class ProductRanker(nn.Module):
    def __init__(self, args, device, vocab_size, review_count, word_dists=None):
        super(ProductRanker, self).__init__()
        self.args = args
        self.device = device
        self.learn_neg = args.learn_neg
        self.embedding_size = args.embedding_size
        self.word_dists = torch.tensor(word_dists).to(device)
        self.word_pad_idx = vocab_size-1
        self.seg_pad_idx = 3
        self.review_pad_idx = review_count-1
        self.word_embeddings = nn.Embedding(
            vocab_size, self.embedding_size, padding_idx=self.word_pad_idx)
        self.transformer_encoder = TransformerEncoder(
                self.embedding_size, args.ff_size, args.heads,
                args.dropout, args.inter_layers)
        self.review_encoder_name = args.review_encoder_name

        if args.review_encoder_name == "pv":
            self.review_encoder = ParagraphVector(
                    self.word_embeddings, self.word_dists,
                    review_count, args.dropout)
        elif args.review_encoder_name == "pvc":
            self.review_encoder = ParagraphVectorCorruption(
                    self.word_embeddings, self.word_dists, args.corrupt_rate, args.dropout)
        elif args.review_encoder_name == "fs":
            self.review_encoder = FSEncoder(self.embedding_size, args.dropout)
        else:
            self.review_encoder = AVGEncoder(self.embedding_size, args.dropout)

        if args.query_encoder_name == "fs":
            self.query_encoder = FSEncoder(self.embedding_size, args.dropout)
        else:
            self.query_encoder = AVGEncoder(self.embedding_size, args.dropout)
        self.seg_embeddings = nn.Embedding(4, self.embedding_size, padding_idx=self.seg_pad_idx)
        #for each q,u,i
        #Q, previous purchases of u, current available reviews for i, padding value
        #self.logsoftmax = torch.nn.LogSoftmax(dim = -1)

        self.initialize_parameters(logger) #logger
        self.to(device)

    def load_cp(self, pt, strict=True):
        self.load_state_dict(pt['model'], strict=strict)

    def forward(self, batch_data):
        query_word_idxs = batch_data.query_word_idxs
        pos_prod_ridxs = batch_data.pos_prod_ridxs
        pos_seg_idxs = batch_data.pos_seg_idxs
        pos_prod_rword_idxs= batch_data.pos_prod_rword_idxs
        pos_prod_rword_masks = batch_data.pos_prod_rword_masks
        neg_prod_ridxs = batch_data.neg_prod_ridxs
        neg_seg_idxs = batch_data.neg_seg_idxs
        neg_prod_rword_idxs = batch_data.neg_prod_rword_idxs
        neg_prod_rword_masks = batch_data.neg_prod_rword_masks
        #rand_pos_prod_rword_idxs = batch_data.rand_pos_prod_rword_idxs
        #rand_neg_prod_rword_idxs = batch_data.rand_neg_prod_rword_idxs
        pos_prod_rword_idxs_pvc = batch_data.pos_prod_rword_idxs_pvc
        neg_prod_rword_idxs_pvc = batch_data.neg_prod_rword_idxs_pvc
        #u, Q, i (positive, negative)
        #Q; ru1,ru2,ri1,ri2 and k negative (ru1,ru2,rn1i1,rn1i2; ru1,ru2,rnji1,rnji2)
        #segs 0; 1,1;#pos 2,2, -1,-1 #neg_1, neg_2
        #r: word_id1, word_id2, ...
        #pos_seg_idxs:0,1,1,2,2,-1
        #word_count can be computed with words that are not padding
        #batch_size, query_term_count
        #print(query_word_idxs)
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        #review of u concat with review of i
        #review of u concat with review of each negative i
        # batch_size, review_count (u+i), max_word_count_per_review
        # batch_size, neg_k, review_count (u+i), max_word_count_per_review
        max_pos_review_count = pos_prod_ridxs.size(1)
        batch_size, neg_k, max_neg_review_count, max_rword_count = neg_prod_rword_idxs.size()
        update_posr_word_idxs = pos_prod_rword_idxs.view(-1, max_rword_count)
        update_negr_word_idxs = neg_prod_rword_idxs.view(-1, max_rword_count)
        update_pos_prod_rword_masks = pos_prod_rword_masks.view(-1, max_rword_count)
        update_neg_prod_rword_masks = neg_prod_rword_masks.view(-1, max_rword_count)
        #update_negr_word_count = update_negr_word_idxs.ne(-1).sum(dim=-1)
        posr_word_emb = self.word_embeddings(update_posr_word_idxs)
        negr_word_emb = self.word_embeddings(update_negr_word_idxs)
        pv_loss = None
        if "pv" in self.review_encoder_name:
            if self.review_encoder_name == "pv":
                pos_review_emb, pos_prod_loss = self.review_encoder(
                        pos_prod_ridxs.view(-1), posr_word_emb,
                        update_pos_prod_rword_masks, self.args.neg_per_pos)
                if self.learn_neg:
                    neg_review_emb, neg_prod_loss = self.review_encoder(
                            neg_prod_ridxs.view(-1), negr_word_emb,
                            update_neg_prod_rword_masks, self.args.neg_per_pos)
            elif self.review_encoder_name == "pvc":
                review_word_limit = pos_prod_rword_idxs_pvc.size()[-1]
                pos_review_emb, pos_prod_loss = self.review_encoder(
                        posr_word_emb, update_pos_prod_rword_masks,
                        pos_prod_rword_idxs_pvc.view(-1, review_word_limit),
                        self.args.neg_per_pos)
                if self.learn_neg:
                    neg_review_emb, neg_prod_loss = self.review_encoder(
                            negr_word_emb, update_neg_prod_rword_masks,
                            neg_prod_rword_idxs_pvc.view(-1, review_word_limit),
                            self.args.neg_per_pos)

            sample_count = pos_prod_ridxs.ne(self.review_pad_idx).float().sum() \
                    + neg_prod_ridxs.ne(self.review_pad_idx).float().sum()
            sample_count = sample_count.masked_fill(sample_count.eq(0),1)
            pv_loss = (pos_prod_loss.sum() + neg_prod_loss.sum()) / sample_count

        else:
            pos_review_emb = self.review_encoder(posr_word_emb, update_pos_prod_rword_masks)
            neg_review_emb = self.review_encoder(negr_word_emb, update_neg_prod_rword_masks)
        pos_review_emb = pos_review_emb.view(batch_size, max_pos_review_count, -1)
        neg_review_emb = neg_review_emb.view(batch_size, neg_k, max_neg_review_count, -1)
        #concat query_emb with pos_review_emb and neg_review_emb
        query_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        pos_review_mask = torch.cat([query_mask, pos_prod_ridxs.ne(self.review_pad_idx)], dim=1) #batch_size, 1+max_review_count
        neg_prod_ridx_mask = neg_prod_ridxs.ne(self.review_pad_idx)
        neg_review_mask = torch.cat([query_mask.unsqueeze(1).expand(-1,neg_k,-1), neg_prod_ridx_mask], dim=2)
        neg_prod_mask = neg_prod_ridx_mask.sum(-1).ne(0) #batch_size, neg_k (valid products, some are padded)
        #batch_size, 1, embedding_size
        pos_sequence_emb = torch.cat((query_emb.unsqueeze(1), pos_review_emb), dim=1)
        pos_seg_emb = self.seg_embeddings(pos_seg_idxs) #batch_size, max_review_count+1, embedding_size
        neg_sequence_emb = torch.cat(
                (query_emb.unsqueeze(1).expand(-1, neg_k, -1).unsqueeze(2), neg_review_emb), dim=2)
        #batch_size, neg_k, max_review_count+1, embedding_size
        neg_seg_emb = self.seg_embeddings(neg_seg_idxs) #batch_size, neg_k, max_review_count+1, embedding_size
        pos_sequence_emb += pos_seg_emb
        neg_sequence_emb += neg_seg_emb

        pos_scores = self.transformer_encoder(pos_sequence_emb, pos_review_mask)
        neg_scores = self.transformer_encoder(
                neg_sequence_emb.view(batch_size*neg_k, max_neg_review_count+1, -1),
                neg_review_mask.view(batch_size*neg_k, max_neg_review_count+1))
        neg_scores = neg_scores.view(batch_size, neg_k)
        #print(pos_scores)
        #print(neg_scores)
        oloss = pos_scores.sigmoid().log()
        nloss = neg_scores.neg().sigmoid().log() * neg_prod_mask.float()
        ps_loss = (-oloss + nloss.sum(1)).mean()

        #prod_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        #labels = torch.cat([torch.ones(batch_size, 1, device=query_word_idxs.device),
        #    torch.zeros(batch_size, neg_k, device=query_word_idxs.device)], dim=-1)
        #ps_loss = -self.logsoftmax(prod_scores) * labels.float()
        #ps_loss = ps_loss.sum().item()
        loss = ps_loss
        if pv_loss is not None:
            loss = ps_loss + pv_loss
        return loss

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" ProductRanker initialization started.")
        nn.init.normal_(self.word_embeddings.weight)
        nn.init.normal_(self.seg_embeddings.weight)
        self.review_encoder.initialize_parameters(logger)
        self.query_encoder.initialize_parameters(logger)
        self.transformer_encoder.initialize_parameters(logger)
        if logger:
            logger.info(" ProductRanker initialization finished.")

