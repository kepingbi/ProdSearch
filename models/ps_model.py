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
        #self.start_decay_steps take effect when decay_method is not noam

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
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean

        self.initialize_parameters(logger) #logger
        self.to(device)

    def load_cp(self, pt, strict=True):
        self.load_state_dict(pt['model'], strict=strict)

    def forward(self, batch_data_arr):
        loss = []
        for batch_data in batch_data_arr:
            cur_loss = self.pass_one_batch(batch_data)
            loss.append(cur_loss)
        return sum(loss) / len(loss)

    def pass_one_batch(self, batch_data):
        query_word_idxs = batch_data.query_word_idxs
        pos_prod_ridxs = batch_data.pos_prod_ridxs
        pos_seg_idxs = batch_data.pos_seg_idxs
        pos_prod_rword_idxs= batch_data.pos_prod_rword_idxs
        pos_prod_rword_masks = batch_data.pos_prod_rword_masks
        neg_prod_ridxs = batch_data.neg_prod_ridxs
        neg_seg_idxs = batch_data.neg_seg_idxs
        neg_prod_rword_idxs = batch_data.neg_prod_rword_idxs
        neg_prod_rword_masks = batch_data.neg_prod_rword_masks
        pos_prod_rword_idxs_pvc = batch_data.pos_prod_rword_idxs_pvc
        neg_prod_rword_idxs_pvc = batch_data.neg_prod_rword_idxs_pvc
        query_word_emb = self.word_embeddings(query_word_idxs)
        query_emb = self.query_encoder(query_word_emb, query_word_idxs.ne(self.word_pad_idx))
        batch_size, pos_rcount, posr_word_limit = pos_prod_rword_idxs.size()
        _, neg_k, neg_rcount = neg_prod_ridxs.size()
        posr_word_emb = self.word_embeddings(pos_prod_rword_idxs.view(-1, posr_word_limit))
        update_pos_prod_rword_masks = pos_prod_rword_masks.view(-1, posr_word_limit)
        pv_loss = None
        if "pv" in self.review_encoder_name:
            if self.review_encoder_name == "pv":
                pos_review_emb, pos_prod_loss = self.review_encoder(
                        pos_prod_ridxs.view(-1), posr_word_emb,
                        update_pos_prod_rword_masks, self.args.neg_per_pos)
                neg_review_emb = self.review_encoder.get_para_vector(neg_prod_ridxs)
            elif self.review_encoder_name == "pvc":
                pos_review_emb, pos_prod_loss = self.review_encoder(
                        posr_word_emb, update_pos_prod_rword_masks,
                        pos_prod_rword_idxs_pvc.view(-1, pos_prod_rword_idxs_pvc.size(-1)),
                        self.args.neg_per_pos)
                neg_review_emb = self.review_encoder.get_para_vector(
                        neg_prod_rword_idxs_pvc.view(-1, neg_prod_rword_idxs_pvc.size(-1)))
                neg_review_emb = neg_review_emb.view(batch_size, neg_k, neg_rcount, -1)

            sample_count = pos_prod_ridxs.ne(self.review_pad_idx).float().sum()
            sample_count = sample_count.masked_fill(sample_count.eq(0),1)
            pv_loss = pos_prod_loss.sum() / sample_count
        else:
            negr_word_limit = neg_prod_rword_idxs.size()[-1]
            negr_word_emb = self.word_embeddings(neg_prod_rword_idxs.view(-1, negr_word_limit))
            pos_review_emb = self.review_encoder(posr_word_emb, update_pos_prod_rword_masks)
            neg_review_emb = self.review_encoder(negr_word_emb, neg_prod_rword_masks.view(-1, negr_word_limit))
            neg_review_emb = neg_review_emb.view(batch_size, neg_k, neg_rcount, -1)

        pos_review_emb = pos_review_emb.view(batch_size, pos_rcount, -1)
        #concat query_emb with pos_review_emb and neg_review_emb
        query_mask = torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device)
        pos_review_mask = torch.cat([query_mask, pos_prod_ridxs.ne(self.review_pad_idx)], dim=1) #batch_size, 1+max_review_count
        neg_prod_ridx_mask = neg_prod_ridxs.ne(self.review_pad_idx)
        neg_review_mask = torch.cat([query_mask.unsqueeze(1).expand(-1,neg_k,-1), neg_prod_ridx_mask], dim=2)
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
                neg_sequence_emb.view(batch_size*neg_k, neg_rcount+1, -1),
                neg_review_mask.view(batch_size*neg_k, neg_rcount+1))
        neg_scores = neg_scores.view(batch_size, neg_k)
        prod_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.uint8, device=query_word_idxs.device),
            neg_prod_ridx_mask.sum(-1).ne(0)], dim=-1) #batch_size, neg_k (valid products, some are padded)
        prod_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        target = torch.cat([torch.ones(batch_size, 1, device=query_word_idxs.device),
            torch.zeros(batch_size, neg_k, device=query_word_idxs.device)], dim=-1)
        ps_loss = self.bce_logits_loss(prod_scores, target) * prod_mask.float()
        ps_loss = ps_loss.sum(-1).mean()
        loss = ps_loss + pv_loss if pv_loss is not None else ps_loss
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

