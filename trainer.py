from tqdm import tqdm
from others.logging import logger
#from data.prod_search_dataloader import ProdSearchDataloader
#from data.prod_search_dataset import ProdSearchDataset
import shutil
import torch
import numpy as np
import data
import os
import time
import sys

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

class Trainer(object):
    """
    Class that controls the training process.
    """
    def __init__(self,  args, model,  optim):
        # Basic attributes.
        self.args = args
        self.model = model
        self.optim = optim
        if (model):
            n_params = _tally_parameters(model)
            logger.info('* number of parameters: %d' % n_params)
        #self.device = "cpu" if self.n_gpu == 0 else "cuda"

    def train(self, args, global_data, train_prod_data, valid_prod_data=None):
    #def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        """
        logger.info('Start training...')
        # Set model in training mode.
        #model_dir = "%s/model" % (args.save_dir)
        model_dir = args.save_dir
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        valid_dataset = data.ProdSearchDataset(args, global_data, valid_prod_data)
        step_time, loss = 0.,0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        best_mrr = 0.
        best_checkpoint_path = ''
        for current_epoch in range(args.start_epoch+1, args.max_train_epoch+1):
            self.model.train()
            logger.info("Initialize epoch:%d" % current_epoch)
            train_prod_data.initialize_epoch()
            dataset = data.ProdSearchDataset(args, global_data, train_prod_data)
            prepare_pv = current_epoch < args.train_pv_epoch+1
            print(prepare_pv)
            dataloader = data.ProdSearchDataLoader(
                    args, dataset, prepare_pv=prepare_pv, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers)
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(current_epoch))
            time_flag = time.time()
            for batch_data_arr in pbar:
                batch_data_arr = [x.to(args.device) for x in batch_data_arr]
                get_batch_time += time.time() - time_flag
                time_flag = time.time()
                step_loss = self.model(batch_data_arr, train_pv=prepare_pv)
                #self.optim.optimizer.zero_grad()
                self.model.zero_grad()
                step_loss.backward()
                self.optim.step()
                step_loss = step_loss.item()
                pbar.set_postfix(step_loss=step_loss, lr=self.optim.learning_rate)
                loss += step_loss / args.steps_per_checkpoint #convert an tensor with dim 0 to value
                current_step += 1
                step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % args.steps_per_checkpoint == 0:
                    logger.info("Epoch %d lr = %5.6f loss = %6.2f time %.2f prepare_time %.2f step_time %.2f" %
                            (current_epoch, self.optim.learning_rate, loss,
                                time.time()-start_time, get_batch_time, step_time))#, end=""
                    step_time, get_batch_time, loss = 0., 0.,0.
                    sys.stdout.flush()
                    start_time = time.time()
            #save model after each epoch
            checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % current_epoch)
            self._save(current_epoch, checkpoint_path)
            mrr, prec = self.validate(args, global_data, valid_dataset)
            logger.info("Epoch {}: MRR:{} P@1:{}".format(current_epoch, mrr, prec))
            if mrr > best_mrr:
                best_mrr = mrr
                best_checkpoint_path = os.path.join(model_dir, 'model_best.ckpt')
                logger.info("Copying %s to checkpoint %s" % (checkpoint_path, best_checkpoint_path))
                shutil.copyfile(checkpoint_path, best_checkpoint_path)
        return best_checkpoint_path

    def _save(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'opt': self.args,
            'optim': self.optim,
        }
        #model_dir = "%s/model" % (self.args.save_dir)
        #checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % epoch)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    def validate(self, args, global_data, valid_dataset):
        """ Validate model.
        """
        candidate_size = args.valid_candi_size
        if args.valid_candi_size < 1:
            candidate_size = global_data.product_size
        dataloader = data.ProdSearchDataLoader(
                args, valid_dataset, batch_size=args.valid_batch_size,
                shuffle=False, num_workers=args.num_workers)
        all_prod_idxs, all_prod_scores, all_target_idxs, \
                all_query_idxs, all_user_idxs \
                = self.get_prod_scores(args, global_data, valid_dataset, dataloader, "Validation", candidate_size)
        sorted_prod_idxs = all_prod_scores.argsort(axis=-1)[:,::-1] #by default axis=-1, along the last axis
        mrr, prec = self.calc_metrics(all_prod_idxs, sorted_prod_idxs, all_target_idxs, candidate_size)
        return mrr, prec

    def test(self, args, global_data, test_prod_data, rankfname="test.best_model.ranklist", cutoff=100):
        candidate_size = args.test_candi_size
        if args.test_candi_size < 1:
            candidate_size = global_data.product_size
        test_dataset = data.ProdSearchDataset(args, global_data, test_prod_data)
        dataloader = data.ProdSearchDataLoader(
                args, test_dataset, batch_size=args.valid_batch_size, #batch_size
                shuffle=False, num_workers=args.num_workers)

        all_prod_idxs, all_prod_scores, all_target_idxs, \
                all_query_idxs, all_user_idxs \
                = self.get_prod_scores(args, global_data, test_dataset, dataloader, "Test", candidate_size)
        sorted_prod_idxs = all_prod_scores.argsort(axis=-1)[:,::-1] #by default axis=-1, along the last axis
        mrr, prec = self.calc_metrics(all_prod_idxs, sorted_prod_idxs, all_target_idxs, candidate_size)
        logger.info("Test: MRR:{} P@1:{}".format(mrr, prec))
        output_path = os.path.join(args.save_dir, rankfname)
        eval_count = all_prod_scores.shape[0]
        print(all_prod_scores.shape)
        with open(output_path, 'w') as rank_fout:
            for i in range(eval_count):
                user_id = global_data.user_ids[all_user_idxs[i]]
                qidx = all_query_idxs[i]
                ranked_product_ids = all_prod_idxs[i][sorted_prod_idxs[i]]
                ranked_product_scores = all_prod_scores[i][sorted_prod_idxs[i]]
                for rank in range(min(cutoff, candidate_size)):
                    product_id = global_data.product_ids[ranked_product_ids[rank]]
                    score = ranked_product_scores[rank]
                    line = "%s_%d Q0 %s %d %f ReviewTransformer\n" \
                            % (user_id, qidx, product_id, rank+1, score)
                    rank_fout.write(line)

    def calc_metrics(self, all_prod_idxs, sorted_prod_idxs, all_target_idxs, candidate_size):
        eval_count = all_prod_idxs.shape[0]
        mrr, prec = 0, 0
        for i in range(eval_count):
            result = np.where(all_prod_idxs[i][sorted_prod_idxs[i]] == all_target_idxs[i])
            if len(result[0]) == 0: #not occur in the list
                pass
            else:
                rank = result[0][0] + 1
                mrr += 1/rank
                if rank == 1:
                    prec +=1
        mrr /= eval_count
        prec /= eval_count
        print("MRR:{} P@1:{}".format(mrr, prec))
        return mrr, prec

    def get_prod_scores(self, args, global_data, dataset, dataloader, description, candidate_size):
        self.model.eval()
        with torch.no_grad():
            self.model.get_review_embeddings() #get model.review_embeddings
            pbar = tqdm(dataloader)
            pbar.set_description(description)
            seg_count = int((candidate_size - 1) / args.candi_batch_size) + 1
            all_prod_scores, all_target_idxs, all_prod_idxs = [], [], []
            all_user_idxs, all_query_idxs = [], []
            for batch_data in pbar:
                batch_data = batch_data.to(args.device)
                batch_scores = self.model.test(batch_data)
                #batch_size, candidate_batch_size
                all_user_idxs.append(np.asarray(batch_data.user_idxs))
                all_query_idxs.append(np.asarray(batch_data.query_idxs))
                all_prod_idxs.append(np.asarray(batch_data.candi_prod_idxs))
                all_prod_scores.append(batch_scores.cpu().numpy())
                all_target_idxs.append(np.asarray(batch_data.target_prod_idxs))
                #use MRR
        assert args.candi_batch_size <= candidate_size #otherwise results are wrong
        padded_length = seg_count * args.candi_batch_size
        all_prod_idxs = np.concatenate(all_prod_idxs, axis=0).reshape(-1, padded_length)[:, :candidate_size]
        all_prod_scores = np.concatenate(all_prod_scores, axis=0).reshape(-1, padded_length)[:, :candidate_size]
        all_target_idxs = np.concatenate(all_target_idxs, axis=0).reshape(-1, seg_count)[:,0]
        all_user_idxs = np.concatenate(all_user_idxs, axis=0).reshape(-1, seg_count)[:,0]
        all_query_idxs = np.concatenate(all_query_idxs, axis=0).reshape(-1, seg_count)[:,0]
        #target_scores = all_prod_scores[np.arange(eval_count), all_target_idxs]
        #all_prod_scores.sort(axis=-1) #ascending
        self.model.clear_review_embbeddings()
        return all_prod_idxs, all_prod_scores, all_target_idxs, all_query_idxs, all_user_idxs

