from tqdm import tqdm
from others.logging import logger
#from data.prod_search_dataloader import ProdSearchDataloader
#from data.prod_search_dataset import ProdSearchDataset
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

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
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
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')
        #orig_init_learning_rate = args.init_learning_rate
        #model, optimizer = create_model(args, data_set)
        # Set model in training mode.
        self.model.train()
        model_dir = "%s/model" % (args.save_dir)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        step_time, loss = 0.,0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        total_norm = 0.
        is_best = False
        for current_epoch in range(args.start_epoch+1, args.max_train_epoch+1):
            logger.info("Initialize epoch:%d" % current_epoch)
            train_prod_data.initialize_epoch()
            dataset = data.ProdSearchDataset(args, global_data, train_prod_data)
            dataloader = data.ProdSearchDataLoader(
                    dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers)
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(current_epoch))
            time_flag = time.time()
            for batch_data_arr in pbar:
                batch_data_arr = [x.to(args.device) for x in batch_data_arr]
                get_batch_time += time.time() - time_flag
                time_flag = time.time()
                step_loss = self.model(batch_data_arr)
                #self.optim.optimizer.zero_grad()
                self.model.zero_grad()
                step_loss.backward()
                self.optim.step()
                step_loss = step_loss.item()
                pbar.set_postfix(step_loss=step_loss)
                loss += step_loss / args.steps_per_checkpoint #convert an tensor with dim 0 to value
                current_step += 1
                step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % args.steps_per_checkpoint == 0:
                    logger.info("Epoch %d lr = %5.6f loss = %6.2f time %.2f prepare_time %.2f step_time %.2f" %
                            (current_epoch, self.optim.learning_rate, loss, time.time()-start_time, get_batch_time, step_time))#, end=""

                    step_time, get_batch_time, loss = 0., 0.,0.
                    sys.stdout.flush()
                    start_time = time.time()
            #save model after each epoch
            self._save(current_epoch)
            #use for validation

    def _save(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'opt': self.args,
            'optim': self.optim,
        }
        model_dir = "%s/model" % (self.args.save_dir)
        checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.pt' % epoch)
        logger.info("Saving checkpoint %s" % checkpoint_path)

    def validate(self, global_data, valid_prod_data):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        valid_dataset = data.ProdSearchDataset(args, global_data, valid_prod_data)
        valid_dataloader = data.ProdSearchDataLoader(
                valid_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers)

    def test(self, global_data, valid_prod_data):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        pass
