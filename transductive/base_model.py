import torch
import numpy as np
import time

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import MoKGR_trans
from utils import cal_ranks, cal_performance
import logging
from tqdm import tqdm

class BaseModel(object):
    def __init__(self, args, loader):
        self.current_epoch = 0

        self.logger = logging.getLogger('BaseModelLogger')
        self.logger.setLevel(logging.INFO)  # Setting the log level
        self.model = MoKGR_trans(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.args = args
        # Create a file processor to write logs to a file
        if not self.logger.handlers:
            fh = logging.FileHandler(args.log_file)  # Use the log file name entered by the user
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        self.logger.info('Model initialization is complete.')
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0
        self.lambda_importance = args.lambda_importance
        self.lambda_load = args.lambda_load
        self.lambda_importance_pruning = args.lambda_importance_pruning  # New hyperparameter for pruning importance loss

    def train_batch(self,):
        self.current_epoch += 1
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)

        t_time = time.time()
        self.model.train()
        for i in tqdm(range(n_batch), desc="Training Epoch Progress", unit="batch"):
            start = i*batch_size
            end = min(self.loader.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx, data='train')  # Specify data='train'

            self.model.zero_grad()
            scores, G_full, Q, L_importance_pruning = self.model(triple[:,0], triple[:,1])

            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:,2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]

            # Calculating L_importance and L_load
            L_importance = self.model.moe_for_hops.compute_importance_loss(G_full)
            L_load = self.model.moe_for_hops.compute_load_loss(Q, G_full)
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))

            total_loss = loss + self.lambda_importance * L_importance + self.lambda_load * L_load + self.lambda_importance_pruning * L_importance_pruning

            total_loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.scheduler.step()
        self.t_time += time.time() - t_time

        valid_mrr, out_str = self.evaluate()
        self.loader.shuffle_train()
        return valid_mrr, out_str

    def evaluate(self, ):
        batch_size = self.n_tbatch

        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels, mode='valid').data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        v_mrr, v_h1, v_h10 = cal_performance(ranking)

        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, mode='test').data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1, t_h10 = cal_performance(ranking)
        i_time = time.time() - i_time

        out_str = f'[Epoch {self.current_epoch}] [VALID] MRR: {v_mrr:.4f} H@1: {v_h1:.4f} H@10: {v_h10:.4f} \t [TEST] MRR: {t_mrr:.4f} H@1: {t_h1:.4f} H@10: {t_h10:.4f} \t [TIME] train: {self.t_time:.4f} inference: {i_time:.4f}\n'
        self.logger.info(out_str)

        return v_mrr, out_str
