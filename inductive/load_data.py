import os
import torch
import numpy as np
import pickle
import cupy as cp
from tqdm import tqdm
from scipy.sparse import csr_matrix
from collections import defaultdict

from PPR_sampler import PPRSampler


class DataLoader:
    def __init__(
            self,
            task_dir,
            n_batch=32,
            sampling_percentage=1,
            PPR_alpha=0.85,
            max_iter=100
    ):
        self.trans_dir = task_dir
        self.ind_dir = task_dir + '_ind'
        self.n_batch = n_batch

        self.sampling_percentage = sampling_percentage
        self.PPR_alpha = PPR_alpha
        self.max_iter = max_iter

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id[entity] = int(eid)

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            id2relation = []
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
                id2relation.append(relation)

        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id_ind = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id_ind[entity] = int(eid)

        for i in range(len(self.relation2id)):
            id2relation.append(id2relation[i] + '_inv')
        id2relation.append('idd')
        self.id2relation = id2relation

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.n_ent_ind = len(self.entity2id_ind)

        self.tra_train = self.read_triples(self.trans_dir, 'train.txt')
        self.tra_valid = self.read_triples(self.trans_dir, 'valid.txt')
        self.tra_test = self.read_triples(self.trans_dir, 'test.txt')

        self.ind_train = self.read_triples(self.ind_dir, 'train.txt', 'inductive')
        self.ind_valid = self.read_triples(self.ind_dir, 'valid.txt', 'inductive')
        self.ind_test = self.read_triples(self.ind_dir, 'test.txt', 'inductive')

        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')
        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])

        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)
        self.ind_KG, self.ind_sub = self.load_graph(self.ind_train, 'inductive')


        self.tra_train = np.array(self.tra_valid)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)

        self.ind_val_qry, self.ind_val_ans = self.load_query(self.ind_valid)
        self.ind_tst_qry, self.ind_tst_ans = self.load_query(self.ind_test)

        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.test_q, self.test_a = self.ind_val_qry + self.ind_tst_qry, self.ind_val_ans + self.ind_tst_ans

        self.n_train = len(self.tra_train)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

        self.trans_all_edges = [(h, t) for (h, r, t) in self.tra_train]  # 只取正向, 也可合并逆向
        # inductive
        self.ind_all_edges = [(h, t) for (h, r, t) in self.ind_train]


        self.trans_ppr_sampler = PPRSampler(
            n_ent=self.n_ent,
            edges=self.trans_all_edges,
            sampling_percentage=self.sampling_percentage,
            PPR_alpha=self.PPR_alpha,
            max_iter=self.max_iter
        )
        self.ind_ppr_sampler = PPRSampler(
            n_ent=self.n_ent_ind,
            edges=self.ind_all_edges,
            sampling_percentage=self.sampling_percentage,
            PPR_alpha=self.PPR_alpha,
            max_iter=self.max_iter
        )

        dataset_name = os.path.basename(os.path.normpath(task_dir))
        self.cache_file_trans = f'{dataset_name}_trans_ppr_cache.pkl'
        self.cache_file_ind = f'{dataset_name}_ind_ppr_cache.pkl'

        self.ppr_cache_trans = {}
        self.ppr_cache_ind = {}

        self._load_or_compute_ppr_cache(
            cache_file=self.cache_file_trans,
            ppr_sampler=self.trans_ppr_sampler,
            num_nodes=self.n_ent,
            ppr_cache=self.ppr_cache_trans
        )

        self._load_or_compute_ppr_cache(
            cache_file=self.cache_file_ind,
            ppr_sampler=self.ind_ppr_sampler,
            num_nodes=self.n_ent_ind,
            ppr_cache=self.ppr_cache_ind
        )


    def read_triples(self, directory, filename, mode='transductive'):

        triples = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                # 正向
                triples.append([h, r, t])
                # 逆向
                triples.append([t, r + self.n_rel, h])
        return triples

    def load_graph(self, triples, mode='transductive'):

        if mode == 'transductive':
            n_ent = self.n_ent
        else:
            n_ent = self.n_ent_ind

        KG = np.array(triples)
        idd = np.concatenate([
            np.expand_dims(np.arange(n_ent), 1),
            2 * self.n_rel * np.ones((n_ent, 1)),
            np.expand_dims(np.arange(n_ent), 1)
        ], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]

        M_sub = csr_matrix(
            (np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])),
            shape=(n_fact, n_ent)
        )
        return KG, M_sub

    def load_query(self, triples):

        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(list)
        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)

        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_filter(self, data='valid'):

        filters = defaultdict(lambda: set())
        if data == 'valid':
            for triple in self.tra_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        else:
            for triple in self.ind_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        return filters

    def shuffle_train(self):

        rand_idx = np.random.permutation(self.n_train)
        self.tra_train = self.tra_train[rand_idx]

    def get_neighbors(self, nodes, mode='transductive'):

        if mode == 'transductive':
            KG = self.tra_KG
            M_sub = self.tra_sub
            n_ent = self.n_ent
        else:
            KG = self.ind_KG
            M_sub = self.ind_sub
            n_ent = self.n_ent_ind

        node_1hot = csr_matrix(
            (np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
            shape=(n_ent, nodes.shape[0])
        )
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        selected_edges = np.concatenate(
            [np.expand_dims(edges[1], 1), KG[edges[0]]],
            axis=1
        )  # (batch_idx, head, rela, tail)
        selected_edges = torch.LongTensor(selected_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(selected_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(selected_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        mask = selected_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        selected_edges = torch.cat([selected_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        return tail_nodes, selected_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):

        if data == 'train':
            return self.tra_train[batch_idx]

        if data == 'valid':
            query, answer = self.valid_q, self.valid_a
            n_ent = self.n_ent  # 使用 transductive 实体
        elif data == 'test':
            query, answer = self.test_q, self.test_a
            n_ent = self.n_ent_ind
        else:
            raise ValueError(f"Unknown data split: {data}")

        subs = [query[i][0] for i in batch_idx]
        rels = [query[i][1] for i in batch_idx]
        objs = np.zeros((len(batch_idx), n_ent))

        for i, idx in enumerate(batch_idx):
            objs[i][answer[idx]] = 1
        return subs, rels, objs

    def _load_or_compute_ppr_cache(self, cache_file, ppr_sampler, num_nodes, ppr_cache):

        if os.path.exists(cache_file):
            try:
                print(f'==> Loading PPR cache from {cache_file}...')
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                ppr_cache.update(data)
                print('==> PPR cache loaded.')
            except (EOFError, pickle.UnpicklingError):
                print(f"Warning: Cache file {cache_file} is empty or corrupted. Recomputing PPR...")
                self._compute_ppr_and_save(ppr_sampler, num_nodes, ppr_cache, cache_file)
        else:
            print(f'==> PPR cache file not found: {cache_file}. Computing and saving...')
            self._compute_ppr_and_save(ppr_sampler, num_nodes, ppr_cache, cache_file)

    def _compute_ppr_and_save(self, ppr_sampler, num_nodes, ppr_cache, cache_file):

        print(f'==> Computing PPR for all {num_nodes} nodes...')
        all_nodes = list(range(num_nodes))
        ppr_scores = ppr_sampler.sample_nodes(seeds=all_nodes)
        for node in tqdm(all_nodes, desc='Caching PPR scores'):
            ppr_cache[node] = {'ppr_scores': ppr_scores[node]}

        print("==> Saving PPR cache to file...")
        with open(cache_file, 'wb') as f:
            pickle.dump(ppr_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("==> PPR cache computed and saved to:", cache_file)

    def build_subgraph(self, subs, k, ppr_cache, node_to_edges):

        combined_scores = cp.zeros(len(ppr_cache), dtype=cp.float32)

        for sub in subs:
            combined_scores += cp.array(ppr_cache[sub]['ppr_scores'], dtype=cp.float32)

        combined_scores_cpu = cp.asnumpy(combined_scores)
        top_k_nodes = np.argsort(combined_scores_cpu)[-k:]
        subgraph_nodes = set(subs) | set(top_k_nodes.tolist())

        subgraph_edges = []
        for nd in subgraph_nodes:
            if nd in node_to_edges:
                for edge in node_to_edges[nd]:
                    if edge[2] in subgraph_nodes:
                        subgraph_edges.append(edge)

        return subgraph_nodes, subgraph_edges
