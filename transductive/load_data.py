import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from PPR_sampler import PPRSampler
import pickle
from tqdm import tqdm
import cupy as cp

# Load_data using PPR_sampler
class DataLoader:
    def __init__(self, task_dir, fact_ratio, remove_1hop_edges=False, sampling_percentage=0.8, PPR_alpha=0.85, max_iter=100, active_PPR = False):
        self.task_dir = task_dir
        self.fact_ratio = fact_ratio
        self.remove_1hop_edges = remove_1hop_edges
        self.active_PPR = active_PPR
        self.sampling_percentage = sampling_percentage
        self.PPR_alpha = PPR_alpha
        self.max_iter = max_iter
        # Dynamically generate cache file name
        dataset_name = os.path.basename(os.path.normpath(task_dir))  # Get the dataset name from the path
        self.cache_file = f'{dataset_name}_ppr_cache.pkl'

        # Read entities and relations
        self.entity2id, self.n_ent = self._read_entities(os.path.join(task_dir, 'entities.txt'))
        self.relation2id, self.n_rel = self._read_relations(os.path.join(task_dir, 'relations.txt'))

        self.filters = defaultdict(set)

        # Read triple data
        self.fact_triple = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple = self.read_triples('test.txt')

        # Add inverse relations
        self.fact_data = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data = self.double_triple(self.test_triple)

        # Build undirected graph edge list
        homo_edges = [(h, t) for h, r, t in (self.fact_triple + self.train_triple)]
        self.all_edges = homo_edges

        # Initialize PPRSampler
        self.ppr_sampler = PPRSampler(
            self.n_ent,
            self.all_edges,
            sampling_percentage=self.sampling_percentage,
            PPR_alpha=self.PPR_alpha,
            max_iter=self.max_iter
        )

        # Prepare data
        self.shuffle_train()

        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q, self.test_a = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

        # Initialize PPR cache
        self.ppr_cache = {}
        if os.path.exists(self.cache_file):
            try:
                print(f'==> Loading PPR cache from {self.cache_file}...')
                with open(self.cache_file, 'rb') as f:
                    self.ppr_cache = pickle.load(f)
                print('==> PPR cache loaded.')
            except (EOFError, pickle.UnpicklingError):
                print(f"Warning: Cache file {self.cache_file} is empty or corrupted. Initializing a new cache.")
                self.ppr_cache = {}
        else:
            print(f'==> PPR cache file not found: {self.cache_file}. It will be created after computing PPR scores.')

        # Compute PPR scores if cache is incomplete
        if len(self.ppr_cache) < self.n_ent:
            print(f'==> Computing PPR for all {self.n_ent} nodes...')
            all_nodes = list(range(self.n_ent))
            ppr_scores = self.ppr_sampler.sample_nodes(seeds=all_nodes)
            self.ppr_cache = {
                node: {'ppr_scores': ppr_scores[node]}
                for node in tqdm(all_nodes, desc='Caching PPR scores')
            }

            # Save cache to file
            print("==> Saving PPR cache to file...")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.ppr_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("==> PPR cache for all nodes computed and saved.")

        # Precompute tKG_triples
        self.tKG_triples = self.double_triple(self.fact_triple) + self.double_triple(self.train_triple)

        # Precompute node to edges mappings
        self.build_node_to_edges()
        self.build_t_node_to_edges()

    def _read_entities(self, filepath):
        """Read entities from file and create a mapping."""
        entity2id = {}
        n_ent = 0
        with open(filepath) as f:
            for line in f:
                entity = line.strip()
                entity2id[entity] = n_ent
                n_ent += 1
        return entity2id, n_ent

    def _read_relations(self, filepath):
        """Read relations from file and create a mapping."""
        relation2id = {}
        n_rel = 0
        with open(filepath) as f:
            for line in f:
                relation = line.strip()
                relation2id[relation] = n_rel
                n_rel += 1
        return relation2id, n_rel

    def read_triples(self, filename):
        """Read triples from a file."""
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h, r, t])
                self.filters[(h, r)].add(t)
                self.filters[(t, r + self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        """Add inverse relations to triples."""
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r + self.n_rel, h])
        return triples + new_triples

    def load_query(self, triples):
        """Load queries and answers for validation/testing."""
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

    def build_node_to_edges(self):
        """Build mapping from node to its outgoing edges for fact_data."""
        self.node_to_edges = defaultdict(list)
        for triple in self.fact_data:
            h, r, t = triple
            self.node_to_edges[h].append((h, r, t))

    def build_t_node_to_edges(self):
        """Build mapping from node to its outgoing edges for tKG_triples."""
        self.t_node_to_edges = defaultdict(list)
        for triple in self.tKG_triples:
            h, r, t = triple
            self.t_node_to_edges[h].append((h, r, t))

    def build_subgraph(self, subs, k, merge_strategy='sum', weights=None, triples=None, node_to_edges=None):
        """
        Construct a subgraph based on PPR scores for a batch of nodes using GPU.

        Args:
            subs (array-like): Subject nodes.
            k (int): Number of top nodes to select based on combined PPR scores.
            merge_strategy (str): Strategy to merge PPR scores ('sum', 'max', 'weighted_average').
            weights (dict, optional): Weights for weighted average.
            triples (list, optional): List of triples to consider.
            node_to_edges (dict, optional): Mapping from node to its outgoing edges.

        Returns:
            subgraph_nodes (set): Nodes included in the subgraph.
            subgraph_edges (list): Edges included in the subgraph.
        """
        combined_scores = cp.zeros(self.n_ent, dtype=cp.float32)

        # Addition on GPU using CuPy
        for sub in subs:
            combined_scores += cp.array(self.ppr_cache[sub]['ppr_scores'])

        # Switch back to the CPU for sorting
        combined_scores_cpu = cp.asnumpy(combined_scores)
        top_k_nodes = np.argsort(combined_scores_cpu)[-k:]
        subgraph_nodes = set(subs.tolist()) | set(top_k_nodes.tolist())

        # Using collections to speed up lookups
        subgraph_nodes = list(subgraph_nodes)
        subgraph_nodes_set = set(subgraph_nodes)

        # Extract subgraph edges
        subgraph_edges = []
        for node in subgraph_nodes:
            if node in node_to_edges:
                for edge in node_to_edges[node]:
                    if edge[2] in subgraph_nodes_set:
                        subgraph_edges.append(edge)

        return set(subgraph_nodes), subgraph_edges

    def get_neighbors(self, nodes, mode='train'):
        if mode == 'train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub

        # nodes: n_node x 2 with (batch_idx, node_idx)
        max_batch_idx = nodes[:, 0].max()

        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, max_batch_idx + 1))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([
            np.expand_dims(edges[1], 1),
            KG[edges[0]]
        ], axis=1)  # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([
            sampled_edges,
            head_index.unsqueeze(1),
            tail_index.unsqueeze(1)
        ], 1)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, merge_strategy='sum', data='train'):
        """
        Retrieve a batch of data.

        Args:
            batch_idx (array-like): Indices of the batch.
            merge_strategy (str): Strategy to merge PPR scores.
            data (str): Type of data ('train', 'valid', 'test').

        Returns:
            Depending on the data type:
                - 'train': Returns triple_batch.
                - 'valid'/'test': Returns subs, rels, objs.
        """
        # Calculate k based on sampling_percentage
        k = int(self.n_ent * self.sampling_percentage)
        if k < 1:
            k = 1  # Ensure at least one node is selected

        if data == 'train':
            triple_batch = np.array(self.train_data)[batch_idx]
            subs = triple_batch[:, 0]
            # Build subgraph using node_to_edges
            subgraph_nodes, subgraph_edges = self.build_subgraph(
                subs, k, merge_strategy, triples=self.fact_data, node_to_edges=self.node_to_edges)
            # Build KG and M_sub based on subgraph_edges
            idd = np.concatenate([
                np.expand_dims(np.arange(self.n_ent), 1),
                2 * self.n_rel * np.ones((self.n_ent, 1)),
                np.expand_dims(np.arange(self.n_ent), 1)
            ], 1)
            if len(subgraph_edges) == 0:
                subgraph_edges_array = np.zeros((0, 3), dtype=int)
            else:
                subgraph_edges_array = np.array(subgraph_edges)
            self.KG = np.concatenate([subgraph_edges_array, idd], 0)
            self.n_fact = len(self.KG)
            self.M_sub = csr_matrix(
                (np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:, 0])),
                shape=(self.n_fact, self.n_ent)
            )

            return triple_batch

        elif data in ['valid', 'test']:
            if data == 'valid':
                query, answer = np.array(self.valid_q), self.valid_a
            else:
                query, answer = np.array(self.test_q), self.test_a

            subs = query[batch_idx, 0]
            rels = query[batch_idx, 1]
            objs = np.zeros((len(batch_idx), self.n_ent))
            for i in range(len(batch_idx)):
                objs[i][answer[batch_idx[i]]] = 1

            # Build subgraph using t_node_to_edges
            subgraph_nodes, subgraph_edges = self.build_subgraph(
                subs, k, merge_strategy, triples=self.tKG_triples, node_to_edges=self.t_node_to_edges)
            # Build tKG and tM_sub based on subgraph_edges
            idd = np.concatenate([
                np.expand_dims(np.arange(self.n_ent), 1),
                2 * self.n_rel * np.ones((self.n_ent, 1)),
                np.expand_dims(np.arange(self.n_ent), 1)
            ], 1)
            if len(subgraph_edges) == 0:
                subgraph_edges_array = np.zeros((0, 3), dtype=int)
            else:
                subgraph_edges_array = np.array(subgraph_edges)
            self.tKG = np.concatenate([subgraph_edges_array, idd], 0)
            self.tn_fact = len(self.tKG)
            self.tM_sub = csr_matrix(
                (np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:, 0])),
                shape=(self.tn_fact, self.n_ent)
            )

            return subs, rels, objs

        else:
            raise ValueError(f"Unknown data type: {data}")

    def shuffle_train(self):
        """
        Shuffle training data and rebuild node_to_edges mapping.
        """
        fact_triple = np.array(self.fact_triple)
        train_triple = np.array(self.train_triple)
        all_triple = np.concatenate([fact_triple, train_triple], axis=0)
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]

        bar = int(n_all * self.fact_ratio)
        self.fact_data = np.array(self.double_triple(all_triple[:bar].tolist()))
        self.train_data = np.array(self.double_triple(all_triple[bar:].tolist()))



        self.n_train = len(self.train_data)

        # Rebuild node_to_edges after shuffling
        self.build_node_to_edges()

        if self.remove_1hop_edges:
            print('==> removing 1-hop links...')
            print(f'Before removal: {len(self.fact_data)} facts')
            tmp_index = np.ones((self.n_ent, self.n_ent))
            tmp_index[self.train_data[:, 0], self.train_data[:, 2]] = 0
            save_facts = tmp_index[self.fact_data[:, 0], self.fact_data[:, 2]].astype(bool)
            self.fact_data = self.fact_data[save_facts]
            print(f'After removal: {len(self.fact_data)} facts')
            print('==> done')
