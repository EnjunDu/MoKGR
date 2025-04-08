from tqdm import tqdm
import cupy as cp

class PPRSampler:
    def __init__(self, n_ent, edges, sampling_percentage=0.8, PPR_alpha=0.85, max_iter=100, tol=1e-9):
        '''
        n_ent: number of entities
        edges: [(h, t)] edge list (undirected graph)
        sampling_percentage: percentage of sampling nodes (between 0 and 1)
        PPR_alpha: teleport probability
        max_iter: maximum iterations for PageRank
        tol: convergence tolerance
        '''
        print('==> Initializing PPRSampler...')
        self.n_ent = n_ent
        self.edges = edges
        self.sampling_percentage = sampling_percentage
        self.PPR_alpha = PPR_alpha
        self.max_iter = max_iter
        self.tol = tol

        # Build a sparse adjacency matrix
        self.adjacency_matrix = self._build_adjacency_matrix()
        print('==> Initialization completed.')

    def _build_adjacency_matrix(self):
        '''
        Constructing a sparse adjacency matrix (on the GPU)
        '''
        src, dst = zip(*self.edges)
        src = cp.array(src, dtype=cp.int32)
        dst = cp.array(dst, dtype=cp.int32)
        data = cp.ones(len(src), dtype=cp.float32)

        # Constructing a sparse matrix
        adjacency_matrix = cp.sparse.coo_matrix((data, (src, dst)), shape=(self.n_ent, self.n_ent))

        # Undirected graph: matrix plus transpose
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T

        # Normalized adjacency matrix (column normalization)
        row_sums = cp.array(adjacency_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        adjacency_matrix = adjacency_matrix.multiply(1 / row_sums[:, None])

        return adjacency_matrix

    def compute_ppr_for_seed(self, seed):
        '''
        Compute Personalized PageRank for a single seed on the GPU, returning the scores of all nodes
        '''
        teleport = cp.zeros(self.n_ent, dtype=cp.float32)
        teleport[seed] = 1.0

        # Initial PageRank value
        pr = cp.ones(self.n_ent, dtype=cp.float32) / self.n_ent

        for i in range(self.max_iter):
            new_pr = self.PPR_alpha * self.adjacency_matrix.dot(pr) + (1 - self.PPR_alpha) * teleport
            if cp.linalg.norm(new_pr - pr, ord=1) < self.tol:
                pr = new_pr
                break
            pr = new_pr

        # Return results to the CPU
        pr_scores = cp.asnumpy(pr)

        return pr_scores  # Returns the scores of all nodes

    def sample_nodes(self, seeds):
        '''
        Compute the PPR of multiple seeds in parallel and return the scores of all nodes
        '''
        ppr_scores_per_seed = {}

        print('==> Calculating PPR Scores Using GPUs...')
        for seed in tqdm(seeds, desc='Calculating PPR'):
            ppr_scores = self.compute_ppr_for_seed(seed)
            ppr_scores_per_seed[seed] = ppr_scores

        return ppr_scores_per_seed
