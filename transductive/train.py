import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from load_data2 import DataLoader2
from base_model import BaseModel
import pickle

parser = argparse.ArgumentParser(description="Parser for MoKGR")
parser.add_argument('--data_path', type=str, default='data/family/')
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--fact_ratio', type=float, default=0.90)
parser.add_argument('--remove_1hop_edges', action='store_true')
# gate
parser.add_argument('--gate_threshold', type=float, default=0.1, help='the less the value, the more the layers to be continued')
parser.add_argument('--active_gate', action='store_true')
# PPR
parser.add_argument('--sampling_percentage', type=float, default=1, help='The proportion of sub-images retained by PPR')
parser.add_argument('--PPR_alpha',type=float, default=0.85)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--active_PPR', action="store_true")
# moe for hops
parser.add_argument('--num_experts', type=int, default=3)
parser.add_argument('--min_hop',type=int, default=3)
parser.add_argument('--max_hop',type=int, default=8)
parser.add_argument('--lambda_importance', type=float, default=1e-7, help= 'Importance loss weight')
parser.add_argument('--lambda_load', type=float, default=0, help= 'Load balancing loss weight')
parser.add_argument('--lambda_noise', type=float, default=1)
parser.add_argument('--hop_temperature', type=float, default=1.1, help='temperature of Softmax, The smaller the value, the sharper the distribution.')
# moe for pruning
parser.add_argument('--pruning_temperature', type=float, default=1.5, help='temperature of Softmax, The smaller the value, the sharper the distribution.')
parser.add_argument('--K_source', type=int, default=1000)
parser.add_argument('--K_min', type=int, default=1000)
parser.add_argument('--K_max', type=int, default=2000)
parser.add_argument('--l_inflection', type=int, default=3, help='Define at which layer the peak value of 𝐾 is reached')
parser.add_argument('--a', type=float, default=3.0)
parser.add_argument('--num_pruning_experts', type=int, default=2)
parser.add_argument('--lambda_importance_pruning', type=float, default=1e-7)
parser.add_argument('--log_file', type=str, default='train.log', help='Log file name')
parser.add_argument('--cache_file', type=str, default='ppr_cache.pkl', help='PPR cache file name')
parser.add_argument('--lambda_noise_pruning', type=float, default=1.0)
args = parser.parse_args()

if __name__ == '__main__':
    args.n_layer = args.max_hop

    opts = args

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    torch.cuda.set_device(opts.gpu)
    print('==> gpu:', opts.gpu)
    if args.active_PPR:
        loader = DataLoader(args.data_path, args.fact_ratio, args.remove_1hop_edges, args.sampling_percentage, args.PPR_alpha, args.max_iter, args.active_PPR)
    else:
        loader = DataLoader2(args)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    if dataset == 'family':
        opts.lr = 0.0036
        opts.decay_rate = 0.999
        opts.lamb = 0.000017
        opts.hidden_dim = 48
        opts.attn_dim = 5
        #opts.n_layer = 3
        opts.n_layer = args.n_layer
        opts.dropout = 0.29
        opts.act = 'relu'
        opts.n_batch = 20
        opts.n_tbatch = 20
    elif dataset == 'umls':
        opts.lr = 0.0012
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        #opts.n_layer = 5
        opts.n_layer = args.n_layer
        opts.dropout = 0.01
        opts.act = 'tanh'
        opts.n_batch = 10
        opts.n_tbatch = 10
    elif dataset == 'WN18RR':
        opts.lr = 0.0030
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        #opts.n_layer = 5
        opts.n_layer = args.n_layer
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50
    elif dataset == 'fb15k-237':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        #opts.n_layer = 4
        opts.n_layer = args.n_layer
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 10
        opts.n_tbatch = 10
    elif dataset == 'nell':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        #opts.n_layer = 5
        opts.n_layer = args.n_layer
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 10
        opts.n_tbatch = 10
    elif dataset == 'YAGO':
        opts.lr = 0.001
        opts.decay_rate = 0.9429713470775948
        opts.lamb = 0.000946516892415447
        opts.hidden_dim = 64
        opts.attn_dim = 2
        #opts.n_layer = 8
        opts.n_layer = args.n_layer
        opts.dropout = 0.19456805575101324
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 5

    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    best_mrr = 0
    best_str = ''
    for epoch in range(args.epoch):
        with open(opts.perf_file, 'a+') as f:
            mrr, out_str = model.train_batch()
            f.write(out_str)
        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
        print(f'{epoch}: best:{best_str}')
    print(best_str)

    # Ensure cache is saved at the end
    with open(loader.cache_file, 'wb') as f:
        pickle.dump(loader.ppr_cache, f)
    print('==> Final PPR cache saved.')

