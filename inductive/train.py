import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel

parser = argparse.ArgumentParser(description="Parser for AdaProp")
parser.add_argument('--data_path', type=str, default='./data/WN18RR_v2')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=0)
# gate
parser.add_argument('--gate_threshold', type=float, default=0, help='the less the value, the more the layers to be continued')
# PPR
parser.add_argument('--sampling_percentage', type=float, default=1, help='The proportion of sub-images retained by PPR')
parser.add_argument('--PPR_alpha',type=float, default=0.85)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--pruning_temperature', type=float, default=1, help='temperature of Softmax, The smaller the value, the sharper the distribution.')
parser.add_argument('--lambda_noise_pruning', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.0021)
parser.add_argument('--decay_rate', type=float, default=0.9968)
parser.add_argument('--lamb', type=float, default=0.000018)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--init_dim', type=int, default=64)
parser.add_argument('--attn_dim', type=int, default=3)
parser.add_argument('--n_layer', type=int, default=7)
parser.add_argument('--n_batch', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0.4237)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--topk', type=int, default=100)
parser.add_argument('--increase', type=bool, default=True)
parser.add_argument('--max_hop', type=int, default=8)
parser.add_argument('--min_hop', type=int, default=2)
parser.add_argument('--num_experts', type=int, default=5)
parser.add_argument('--lambda_importance', type=float, default=0.0)
parser.add_argument('--lambda_load', type=float, default=0.0)
parser.add_argument('--lambda_noise', type=float, default=1.0)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--K_source', type=int, default=1000)
parser.add_argument('--K_min', type=int, default=750)
parser.add_argument('--K_max', type=int, default=1275)
parser.add_argument('--l_inflection', type=int, default=3)
parser.add_argument('--a', type=float, default=3.5)
parser.add_argument('--num_pruning_experts', type=int, default=2)
parser.add_argument('--log_file', type=str, default='WN18RR_v2.log')

args = parser.parse_args()



dataset = args.data_path.split('/')
if len(dataset[-1]) > 0:
    dataset = dataset[-1]
else:
    dataset = dataset[-2]

opts = args
opts.hidden_dim = 64
opts.init_dim = 10
opts.attn_dim = 5
opts.n_layer = 3
opts.n_batch = 50
opts.lr = 0.001
opts.decay_rate = 0.999
opts.perf_file = './results.txt'

torch.cuda.set_device(args.gpu)
print('==> gpu:', args.gpu)

loader = DataLoader(args.data_path, n_batch=opts.n_batch)
opts.n_ent = loader.n_ent
opts.n_rel = loader.n_rel

def run_model():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %d, %.4f, %s, %d, %s\n' % (
        opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.init_dim, opts.attn_dim,
        opts.n_layer, opts.n_batch, opts.dropout, opts.act, opts.topk, str(opts.increase)
    )
    print(args.data_path)
    print(config_str)

    best_str = "No improvement during training."

    try:
        model = BaseModel(opts, loader)
        best_mrr = 0
        best_tmrr = 0
        early_stop = 0
        for epoch in range(75):
            mrr, t_mrr, out_str = model.train_batch()
            if mrr > best_mrr:
                best_mrr = mrr
                best_tmrr = t_mrr
                best_str = out_str
                early_stop = 0
            else:
                early_stop += 1

        with open(opts.perf_file, 'a') as f:
            f.write(args.data_path + '\n')
            f.write(config_str)
            f.write(best_str + '\n')
            print('\n\n')

    except RuntimeError as e:
        best_tmrr = 0
        best_str = f"RuntimeError occurred: {str(e)}"

    print('self.time_1, self.time_2, time_3, v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050, t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050')
    print(best_str)
    return

run_model()
