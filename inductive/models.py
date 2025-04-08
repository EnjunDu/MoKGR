import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from moe import MoE_for_hops, MoE_for_Pruning

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]
        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:,0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        message = hs + hr
        alpha_temp = self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        alpha = torch.sigmoid(alpha_temp)
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new, alpha, message, obj, alpha_temp, self.act

class GNNModel(nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        self.max_hop = params.max_hop
        self.min_hop = params.min_hop
        self.temperature = params.temperature
        self.lambda_noise = params.lambda_noise
        self.K_source = params.K_source
        self.K_min = params.K_min
        self.K_max = params.K_max
        self.l_inflection = params.l_inflection
        self.a = params.a
        self.num_pruning_experts = params.num_pruning_experts
        self.pruning_temperature = params.pruning_temperature
        self.lambda_noise_pruning = params.lambda_noise_pruning

        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.moe_for_hops = MoE_for_hops(
            loader,
            hidden_dim=self.hidden_dim,
            num_experts=params.num_experts,
            min_hop=self.min_hop,
            max_hop=self.max_hop,
            lambda_noise=self.lambda_noise,
            temperature=self.temperature
        )
        self.MoE_for_Pruning = MoE_for_Pruning(
            K_source=self.K_source,
            K_min=self.K_min,
            K_max=self.K_max,
            l_inflection=self.l_inflection,
            a=self.a,
            n_rel=self.n_rel,
            n_ent=self.n_ent,
            n_ent_ind = self.n_ent_ind,# Using n_ent_ind in inductive mode
            hidden_dim=self.hidden_dim,
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_pruning_experts=self.num_pruning_experts,
            temperature=self.pruning_temperature,
            lambda_noise_pruning=self.lambda_noise_pruning
        )

    def forward(self, subs, rels, mode='transductive'):
        n = len(subs)

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        if mode == 'transductive':
            n_ent = self.n_ent
        else:
            n_ent = self.n_ent_ind

        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        scores_all = torch.zeros((n, n_ent)).cuda()
        G_full, Q = self.moe_for_hops(subs, rels)
        j = 0
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)

            hidden, alpha, message, obj, alpha_temp, act = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0))
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            if i >= self.min_hop - 1 and i <= self.max_hop - 1:
                scores = self.W_final(hidden).squeeze(-1)
                hidden, nodes, scores, h0 = self.MoE_for_Pruning(hidden, nodes, scores, h0, alpha, q_sub, q_rel, edges, old_nodes_new_idx, n, message, obj, alpha_temp, act, (i+1))

                scores = scores * G_full[j]
                scores_all[[nodes[:,0], nodes[:,1]]] += scores
                j += 1

        if self.training:
            return scores_all, G_full, Q
        else:
            return scores_all
