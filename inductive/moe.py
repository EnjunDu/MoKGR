import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter

class MoE_for_Pruning(nn.Module):
    def __init__(self, K_source, K_min, K_max, l_inflection, a, n_rel, n_ent, n_ent_ind, hidden_dim, in_dim, out_dim, num_pruning_experts, temperature, lambda_noise_pruning=1.0):
        super(MoE_for_Pruning, self).__init__()
        self.K_source = K_source
        self.K_min = K_min
        self.K_max = K_max
        self.l_inflection = l_inflection
        self.a = a
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_ent = n_ent
        self.n_ent_ind = n_ent_ind  # Inductive entities
        self.softmax = lambda x :F.softmax(x, dim=-1)
        self.temperature =temperature
        self.lambda_noise_pruning= lambda_noise_pruning

        self.num_pruning_experts = num_pruning_experts
        assert 1 <= self.num_pruning_experts <= 3, "num_pruning_experts must be between 1 and 3"

        # Define the MLP for generating expert weights
        self.weight_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)  # Ensure w1 + w2 + w3 = 1
        )
        # For inductive mode, we cannot use embedding layers directly
        # We can initialize embeddings for inductive entities and relations differently
        self.sub_embed = nn.Embedding(n_ent + n_ent_ind, in_dim)
        self.rel_embed = nn.Embedding(2 * n_rel + 1, in_dim)

        # Initialize the pruning experts
        self.W_h_alpha = nn.Linear(in_dim, out_dim, bias=False)  # For Alpha_Pruning
        self.W_h_similarity = nn.Linear(in_dim, out_dim, bias=False)  # For Similarity_Pruning
        self.softplus = nn.Softplus()
        self.W_n = nn.Linear(2 * hidden_dim, 1, bias=False)


    def forward(self, hidden, nodes, scores, h0, alpha, q_sub, q_rel, edges, old_nodes_new_idx, batch_size, message, obj, alpha_temp, act, l, mode='transductive'):
        n_node = nodes.size(0)
        if mode == 'transductive':
            n_ent = self.n_ent
        else:
            n_ent = self.n_ent_ind
        hidden_dim = hidden.size(1)

        # Get embeddings for sub-entities and relations
        if mode == 'transductive':
            q_sub_embed = self.sub_embed(q_sub)  # [batch_size, in_dim]
        else:
            # For inductive entities, we can use random embeddings or some feature-based embeddings
            q_sub_embed = self.sub_embed(q_sub + self.n_ent)  # Shift indices for inductive entities

        q_rel_embed = self.rel_embed(q_rel)  # [batch_size, in_dim]

        # Concatenate embeddings as input for gating weight generation
        combined_embed = torch.cat([q_sub_embed, q_rel_embed], dim=-1)  # [batch_size, 2 * in_dim]
        # Generate three weight values through MLP
        weights_temp = self.weight_mlp(combined_embed)  # [batch_size, 3]

        # Add noise term
        noise = torch.randn_like(weights_temp) * self.softplus(self.W_n(combined_embed)).squeeze(0)  # [hop_range]

        weights = weights_temp + self.lambda_noise_pruning * noise

        # Apply temperature-scaled softmax with temperature
        weights = F.softmax(weights / self.temperature, dim=-1)  # [batch_size, 3]

        # Compute importance per expert
        importance = weights.sum(dim=0) / weights.sum()

        # Select the top num_pruning_experts weights and their indices
        topk_weights, topk_indices = torch.topk(importance, self.num_pruning_experts)

        # Normalize the retained weights
        selected_weights = topk_weights / topk_weights.sum()  # [num_pruning_experts]
        selected_indices = topk_indices.tolist()

        # Prepare shared data for experts
        device = hidden.device

        # Identify new nodes (nodes not in old_nodes_new_idx)
        tmp_diff_node_idx = torch.ones(n_node, dtype=torch.bool, device=device)
        tmp_diff_node_idx[old_nodes_new_idx] = False
        bool_diff_node_idx = tmp_diff_node_idx  # [n_node]

        diff_node = nodes[bool_diff_node_idx]  # [num_diff_nodes, 2]
        hidden_diff = hidden[bool_diff_node_idx]  # [num_diff_nodes, hidden_dim]
        scores_diff = scores[bool_diff_node_idx]  # [num_diff_nodes]
        h0_diff = h0[:, bool_diff_node_idx, :]  # [1, num_diff_nodes, hidden_dim]

        # Initialize combined outputs
        # Use the global node list to ensure node alignment
        # Create a global node list containing all nodes
        all_nodes = nodes  # [n_node, 2]
        n_all_nodes = all_nodes.size(0)
        hidden_combined = torch.zeros((n_all_nodes, hidden_dim), device=device)
        scores_combined = torch.zeros((n_all_nodes,), device=device)
        h0_combined = torch.zeros((1, n_all_nodes, hidden_dim), device=device)

        # Create a node mapping: a mapping from node ID to global index
        node_ids = all_nodes[:, 0] * n_ent + all_nodes[:, 1]  # [n_node]
        max_node_id = node_ids.max().item()
        node_id_to_index_tensor = torch.full((max_node_id + 1,), -1, dtype=torch.long, device=device)
        node_id_to_index_tensor[node_ids] = torch.arange(n_all_nodes, device=device)

        # Calculate the Top-K value for the current layer
        K_l = self.compute_topk(l)

        # For the selected experts, compute their outputs
        for idx, weight in zip(selected_indices, selected_weights):
            if idx == 0:
                # NodesPruner
                hidden_i, nodes_i, scores_i, h0_i = self.nodes_pruner(
                    hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                    hidden_diff, scores_diff, h0_diff, batch_size, K_l, mode=mode
                )
            elif idx == 1:
                # Similarity_Pruning
                hidden_i, nodes_i, scores_i, h0_i = self.similarity_pruner(
                    hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                    hidden_diff, scores_diff, h0_diff, q_rel_embed, batch_size, K_l, mode=mode
                )
            elif idx == 2:
                # Alpha_Pruning
                hidden_i, nodes_i, scores_i, h0_i = self.alpha_pruner(
                    hidden, nodes, scores, h0, edges, alpha_temp, message, obj,
                    act, bool_diff_node_idx, old_nodes_new_idx, n_node, K_l, mode=mode
                )
            else:
                continue  # Prevent accidental idx

            # Create an index mapping for the nodes of the current expert
            node_ids_i = nodes_i[:, 0] * n_ent + nodes_i[:, 1]  # [n_nodes_i]
            indices_in_all = node_id_to_index_tensor[node_ids_i]  # [n_nodes_i]

            # Accumulate the output using the selected weights
            hidden_combined.index_add_(0, indices_in_all, weight * hidden_i)  # [n_all_nodes, hidden_dim]
            scores_combined.index_add_(0, indices_in_all, weight * scores_i)  # [n_all_nodes]
            h0_combined.index_add_(1, indices_in_all, weight * h0_i)  # [1, n_all_nodes, hidden_dim]

        # Filter out nodes with zero scores
        mask = scores_combined != 0  # Create a mask
        hidden_combined = hidden_combined[mask]  # [n_filtered_nodes, hidden_dim]
        nodes = all_nodes[mask]  # [n_filtered_nodes, 2]
        scores_combined = scores_combined[mask]  # [n_filtered_nodes]
        h0_combined = h0_combined[:, mask, :]  # [1, n_filtered_nodes, hidden_dim]

        # Return the filtered output
        return hidden_combined, nodes, scores_combined, h0_combined

    def compute_topk(self, l):
        """
        Calculate the Top-K value of layer l
        """
        if l < self.l_inflection:
            # Increasing phase
            S_l = 1 / (1 + math.exp(-self.a * (l - self.l_inflection / 2)))
            K_l = self.K_source + (self.K_max - self.K_source) * S_l
        else:
            # Decreasing phase
            S_l = 1 / (1 + math.exp(-self.a * (l - 3 * self.l_inflection / 2)))
            K_l = self.K_min + (self.K_max - self.K_min) * (1 - S_l)
        return int(K_l)

    def nodes_pruner(self, hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                     hidden_diff, scores_diff, h0_diff, batch_size, K_l, mode='transductive'):
        if mode == 'transductive':
            n_ent = self.n_ent
        else:
            n_ent = self.n_ent_ind
        device = hidden.device

        # Project logits to fixed-size tensor via indexing
        node_scores = torch.full((batch_size, n_ent), float('-inf'), device=device)
        node_scores[diff_node[:, 0], diff_node[:, 1]] = scores_diff

        # Apply softmax to get probabilities
        node_probs = self.softmax(node_scores)  # [batch_size, n_ent]

        # Select top K_l nodes
        topk_probs, topk_indices = torch.topk(node_probs, K_l, dim=1)
        topk_mask = torch.zeros_like(node_probs).scatter_(1, topk_indices, 1).bool()

        # Generate mask to keep nodes
        bool_sampled_diff_nodes_idx = topk_mask[diff_node[:, 0], diff_node[:, 1]]

        # Compute diff_node_prob and diff_node_prob_hard
        diff_node_prob = node_probs[diff_node[:, 0], diff_node[:, 1]]
        diff_node_prob_hard = topk_mask[diff_node[:, 0], diff_node[:, 1]].float()

        # Update hidden states with gradient preserving operation
        hidden_updated = hidden.clone()
        hidden_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)

        # Update h0 and scores
        h0_updated = h0.clone()
        h0_updated[:, bool_diff_node_idx, :] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(0).unsqueeze(-1)
        scores_updated = scores.clone()
        scores_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob)

        return hidden_updated, nodes, scores_updated, h0_updated

    def similarity_pruner(self, hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                          hidden_diff, scores_diff, h0_diff, q_rel_embed, batch_size, K_l, mode='transductive'):
        if mode == 'transductive':
            n_ent = self.n_ent
        else:
            n_ent = self.n_ent_ind
        device = hidden.device

        # Get query relation embeddings for corresponding batches
        node_batch_indices = diff_node[:, 0]  # [num_diff_nodes]
        node_q_rel_embed = q_rel_embed[node_batch_indices]  # [num_diff_nodes, in_dim]

        # Compute similarity between node embedding and query relation embedding
        similarities = torch.sum(hidden_diff * node_q_rel_embed, dim=1)  # [num_diff_nodes]

        # Project similarities to fixed-size tensor via indexing
        similarity_scores = torch.full((batch_size, n_ent), float('-inf'), device=device)
        similarity_scores[diff_node[:, 0], diff_node[:, 1]] = similarities

        # Apply softmax to get probabilities
        node_probs = self.softmax(similarity_scores)  # [batch_size, n_ent]

        # Select top K_l nodes
        topk_probs, topk_indices = torch.topk(node_probs, K_l, dim=1)
        topk_mask = torch.zeros_like(node_probs).scatter_(1, topk_indices, 1).bool()

        # Generate mask to keep nodes
        bool_sampled_diff_nodes_idx = topk_mask[diff_node[:, 0], diff_node[:, 1]]

        # Compute diff_node_prob and diff_node_prob_hard
        diff_node_prob = node_probs[diff_node[:, 0], diff_node[:, 1]]
        diff_node_prob_hard = topk_mask[diff_node[:, 0], diff_node[:, 1]].float()

        # Update hidden states with gradient preserving operation
        hidden_updated = hidden.clone()
        hidden_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)

        # Update h0 and scores
        h0_updated = h0.clone()
        h0_updated[:, bool_diff_node_idx, :] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(0).unsqueeze(-1)
        scores_updated = scores.clone()
        scores_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob)

        return hidden_updated, nodes, scores_updated, h0_updated

    def alpha_pruner(self, hidden, nodes, scores, h0, edges, alpha_temp, message, obj,
                     act, bool_diff_node_idx, old_nodes_new_idx, n_node, K_l, mode='transductive'):
        device = hidden.device

        # Edge-related computations
        n_edges = edges.size(0)
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj_indices = edges[:, 5]

        # Identify edges pointing to new nodes
        node_is_new = bool_diff_node_idx
        edge_target_is_new = node_is_new[obj_indices]

        # Prune edges pointing to new nodes
        alpha = alpha_temp.squeeze(-1)  # [n_edges]
        edge_prob = self.softmax(alpha)  # [n_edges]

        # Edges pointing to new nodes
        edge_indices_new = torch.nonzero(edge_target_is_new).view(-1)
        edge_prob_new = edge_prob[edge_indices_new]

        if edge_prob_new.numel() > 0:
            topk = min(K_l, edge_prob_new.numel())
            topk_values, topk_indices = torch.topk(edge_prob_new, topk)
            topk_edge_indices = edge_indices_new[topk_indices]
        else:
            topk_edge_indices = torch.tensor([], dtype=torch.long, device=device)

        # Keep edges pointing to old nodes
        edge_indices_old = torch.nonzero(~edge_target_is_new).view(-1)

        # Merge retained edges and calculate message passing
        edge_prob_hard = torch.zeros_like(edge_prob)
        edge_prob_hard[topk_edge_indices] = 1
        edge_prob_hard[edge_indices_old] = 1

        alpha = alpha * (edge_prob_hard - edge_prob.detach() + edge_prob)  # [n_edges]
        alpha = torch.sigmoid(alpha).unsqueeze(-1)  # [n_edges, 1]

        message = alpha * message  # [n_edges, hidden_dim]

        message_agg = scatter(message, index=obj_indices, dim=0, dim_size=n_node, reduce='sum')  # [n_node, hidden_dim]

        hidden_new = act(self.W_h_alpha(message_agg))  # [n_node, out_dim]

        # Prepare updated hidden, h0, and scores
        hidden_updated = hidden.clone()
        hidden_updated = hidden_new  # Since Alpha_Pruning recomputes hidden states

        # For nodes, determine which ones to keep
        # New nodes are those in bool_diff_node_idx
        new_nodes_indices = torch.nonzero(bool_diff_node_idx).view(-1)
        new_nodes_messages = message_agg[new_nodes_indices]

        # Determine which new nodes have received messages
        if new_nodes_messages.numel() > 0:
            new_nodes_keep = (new_nodes_messages.abs().sum(dim=-1) > 0)
            bool_keep_node = torch.ones(n_node, dtype=torch.bool, device=device)
            bool_keep_node[new_nodes_indices] = new_nodes_keep
        else:
            bool_keep_node = torch.ones(n_node, dtype=torch.bool, device=device)

        # Apply mask to nodes and hidden states
        hidden_updated = hidden_updated[bool_keep_node]
        nodes_updated = nodes[bool_keep_node]
        scores_updated = scores[bool_keep_node]
        h0_updated = h0[:, bool_keep_node, :]

        # Since nodes are filtered, need to adjust hidden and scores accordingly
        return hidden_updated, nodes_updated, scores_updated, h0_updated


class MoE_for_hops(nn.Module):
    def __init__(self, loader, hidden_dim, num_experts, min_hop, max_hop, lambda_noise=1.0, temperature=1.0):
        """
        Initialize the MoE module.

        Parameters:
        hidden_dim (int): Dimension of hidden embedding.
        num_experts (int): Number of experts to choose (i.e. TopK).
        min_hop (int): Minimum number of hops.
        max_hop (int): Maximum number of hops.
        lambda_noise (float): Coefficient for the noise term.
        temperature (float): Temperature for softmax.
        """
        super(MoE_for_hops, self).__init__()
        self.num_experts = num_experts
        self.min_hop = min_hop
        self.max_hop = max_hop
        self.hop_range = max_hop - min_hop + 1
        self.lambda_noise = lambda_noise
        self.embed_dim = hidden_dim
        self.temperature = temperature

        # MLP: Generate contextual embedding c_i from [h_i, h_q]
        self.context_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # The embedding φ(hop) of each hop expert
        self.hop_embedding = nn.Embedding(self.hop_range, hidden_dim)
        # Initialize entity and relation embeddings, learn hop count embeddings
        self.entity_embed = nn.Embedding(loader.n_ent + loader.n_ent_ind, self.embed_dim)
        self.relation_embed = nn.Embedding(loader.n_rel * 2 + 1, self.embed_dim)

        # Linear layer for the noise term
        self.W_n = nn.Linear(hidden_dim, 1, bias=False)
        # Linear layer W_m for load balancing
        self.W_m = nn.Linear(1, 1, bias=False)
        # Softplus activation function is used to smooth nonlinearity
        self.softplus = nn.Softplus()

    def forward(self, subs, rels, mode='transductive'):
        """
        Forward propagation, calculate expert selection probability.

        Parameters:
        subs (torch.Tensor): Subject indices [batch_size].
        rels (torch.Tensor): Relation indices [batch_size].

        Returns:
        G_full (torch.Tensor): Expert selection probability list [hop_range].
        Q (torch.Tensor): Scores before top-k selection [hop_range].
        """
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        if mode == 'transductive':
            q_sub_embed = self.entity_embed(q_sub)  # [batch_size, embed_dim]
        else:
            q_sub_embed = self.entity_embed(q_sub + self.num_entities)  # For inductive entities

        q_rel_embed = self.relation_embed(q_rel)  # [batch_size, embed_dim]

        # Generate context embedding c_i
        c_i = torch.cat([q_sub_embed, q_rel_embed], dim=-1)  # [batch_size, 2 * hidden_dim]
        c_i = self.context_mlp(c_i)  # [batch_size, hidden_dim]
        c_i = torch.mean(c_i, dim=0, keepdim=True)  # [1, hidden_dim]

        # Get the embedding φ(hop) for each hop
        hops = torch.arange(self.min_hop, self.max_hop + 1).cuda()  # [hop_range]
        phi_hop = self.hop_embedding(hops - self.min_hop)  # [hop_range, hidden_dim]

        # Calculate Q(hop)  = c_i^T φ(hop)
        Q = torch.matmul(c_i, phi_hop.t()).squeeze(0)  # [hop_range]

        # Add noise term
        noise = torch.randn_like(Q) * self.softplus(self.W_n(c_i)).squeeze(0)  # [hop_range]
        Q = Q + self.lambda_noise * noise

        # Select TopK hops
        topk_values, topk_indices = torch.topk(Q, self.num_experts, dim=0)  # [num_experts], [num_experts]

        # Apply Softmax to the TopK values to get the selection probability
        G_topk = F.softmax(topk_values / self.temperature, dim=0)  # [num_experts]

        G_full = torch.zeros(self.hop_range).cuda()  # [hop_range]

        # Assign the selection probability of TopK to the corresponding hop position
        G_full[topk_indices] = G_topk  # [hop_range]

        return G_full, Q  # [hop_range], [hop_range]

    def compute_importance_loss(self, G_full):
        """
        Calculate L_importance loss: CV squared
        Use G_full as the selection probability of each expert
        """
        importance = G_full  # [hop_range]
        mean_importance = importance.mean()
        std_importance = importance.std()
        CV_importance = std_importance / (mean_importance + 1e-5)  # Prevent division by zero
        L_importance = CV_importance ** 2
        return L_importance

    def compute_load_loss(self, Q, G_full):
        """
        Calculate L_load loss: Use load balancing loss
        """
        # Get the k-th largest value in the selection probability (take the k-th largest value in num_experts)
        kth_ex = Q.topk(self.num_experts, dim=0, largest=True, sorted=True)[0][-1]  # scalar

        # Use Q to calculate sigma_hop from W_m
        sigma_hop = self.softplus(self.W_m(Q.unsqueeze(1))).squeeze(1)  # [hop_range]

        # Calculate P(hop, o) = Φ((Q(hop) - kth_ex) / sigma_hop)
        P_hi_o = 0.5 * (1 + torch.erf((Q - kth_ex) / (sigma_hop * math.sqrt(2))))  # [hop_range]

        # Calculate the coefficient of variation of the load CV
        mean_p = P_hi_o.mean()
        std_p = P_hi_o.std()
        CV_load = std_p / (mean_p + 1e-8)
        L_load = CV_load ** 2
        return L_load
