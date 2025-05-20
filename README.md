<!--
 ┌────────────────────────────────────────────────────────────────────┐
 │  MoKGR · Mixture-of-Experts for Personalized KG Reasoning         │
 └────────────────────────────────────────────────────────────────────┘
-->

<p align="center">
  <img src="images/MoKGR.png" alt="MoKGR architecture" width="55%"/>
</p>

# MoKGR

**MoKGR (Mixture of Length and Pruning Experts for Knowledge-Graph Reasoning)** is a relation-centric framework that *personalizes* path exploration to deliver state-of-the-art KG reasoning in both **transductive** and **inductive** settings.

<div align="center">
<strong>Key ideas</strong> · adaptive path-length selection · complementary pruning experts · fast & memory-efficient message passing
</div>

---

## ✨ Highlights
* **Adaptive Length Experts** – query-aware gating selects the most relevant hop distances and stops early with a Gumbel-Sigmoid binary gate.  
* **Complementary Pruning Experts** – score-, attention- and semantic-based experts collaboratively retain the most informative entities.  
* **Unified Pipeline** – handles *fully inductive*, *transductive* and *cross-domain* KGs with a single codebase.  
* **Scalable** – tested on large KGs (e.g. **YAGO3-10**) without GPU out-of-memory errors.  
* **Plug-and-Play** – lightweight implementation; a single modern GPU is sufficient for all benchmarks.

---

## 🔧 Installation
```bash
cd MoKGR
pip install -r requirements.txt   
```

## 🚀 Quick Start

### 1. Transductive Reasoning

```
cd transductive
# Family (small-scale)
python train.py \
  --data_path data/family --gpu 0 \
  --max_hop 8 --min_hop 2 \
  --num_experts 4 --num_pruning_experts 2 \
  --active_PPR --sampling_percentage 0.85 \
  --active_gate --gate_threshold 0.25
```

```
# YAGO3-10 (large-scale)
python train.py \
  --data_path data/YAGO --gpu 0 \
  --max_hop 8 --min_hop 1 \
  --num_experts 6 --num_pruning_experts 2 \
  --active_PPR --sampling_percentage 0.475
```

*📝 Tip*: Encounter **OOM**? Increase `--sampling_percentage` *or* disable `--active_PPR` to reduce subgraph size.

### 2. Inductive Reasoning

```
cd inductive
python train.py \
  --data_path ./data/WN18RR_v2 --gpu 0 \
  --max_hop 8 --min_hop 2 --num_experts 5 \
  --active_gate --gate_threshold 0.05
```

## 📊 Reproducing Paper Results

| Dataset   | MRR       | Hit@1     | Hit@10    |
| --------- | --------- | --------- | --------- |
| WN18RR    | 0.611     | 0.539     | 0.702     |
| FB15k-237 | 0.443     | 0.368     | 0.607     |
| YAGO3-10  | **0.657** | **0.577** | **0.758** |



Full benchmark tables & ablation studies can be found in our paper’s Appendix B–D.

## 🛠  Project Structure

```
MoKGR/
├─ transductive/     # training & evaluation scripts (fixed entity set)
├─ inductive/        # inductive split loader + training scripts
├─ images/        # logo of MoKGR
└─ requirements.txt
```
