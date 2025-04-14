# MoKGR

<p align="center">
<img src="images/MoKGR.png" alt="MoKGR" , width= 40%/>
</p>


## Dependencies

* cupy=13.3.0
* numpy=2.2.1
* scipy=1.14.1
* torch=2.5.1
* torch_scatter=2.1.2
* tqdm=4.67.1

## Usage

```
pip install -r requirements.txt
```

### Transductive settings (in `\transductive`)

```
cd transductive
```



* If there occur any **OoM** (Out-of-Memory) problem, just **active** the PPR or **turn down** (if you already activated it) the sampling_percentage.
* For the gated function, enable **--active_gate** to turn on the gate. Generally speaking, we recommend setting **--gate_threshold** to between 0.1 and 0.4. 

#### For Small-Scale Dataset like Family

```
python -W ignore train.py --data_path=data/family --max_hop 8 --min_hop 2 --K_min 90 --K_max 150 --l_inflection 3 --a 3.5 --log_file family.log  --gpu 0 --K_source 100 --num_pruning_experts 2 --fact_ratio 0.90 --lambda_importance 0.000000000000005725039842729399 --lambda_load 4.377277390633418e-07 --lambda_noise 1.2079892684178535 --hop_temperature 1.7064284191836472 --l_inflection 4 --num_experts 4 --a 5.336083198086804 --seed 9582 --pruning_temperature 2.049253025798915 --arctive_PPR --sampling_percentage 0.8502985376570376 
```

#### For Large-Scale Dataset like YAGO

```
python3 -W ignore train.py --data_path=data/YAGO --max_hop 8 --min_hop 1 --num_experts 6 --K_min 1750 --K_max 2550 --l_inflection 3 --a 3.5 --log_file YAGO.log  --gpu 0 --K_source 2000 --num_pruning_experts 2 --fact_ratio 0.995 --sampling_percentage 0.475 --active_PPR --lambda_load 1e-8 --lambda_importance 1e-07 --lambda_importance_pruning 1e-08
```

### Inductive settings (in `\inductive`)

```
cd inductive
```



```
python train.py --data_path ./data/WN18RR_v2 --seed 1234 --gpu 0 --gate_threshold 0.05 --sampling_percentage 1 --PPR_alpha 0.85 --max_iter 100 --pruning_temperature 1 --lambda_noise_pruning 0 --lr 0.0021 --decay_rate 0.9968 --lamb 0.000018 --hidden_dim 64 --init_dim 64 --attn_dim 3 --n_layer 7 --n_batch 20 --dropout 0.4237 --act relu --topk 100 --increase True --max_hop 8 --min_hop 2 --num_experts 5 --lambda_importance 0.0 --lambda_load 0.0 --lambda_noise 1.0 --temperature 1.0 --K_source 1000 --K_min 750 --K_max 1275 --l_inflection 3 --a 3.5 --num_pruning_experts 2 --log_file WN18RR_v2.log
```



