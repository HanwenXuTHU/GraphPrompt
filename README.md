# GraphPrompt: Graph-based Prompt Templates For Biomedical Synonym Prediction
GraphPrompt code

## Environment

- linux (`./tmp` folder for storing tmp files)
- pytorch
## Folder Structure
- `./baselines/` folder contains code for baselines that we have compared graphprompt with. Please navigate to the `./baselines/` folder to see how to run and reproduce the results of the baselines.

## Running the code
Take this command as an example:

```
CUDA_VISIBLE_DEVICES=1 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main.py --filename=../data/datasets/cl.obo --is_unseen --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=0 --cst_ratio=1 --lr=1.000e-04 --epoch_num=10 --pretrain_emb_iter=400 --use_get_ent_emb  --path_depth=1 --seed=0 --exp_path=../exp/0cl.obo_unseen__path1/    > 0cl.obo_unseen_log_simplepath10_ratio210_pretrain400_depth1_1.000e-04_get_ent  2>&1
```

- `CUDA_VISIBLE_DEVICES=1`: select CUDA device.
- `DEBUG_DECODE_EVAL=1`: (*you can ignore this*) define this var if you want to print output when evaluating.
- `DEBUG_DECODE_OUTPUT=1`: (*you can ignore this*) define this var if you want to print output when training.
- `--filename=../data/datasets/cl.obo`: select training file as `../data/datasets/cl.obo`
- `--is_unseen`: add this flag if you are doing zero-shot learning.
- `--use_scheduler`: add this flag if you want to use lr scheduler.
- `--syn_ratio=2`, `--ent_ratio=1`, `--cst_ratio=1` and `--hrt_ratio=0`: the ratio of training synonyms representation, entity representation (contrastive loss term1), contrastive loss term2 and knowledge graph representation (will be normalized to 1 in the code).
- `--lr=1.000e-04`: learning rate
- `--epoch_num=10`: number of epochs
- `--pretrain_emb_iter=400`: the iteration of warming up the entity representation. 
- `--use_get_ent_emb`: use encoder to get entity embedding (concept embedding). If the flag is not added, the entity embedding will be looked up from an `Embedding` layer. Usually, turn on this flag in the zero-shot setting, and turn it off in the few-shot setting.
- `--path_depth=1`: the depth of path, which is also the number of verbs in the sentence.
- `--seed=0`: random seed.
- `--exp_path=../exp/0cl.obo_unseen__path1/` path for the checkpoint.    
- `> 0cl.obo_unseen_log_simplepath10_ratio210_pretrain400_depth1_1.000e-04_get_ent  2>&1`: redirect stdout and stderr to a log file.

## Reproduce our results
Description of the computing infrastructure: 2 GPUs, 20 cores, 10TB hard disk, 128G memory

Average run time: we estimate that it takes around 1 hour

hyperparamter seach method: grid sample

Total parameters: 116741958

The following are the running commands for our GraphPromp model. There are 5 datasets and 2 settings (zero-shot and few-shot), so we have 5*2=10 commands for different datasets and settings. Please replace the cuda device in the command:

Reproduce zero shot task performance
 
```
python ../main.py \
  --filename=../data/datasets/cl.obo \
  --is_unseen --use_scheduler \
  --syn_ratio=3 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=1 \
  --lr=2.500e-4 --epoch_num=10 --pretrain_emb_iter=400 --use_get_ent_emb  --path_depth=2 --seed=0 \
  --save-model \
  --exp_path=exp/0cl.obo_unseen_log_simplepath_cst_kg10_ratio3121_pretrain400_depth2_lr2.500e-4_get_ent/    \
  > cache/0cl.obo_unseen_log_simplepath_cst_kg10_ratio3121_pretrain400_depth2_lr2.500e-4_get_ent \

python ../main.py \
  --filename=../data/datasets/doid.obo \
  --is_unseen --use_scheduler \
  --gpu_ids 1 \
  --syn_ratio=2 --ent_ratio=1 --hrt_ratio=1 --cst_ratio=1 \
  --lr=1.000e-4 --epoch_num=10 --pretrain_emb_iter=400 --use_get_ent_emb  --path_depth=2 --seed=0 \
  --exp_path=exp/0doid.obo_unseen_log_simplepath_cst_kg10_ratio2111_pretrain400_depth2_lr1.000e-4_get_ent/    \
  > cache/0doid.obo_unseen_log_simplepath_cst_kg10_ratio2111_pretrain400_depth2_lr1.000e-4_get_ent \

python ../main.py \
  --filename=../data/datasets/hp.obo \
  --is_unseen --use_scheduler \
  --gpu_ids 1\
  --syn_ratio=3 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=1 \
  --lr=1.500e-4 --epoch_num=10 --pretrain_emb_iter=400 --use_get_ent_emb  --path_depth=2 --seed=0 \
  --exp_path=exp/0hp.obo_unseen_log_simplepath_cst_kg10_ratio3121_pretrain400_depth2_lr1.500e-4_get_ent/    \
  > cache/0hp.obo_unseen_log_simplepath_cst_kg10_ratio3121_pretrain400_depth2_lr1.500e-4_get_ent \

python ../main.py \
  --filename=../data/datasets/fbbt.obo \
  --is_unseen --use_scheduler \
  --gpu_ids 1\
  --syn_ratio=3 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=1 \
  --lr=1.500e-4 --epoch_num=40 --pretrain_emb_iter=400 --use_get_ent_emb  --path_depth=2 --seed=0 \
  --exp_path=exp/0fbbt.obo_unseen_log_simplepath_cst_kg10_ratio3121_pretrain400_depth2_lr1.500e-4_get_ent/    \
  > cache/0fbbt.obo_unseen_log_simplepath_cst_kg10_ratio3121_pretrain400_depth2_lr1.500e-4_get_ent \

python ../main.py \
  --filename=../data/datasets/mp.obo \
  --is_unseen --use_scheduler \
  --gpu_ids 1\
  --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=1 \
  --lr=1.500e-4 --epoch_num=40 --pretrain_emb_iter=400 --use_get_ent_emb  --path_depth=2 --seed=0 \
  --exp_path=exp/0mp.obo_unseen_log_simplepath_cst_kg10_ratio2121_pretrain400_depth2_lr1.500e-4_get_ent/    \
  > cache/0mp.obo_unseen_log_simplepath_cst_kg10_ratio2121_pretrain400_depth2_lr1.500e-4_get_ent \

```

Reproduce few shot performance

```
python main.py \
  --filename=data/datasets/cl.obo  \
  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=0 \
  --lr=1.500e-04 --epoch_num=10 --pretrain_emb_iter=400   \
  --path_depth=2 --seed=0 --exp_path=exp/0cl.obo_seen_kg_path2/    \
  > 0cl.obo_seen_log_simplepath_cst_kg10_ratio2111_pretrain400_depth2_1.500e-04  2>&1\

python main.py \
  --filename=data/datasets/hp.obo  \
  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=0 \
  --lr=1.500e-04 --epoch_num=10 --pretrain_emb_iter=400   \
  --path_depth=2 --seed=0 --exp_path=exp/0hp.obo_seen_kg_path2/    \
  > 0hp.obo_seen_log_simplepath_cst_kg10_ratio2111_pretrain400_depth2_1.500e-04  2>&1\

python main.py \
  --filename=data/datasets/doid.obo  \
  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=0 \
  --lr=1.500e-04 --epoch_num=10 --pretrain_emb_iter=400   \
  --path_depth=2 --seed=0 --exp_path=exp/0doid.obo_seen_kg_path2/    \
  > 0doid.obo_seen_log_simplepath_cst_kg10_ratio2111_pretrain400_depth2_1.500e-04  2>&1\

python main.py \
  --filename=data/datasets/fbbt.obo  \
  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=0 \
  --lr=1.500e-04 --epoch_num=40 --pretrain_emb_iter=400   \
  --path_depth=2 --seed=0 --exp_path=exp/0fbbt.obo_seen_kg_path2/ \
  > 0fbbt.obo_seen_log_simplepath_cst_kg40_ratio2111_pretrain400_depth2_1.500e-04  2>&1\

python main.py \
  --filename=data/datasets/mp.obo  \
  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --cst_ratio=0 \
  --lr=2.000e-04 --epoch_num=10 --pretrain_emb_iter=400   \
  --path_depth=2 --seed=0 --exp_path=exp/0mp.obo_seen_kg_path2/    \
  > 0mp.obo_seen_log_simplepath_cst_kg10_ratio2111_pretrain400_depth2_2.000e-04  2>&1
```