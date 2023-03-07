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
