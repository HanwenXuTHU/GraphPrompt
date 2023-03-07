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