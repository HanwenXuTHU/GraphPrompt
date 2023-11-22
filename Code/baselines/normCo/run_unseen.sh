#DATA_DIR='../data/datasets'
#
#MAX_DEPTH=15
#MAX_NODES=30
#SEARCH_METHOD=bfs
#MODEL=LSTM
#NUM_EPOCHS_MENTION_ONLY=200
#NUM_EPOCHS_WITH_COHERENCE=400
#BATCH_SIZE=32
#SEQUENCE_LEN=20
#EMBEDDING_DIM=200
#LR=0.0005
#EVAL_EVERY=1
#NUM_NEG=30
#python run_normco.py\
#  --data_dir $DATA_DIR\
#	--max_depth $MAX_DEPTH\
#	--max_nodes $MAX_NODES\
#	--search_method $SEARCH_METHOD\
#	--model $MODEL\
#	--num_epochs_mention_only $NUM_EPOCHS_MENTION_ONLY\
#	--num_epochs_with_coherence $NUM_EPOCHS_WITH_COHERENCE\
#	--batch_size $BATCH_SIZE\
#	--sequence_len $SEQUENCE_LEN\
#	--lr $LR\
#	--eval_every $EVAL_EVERY\
#	--num_neg $NUM_NEG\
#	--dataset_nm $1\
#	--save_only\
#	--is_unseen


DATA_DIR='../data/datasets'

MAX_DEPTH=15
MAX_NODES=30
SEARCH_METHOD=bfs
MODEL=GRU
NUM_EPOCHS_MENTION_ONLY=10
NUM_EPOCHS_WITH_COHERENCE=40

BATCH_SIZE=256
SEQUENCE_LEN=10
EMBEDDING_DIM=512
LR=0.0005
EVAL_EVERY=50
NUM_NEG=128
python run_normco.py\
  --data_dir $DATA_DIR\
	--max_depth $MAX_DEPTH\
	--max_nodes $MAX_NODES\
	--search_method $SEARCH_METHOD\
	--model $MODEL\
	--num_epochs_mention_only $NUM_EPOCHS_MENTION_ONLY\
	--num_epochs_with_coherence $NUM_EPOCHS_WITH_COHERENCE\
	--batch_size $BATCH_SIZE\
	--sequence_len $SEQUENCE_LEN\
	--lr $LR\
	--eval_every $EVAL_EVERY\
	--num_neg $NUM_NEG\
	--dataset_nm "all"\
	--start 0\
	--end 1\
	--is_unseen\



