TASK_NAME=COPA
EXP_NAME=copa-${NAME}-${TIMESTAMP}
DATA_PATH="${DATA_ROOT}/COPA"
MAX_SEQ_LEN=256

LR_SINGLE=8e-3
EPOCH_SINGLE=100
XXLARGE_EPOCH=100

TRAIN_ARGS="--lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --weight-decay 1.0e-4 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 1000000 \
             --log-interval 100 \
             --eval-interval 1000000 \
             --eval-iters 100 \
             --fast-decode \
             --tgt-seq-length 32"

PATTERN_IDS=(0 1)
PROMPT_IDS=(1 2)

BATCH_SIZE=8