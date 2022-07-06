EXP_NAME=record-${NAME}-${TIMESTAMP}
TASK_NAME=ReCoRD
DATA_PATH="${DATA_ROOT}/ReCoRD"
MAX_SEQ_LEN=512

LR_SINGLE=3e-3
EPOCH_SINGLE=12
XXLARGE_EPOCH=3

TRAIN_ARGS="--lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --weight-decay 1.0e-4 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 1000000 \
             --log-interval 1 \
             --eval-interval 1000000 \
             --eval-iters 100 \
             --fast-decode \
             --tgt-seq-length 32"


PATTERN_IDS=(0)

BATCH_SIZE=64