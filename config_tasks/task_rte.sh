TASK_NAME=RTE
EXP_NAME=rte-${NAME}-${TIMESTAMP}
DATA_PATH="${DATA_ROOT}/RTE"
MAX_SEQ_LEN=256

LR_SINGLE=3e-3
EPOCH_SINGLE=100
XXLARGE_EPOCH=50

TRAIN_ARGS="
            --lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --weight-decay 1.0e-4 \
            --pattern-id 6"

COMMON_ARGS="--save-interval 10000000 \
             --log-interval 100 \
             --eval-interval 10000000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3 4 5 6)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16