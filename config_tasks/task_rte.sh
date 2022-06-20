TASK_NAME=RTE
EXP_NAME=rte-${NAME}-${TIMESTAMP}
DATA_PATH="${DATA_ROOT}/RTE"
MAX_SEQ_LEN=256

LR_SINGLE=2e-5
EPOCH_SINGLE=30
XXLARGE_EPOCH=50

TRAIN_ARGS="
            --lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50000 \
             --eval-interval 10000000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=8