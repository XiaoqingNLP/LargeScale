TASK_NAME=WiC
EXP_NAME=wic-${NAME}-${TIMESTAMP}
DATA_PATH="${DATA_ROOT}/WiC"
MAX_SEQ_LEN=256

LR_SINGLE=1e-5
EPOCH_SINGLE=100
XXLARGE_EPOCH=50

TRAIN_ARGS="
            --lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --weight-decay 1.0e-4 \
            --fast-decode \
            --pattern-id 3"

COMMON_ARGS="--save-interval 10000000 \
             --log-interval 100 \
             --eval-interval 10000000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16