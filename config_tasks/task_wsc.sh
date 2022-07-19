TASK_NAME=WSC
EXP_NAME=wsc-${NAME}-${TIMESTAMP}
DATA_PATH="${DATA_ROOT}/WSC-negative"
MAX_SEQ_LEN=128

LR_SINGLE=1e-5
EPOCH_SINGLE=100
XXLARGE_EPOCH=50

TRAIN_ARGS="--lr-decay-style linear \
            --lr-warmup-fraction 0.1 \
            --weight-decay 1.0e-4 \
            --wsc-negative \
            --length-penalty 1 \
            --loss-func mix \
            --fast-decode \
            --pattern-id 3"

COMMON_ARGS="--save-interval 10000000 \
             --log-interval 1 \
             --eval-interval 10000000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16