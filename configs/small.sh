#! /bin/bash

DATA_PATH="/root/wudao_corpus.part_0"
CHECKPOINT_PATH="/thudm/checkpoints/debug"

NAME="glm-base-batch-collator"-${TIMESTAMP}

config_json="./logs/${NAME}/ds_config.json"

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=512

TP_SIZE=1
PP_SIZE=1

NHIDDEN=768
NLAYERS=12
NHEADS=12
LENGTH_PER_SAMPLE=500 # sequence length per sample from BinaryDataset
SEQ_LEN=512 # actual length during training (pad to this)

SAVE_INTERVAL=500

TRAIN_ITERS=120000 # 10B tokens

ZERO_STAGE=1

script_path="pretrain_glm.py"

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --adam-eps 1e-8 \
    --lr 4e-4 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.05 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 3 \
    --tensorboard-dir runs/${NAME} \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

GLM_ARGS="
       --glm \
       --gpt-prob 0.2 \
       --single-span-prob 0.02 \
       --short-seq-prob 0.02 \
       --mask-prob 0.15 \
       --average-block-length 3 \
       --min-gmask-ratio 0.2 \
       --sandwich-ln \
       --apply-pb-relax \
       --num-workers 1 \
    "

DEEPSPEED_ARGS=" \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --zero-stage $ZERO_STAGE \
       --deepspeed-activation-checkpointing \
    "

#       --rampup-batch-size 128 16 9_765_625 \
#       --save $CHECKPOINT_PATH \
gpt_options=" \
       $GLM_ARGS \
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --pp-partition-method 'type:transformer|embedding' \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --num-attention-heads $NHEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-iters $TRAIN_ITERS \
       --length-per-sample $LENGTH_PER_SAMPLE \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --data-path $DATA_PATH \
       --abort-on-unmet-fused-kernel-constraints \
       --split 949,50,1 \
       --distributed-backend nccl \
       --checkpoint-activations \
       --fp16 \
       $OPTIMIZER_ARGS \
       $DEEPSPEED_ARGS \
       $OUTPUT_ARGS
"
#       --init-method-std 0.0048 \
#       --shrink-embedding-gradient-alpha 0.1 \
#       --embed-layernorm \
#       --embedding-init-std 1e-5 \

mkdir -p logs/${NAME}
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 200,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "steps_per_print": 1000,
  "wall_clock_breakdown": false
}
EOT
