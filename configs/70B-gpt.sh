#! /bin/bash

DATA_PATH="/thudm/data/wudao_corpus.part_0"
CHECKPOINT_PATH="/thudm/checkpoints/debug"

NAME="gpt-70B-embed-norm-0.015%-warmup-4%-bszwarmup"-${TIMESTAMP}

config_json="./logs/${NAME}/ds_config.json"

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024

TP_SIZE=4
PP_SIZE=4

NHIDDEN=9216
NLAYERS=54
NHEADS=64
LENGTH_PER_SAMPLE=513 # sequence length per sample from BinaryDataset
SEQ_LEN=512 # actual length during training (pad to this)

SAVE_INTERVAL=500

TRAIN_TOKENS=300000000000 # 300B tokens
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))  # Decay for the first 90% tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 15 / 10000))  # 0.015% warmup
#LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 3 / 100))  # 3% warmup
BATCH_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 4 / 100))  # 4% warmup

ZERO_STAGE=1

script_path="pretrain_gpt.py"

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
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
       --tokenizer-type IceTokenizer \
       --sandwich-ln \
       --apply-pb-relax \
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
       --rampup-batch-size 128 16 $BATCH_WARMUP_SAMPLES \
       --train-samples $TRAIN_SAMPLES \
       --length-per-sample $LENGTH_PER_SAMPLE \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --data-path $DATA_PATH \
       --abort-on-unmet-fused-kernel-constraints \
       --split 949,50,1 \
       --distributed-backend nccl \
       --init-method-std 0.0048 \
       --checkpoint-activations \
       --embed-layernorm \
       --fp16 \
       $OPTIMIZER_ARGS \
       $DEEPSPEED_ARGS \
       $OUTPUT_ARGS
"
#       --shrink-embedding-gradient-alpha 0.1 \
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
