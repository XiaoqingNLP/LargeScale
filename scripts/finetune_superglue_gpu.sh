#! /bin/bash

WORLD_SIZE=8
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

DATA_ROOT="/sharefs/cognitive/dataset/fd5061f60d4dd7a8e055690bd68d1e2c/english_data_new/superglue"

source $1 # model
source $2 # task

mkdir -p logs/${EXP_NAME}

MICRO_BATCH_SIZE=$(($BATCH_SIZE/$WORLD_SIZE))

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
       --seed 1234 \
       --task ${TASK_NAME} \
       --train-data ${DATA_PATH} \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --seq-length ${MAX_SEQ_LEN} \
       --epochs ${EPOCH_SINGLE} \
       --lr ${LR_SINGLE} \
       --optimizer adam \
       --tokenizer-type IceTokenizer \
       --fp16 \
       ${GLM_ARGS} \
       ${TRAIN_ARGS} \
       ${COMMON_ARGS} \
       --prefix-prompt-length 16 \
       2>&1 | tee logs/${EXP_NAME}/output.log
#       --checkpoint-activations \
