#! /bin/bash

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/thudm/LargeScale/group2"
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

DATA_ROOT="/thudm/LargeScale/data/superglue"

source $1 # model
source $2 # task

mkdir -p logs/${EXP_NAME}

MICRO_BATCH_SIZE=$(($BATCH_SIZE/$NUM_WORKERS/$NUM_GPUS_PER_WORKER))

args="./tasks/main.py \
       --seed 1234 \
       --task ${TASK_NAME} \
       --pretrained-checkpoint ${CHECKPOINT_PATH} \
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
        "
#       --checkpoint-activations \

run_cmd="PYTHONPATH=/thudm/LargeScale/packages ${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${args}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}/output.log
