#! /bin/bash

NUM_WORKERS=8
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/root/hostfile"
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=1"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

source $1
mkdir -p logs/${NAME}

run_cmd="PYTHONPATH=/thudm/packages ${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${script_path} ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee logs/${NAME}/output.log
