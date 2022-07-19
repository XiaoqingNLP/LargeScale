#! /bin/bash

TP_SIZE=8
PP_SIZE=1

NHIDDEN=12288
FFN_HIDDEN=$((NHIDDEN * 8 / 3))
NLAYERS=70
NHEADS=96

GLM_ARGS="
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --num-attention-heads $NHEADS \
       --make-vocab-size-divisible-by 384 \
       --glm \
       --gpt-prob 0.7 \
       --single-span-prob 0.02 \
       --short-seq-prob 0.02 \
       --mask-prob 0.15 \
       --average-block-length 3 \
       --min-gmask-ratio 0.2 \
       --deepnorm \
       --apply-residual-connection-post-layernorm \
       --position-embedding-type rotary \
       --ffn-hidden-size $FFN_HIDDEN \
       --glu-activation geglu \
       --no-bias-gelu-fusion \
    "