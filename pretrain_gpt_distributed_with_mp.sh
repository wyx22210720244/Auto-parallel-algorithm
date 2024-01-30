#!/bin/bash

# Runs the "7B" parameter model
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
GPUS_PER_NODE=4
# Change for multinode config
#MASTER_ADDR=localhost
#MASTER_PORT=6000
#NNODES=1
#NODE_RANK=0
#WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/root/Megatron-LM/checkpoints/gpt2 #自定义ckp路径
VOCAB_FILE=/root/Megatron-LM/data/gpt2-vocab.json
MERGE_FILE=/root/Megatron-LM/data/gpt2-merges.txt
DATA_PATH=/root/Megatron-LM/data/meg-gpt2-oscar-en-10k_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --num-layers 4 \
    --hidden-size 512 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 50000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"
#export CUDA_VISIBLE_DEVICES=0
torchrun $DISTRIBUTED_ARGS /root/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
