#!/bin/bash
source scripts/models/megatron/source.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3

TP=4
PP=1
LR=1e-5
MODEL_NAME=Mistral-7b

SEQ_LENGTH=24576

# dir name under data/megatron_format: e.g., data/megatron_format/{DATA_NAME}/data
DATA_NAME=pvd_160k
N_EXAMPLES=7842 # packed document length
N_EPOCHS=3

echo "DATA_NAME=$DATA_NAME"
echo "N_EXAMPLES=$N_EXAMPLES"
echo "N_EPOCHS=$N_EPOCHS"

GLOBAL_BATCH_SIZE=32
N_GPU_PER_NODE=4

WANDB_ENTITY="your_entity" # TODO: change this to your wandb entity

# set warmup steps
N_BATCHES_PER_EPOCH=$((($N_EXAMPLES/$GLOBAL_BATCH_SIZE)+1))
TRAIN_ITERATIONS=$(($N_BATCHES_PER_EPOCH*$N_EPOCHS))
N_WARMUP_STEPS=$(($TRAIN_ITERATIONS/10))
echo "N_WARMUP_STEPS=$N_WARMUP_STEPS"
echo "TRAIN_ITERATIONS=$TRAIN_ITERATIONS"

SAVE_INTERVAL=$N_BATCHES_PER_EPOCH
echo "SAVE_INTERVAL=$SAVE_INTERVAL"

train \
 --tp $TP \
 --pp $PP \
 --lr $LR \
 --n_warmup_steps $N_WARMUP_STEPS \
 --seq_length $SEQ_LENGTH \
 --model_name $MODEL_NAME \
 --data_name $DATA_NAME \
 --n_examples $N_EXAMPLES \
 --global_batch_size $GLOBAL_BATCH_SIZE \
 --n_epochs $N_EPOCHS \
 --n_gpu_per_node $N_GPU_PER_NODE \
 --recompute_num_layers 2 \
 --wandb_entitry $WANDB_ENTITY \
 --use_flash_attn \
 --packed_input