#!/bin/bash

# 启动分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --nnodes=$NODE_NUM \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py\
    --do_train \
    --do_eval \
    --train_data_path ../data/Time-300B-4Test/ \
    --eval_data_path ../data/Benchmark/ETT-small/ETTh1.csv \
    --output_path  ../output/models/ \
    --lr 0.001 \
    --batch_size 2 \
    --context_length 4096 \
	--evaluate_step_num 10240 \
    --eval_context_length 512 \
    --prediction_length 96 \
    --model_path ../cfg/mofe_100m.json \
    --epochs 1 \
    --use_ds \
    --version mofe_100m_pre

echo "Training started on node $RANK"
