# 激活环境
# conda activate diffusers-0-27-0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main.py  \
    --do_finetune \
    --do_eval \
    --ckpt_path /lpai/inputs/models/time-series-timemoe-res-25-06-12-mofe-20M/time_moe_timemoe_rep/ \
    --eval_data_path /mnt/datasets/ts-benchmark/0-1-0/Benchmark/traffic/traffic.csv \
    --output_path /lpai/output/models \
    --lr 0.00001 \
    --batch_size 16 \
    --eval_context_length 512 \
    --prediction_length 96 \
    --model_path ../cfg/mofe_20m.json \
    --epochs 1 \
    --use_ds 2>&1 | tee -a /lpai/output/logs/log.txt
