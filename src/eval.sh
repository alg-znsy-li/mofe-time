export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 main.py  \
    --do_eval \
    --eval_data_path $1 \
    --output_path  /lpai/output/data \
    --batch_size 32 \
    --eval_context_length $3 \
    --prediction_length $4 \
    --model_path $5 \
    --ckpt_path $2 \
    --epochs 1 \
    --use_ds \
    --version tm_v0.1 2>&1 | tee -a /lpai/output/logs/log.txt
