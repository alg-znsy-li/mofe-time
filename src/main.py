import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import argparse

from mofe_trainer import MoFETrainer,MoFEDeepspeedTrainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path',  type=str, default="../out", help='Path to output_path (Folder).')
    parser.add_argument('--ckpt_path',  type=str, default=None, help='Path to ckpt_path (Folder).')
    parser.add_argument('--train_data_path',  type=str, default="../data/Time-300B-4Test/", help='Path to training data. (Folder contains data files, or data file)')
    parser.add_argument('--eval_data_path',  type=str, default="../data/Benchmark/ETT-small/ETTh1.csv", help='Path to eval data. (Folder contains data files, or data file)')
    parser.add_argument('--model_path', '-m', type=str, default='../cfg/horae_50m.json', help='Path to model config.')

    parser.add_argument('--normalization_method', type=str, choices=['none', 'zero', 'max'], default='zero',help='normalization method for sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size per gpu')
    parser.add_argument('--warmup_steps', type=int, default=128, help='warmup steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=5e-7, help='min learning rate')
    parser.add_argument('--gamma', type=float, default=0.9999, help='learning rate decay')
    parser.add_argument('--epochs', type=int, default=9899, help='epoch number')
    parser.add_argument('--seed', type=int, default=9899, help='seed number')
    parser.add_argument('--version', type=str, default='v1.0', help='version number')
    parser.add_argument('--print_step_num', type=int, default=10, help='print_step_num')
    parser.add_argument('--evaluate_step_num', type=int, default=1024, help='evaluate_step_num')
    parser.add_argument('--save_step_num', type=int, default=10240, help='evaluate_step_num')

    parser.add_argument('--context_length', type=int, default=1024, help='context_length')
    parser.add_argument('--eval_context_length', type=int, default=512, help='eval context_length')
    parser.add_argument('--prediction_length', type=int, default=1, help='prediction_length')

    parser.add_argument('--do_train', action='store_true', default=False, help='whether to train model')
    parser.add_argument('--do_eval', action='store_true', default=False, help='whether to eval model')
    parser.add_argument('--do_finetune', action='store_true', default=False, help='whether to finetune model')

    parser.add_argument('--use_ds', action='store_true', default=False, help='whether to use deepspeed')

    args = parser.parse_args()

    print('============== args ================')
    print('train_data_path: ', args.train_data_path)
    print('eval_data_path: ', args.eval_data_path)
    print('output_path: ', args.output_path)
    print('lr: ', args.lr)
    print('batch_size: ', args.batch_size)
    print('context_length: ', args.context_length)
    print('eval_context_length: ', args.eval_context_length)
    print('prediction_length: ', args.prediction_length)
    print('model_path: ', args.model_path)
    print('epochs: ', args.epochs)
    print('version: ', args.version)

    if args.normalization_method == 'none':
        args.normalization_method = None
    CLS_NM= {
        "single_gpu": MoFETrainer,
        "multi_gpu": MoFEDeepspeedTrainer
    }
    TrainerCLS = CLS_NM['single_gpu']
    if args.use_ds:
        TrainerCLS = CLS_NM['multi_gpu']


    runner = TrainerCLS(
        model_path=args.model_path,
        output_path=args.output_path,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
        train_config=args
    )
    if args.do_finetune:
        runner.fine_tune_model()
    else:
        runner.train_and_eval_model()
