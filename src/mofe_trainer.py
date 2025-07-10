import os
from os import path

import deepspeed
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from time_moe.datasets.benchmark_dataset import BenchmarkTestDataset, BenchmarkDataset, BenchmarkFinetuneDataset
from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
from time_moe.utils.log_util import get_logger, is_local_rank_0
from models.modeling_mofe import MoFEPrediction
from models.configuration_mofe import MoFEConfig


class WarmupExponentialLR(LRScheduler):
    """linear warmups and then decays the learning rate of each parameter group by gamma every step."""

    def __init__(self, optimizer, gamma, warmup_step, min_lr=0.0000001, verbose=False):
        self.gamma = gamma
        self.warmup_step = warmup_step
        self.min_lr = min_lr
        super().__init__(optimizer, -1, verbose)

    def get_lr(self):

        if self._step_count < self.warmup_step:
            ratio = float(self._step_count) / float(self.warmup_step)
            return [group['initial_lr'] * ratio for group in self.optimizer.param_groups]

        else:
            return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]


class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0, device='cuda:0'):
        self.name = name
        self.value = torch.tensor(data=init_val, dtype=torch.float64, device=device)

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass

class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        tmp = (preds - labels) ** 2
        tmp = tmp.to(dtype=torch.float64)
        return torch.sum(tmp)

class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        tmp = torch.abs(preds - labels)

        tmp = tmp.to(dtype=torch.float64)
        return torch.sum(tmp)

class MoFETrainer:

    def __init__(self,
            output_path = '../out',
            ckpt_path = None,
            seed = 9899,
            model_path = 'Maple728/TimeMoE-50M',
            train_config = None
    ):
        self.output_path = output_path
        self.seed = seed
        self.model_path = model_path
        self.train_config = train_config
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
        else:
            self.ckpt_path = os.path.join(self.output_path,
                                      f"./checkpoint/time_moe_{self.train_config.version}")
        self.save_ckpt_path = os.path.join(self.output_path, f"./checkpoint/{self.train_config.version}")
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_path, 'tensorboard', self.train_config.version),
                                    filename_suffix="lixiang")

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.dtype = torch.float32

        self.print_step_num = self.train_config.print_step_num
        self.evaluate_step_num = self.train_config.evaluate_step_num
        self.save_step_num = self.train_config.save_step_num

        self.bc_dataset = BenchmarkDataset(csv_path=self.train_config.eval_data_path,
                                           context_length=self.train_config.eval_context_length,
                                           prediction_length=self.train_config.prediction_length)

    def count_num_tensor_elements(self, a):
        n = 1
        for s in a.shape:
            n = n * s
        return n

    def load_model(self):
        config = MoFEConfig.from_pretrained(pretrained_model_name_or_path=self.model_path)
        inst = MoFEPrediction(config=config)

        if self.train_config.use_ds == False:
            inst.to(self.device)
            return inst
        return inst

    def get_train_dataset(self, data_path, max_length, normalization_method):

        dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
        window_dataset = TimeMoEWindowDataset(dataset, context_length=max_length,
                                              prediction_length=0, shuffle=True)
        return window_dataset

    def get_eval_dataset(self):
        dataset = BenchmarkTestDataset(dataset_inst=self.bc_dataset)
        return dataset

    def get_finetune_dataset(self):
        dataset = BenchmarkFinetuneDataset(dataset_inst=self.bc_dataset)
        return dataset

    def get_tensorboard(self ):
        writer_path = path.join(self.output_path, "tensorboard")
        writer = SummaryWriter(log_dir=writer_path, filename_suffix="lixiang")
        return writer

    def create_optimizer(self,
                         params,
                         num_warmup_steps,
                         lr,
                         min_lr,
                         gamma=0.9999):

        optimizer = torch.optim.AdamW(params=params, lr=lr)
        lr_scheduler = WarmupExponentialLR(optimizer=optimizer, warmup_step=num_warmup_steps, min_lr=min_lr, gamma=gamma)

        return optimizer, lr_scheduler

    def save_checkpoint(self, step, model, optimizer, loss, file_name="./ckpt/time_moe.pt"):
        state = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        get_logger("info").info(f'loss: {loss.item():.4f}\tepoch: {step}\tstep: {step}')
        torch.save(state, file_name)

    def load_checkpoint(self,checkpoint_path, model, optimizer):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step = checkpoint['step']
            loss = checkpoint['loss']
            get_logger("log").info("=> Loaded checkpoint '{}' (step {}；loss {})".format(checkpoint_path,
                                                                                   step, loss.item()))
            return step
        else:
            get_logger("log").info("=> No checkpoint found at '{}'".format(checkpoint_path))
            return 0

    def load_checkpoint_bin(self,checkpoint_path, model, optimizer):
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)

            get_logger("log").info("=> Loaded checkpoint '{}' ".format(checkpoint_path))
            return 0
        else:
            get_logger("log").info("=> No checkpoint found at '{}'".format(checkpoint_path))
            return 0

    def predict(self, model_inst, batch, prediction_length=16):

        model = model_inst
        prediction_length = prediction_length

        outputs = model.generate(
            inputs=batch['inputs'].to(model.dtype),
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels']
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels

    def evaluate_model(self, model_inst=None):

        model_inst.eval()

        ds = self.get_eval_dataset()
        self.eval_ds = ds
        sampler = None
        if self.train_config.use_ds:
            sampler = DistributedSampler(ds)

        evaluation_dataloader = DataLoader(
            dataset=ds,
            batch_size=self.train_config.batch_size,
            sampler=sampler,
            drop_last=False,
        )



        acc_count = 0
        ii_idx = 0
        self.device = model_inst.device
        self.dtype = model_inst.dtype
        metric_list = [
            MSEMetric(name='mse', init_val=0.0, device=self.device),
            MAEMetric(name='mae', init_val=0.0, device=self.device),
        ]
        with torch.no_grad():
            for batch in evaluation_dataloader:
                for k,v in batch.items():
                    batch[k] = batch[k].to(self.device).to(self.dtype)

                preds, labels = self.predict(
                    model_inst=model_inst, batch=batch, prediction_length=self.train_config.prediction_length)

                #preds = preds * torch.unsqueeze(batch['var'], dim=-1) + torch.unsqueeze(batch['mean'], dim=-1)
                #labels = labels * torch.unsqueeze(batch['var'], dim=-1) + torch.unsqueeze(batch['mean'], dim=-1)


                for metric in metric_list:
                    metric.push(preds, labels)

                acc_count += self.count_num_tensor_elements(preds)
                ii_idx += 1
                if ii_idx % 100 == 0:
                    try:
                        if is_local_rank_0():
                            for metric in metric_list:
                                get_logger("log").info(f"=> eval {ii_idx} of {len(evaluation_dataloader)} samples :"
                                                       f" metric/{metric.name}:{metric.value.item()/acc_count}")
                    except:
                        get_logger("log").info(f"=> eval {ii_idx} of {len(evaluation_dataloader)} samples")

        model_inst.train()

        ret_metrics = {}
        for metric in metric_list:
            ret_metrics[metric.name] = metric.value.item()
        return ret_metrics, acc_count

    def train_and_eval_model(self):

        train_ds = self.get_train_dataset(self.train_config.train_data_path,
                                          max_length=self.train_config.context_length,
                                          normalization_method=self.train_config.normalization_method)

        training_data_loader = DataLoader(train_ds, batch_size=self.train_config.batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)

        model_inst = self.load_model()
        model_inst.train()


        #writer = self.writer
        CKPT_PATH = self.ckpt_path + ".bin"
        optimizer, lr_scheduler = self.create_optimizer(params=model_inst.parameters(),
                                                        num_warmup_steps=self.train_config.warmup_steps,
                                                        lr=self.train_config.lr,
                                                        min_lr=self.train_config.min_lr,
                                                        gamma=self.train_config.gamma)

        epochs = self.train_config.epochs
        ii_steps = 0
        step = self.load_checkpoint(checkpoint_path=CKPT_PATH, model=model_inst, optimizer=optimizer)

        get_logger("log").info("TTTT=> Training dataset length: {}".format(len(training_data_loader)))
        loss = -1
        if not self.train_config.do_train:
            epochs = 0
        for epoch in range(epochs):
            for ele in training_data_loader:
                if ii_steps < step:
                    ii_steps += 1
                    continue
                for k,v in ele.items():
                    ele[k] = ele[k].to(self.device).to(self.dtype)
                ii_steps += 1
                ret = model_inst(**ele)
                loss = ret["loss"]
                #get_logger("log").info("\n\n\nloss is {}".format(loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()


                if ii_steps % self.print_step_num == 0:
                    self.writer.add_scalar('loss', loss, ii_steps)
                    msg = f'loss: {loss:.4f}\tepoch: {epoch}\tstep: {ii_steps}\tlr: {optimizer.param_groups[0]["lr"]}'
                    get_logger("log").info(msg)

                if ii_steps % self.evaluate_step_num == 0:
                    if self.train_config.do_eval:
                        get_logger("log").info("evaluation ...")
                        self.evaluate_model(model_inst)

                    get_logger("log").info("save model ...")
                    self.save_checkpoint(step=ii_steps, model=model_inst, optimizer=optimizer,
                                         loss=loss, file_name=CKPT_PATH)
                self.writer.flush()

        if self.train_config.do_eval:
            get_logger("log").info("evaluation ...")
            self.evaluate_model(model_inst)
        if self.train_config.do_train:
            get_logger("log").info("save model ...")
            self.save_checkpoint(step=ii_steps, model=model_inst, optimizer=optimizer,
                                 loss=loss, file_name=CKPT_PATH)


        self.writer.close()


class MoFEDeepspeedTrainer(MoFETrainer):
    DS_CONFIG_BF16_STAGE2 = {
        "train_batch_size": 1024,
        "train_micro_batch_size_per_gpu": 16,
        "gradient_clipping": 0.6,
        "steps_per_print": 100,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "Adam",
            "params": {
              "lr": 0.001,
              "weight_decay": 0.1,
              "betas": [0.9, 0.95]
            }
        }
    }

    def __init__(self, output_path = '../out', seed = 9899,
                 model_path = 'Maple728/TimeMoE-50M', ckpt_path=None, train_config =None):
        super().__init__(output_path, ckpt_path, seed, model_path, train_config)
        MoFEDeepspeedTrainer.DS_CONFIG_BF16_STAGE2["train_micro_batch_size_per_gpu"] = train_config.batch_size

    def save_checkpoint(self, step, model, optimizer, loss, file_name="./ckpt/time_moe_ds"):
        client_sd = {
                'step': step,
                'loss': loss,
        }
        get_logger("log").info(f'loss: {loss:.4f} \tstep: {step}')
        model.save_checkpoint(save_dir=file_name, client_state = client_sd)

    def load_checkpoint(self, checkpoint_path, model, optimizer):
        if os.path.exists(checkpoint_path):

            get_logger("log").info("=> Checkpoint found at '{}' ".format(checkpoint_path))
            _, client_sd = model.load_checkpoint(checkpoint_path)
            step = client_sd['step']
            loss = client_sd['loss']
            get_logger("log").info("=> Checkpoint found at '{}' and loss is {} after step {}"
                                   .format(checkpoint_path, loss, step))

            return step

        else:
            get_logger("log").info("=> No checkpoint found at '{}'".format(checkpoint_path))
            return 0

    def train_and_eval_model(self, ds_config=DS_CONFIG_BF16_STAGE2):

        CKPT_PATH = self.ckpt_path
        model_inst = self.load_model()
        if is_local_rank_0():
            get_logger("log").info(model_inst)
        model_inst.train()
        if ds_config['bf16']['enabled']:
            model_inst.half()
        optimizer, lr_scheduler = self.create_optimizer(params=model_inst.parameters(),
                                                        num_warmup_steps=self.train_config.warmup_steps,
                                                        lr=self.train_config.lr,
                                                        min_lr=self.train_config.min_lr,
                                                        gamma=self.train_config.gamma)

        epochs = self.train_config.epochs
        step = 0
        ii_steps: int = 0
        step = super().load_checkpoint_bin(checkpoint_path=CKPT_PATH, model=model_inst,
                                    optimizer=optimizer)
        if self.train_config.do_train:
            train_ds = self.get_train_dataset(self.train_config.train_data_path,
                                            max_length=self.train_config.context_length,
                                            normalization_method=self.train_config.normalization_method)
            model_engine_ds, optimizer_ds, training_data_loader_ds, lr_scheduler_ds \
                = deepspeed.initialize(args=None,
                                    model=model_inst,
                                    training_data=train_ds,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    config=ds_config)
            get_logger("log").info("TTTT=> Train Dataset length:{} \t Training DataLoader length: {}"
                               .format(len(train_ds), len(training_data_loader_ds)))

        else:
            epochs = 0
            model_engine_ds = deepspeed.initialize(model=model_inst, optimizer=optimizer, config=ds_config)[0]
            get_logger("log").info("EEEE=> initialize model for prediction")

        # step = self.load_checkpoint(checkpoint_path=CKPT_PATH, model=model_engine_ds,
        #                             optimizer=optimizer)
                               

        self.device=model_engine_ds.device
        self.dtype = model_engine_ds.dtype

        loss = -1
        for epoch in range(epochs):
            for batch in training_data_loader_ds:
                if ii_steps < step:
                    ii_steps += 1
                    continue

                # 数据发送设备
                for k, v in batch.items():
                    tmp = batch[k].to(device=self.device, dtype=self.dtype)
                    batch[k] = tmp

                ii_steps += 1

                # model_engine_ds.zero_grad()
                ret = model_engine_ds(**batch)
                loss = ret["loss"]

                model_engine_ds.backward(loss)
                model_engine_ds.step()

                if ii_steps % self.print_step_num == 0:
                    if is_local_rank_0():
                        self.writer.add_scalar('loss', loss.to(dtype=torch.float32), ii_steps)
                        msg = f'loss: {loss:.4f}\tepoch: {epoch}\tstep: {ii_steps}\tlr: {optimizer.param_groups[0]["lr"]}'
                        get_logger("log").info(msg)

                if ii_steps % self.evaluate_step_num == 0:

                    if self.train_config.do_eval:
                        get_logger("log").info("evaluation ...")
                        self.evaluate_model(model_engine_ds, train_step=ii_steps)

                if ii_steps % self.save_step_num == 0:
                    get_logger("log").info("save model ...")
                    self.save_checkpoint(step=ii_steps, model=model_engine_ds, optimizer=optimizer,
                                         loss=loss, file_name=CKPT_PATH)
        if self.train_config.do_eval:
            get_logger("log").info("evaluation ...")
            self.evaluate_model(model_engine_ds, train_step=ii_steps)
        if self.train_config.do_train:
            get_logger("log").info("save model ...")
            self.save_checkpoint(step=ii_steps, model=model_engine_ds, optimizer=optimizer,
                                 loss=loss, file_name=CKPT_PATH)

    def evaluate_model(self, model_inst=None, train_step=0):

        self.device = model_inst.device

        metric_dict, cnt = super().evaluate_model(model_inst=model_inst)

        cnt_tensor = torch.tensor(data=cnt, device=self.device, dtype=torch.float32)
        metric_dict_tensor = {}
        for ele in metric_dict.keys():
            metric_dict_tensor[ele]= torch.tensor(data=metric_dict[ele], device=self.device, dtype=torch.float32)

        dist.barrier()
        dist.all_reduce(tensor=cnt_tensor, op=dist.ReduceOp.SUM, async_op=True)
        dist.barrier()
        for ele in metric_dict.keys():
            dist.all_reduce(tensor=metric_dict_tensor[ele], op=dist.ReduceOp.SUM, async_op=True)
            dist.barrier()
            if is_local_rank_0():
                get_logger("log").info(
                    msg=f"\n\n==============================================\n"
                        f"metric/{ele}:{metric_dict_tensor[ele].item() / cnt_tensor.item()}\n"
                        f"ckpt_path: {self.ckpt_path}\n"
                        f"prediction_length: {self.train_config.prediction_length}\n"
                        f"eval_data_path: {self.train_config.eval_data_path}\n"
                        f"\n==============================================\n\n"
                )
                self.writer.add_scalar(f"metric/{ele}", metric_dict_tensor[ele] / cnt_tensor, train_step)

    def fine_tune_model(self, ds_config=DS_CONFIG_BF16_STAGE2):
        get_logger("log").info("fine_tune_model ...\n\n\n")
        model_inst = self.load_model()
        model_inst.train()
        CKPT_PATH = self.ckpt_path
        epochs = self.train_config.epochs
        if ds_config['bf16']['enabled']:
            model_inst.half()

        train_ds = self.get_finetune_dataset()
        optimizer, lr_scheduler = self.create_optimizer(params=model_inst.parameters(),
                                                        num_warmup_steps=self.train_config.warmup_steps,
                                                        lr=self.train_config.lr,
                                                        min_lr=self.train_config.min_lr,
                                                        gamma=self.train_config.gamma)

        _ = super().load_checkpoint_bin(checkpoint_path=CKPT_PATH, model=model_inst, optimizer=optimizer)
        model_engine_ds, optimizer_ds, training_data_loader_ds, lr_scheduler_ds \
            = deepspeed.initialize(args=None,
                                   model=model_inst,
                                   training_data=train_ds,
                                   optimizer=optimizer,
                                   lr_scheduler=lr_scheduler,
                                   config=ds_config)

        get_logger("log").info("TTTT=> Finetune Dataset length:{} \t Finetune DataLoader length: {}"
                               .format(len(train_ds), len(training_data_loader_ds)))
        # if not self.train_config.do_finetune:
        #     step = self.load_checkpoint(checkpoint_path=os.path.join(CKPT_PATH,".ft"),
        #                                 model=model_engine_ds,
        #                                 optimizer=optimizer)
        #     epochs = 0
        # else:
        #     step = self.load_checkpoint(checkpoint_path=CKPT_PATH,
        #                                 model=model_engine_ds,
        #                                 optimizer=optimizer)

        self.device = model_engine_ds.device
        self.dtype = model_engine_ds.dtype
        loss = -1
        ii_steps = 0
        for epoch in range(epochs):
            for batch in training_data_loader_ds:
                # if ii_steps < step:
                #     ii_steps += 1
                #     continue

                # 数据发送设备
                for k, v in batch.items():
                    tmp = batch[k].to(device=self.device, dtype=self.dtype)
                    batch[k] = tmp

                ii_steps += 1

                # model_engine_ds.zero_grad()
                ret = model_engine_ds(**batch)
                loss = ret["loss"]

                model_engine_ds.backward(loss)
                model_engine_ds.step()
                # lyw
                optimizer.param_groups[0]["lr"] = self.train_config.lr
                if ii_steps % self.print_step_num == 0:
                    if is_local_rank_0():
                        self.writer.add_scalar('loss', loss.to(dtype=torch.float32), ii_steps)
                        msg = f'loss: {loss:.4f}\tepoch: {epoch}\tstep: {ii_steps}\tlr: {optimizer.param_groups[0]["lr"]}'
                        get_logger("log").info(msg)

                if ii_steps % self.evaluate_step_num == 0:

                    if self.train_config.do_eval:
                        get_logger("log").info("evaluation ...")
                        self.evaluate_model(model_engine_ds, train_step=ii_steps)

                if ii_steps % self.save_step_num == 0:
                    get_logger("log").info("save model ...")
                    self.save_checkpoint(step=ii_steps, model=model_engine_ds, optimizer=optimizer,
                                         loss=loss, file_name=self.save_ckpt_path)
        if self.train_config.do_eval:
            get_logger("log").info("evaluation ...")
            self.evaluate_model(model_engine_ds, train_step=ii_steps)
        if self.train_config.do_train:
            get_logger("log").info("save model ...")
            self.save_checkpoint(step=ii_steps, model=model_engine_ds, optimizer=optimizer,
                                 loss=loss, file_name=self.save_ckpt_path)
