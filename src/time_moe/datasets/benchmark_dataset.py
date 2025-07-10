#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from time_moe.utils.log_util import log_in_local_rank_0


class BenchmarkDataset:

    def __init__(self, csv_path, context_length: int, prediction_length: int):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        df = pd.read_csv(csv_path)

        base_name = os.path.basename(csv_path).lower()
        if 'etth' in base_name:
            border1s = [0, 12 * 30 * 24 - context_length, 12 * 30 * 24 + 4 * 30 * 24 - context_length]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ettm' in base_name:
            border1s = [0, 12 * 30 * 24 * 4 - context_length, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_length]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif 'm4' in base_name:
            log_in_local_rank_0("++++++ Use m4 test dataset")
            border1s = [0, 0, 0]
            border2s = [0, 0, len(df)]

        else:
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [0, num_train - context_length, len(df) - num_test - context_length]
            border2s = [num_train, num_train + num_vali, len(df)]

        if 'm4' in base_name:
            log_in_local_rank_0(">>> Use m4 test dataset")
        else:
            start_dt = df.iloc[border1s[2]]['date']
            eval_start_dt = df.iloc[border1s[2] + context_length]['date']
            end_dt = df.iloc[border2s[2] - 1]['date']
            log_in_local_rank_0(f'>>> Split test data from {start_dt} to {end_dt}, '
                                f'and evaluation start date is: {eval_start_dt}')

        cols = df.columns[1:]
        df_values = df[cols].values

        train_data = df_values[border1s[0]:border2s[0]]
        valid_data = df_values[border1s[1]:border2s[1]]
        test_data = df_values[border1s[2]:border2s[2]]

        # scaler fitting
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

        scaled_train_data = self.scaler.transform(train_data)
        scaled_valid_data = self.scaler.transform(valid_data)
        scaled_test_data = self.scaler.transform(test_data)

        # assignment
        self.hf_train_dataset = scaled_train_data.transpose(1, 0)
        self.hf_valid_dataset = scaled_valid_data.transpose(1, 0)
        self.hf_test_dataset = scaled_test_data.transpose(1, 0)

        # 1 for the label
        self.window_length = self.context_length + self.prediction_length

        self.train_sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_train_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.train_sub_seq_indexes.append((seq_idx, offset_idx))

        self.valid_sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_valid_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.valid_sub_seq_indexes.append((seq_idx, offset_idx))


        self.test_sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_test_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.test_sub_seq_indexes.append((seq_idx, offset_idx))

class BenchmarkTestDataset(Dataset):

    def __init__(self, dataset_inst):
        self.dataset_inst = dataset_inst
        self.sub_seq_indexes = self.dataset_inst.test_sub_seq_indexes
        self.hf_dataset = self.dataset_inst.hf_test_dataset
        self.scaler = self.dataset_inst.scaler
        self.window_length = self.dataset_inst.context_length + self.dataset_inst.prediction_length
        self.context_length = self.dataset_inst.context_length
        self.prediction_length = self.dataset_inst.prediction_length

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]

        # 这个地方有点化简了，因为是切好的，而且有中间的valid 集合，所以一定不会穿越，而且丢掉了最后一个数据，所以不用pad
        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)

        return {
            'inputs': np.array(window_seq[: self.context_length], dtype=np.float32),
            'labels': np.array(window_seq[-self.prediction_length:], dtype=np.float32),
            'var': self.scaler.scale_[seq_i],
            'mean': self.scaler.mean_[seq_i]
        }

class BenchmarkFinetuneDataset(Dataset):

    def __init__(self, dataset_inst):
        self.dataset_inst = dataset_inst
        self.sub_seq_indexes = self.dataset_inst.train_sub_seq_indexes
        self.hf_dataset = self.dataset_inst.hf_train_dataset
        self.scaler = self.dataset_inst.scaler
        self.context_length = self.dataset_inst.context_length
        self.prediction_length = self.dataset_inst.prediction_length

        self.window_size = self.dataset_inst.context_length + self.dataset_inst.prediction_length
        self.window_size_plus_one = self.window_size + 1


        random.shuffle(self.sub_seq_indexes)

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        seq = self.hf_dataset[seq_i][offset_i: offset_i + self.window_size_plus_one]
        seq = np.array(seq, dtype=np.float32)

        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        n_pad = self.window_size_plus_one - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)

        if len(seq) != len(loss_mask) + 1:
            raise ValueError()

        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
            'loss_masks': loss_mask
        }
