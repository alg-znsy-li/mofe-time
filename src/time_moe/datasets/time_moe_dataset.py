#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import numpy as np
import random
from .ts_dataset import TimeSeriesDataset
from .general_dataset import GeneralDataset
from .binary_dataset import BinaryDataset


class TimeMoEDataset(TimeSeriesDataset):

    def __init__(self, data_folder, normalization_method=None):
        self.data_folder = data_folder
        self.normalization_method = normalization_method
        self.datasets = []
        self.num_tokens = None

        if normalization_method is None:
            self.normalization_method = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == 'max':
                self.normalization_method = max_scaler
            elif normalization_method.lower() == 'zero':
                self.normalization_method = zero_scaler
            else:
                raise ValueError(f'Unknown normalization method: {normalization_method}')
        else:
            self.normalization_method = normalization_method

        if BinaryDataset.is_valid_path(self.data_folder):
            ds = BinaryDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        elif GeneralDataset.is_valid_path(self.data_folder):
            ds = GeneralDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        else:
            # walk through the data_folder
            for root, dirs, files in os.walk(self.data_folder):
                # if 'synthetic' not in root:
                #     continue
                for file in files:
                    fn_path = os.path.join(root, file)
                    if file != BinaryDataset.meta_file_name and GeneralDataset.is_valid_path(fn_path):
                        ds = GeneralDataset(fn_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                for sub_folder in dirs:
                    folder_path = os.path.join(root, sub_folder)
                    if BinaryDataset.is_valid_path(folder_path):
                        ds = BinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
        self.cumsum_lengths = [0]
        for ds in self.datasets:
            self.cumsum_lengths.append(
                self.cumsum_lengths[-1] + len(ds)
            )
        self.num_sequences = self.cumsum_lengths[-1]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
        elif seq_idx < 0:
            raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        seq = self.datasets[dataset_idx][dataset_offset]
        if self.normalization_method is not None:
            seq = self.normalization_method(seq)
        return seq

    def get_sequence_length_by_idx(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
        elif seq_idx < 0:
            raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_sequence_length_by_idx(dataset_offset)

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([ds.get_num_tokens() for ds in self.datasets])

        return self.num_tokens

class TimeMoEDataset_mix:
    def __init__(self, dataset: TimeSeriesDataset, dataset_flow: TimeSeriesDataset, random_length: int):
        self.dataset = dataset
        self.dataset_flow = dataset_flow
        self.random_length = random_length
        self.dataset_len = len(dataset)
        self.dataset_flow_len = len(dataset_flow)
        self.random_numbers = random.sample(range(0, self.dataset_len), self.random_length)

    def __len__(self):
        return self.random_length + self.dataset_flow_len

    def __getitem__(self, seq_idx):
        if seq_idx < self.random_length:
            return self.dataset[self.random_numbers[seq_idx]]
        else:
            return self.dataset_flow[seq_idx-self.random_length]

    def get_sequence_length_by_idx(self, seq_idx):
        if seq_idx < self.random_length:
            return self.dataset.get_sequence_length_by_idx(self.random_numbers[seq_idx])
        else:
            return self.dataset_flow.get_sequence_length_by_idx(seq_idx-self.random_length)
def zero_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    std_val = seq.std(dtype=np.float64)
    if std_val == 0:
        normed_seq = seq
    else:
        mean_val = seq.mean(dtype=np.float64)
        normed_seq = (seq - mean_val) / std_val
    return normed_seq


def zero_scalar2(seq, std=12.0, mean=120.0):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype
    std_val = std
    if std_val == 0:
        normed_seq = seq
    else:
        mean_val = mean
        normed_seq = (seq - mean_val) / std_val

    return normed_seq.astype(origin_dtype)


def max_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype
    max_val = np.abs(seq).max(dtype=np.float64)
    if max_val == 0:
        normed_seq = seq
    else:
        normed_seq = seq / max_val

    return normed_seq.astype(origin_dtype)


def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    best_index = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= value:
            best_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_index

