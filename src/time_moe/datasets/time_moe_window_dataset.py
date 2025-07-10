#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
import numpy as np

from time_moe.datasets.ts_dataset import TimeSeriesDataset


class TimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data.
    """
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0, **kwrags):
        self.dataset = dataset
        self.shuffle = kwrags['shuffle']
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1

        num_seqs = len(self.dataset)
        iterator = range(num_seqs)
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
        self.sub_seq_indexes = []
        for seq_idx in iterator:
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            #self.sub_seq_indexes.append((seq_idx, 0))
            for offset_idx in range(0, n_points, self.window_size):
                self.sub_seq_indexes.append((seq_idx, offset_idx))
        if self.shuffle:
            random.shuffle(self.sub_seq_indexes)

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        seq = self.dataset[seq_i][offset_i: offset_i + self.window_size_plus_one]
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