# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/01/11 21:01:51
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import pickle

from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    def __init__(self, path, process_fn):
        import lmdb
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.process_fn = process_fn
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = str(idx).encode('utf-8')
            row = pickle.loads(txn.get(key))
            return self.process_fn(row)


class BinaryDataset(Dataset):
    def __init__(self, path, process_fn, length_per_sample=64 + 1024 + 4096, dtype='int32', preload=False,
                 **kwargs):  # TODO ARGS
        assert length_per_sample is not None
        self.length_per_sample = length_per_sample
        self.dtype = np.dtype(dtype)
        self.process_fn = process_fn
        if preload:
            self.bin = np.fromfile(path, dtype=self.dtype).reshape(-1, length_per_sample)
        else:
            with open(path, 'r') as fid:
                nbytes = fid.seek(0, 2)
                flen = fid.tell() // self.dtype.itemsize
            self.bin = np.memmap(path, dtype=self.dtype, shape=(flen // length_per_sample, length_per_sample))

    def __len__(self):
        return self.bin.shape[0]

    def __getitem__(self, index):
        return self.process_fn(self.bin[index], index)


class TSVDataset(Dataset):
    def __init__(self, path, process_fn, with_heads=True, **kwargs):
        self.process_fn = process_fn
        with open(path, 'r') as fin:
            if with_heads:
                self.heads = fin.readline().split('\t')
            else:
                self.heads = None
            self.items = [line.split('\t') for line in fin]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.process_fn(self.items[index])


class RandomMappingDataset(Dataset):
    '''
    Dataset wrapper to randomly mapping indices to original order.
    Will also enlarge the length
    '''
    def __init__(self, ds, scale=200, **kwargs):
        self.wrapped_data = ds
        self.scale = scale

    def __len__(self):
        return len(self.wrapped_data) * self.scale

    def __getitem__(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[rng.randint(0, 2**32-1) for _ in range(16)])
        index = rng.randint(len(self.wrapped_data))
        return self.wrapped_data[index]


class BlockedRandomSplitDataset(Dataset):
    '''
    Dataset wrapper to access a subset of another dataset.
    Use block algorithm to reduce memory.
    In each block, using the `indices` items.
    '''
    def __init__(self, ds, indices, block_size,**kwargs):
        if type(indices) is not np.ndarray:
            indices = np.array(indices)
        indices = np.sort(indices)
        self.block_size = block_size
        self.wrapped_data = ds
        self.wrapped_data_len = len(ds)
        self.indices = indices
        self.len = len(indices) * (len(ds) // block_size) + np.sum(indices < (len(ds) % block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.wrapped_data[(index // len(self.indices)) * self.block_size + self.indices[index % len(self.indices)]]


def split_ds(ds, split=[.8,.2,.0], block_size = 10000):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
    """
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split, dtype=np.float32)
    split /= split.sum()

    assert block_size <= len(ds)

    start_idx = 0
    residual_idx = 0
    rtn_ds = [None]*len(split)
    indices = np.random.permutation(np.array(range(block_size)))
    for i, f in enumerate(split):
        if f != 0:
            proportion = block_size*split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            rtn_ds[i] = BlockedRandomSplitDataset(ds, indices[range(start_idx, start_idx+max(split_, 1))], block_size)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds
