"""Zero-shot datasets."""

import os
import json

import numpy as np
import torch

from megatron import print_rank_0
from megatron import get_tokenizer


def build_dataset(path):
    """Helper function to select and build dataset."""
    return ZeroShotDataset(path)


def pad_batch(tokens, targets, loss_masks, position_ids, max_seq_length=None):
    if len(tokens) >= max_seq_length:
        tokens = tokens[: max_seq_length]
        targets = targets[: max_seq_length]
        loss_masks = loss_masks[: max_seq_length]
        position_ids = position_ids[: max_seq_length]
    else:
        tokens = np.concatenate(
            (
                tokens,
                np.zeros(max_seq_length - len(tokens), dtype=np.int),
            )
        )
        targets = np.concatenate(
            (
                targets,
                np.zeros(max_seq_length - len(targets), dtype=np.int),
            )
        )
        loss_masks = np.concatenate(
            (
                loss_masks,
                np.zeros(
                    max_seq_length - len(loss_masks), dtype=np.int
                ),
            )
        )
        position_ids = np.concatenate(
            (
                position_ids,
                np.zeros(
                    max_seq_length - len(position_ids), dtype=np.int
                ),
            )
        )
    return tokens, targets, loss_masks, position_ids


class ZeroShotDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_seq_length=2048):
        self.path = path
        self.max_seq_length = max_seq_length
        self.data = []
        self.answer = []

        tokenizers = get_tokenizer()
        self.mask_id = tokenizers.get_special_token('MASK')
        self.sop_id = tokenizers.get_special_token('sop')
        self.eop_id = tokenizers.get_special_token('eop')
        self.dtype = np.int

        with open(os.path.join(path), 'r') as file:
            for line in file:
                item = json.loads(line)
                # assert item['inputs'][-1] == self.sop_id
                # assert item['inputs'][-2] == self.mask_id
                text, target = item['inputs'], item['targets']
                if len(text) + len(target) + 2 >= max_seq_length:
                    text_length = max_seq_length - len(target) - 2
                    text = text[len(text) - text_length:len(text)]
                assert len(text) + len(target) + 2 <= max_seq_length
                self.data.append((text, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, target = self.data[idx]

        # tokens = np.concatenate((text, [self.mask_id, self.sop_id], target))
        # targets = np.concatenate((text, [self.mask_id], target, [self.eop_id]))
        # target_ids = np.arange(len(text) + 1, len(text) + len(target) + 1, dtype=self.dtype)
        # loss_masks = np.concatenate(
        #     (np.zeros(len(text) + 1, dtype=self.dtype), np.ones(len(target) + 1, dtype=self.dtype)))
        # division = len(text) + 1
        assert text[-1] == self.sop_id
        text = text[:-1]
        tokens = np.concatenate((text, [self.sop_id], target))
        targets = np.concatenate((text, target, [self.eop_id]))
        target_ids = np.arange(len(text), len(text) + len(target), dtype=self.dtype)
        loss_masks = np.concatenate(
            (np.zeros(len(text), dtype=self.dtype), np.ones(len(target) + 1, dtype=self.dtype)))
        division = len(text)
        mask_position = text.index(self.mask_id)
        position_ids = np.arange(len(tokens), dtype=self.dtype)
        position_ids[len(text):] = mask_position
        # pad batch
        PAD = 32
        tokens, targets, loss_masks, position_ids = pad_batch(
            tokens, targets, loss_masks, position_ids, ((len(tokens) + PAD - 1) // PAD) * PAD)

        return {
            'text': tokens,
            'target': targets,
            'target_id': target_ids,
            'loss_mask': loss_masks,
            'position_id': position_ids,
            'attention_mask': np.array([division], dtype=self.dtype)
        }
