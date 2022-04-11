import torch

from .datasets import BinaryDataset, RandomMappingDataset, split_ds
from .collator import GLMPreprocessor
from megatron import get_tokenizer


def get_input(tokens, targets, loss_masks, position_ids, division):
    return {
        "text": torch.tensor(tokens, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
        "loss_mask": torch.tensor(loss_masks, dtype=torch.long),
        "position_id": torch.tensor(position_ids, dtype=torch.long),
        "attention_mask": torch.tensor(division, dtype=torch.long),
    }


def build_train_valid_test_datasets(
    data_prefix, splits_string, train_valid_test_num_samples, seq_length,
    length_per_sample, args
):
    tokenizer = get_tokenizer()

    collator = GLMPreprocessor(
        eod_id=tokenizer.get_special_token("eod"),
        mask_id=tokenizer.get_special_token("MASK"),
        gmask_id=tokenizer.get_special_token("gMASK"),
        sop_id=tokenizer.get_special_token("sop"),
        eop_id=tokenizer.get_special_token("eop"),
        max_seq_length=seq_length,
        gpt_prob=args.gpt_prob,
        short_seq_prob=args.short_seq_prob,
        single_span_prob=args.single_span_prob,
        mask_ratio=args.mask_prob,
        average_block_length=args.average_block_length,
        min_gmask_ratio=args.min_gmask_ratio,
        rank=0,
        device_num=1,
    )

    dataset = BinaryDataset(
        f"{data_prefix[0]}.bin",
        lambda *args: get_input(*collator.get_input_data(*args)),
        length_per_sample=length_per_sample,
    )
    train_dataset, valid_dataset, test_dataset = split_ds(
        dataset, [float(s) for s in splits_string.split(",")], block_size=10000
    )
    print(
        f"    train: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}"
    )

    scale = max(200, 1 + train_valid_test_num_samples[0] // len(train_dataset))
    train_dataset = RandomMappingDataset(train_dataset, scale=scale)
    valid_dataset = RandomMappingDataset(valid_dataset, scale=200)
    test_dataset = RandomMappingDataset(test_dataset, scale=200)

    return train_dataset, valid_dataset, test_dataset


def build_mask_matrix(separator, batch_size, seq_length, memory_length=0):
    dtype = torch.float
    m = torch.ones(
        (1, seq_length, seq_length), dtype=dtype, device=separator.device
    )
    m = torch.tril(m)
    is_scalar = torch.numel(separator) == 1
    if is_scalar:
        m[0, :, :separator] = 1
    else:
        m = m.expand(batch_size, -1, -1)
        ids = torch.arange(
            seq_length, device=separator.device, dtype=separator.dtype
        ).view(1, -1)
        mask = ids < separator.view(-1, 1)
        m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
    if memory_length > 0:
        m = m.expand(batch_size, -1, -1)
        m = torch.cat(
            (
                torch.ones(
                    (batch_size, seq_length, memory_length), dtype=dtype
                ),
                m,
            ),
            dim=2,
        )
    m = m.unsqueeze(1)
    m = m < 0.5
    return m
