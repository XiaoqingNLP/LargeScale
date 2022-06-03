import torch
import numpy as np
import os

from megatron.enums import PositionEmbeddingType
from .datasets import BinaryDataset, RandomMappingDataset, LMDBDataset, ConcatDataset, AggregatedDataset, split_ds
from .datasets import TransformingDataset
from .collator import GLMPreprocessor
from megatron import get_tokenizer, print_rank_0, get_num_microbatches
from megatron.mpu import get_tensor_model_parallel_rank, is_pipeline_first_stage

def get_input(tokens, targets, loss_masks, position_ids, division, is_multitask=False):
    return {
        "text": torch.tensor(tokens, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
        "loss_mask": torch.tensor(loss_masks, dtype=torch.long),
        "position_id": torch.tensor(position_ids, dtype=torch.long),
        "attention_mask": torch.tensor(division, dtype=torch.long),
        "is_multitask": torch.tensor([is_multitask], dtype=torch.long)
    }


def get_multitask_data(mutlitask_data_path, collator: GLMPreprocessor, aggregated_samples_per_sequence=1):
    train_datasets, val_datasets = [], []
    for path in sorted(os.listdir(mutlitask_data_path)):
        full_path = os.path.join(mutlitask_data_path, path)
        if os.path.isdir(full_path):
            print_rank_0(f"Loading multitask data {full_path}")
            data = LMDBDataset(full_path, process_fn=lambda row: (np.array(row[0]), np.array(row[1])))
            if path.endswith("eval"):
                val_datasets.append(data)
            else:
                train_datasets.append(data)

    def process_fn(samples):
        # unpack List[(token, label)] to -> (List[tokens], List[labels])
        return get_input(*collator.get_multitask_data(*list(zip(*samples))), is_multitask=True)

    # We need a random mapping here or will lose some task when multitask_ratio < actual data ratio
    train_datasets = AggregatedDataset(RandomMappingDataset(ConcatDataset(train_datasets)),
                                       aggregated_samples_per_sequence, process_fn)
    val_datasets = AggregatedDataset(RandomMappingDataset(ConcatDataset(val_datasets)),
                                     aggregated_samples_per_sequence, process_fn) if len(val_datasets) > 0 else []
    return train_datasets, val_datasets


def make_transforming_dataset(args, ds1, ds2):
    assert args.multitask_data_transform_steps is not None and len(args.multitask_data_transform_steps) == 2
    assert args.num_workers == 1  # for iteration calculation
    start = int(args.multitask_data_transform_steps[0])
    end = start + int(args.multitask_data_transform_steps[1])
    # ds2 = RandomMappingDataset(ds2, scale=len(ds1) / len(ds2))
    return TransformingDataset(ds1, ds2, start=start, end=end, iteration=args.iteration,
        local_batch_size=get_num_microbatches() * args.micro_batch_size * args.multitask_ratio / args.num_workers,
        if_print=is_pipeline_first_stage() and get_tensor_model_parallel_rank() == 0)


def build_train_valid_test_datasets(
    data_prefix, splits_string, train_valid_test_num_samples, seq_length,
    length_per_sample, aggregated_samples_per_sequence, args
):
    tokenizer = get_tokenizer()

    collator = GLMPreprocessor(
        eod_id=tokenizer.get_special_token("eod"),
        mask_id=tokenizer.get_special_token("MASK"),
        gmask_id=tokenizer.get_special_token("gMASK"),
        sop_id=tokenizer.get_special_token("sop"),
        eop_id=tokenizer.get_special_token("eop"),
        max_seq_length=seq_length,
        aggregated_samples_per_sequence=aggregated_samples_per_sequence,
        gpt_prob=args.gpt_prob,
        short_seq_prob=args.short_seq_prob,
        single_span_prob=args.single_span_prob,
        mask_ratio=args.mask_prob,
        average_block_length=args.average_block_length,
        min_gmask_ratio=args.min_gmask_ratio,
        relative_pos_encoding=args.position_embedding_type == PositionEmbeddingType.alibi,
        no_2d_encoding=args.position_embedding_type == PositionEmbeddingType.rotary,
        aggregate_gpt_sample=args.aggregate_gpt_sample,
        rank=0,
        device_num=1,
    )

    dataset = BinaryDataset(
        f"{data_prefix[0]}.bin",
        lambda tokens, index: get_input(*collator.get_input_data(np.array(tokens), index)),  # np.memmap -> np.array
        length_per_sample=length_per_sample,
    )
    train_dataset, valid_dataset, test_dataset = split_ds(
        dataset, [float(s) for s in splits_string.split(",")], block_size=10000
    )
    print_rank_0(
        f"    text_train: {len(train_dataset)}, text_valid: {len(valid_dataset)}, text_test: {len(test_dataset)}"
    )

    if args.multitask_data_path is not None:
        if len(args.multitask_data_path) == 1:
            multitask_train_dataset, multitask_valid_dataset = \
                get_multitask_data(args.multitask_data_path[0], collator, aggregated_samples_per_sequence)
            print_rank_0(f"    multitask_train: {len(multitask_train_dataset)}, multitask_valid: {len(multitask_valid_dataset)}")
        elif len(args.multitask_data_path) == 2:
            multitask_train_dataset1, multitask_valid_dataset1 = \
                get_multitask_data(args.multitask_data_path[0], collator, aggregated_samples_per_sequence)
            multitask_train_dataset2, multitask_valid_dataset2 = \
                get_multitask_data(args.multitask_data_path[1], collator, aggregated_samples_per_sequence)
            print_rank_0(f"    multitask_train1: {len(multitask_train_dataset1)}, multitask_valid1: {len(multitask_valid_dataset1)}")
            print_rank_0(f"    multitask_train2: {len(multitask_train_dataset2)}, multitask_valid2: {len(multitask_valid_dataset2)}")
            multitask_train_dataset = make_transforming_dataset(args, multitask_train_dataset1, multitask_train_dataset2)
            multitask_valid_dataset = multitask_valid_dataset2
            print_rank_0(f"    transforming_multitask_train: {len(multitask_train_dataset)}")
        else:
            assert False
        def calc_weight(ds1, ds2, ratio):
            return [1, 1] if ratio is None else [1, ratio * len(ds1) / (len(ds2) * (1 - ratio))]

        train_dataset = ConcatDataset([train_dataset, multitask_train_dataset],
                                      weights=calc_weight(train_dataset, multitask_train_dataset, args.multitask_ratio))
        if len(multitask_valid_dataset) > 0:
            valid_dataset = ConcatDataset([valid_dataset, multitask_valid_dataset],
                                          weights=calc_weight(valid_dataset, multitask_valid_dataset, args.multitask_ratio))

    scale = max(200, 1 + train_valid_test_num_samples[0] // len(train_dataset))
    train_dataset = RandomMappingDataset(train_dataset, scale=scale, seed=args.data_shuffle_seed)
    valid_dataset = RandomMappingDataset(valid_dataset, scale=200)
    test_dataset = RandomMappingDataset(test_dataset, scale=200)

    print_rank_0(
        f"    all_train: {len(train_dataset)}, all_valid: {len(valid_dataset)}, data_shuffle_seed={args.data_shuffle_seed}")

    return train_dataset, valid_dataset, test_dataset


def build_single_mask_matrix(separator, batch_size, seq_length, memory_length=0):
    dtype = torch.float
    m = torch.ones(
        (1, seq_length, seq_length), dtype=dtype, device=separator.device
    )
    m = torch.tril(m)
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


def build_mask_matrix(separator, batch_size, seq_length):
    if separator.dim() == 1:
        return build_single_mask_matrix(separator, batch_size=batch_size, seq_length=seq_length)
    else:
        aggregated_samples = separator.size(-1)
        assert seq_length % aggregated_samples == 0
        single_length = seq_length // aggregated_samples
        m = torch.ones((batch_size, 1, seq_length, seq_length), dtype=torch.bool, device=separator.device)
        for i in range(aggregated_samples):
            single_mask = build_single_mask_matrix(separator[:, i], batch_size=batch_size, seq_length=single_length)
            m[:, :, single_length * i: single_length * (i + 1), single_length * i: single_length * (i + 1)] = single_mask
        return m


if __name__ == "__main__":
    separator = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int)
    m = build_mask_matrix(separator, batch_size=2, seq_length=12)
