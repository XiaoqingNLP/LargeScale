"""GLM zero-shot evaluation."""

import os
import glob
import json
import torch

from megatron import get_args, get_tokenizer
from megatron import print_rank_0, is_last_rank
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.utils import unwrap_model, report_memory
from megatron.p2p_communication import recv_forward, send_forward

from .datasets import build_dataset

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module

from pretrain_glm import model_provider as glm_model_provider
from pretrain_glm import process_data


def get_model_provider():
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        model = glm_model_provider(pre_process=pre_process, post_process=post_process)
        # model = GLMForMultiTokenCloze(model)
        return model

    return model_provider


def forward_step(batch, model):
    """Forward step."""

    # Get the batch.
    tokens, labels, loss_mask, attention_mask, position_ids = process_data(
        batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)
    assert args.micro_batch_size == 1

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output)

    if mpu.is_pipeline_last_stage():

        output = mpu.gather_from_tensor_model_parallel_region(output)
        # output: [b, sq, vocab]
        output = torch.nn.functional.log_softmax(output, dim=-1)
        target_ids = batch['target_id'][0]

        # tokenizer = get_tokenizer()
        # def decode(seq):
        #     return ' '.join([tokenizer.IdToToken(idx.item()) for idx in seq])

        # print(f'tokens: {decode(tokens[0, target_ids])}')
        # print(f'label: {decode(labels[0, target_ids])}')
        logits = output[0, target_ids, labels[0, target_ids]]
        # print(logits)

        return logits.sum(dim=-1)
        # # For loss, return the unreduced loss.
        # if eval_metric == 'loss':
        #     losses = mpu.vocab_parallel_cross_entropy(
        #         output.contiguous().float(), labels.contiguous())
        #     loss = torch.sum(
        #         losses.view(-1) * loss_mask.contiguous().view(-1).float())
        #     return loss
        #
        # # For accuracy, return the number of correctly predicted samples.
        # if eval_metric == 'accuracy':
        #     outputs = torch.argmax(output, -1)
        #     correct = (outputs == labels).float()
        #     correct[(1 - loss_mask).bool()] = 1
        #     correct = correct.prod(-1)
        #     return correct.sum()

    return None


def evaluate(data_loader, model):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    outputs = []
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            # Forward evaluation.
            output = forward_step(batch, model)
            # Reduce across processes.
            if mpu.is_pipeline_last_stage():
                outputs.append(output)

    return outputs


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last):
    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=micro_batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True)

    return data_loader


def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # # Set up model and load checkpoint.
    model = get_model(get_model_provider())
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # print(model)

    dataloaders = []
    folders = []

    for file_name in sorted(glob.glob(f"{args.train_data[0]}/**/validation.json", recursive=True)):
        dataset = build_dataset(file_name)
        dataloader = build_data_loader(dataset, args.micro_batch_size,
                                       args.num_workers, drop_last=False)
        folder = os.path.dirname(file_name)
        dataloaders.append(dataloader)
        folders.append(folder)
        print_rank_0(f"Loaded {file_name}")

    report_memory("Before train")

    for i in range(len(dataloaders)):
        outputs = evaluate(dataloaders[i], model)
        folder = folders[i]
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            with open(os.path.join(folder, 'predict.json'), 'w') as file:
                for output in outputs:
                    file.write(json.dumps({'prob': output.item()}) + '\n')
            print(f"Finish {folder}")

    torch.distributed.barrier()

    print_rank_0('done :-)')
