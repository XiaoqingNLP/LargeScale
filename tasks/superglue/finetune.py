# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SuperGLUE finetune/evaluation."""
import torch
import os

from collections import OrderedDict
from megatron import get_args, get_timers, print_rank_0, get_tokenizer, mpu
from megatron.enums import PositionEmbeddingType
from tasks.finetune_utils import finetune
from tasks.superglue.dataset import SuperGlueDataset, PROCESSORS, get_output_func
from tasks.superglue.dataset import MULTI_CHOICE_DATASETS
from tasks.superglue.evaluate import qa_exact_match, qa_f1, multirc_em
from tasks.superglue.pvp import PVPS
from tasks.superglue.eval_utils import accuracy_metric, f1_macro_metric, f1_metric, accuracy_func_provider
from pretrain_glm import model_provider as glm_model_provider
from glm.model import GLMForSingleTokenCloze, GLMForMultiTokenClozeFast, GLMForMultiTokenCloze
from tasks.finetune_utils import cross_entropy_loss_func
from functools import partial


DEFAULT_METRICS = {
    "record": [("EM", qa_exact_match), ("F1", qa_f1)],
    "copa": [("accuracy", accuracy_metric)],
    "rte": [("accuracy", accuracy_metric)],
    "boolq": [("accuracy", accuracy_metric)],
    "wic": [("accuracy", accuracy_metric)],
    "wsc": [("accuracy", accuracy_metric)],
    "cb": [("accuracy", accuracy_metric), ("f1-macro", f1_macro_metric)],
    "multirc": [("f1a", f1_metric), ("em", multirc_em), ("acc", accuracy_metric)],
    "mnli": [("accuracy", accuracy_metric)],
    "sst2": [("accuracy", accuracy_metric)],
    "qnli": [("accuracy", accuracy_metric)],
    "qqp": [("accuracy", accuracy_metric)],
    "mrpc": [("accuracy", accuracy_metric)],
    "cola": [("accuracy", accuracy_metric)],
    "squad": [("accuracy", accuracy_metric)],
    "afqmc": [("accuracy", accuracy_metric)],
    "tnews": [("accuracy", accuracy_metric)],
    "cluewsc": [("accuracy", accuracy_metric)],
    "cmrc": [("accuracy", accuracy_metric)],
}


def train_valid_datasets_provider(pattern_text=False):
    """Provide train and validation datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    assert len(args.train_data) == 1

    task_name = args.task.lower()
    train_dataset = SuperGlueDataset(args, task_name, args.train_data[0], args.seq_length, "train", tokenizer,
                                pattern_text=pattern_text)
    valid_dataset = SuperGlueDataset(args, task_name, args.train_data[0], args.seq_length, "dev", tokenizer, for_train=True,
                                pattern_text=pattern_text)

    return train_dataset, valid_dataset


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    print_rank_0('building GLM downstream model for {} ...'.format(
        args.task))
    model = glm_model_provider(pre_process=pre_process, post_process=post_process)
    if args.cloze_eval:
        if args.multi_token:
            if args.fast_decode:
                model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
            else:
                model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
        else:
            model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
    else:
        raise NotImplementedError
    return model


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    keys = ["text", "label"]
    if args.pretrained_bert:
        keys += ["padding_mask", "types"]
    else:
        keys += ["mask", "position"]
        if args.cloze_eval:
            if args.fast_decode:
                keys += ["dec_text", "dec_position", "dec_mask", "dec_target", "dec_logit_mask"]
            else:
                keys += ["target", "logit_mask"]
                if args.segment_length > 0:
                    keys += ["segment_id"]
                if args.continuous_prompt:
                    keys += ["prompt_pos"]
    if args.variable_num_choices:
        keys.append("loss_mask")
    # Broadcast data.
    datatype = torch.int64
    data_b = mpu.broadcast_data(keys, batch, datatype)

    if "padding_mask" in data_b:
        attention_mask = data_b['padding_mask'].float().cuda().contiguous()
        if args.fp16:
            attention_mask = attention_mask.half()
        data_b["padding_mask"] = attention_mask

    return data_b


def finetune_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    args = get_args()
    timers = get_timers()


    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    data = process_batch(batch_, args)
    timers('batch generator').stop()

    # Forward model.
    if args.pretrained_bert:
        tokens, types, labels, attention_mask = data['text'], data['types'], data['label'], data['padding_mask']
        logits = model(tokens, token_type_ids=types, attention_mask=attention_mask, checkpoint_activations=True)
    elif args.cloze_eval:
        tokens, labels, position_ids = data['text'], data['label'], data['position']
        attention_mask = data['mask']

        if not args.fast_decode:
            target_ids, logit_mask = data['target'], data['logit_mask']
            if args.continuous_prompt:
                prompt_pos = data["prompt_pos"]
                result = model(tokens, position_ids, attention_mask, target_ids, logit_mask, prompt_pos=prompt_pos)
            else:
                result = model(tokens, position_ids, attention_mask, target_ids, logit_mask)
            if not args.multi_token:
                logits, lm_logits = result
            else:
                logits = result
        else:
            dec_input_ids, dec_position_ids, dec_attention_mask = data['dec_text'], data['dec_position'], data[
                'dec_mask']
            dec_target_ids, dec_logit_mask = data['dec_target'], data['dec_logit_mask']
            logits = model(tokens, position_ids, attention_mask, dec_input_ids, dec_position_ids,
                                  dec_attention_mask, dec_target_ids, dec_logit_mask)
    else:
        tokens, labels, position_ids, attention_mask = data['text'], data['label'], data['position'], data['mask']
        logits = model(tokens, position_ids, attention_mask)

    if args.adapet:
        raise NotImplementedError
        batch_size, num_classes = logits.size()[:2]
        label_mask = torch.ones(batch_size, num_classes, device=logits.device)
        label_mask.scatter_(1, labels.unsqueeze(1), -1.0)
        if "loss_mask" in data:
            loss_mask = data["loss_mask"]
            label_mask = label_mask * loss_mask
        loss = logits.contiguous().float() * label_mask
        loss = loss.sum() / batch_size
    else:
        if "segment_id" in data:
            raise NotImplementedError
            from torch_scatter import scatter_sum
            if "loss_mask" in data:
                logits = logits * data["loss_mask"]
            logits = scatter_sum(logits, data["segment_id"], dim=1)
        elif "loss_mask" in data:
            loss_mask = data["loss_mask"]
            logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)
        if args.loss_func == "cross_entropy":
            return logits.contiguous().float(), partial(cross_entropy_loss_func, labels)
        elif args.loss_func == "hinge":
            raise NotImplementedError
            correct_logits = logits[range(logits.size(0)), labels]
            hinge_loss = 1 + logits - correct_logits.unsqueeze(1)
            hinge_loss[hinge_loss < 0.0] = 0.0
            loss = hinge_loss.sum(dim=1).mean() - 1.0
        elif args.loss_func == "generative" or args.loss_func == "mix":
            raise NotImplementedError
            batch_size = logits.size(0)
            loss = - logits[range(batch_size), labels].mean()
            if args.loss_func == "mix":
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss + loss_func(logits.contiguous().float(), labels)
        else:
            raise NotImplementedError

    # Reduce loss for logging.

    return loss


def metrics_func_provider(is_test=False):
    """Privde metrics callback function."""
    args = get_args()
    tokenizer = get_tokenizer()

    def single_dataset_provider(split):
        return SuperGlueDataset(args, args.task.lower(), args.train_data[0], args.seq_length, split, tokenizer)

    output_func = get_output_func(args.task.lower(), args)
    eval_func = None
    if args.task.lower() in ['wsc', 'squad'] and args.cloze_eval and not args.wsc_negative:
        from tasks.language_model.finetune import classify_evaluate
        eval_func = classify_evaluate
    metric_dict = OrderedDict(DEFAULT_METRICS[args.task.lower()])
    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test, eval_func=eval_func,
                                  output_func=output_func, only_rank0=False, tokenizer=tokenizer)


def main():
    args = get_args()

    assert args.glm, "Only support GLM for SuperGLUE"
    assert args.tokenizer_type == "IceTokenizer", "Only support IceTokenzier for SuperGLUE"
    assert args.position_embedding_type != PositionEmbeddingType.alibi, "Don't support alibi for finetune"

    # For compability
    args.few_superglue = False
    args.cloze_eval = True
    args.pretrained_bert = False
    args.segment_length = 0
    args.continuous_prompt = False
    args.fast_decode = False
    args.num_prompt_tokens = 0
    args.task_mask = False
    args.prefix_prompt = False
    args.sentinel_token = False
    args.adapet = False
    args.no_block_position = False
    args.eval_batch_size = args.micro_batch_size
    args.master_ip = os.environ.get('MASTER_ADDR')
    args.master_port = os.environ.get('MASTER_PORT')

    # multi_token
    processor = PROCESSORS[args.task.lower()](args)
    pvp = PVPS[args.task.lower()](args, None, processor.get_labels(), args.seq_length,
                                  pattern_id=args.pattern_id, is_multi_token=args.multi_token,
                                  num_prompt_tokens=args.num_prompt_tokens)

    if args.task.lower() in ['wsc', 'squad'] and args.cloze_eval and not args.wsc_negative:
        # from tasks.language_model.finetune import lm_forward_step
        # finetune(args, train_valid_datasets_provider, model_kwargs,
        #          end_of_epoch_callback_provider=metrics_func_provider, forward_step=lm_forward_step)
        raise NotImplementedError
    else:
        if args.cloze_eval:
            multi_token = pvp.is_multi_token
        else:
            multi_token = args.task.lower() in MULTI_CHOICE_DATASETS
        args.multi_token = multi_token

        finetune(train_valid_datasets_provider, model_provider,
                 forward_step=finetune_forward_step,
                 end_of_epoch_callback_provider=metrics_func_provider)
