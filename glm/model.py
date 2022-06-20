import torch
from torch.nn import Module
from glm import build_mask_matrix
from megatron.mpu import vocab_parallel_cross_entropy


class GLMForMultiTokenCloze(torch.nn.Module):
    def __init__(self, language_model, take_softmax=True, length_penalty=0.0):
        super(GLMForMultiTokenCloze, self).__init__()
        self.model = language_model
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # [h.remove() for h in self.hook_handles]
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, target_ids=None, logit_mask=None, prompt_pos=None):
        if target_ids == None:
            return self.model(input_ids, position_ids, attention_mask)
        num_choices = None
        if len(input_ids.shape) == 3:
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
            target_ids = target_ids.reshape(-1, target_ids.size(-1))
            logit_mask = logit_mask.reshape(-1, logit_mask.size(-1))
            if prompt_pos is not None:
                prompt_pos = prompt_pos.reshape(-1, prompt_pos.size(-1))
        outputs = self.model(input_ids, position_ids, build_mask_matrix(attention_mask,
                                input_ids.size(0), input_ids.size(1)).to(torch.bool))
        if self.take_softmax:
            outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # select the target logits
        batch_ids = torch.arange(target_ids.size(0), dtype=torch.long, device=target_ids.device)
        batch_ids = batch_ids.unsqueeze(1).expand_as(target_ids)
        seq_ids = torch.arange(target_ids.size(-1), dtype=torch.long, device=target_ids.device)
        seq_ids = seq_ids.unsqueeze(0).expand_as(target_ids)
        logits = outputs[batch_ids, seq_ids, target_ids]
        logits = (logits * logit_mask).sum(dim=1)
        if self.length_penalty > 0.0:
            logits = logits / logit_mask.sum(dim=1) ** self.length_penalty
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return logits


class GLMForMultiTokenClozeFast(torch.nn.Module):
    def __init__(self, language_model, take_softmax=True, length_penalty=0.0):
        super(GLMForMultiTokenClozeFast, self).__init__()
        self.model = language_model
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def forward(self, input_ids, position_ids, attention_mask,
                dec_input_ids, dec_position_ids, dec_attention_mask, dec_target_ids, dec_logit_mask):
        # encoder
        outputs, *mems = self.model(input_ids, position_ids, attention_mask, return_memory=True, detach_memory=False)
        batch_size, num_choices, max_dec_len = dec_input_ids.size()
        max_enc_len = input_ids.size(-1)

        enc_mems = []
        for hidden in mems:
            hidden = hidden.unsqueeze(1).expand(-1, num_choices, -1, -1).reshape(batch_size * num_choices,
                                                                                 *hidden.size()[1:])
            enc_mems.append(hidden)

        def build_dec_mask_matrix(seq_length, sep, memory_length=0):
            m = enc_mems[0].new_ones((1, seq_length, seq_length))
            m = torch.tril(m)

            # sep = dec_attention_mask
            ids = torch.arange(memory_length, device=sep.device, dtype=sep.dtype).view(1, -1)
            mask = ids < sep.view(-1, 1)  # batch * mem
            mask = mask.unsqueeze(1).float().expand(-1, seq_length, -1)

            m = m.expand(batch_size * num_choices, -1, -1)
            m = torch.cat((mask, m), dim=2)
            m = m.unsqueeze(1)
            return m

        dec_input_ids = dec_input_ids.reshape(-1, max_dec_len)
        dec_position_ids = dec_position_ids.reshape(-1, *dec_position_ids.size()[2:])
        # dec_attention_mask = dec_attention_mask.reshape(-1, *dec_attention_mask.size()[2:]).unsqueeze(1)
        dec_attention_mask = build_dec_mask_matrix(max_dec_len, dec_attention_mask.reshape(-1), max_enc_len)
        dec_target_ids = dec_target_ids.reshape(-1, dec_target_ids.size(-1))
        dec_logit_mask = dec_logit_mask.reshape(-1, dec_logit_mask.size(-1))

        outputs = self.model(dec_input_ids, dec_position_ids, dec_attention_mask, *enc_mems)
        if self.take_softmax:
            outputs = torch.nn.functional.log_softmax(outputs, dim=-1)

        batch_ids = torch.arange(dec_target_ids.size(0), dtype=torch.long, device=dec_target_ids.device)
        batch_ids = batch_ids.unsqueeze(1).expand_as(dec_target_ids)
        seq_ids = torch.arange(dec_target_ids.size(-1), dtype=torch.long, device=dec_target_ids.device)
        seq_ids = seq_ids.unsqueeze(0).expand_as(dec_target_ids)
        logits = outputs[batch_ids, seq_ids, dec_target_ids]
        logits = (logits * dec_logit_mask).sum(dim=1)
        if self.length_penalty > 0.0:
            logits = logits / dec_logit_mask.sum(dim=1) ** self.length_penalty
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return logits


class GLMForSingleTokenCloze(Module):
    def __init__(self, language_model, take_softmax=False):
        super().__init__()
        self.model = language_model
        self.take_softmax = take_softmax

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # [h.remove() for h in self.hook_handles]
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, target_ids=None, logit_mask=None, prompt_pos=None):
        if target_ids is None:
            return self.model(input_ids, position_ids, attention_mask)
        assert len(input_ids.shape) == 2
        # print(f"input_ids: {input_ids.shape}, position_ids: {position_ids.shape}, attention_mask: {attention_mask.shape}")
        # from megatron import get_tokenizer
        # tokenizer = get_tokenizer()
        # print(f"inputs_ids: { ' '.join([tokenizer.IdToToken(t.item()) for t in input_ids[0]])}")
        # print(f"position_ids: {position_ids[0]}")
        # print(f"attention_mask: {attention_mask[0]}")
        # # print("target_ids:", ' ' target_ids[0])
        # exit(0)
        outputs = self.model(input_ids, position_ids, build_mask_matrix(attention_mask,
                                input_ids.size(0), input_ids.size(1)).to(torch.bool))
        batch_size, vocab_size, num_choices = outputs.size(0), outputs.size(-1), target_ids.size(1)
        batch_ids = torch.arange(batch_size, dtype=attention_mask.dtype, device=attention_mask.device)
        target_logits = outputs[batch_ids, attention_mask]
        target_logits = target_logits.repeat(1, target_ids.size(1)).reshape(batch_size, num_choices, vocab_size)
        output = vocab_parallel_cross_entropy(target_logits, target_ids)
        return (output, target_logits)

