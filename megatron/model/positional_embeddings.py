# Extracted from: https://github.com/EleutherAI/gpt-neox
import torch
import torch.nn.functional as F


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # q: [sq, b * np, hn], position_id: [sq, b]
    cos, sin = F.embedding(position_id, cos.squeeze(1)), F.embedding(position_id, sin.squeeze(1))
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@torch.jit.script
def apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id, cos_block, sin_block, block_position_id):
    # q: [sq, b * np, hn], position_id: [sq, b]
    cos, sin = F.embedding(position_id, cos.squeeze(1)), F.embedding(position_id, sin.squeeze(1))
    cos_block, sin_block = F.embedding(block_position_id, cos_block.squeeze(1)), F.embedding(block_position_id, sin_block.squeeze(1))
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    q, k = (q * cos_block) + (rotate_half(q) * sin_block), (k * cos_block) + (rotate_half(k) * sin_block)
    return q, k


@torch.jit.script
def apply_rotary_pos_emb_index_torch(q, k, cos, sin, position_id):  # jitting fails with bf16
    cos, sin = F.embedding(position_id, cos.squeeze(1)), F.embedding(position_id, sin.squeeze(1))
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
