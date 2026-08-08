"""Throughput variant: fusion transformers with fused QKV projections.

One packed linear produces queries, keys and values per attention call instead
of three separate GEMM launches. The fusion semantics are identical to the
base; only the attention arithmetic is repacked.
"""

import jaxtyping as jt
import torch
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.transfuser_backbone import (
    GPT,
    TransfuserBackbone,
)


class FusedSelfAttention(nn.Module):
    """Multi-head self-attention with one packed QKV projection."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        """Initialize fused multi-head self-attention.

        Args:
            n_embd: Embedding dimension (must be divisible by n_head).
            n_head: Number of attention heads.
            attn_pdrop: Dropout probability for attention weights.
            resid_pdrop: Dropout probability for output projection.
        """
        super().__init__()
        assert n_embd % n_head == 0
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T C"]:
        """Compute multi-head self-attention from one packed projection.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_embd).

        Returns:
            Attention output tensor of shape (batch, sequence_length, n_embd).
        """
        b, t, c = x.size()
        q, k, v = (
            self.qkv(x)
            .view(b, t, 3, self.n_head, c // self.n_head)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(b, t, c)
        return self.resid_drop(self.proj(y))


class FusedBlock(nn.Module):
    """Transformer block using the fused attention."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_exp: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        """Initialize a transformer block with fused attention.

        Args:
            n_embd: Embedding dimension (feature channels).
            n_head: Number of attention heads.
            block_exp: Expansion factor for MLP hidden dimension.
            attn_pdrop: Dropout probability for attention weights.
            resid_pdrop: Dropout probability for residual connections.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FusedSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T C"]:
        """Apply the block with pre-normalization and residual connections.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_embd).

        Returns:
            Output tensor of same shape as input.
        """
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class FusedGPT(GPT):
    """The base fusion transformer with its blocks swapped for fused ones."""

    def __init__(self, n_embd: int, lead_config: LeadConfig) -> None:
        """Build the base GPT, then replace its transformer blocks.

        Args:
            n_embd: Embedding dimension (number of feature channels).
            lead_config: Root config tree.
        """
        super().__init__(n_embd, lead_config)
        config = lead_config.policy.transfuser
        self.blocks = nn.Sequential(
            *[
                FusedBlock(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for _ in range(config.n_layer)
            ],
        )
        self.apply(self._init_weights)


class FusedAttentionBackbone(TransfuserBackbone):
    """TransfuserBackbone whose fusion transformers use fused QKV attention."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the base backbone, then swap the fusion transformers.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        image_start = 1 if len(self.image_encoder.return_layers) > 4 else 0
        self.transformers = nn.ModuleList(
            [
                FusedGPT(
                    n_embd=self.image_encoder.feature_info.info[image_start + i][
                        "num_chs"
                    ],
                    lead_config=lead_config,
                )
                for i in range(4)
            ],
        )
