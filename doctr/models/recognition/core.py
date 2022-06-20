# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Tuple

import numpy as np

from doctr.datasets import encode_sequences
from doctr.utils.repr import NestedObject
import torch
from torch.nn import functional as F

__all__ = ['RecognitionPostProcessor', 'RecognitionModel']

def build_targets(gts, vocab, target_size = 32):
    encoded = encode_sequences(
        sequences=gts,
        vocab=vocab,
        target_size=target_size,
        eos=len(vocab))
    seq_len = [len(word) for word in gts]
    encoded = torch.tensor(encoded, device = "cuda")
    seq_len = torch.tensor(seq_len, device = "cuda", dtype = torch.int32)
    return encoded, seq_len

def calc_loss(logits, targets, seq_len, vocab):
    batch_len = logits.shape[0]
    input_length = logits.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
    logits = logits.permute(1, 0, 2)
    probs = F.log_softmax(logits, dim=-1)
    loss = F.ctc_loss(
        probs,
        targets,
        input_length,
        seq_len,
        len(vocab),
        zero_infinity=True,
    )
    return loss

class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""

    vocab: str
    max_length: int

    def build_target(
        self,
        gts: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(
            sequences=gts,
            vocab=self.vocab,
            target_size=self.max_length,
            eos=len(self.vocab)
        )
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:

        self.vocab = vocab
        self._embedding = list(self.vocab) + ['<eos>']

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"
