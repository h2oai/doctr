# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from copy import deepcopy
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from openvino.runtime import Core
from torch import nn
from torch.nn import functional as F

from doctr.models.classification.effnet.pytorch import efficientnet_b0, efficientnet_b3, efficientnetv2_m
from doctr.datasets import VOCABS, decode_sequence
from doctr.utils.data import download_from_url

from ...classification import mobilenet_v3_large_r, mobilenet_v3_small_r, vgg16_bn_r
from ...utils.pytorch import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ['CRNN', 'crnn_vgg16_bn', 'crnn_vgg16_bn_onnx', 'crnn_mobilenet_v3_small',
           'crnn_mobilenet_v3_large', 'crnn_efficientnet_b0', 'crnn_efficientnet_b0_onnx',
           'crnn_efficientnet_b3', 'crnn_efficientnet_b3_onnx', 'crnn_efficientnetv2_m',
           'crnn_efficientnetv2_m_onnx', 'crnn_efficientnetv2_mV2', 'crnn_efficientnetv2_mV2_onnx']

default_cfgs: Dict[str, Dict[str, Any]] = {
    "crnn_vgg16_bn": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["legacy_french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.3.1/crnn_vgg16_bn-9762b0b0.pt&src=0",
    },
    'crnn_vgg16_bn_onnx': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['legacy_french'],
        'url': 'https://github.com/h2oai/doctr/releases/download/onnx_models/crnn_vgg16_bn.onnx',
    },
    'crnn_mobilenet_v3_small': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'],
        'url': "https://github.com/mindee/doctr/releases/download/v0.3.1/crnn_mobilenet_v3_small_pt-3b919a02.pt",
    },
    "crnn_mobilenet_v3_large": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.3.1/crnn_mobilenet_v3_large_pt-f5259ec2.pt&src=0",
    },
    'crnn_efficientnet_b0': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': 'https://github.com/h2oai/doctr/releases/download/efficientnet_crnnv2/crnn_effnet_b0.pt'
    },
    'crnn_efficientnet_b0_onnx': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': 'https://github.com/h2oai/doctr/releases/download/efficientnet_onnx_models/crnn_effnet_b0.onnx'
    },
    'crnn_efficientnet_b3': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': "https://github.com/h2oai/doctr/releases/download/efficientnet_onnx_models/crnn_effnet_b3.pt",
    },
    'crnn_efficientnet_b3_onnx': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': 'https://github.com/h2oai/doctr/releases/download/efficientnet_onnx_models/crnn_effnet_b3.onnx'
    },
    'crnn_efficientnetv2_m': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': 'https://github.com/h2oai/doctr/releases/download/efficientnet_crnnv2/crnn_effnetv2_m.pt'
    },
    'crnn_efficientnetv2_m_onnx': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': 'https://github.com/h2oai/doctr/releases/download/efficientnet_onnx_models/crnn_effnetv2_m.onnx'
    },
    'crnn_efficientnetv2_mV2': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': 'https://github.com/h2oai/doctr/releases/download/efficientnet_onnx_models/crnn_effnetv2_mV2.pt'
    },
    'crnn_efficientnetv2_mV2_onnx': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['french'] + " ",
        'url': 'https://github.com/h2oai/doctr/releases/download/efficientnet_onnx_models/crnn_effnetv2_mV2.onnx'
    },
}


class CTCPostProcessor(RecognitionPostProcessor):
    """
    Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: str = VOCABS["french"],
        blank: int = 0,
    ) -> List[Tuple[str, float]]:
        """Implements best path decoding as shown by Graves (Dissertation, p63), highly inspired from
        <https://github.com/githubharald/CTCDecoder>`_.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            blank: index of blank label

        Returns:
            A list of tuples: (word, confidence)
        """

        # Gather the most confident characters, and assign the smallest conf among those to the sequence prob
        probs = F.softmax(logits, dim=-1).max(dim=-1).values.min(dim=1).values

        # collapse best path (using itertools.groupby), map to chars, join char list to string
        words = [
            decode_sequence([k for k, _ in groupby(seq.tolist()) if k != blank], vocab)
            for seq in torch.argmax(logits, dim=-1)
        ]

        return list(zip(words, probs.tolist()))

    def __call__(self, logits: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Decode CTC
        return self.ctc_best_path(logits=logits, vocab=self.vocab, blank=len(self.vocab))


class CRNN(RecognitionModel, nn.Module):
    """Implements a CRNN architecture as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        exportable: onnx exportable returns only logits
        cfg: configuration dictionary
    """

    _children_names: List[str] = ["feat_extractor", "decoder", "linear", "postprocessor"]

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        rnn_units: int = 128,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        exportable: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.max_length = 32
        self.exportable = exportable
        self.feat_extractor = feature_extractor

        # Resolve the input_size of the LSTM
        self.feat_extractor.eval()
        with torch.no_grad():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape))).shape
        lstm_in = out_shape[1] * out_shape[2]
        # Switch back to original mode
        self.feat_extractor.train()

        self.decoder = nn.LSTM(
            input_size=lstm_in,
            hidden_size=rnn_units,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
        )

        # features units = 2 * rnn_units because bidirectional layers
        self.linear = nn.Linear(in_features=2 * rnn_units, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: List[str],
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.build_target(target)
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)
        ctc_loss = F.ctc_loss(
            probs,
            torch.from_numpy(gt),
            input_length,
            torch.tensor(seq_len, dtype=torch.int),
            len(self.vocab),
            zero_infinity=True,
        )

        return ctc_loss

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.feat_extractor(x)
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        logits, _ = self.decoder(features_seq)
        logits = self.linear(logits)
        
        return logits
        
def _crnn(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[Any], nn.Module],
    pretrained_backbone: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> CRNN:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Feature extractor
    feat_extractor = backbone_fn(pretrained=pretrained_backbone).features  # type: ignore[call-arg]

    kwargs["vocab"] = kwargs.get("vocab", default_cfgs[arch]["vocab"])
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs["vocab"]
    _cfg["input_shape"] = kwargs["input_shape"]

    # Build the model
    model = CRNN(feat_extractor, cfg=_cfg, **kwargs)  # type: ignore[arg-type]
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, _cfg["url"], ignore_keys=_ignore_keys)

    return model


def crnn_vgg16_bn(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a VGG-16 backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    >>> import torch
    >>> from doctr.models import crnn_vgg16_bn
    >>> model = crnn_vgg16_bn(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn("crnn_vgg16_bn", pretrained, vgg16_bn_r, ignore_keys=["linear.weight", "linear.bias"], **kwargs)

class _crnn_onnx(RecognitionModel, nn.Module):
    """Onnx converted models"""
    def __init__(
        self,
        pretrained = True,
        model_name = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.vocab = default_cfgs[model_name]["vocab"]
        self.cfg = default_cfgs[model_name]

        self.postprocessor = CTCPostProcessor(vocab=self.vocab)
        model_path = str(download_from_url(self.cfg["url"], cache_subdir='models'))
        self.ie = Core()
        self.ie.set_property({'CACHE_DIR': os.path.join(os.path.expanduser('~'), '.cache', 'doctr', 'models')})
        self.compiled_model_onnx = self.ie.compile_model(model=model_path, device_name="CPU")
        self.output_layer_onnx = self.compiled_model_onnx.output(0)
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
    ):
        logits = self.compiled_model_onnx([x.detach().cpu().numpy()])[self.output_layer_onnx]
        return logits
    
def crnn_efficientnet_b0_onnx(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnetb0 onnx

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    kwargs["rnn_units"] = 512
    return _crnn_onnx(
        True,
        'crnn_efficientnet_b0_onnx',
        **kwargs,
    )
def crnn_efficientnet_b3_onnx(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnetb3 onnx

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    kwargs["rnn_units"] = 512
    return _crnn_onnx(
        True,
        'crnn_efficientnet_b3_onnx',
        **kwargs,
    )
    
def crnn_efficientnetv2_m_onnx(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnetv2_m onnx

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    kwargs["rnn_units"] = 512
    return _crnn_onnx(
        True,
        'crnn_efficientnetv2_m_onnx',
        **kwargs,
    )
    
def crnn_efficientnetv2_mV2_onnx(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnetv2_m onnx

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    kwargs["rnn_units"] = 512
    return _crnn_onnx(
        True,
        'crnn_efficientnetv2_mV2_onnx',
        **kwargs,
    )

def crnn_vgg16_bn_onnx(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with vgg16_bn onnx

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    kwargs["rnn_units"] = 512
    return _crnn_onnx(
        True,
        "crnn_vgg16_bn_onnx",
        **kwargs,
    )


def crnn_mobilenet_v3_small(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Small backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    >>> import torch
    >>> from doctr.models import crnn_mobilenet_v3_small
    >>> model = crnn_mobilenet_v3_small(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn(
        "crnn_mobilenet_v3_small",
        pretrained,
        mobilenet_v3_small_r,
        ignore_keys=["linear.weight", "linear.bias"],
        **kwargs,
    )


def crnn_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Large backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    >>> import torch
    >>> from doctr.models import crnn_mobilenet_v3_large
    >>> model = crnn_mobilenet_v3_large(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn(
        "crnn_mobilenet_v3_large",
        pretrained,
        mobilenet_v3_large_r,
        ignore_keys=["linear.weight", "linear.bias"],
        **kwargs,
    )

def crnn_efficientnet_b0(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnet_b0

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn(
        'crnn_efficientnet_b0',
        pretrained,
        efficientnet_b0,
        ignore_keys=['linear.weight', 'linear.bias'],
        **kwargs,
    )

def crnn_efficientnet_b3(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnet_b3

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    return _crnn(
        'crnn_efficientnet_b3',
        pretrained,
        efficientnet_b3,
        ignore_keys=['linear.weight', 'linear.bias'],
        **kwargs,
    )

def crnn_efficientnetv2_m(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnetv2_m

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    kwargs["rnn_units"] = 512
    return _crnn(
        'crnn_efficientnetv2_m',
        pretrained,
        efficientnetv2_m,
        ignore_keys=['linear.weight', 'linear.bias'],
        **kwargs,
    )
    
def crnn_efficientnetv2_mV2(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with efficientnetv2_m

    >>> import torch
    >>> from doctr.models import crnn_convnext_tiny
    >>> model = crnn_convnext_tiny(pretrained=True)
    >>> input_tensor = torch.rand(1, 3, 32, 128)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """
    kwargs["rnn_units"] = 512
    return _crnn(
        'crnn_efficientnetv2_mV2',
        pretrained,
        efficientnetv2_m,
        ignore_keys=['linear.weight', 'linear.bias'],
        **kwargs,
    )
