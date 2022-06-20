# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional

from torch import nn
from torchvision.models import convnext as tv_convnext

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params

__all__ = ['convnext_tiny']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'convnext_tiny': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'classes': list(VOCABS['french']),
    },
}


def _convnext(
    arch: str,
    pretrained: bool,
    tv_arch: str,
    num_rect_pools: int = 3,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any
) -> tv_convnext.ConvNeXt:

    kwargs['num_classes'] = kwargs.get('num_classes', len(default_cfgs[arch]['classes']))
    kwargs['classes'] = kwargs.get('classes', default_cfgs[arch]['classes'])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg['num_classes'] = kwargs['num_classes']
    _cfg['classes'] = kwargs['classes']
    kwargs.pop('classes')

    # Build the model
    model = tv_convnext.__dict__[tv_arch](pretrained = True)
    model.features[0][1].stride = (4, 4)
    model.features[2][1].stride = (2, 1)
    model.features[2][1].padding = (1, 1)
    model.features[4][1].stride = (2, 1)
    model.features[4][1].padding = (1, 1)
    model.features[6][1].stride = (2, 1)
    
    # Replace their kernel with rectangular ones


    model.classifier = nn.Linear(512, kwargs['num_classes'])
    # Load pretrained parameters
    model.cfg = _cfg
    # print(model)
    return model


def convnext_tiny(pretrained: bool = False, **kwargs: Any) -> tv_convnext.ConvNeXt:
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_, modified by adding batch normalization, rectangular pooling and a simpler
    classification head.

    >>> import torch
    >>> from doctr.models import vgg16_bn_r
    >>> model = vgg16_bn_r(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        VGG feature extractor
    """

    return _convnext(
        'convnext_tiny',
        pretrained,
        'convnext_tiny',
        3,
        **kwargs,
    )
# convnext_tiny(pretrained = True)