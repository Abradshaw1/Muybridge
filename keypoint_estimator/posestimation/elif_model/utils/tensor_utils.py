# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Optional, Sequence, Union, Type
from collections import abc
import numpy as np
import torch
from torch import Tensor

def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def to_numpy(x: Union[Tensor, Sequence[Tensor]],
             return_device: bool = False,
             unzip: bool = False) -> Union[np.ndarray, tuple]:
    """Convert torch tensor to numpy.ndarray.

    Args:
        x (Tensor | Sequence[Tensor]): A single tensor or a sequence of
            tensors
        return_device (bool): Whether return the tensor device. Defaults to
            ``False``
        unzip (bool): Whether unzip the input sequence. Defaults to ``False``

    Returns:
        np.ndarray | tuple: If ``return_device`` is ``True``, return a tuple
        of converted numpy array(s) and the device indicator; otherwise only
        return the numpy array(s)
    """

    if isinstance(x, Tensor):
        arrays = x.detach().cpu().numpy()
        device = x.device
    elif isinstance(x, np.ndarray) or is_seq_of(x, np.ndarray):
        arrays = x
        device = 'cpu'
    elif is_seq_of(x, Tensor):
        if unzip:
            # convert (A, B) -> [(A[0], B[0]), (A[1], B[1]), ...]
            arrays = [
                tuple(to_numpy(_x[None, :]) for _x in _each)
                for _each in zip(*x)
            ]
        else:
            arrays = [to_numpy(_x) for _x in x]

        device = x[0].device

    else:
        raise ValueError(f'Invalid input type {type(x)}')

    if return_device:
        return arrays, device
    else:
        return arrays


def to_tensor(x: Union[np.ndarray, Sequence[np.ndarray]],
              device: Optional[Any] = None) -> Union[Tensor, Sequence[Tensor]]:
    """Convert numpy.ndarray to torch tensor.

    Args:
        x (np.ndarray | Sequence[np.ndarray]): A single np.ndarray or a
            sequence of tensors
        tensor (Any, optional): The device indicator. Defaults to ``None``

    Returns:
        tuple:
        - Tensor | Sequence[Tensor]: The converted Tensor or Tensor sequence
    """
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device)
    elif is_seq_of(x, np.ndarray):
        return [to_tensor(_x, device=device) for _x in x]
    else:
        raise ValueError(f'Invalid input type {type(x)}')