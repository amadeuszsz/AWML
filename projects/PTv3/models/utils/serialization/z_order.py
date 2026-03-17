# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from typing import Optional, Union

import torch


class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = key | ((x & mask) << (2 * i + 2)) | ((y & mask) << (2 * i + 1)) | ((z & mask) << (2 * i + 0))
        return key

    def key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    r"""Encode :attr:`x`, :attr:`y`, :attr:`z` to z-order (Morton) keys.

    Two code-paths exist:

    * **Static depth** (``int``) — used during pre-computation outside the
      traced graph.  The original LUT-based algorithm is kept; all ``2**``
      values are Python int constants that become ONNX ``Constant`` nodes.

    * **Dynamic depth** (``torch.Tensor``) — used inside the traced graph
      when ``GridPooling`` calls ``serialization()`` on pooled points whose
      depth is data-dependent.  Bitwise ops (``|``, ``<<``) are replaced by
      arithmetic equivalents (``+``, ``*``) and ``pow2()`` LUT to satisfy
      ONNX / TensorRT constraints.  The LUT bit-patterns are non-overlapping,
      so ``+`` ≡ ``|`` for non-negative integers.

    Args:
      x, y, z (torch.Tensor): Coordinates.
      b (torch.Tensor or int, optional): Batch index (< 32768).
      depth (int or torch.Tensor): Serialization depth (< 17).
    """

    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    if isinstance(depth, torch.Tensor):
        # Dynamic depth — all intermediate values are tensors.
        from models.utils.structure import pow2

        # Always take the depth > 8 branch (depth is at least ~10 for
        # any reasonable point cloud range / grid size).
        key = EX[x % 256] + EY[y % 256] + EZ[z % 256]
        mask_hi = pow2(depth - 8) - 1
        key16 = (
            EX[torch.div(x, 256, rounding_mode="trunc") % (mask_hi + 1)]
            + EY[torch.div(y, 256, rounding_mode="trunc") % (mask_hi + 1)]
            + EZ[torch.div(z, 256, rounding_mode="trunc") % (mask_hi + 1)]
        )
        key = key16 * (2**24) + key
        if b is not None:
            b = b.long() if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.long, device=x.device)
            key = b * (2**48) + key
        return key

    # ---- Static (int) depth — original algorithm ----
    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x % (mask + 1)] + EY[y % (mask + 1)] + EZ[z % (mask + 1)]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = (
            EX[torch.div(x, 256, rounding_mode="trunc") % (mask + 1)]
            + EY[torch.div(y, 256, rounding_mode="trunc") % (mask + 1)]
            + EZ[torch.div(z, 256, rounding_mode="trunc") % (mask + 1)]
        )
        key = key16 * (2**24) + key

    if b is not None:
        b = b.long() if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.long, device=x.device)
        key = b * (2**48) + key

    return key


def key2xyz(key: torch.Tensor, depth: int = 16):
    r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    DX, DY, DZ = _key_lut.decode_lut(key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 48
    key = key & ((1 << 48) - 1)

    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 9) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b
