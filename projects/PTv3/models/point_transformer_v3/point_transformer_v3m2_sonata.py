"""
Point Transformer - V3 Mode2

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from addict import Dict
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

try:
    import flash_attn
except ImportError:
    flash_attn = None

from models.builder import MODELS
from models.scatter.functional import argsort, segment_csr, unique
from models.utils.misc import offset2bincount
from models.utils.serialization import encode
from models.utils.structure import Point, bit_length_tensor
from models.modules import PointModule, PointSequential


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    Copied from timm https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        export_mode=False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash and not export_mode
        self.export_mode = export_mode
        if self.enable_flash:
            assert enable_rpe is False, "Set enable_rpe to False when enable Flash Attention"
            assert upcast_attention is False, "Set upcast_attention to False when enable Flash Attention"
            assert upcast_softmax is False, "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if pad_key not in point.keys() or unpad_key not in point.keys() or cu_seqlens_key not in point.keys():
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.maximum(
                    torch.div(
                        bincount + self.patch_size - 1,
                        self.patch_size,
                        rounding_mode="trunc",
                    ),
                    torch.tensor(1, device=bincount.device),
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = (1 - mask_pad.int()) * bincount + mask_pad.int() * bincount_pad

            if not self.export_mode:
                _offset = nn.functional.pad(offset, (1, 0))
                _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
                pad = torch.arange(_offset_pad[-1], device=offset.device)
                unpad = torch.arange(_offset[-1], device=offset.device)
                cu_seqlens = []
                for i in range(len(offset)):
                    unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                    if bincount[i] != bincount_pad[i]:
                        pad[
                            _offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        ] = pad[
                            _offset_pad[i + 1]
                            - 2 * self.patch_size
                            + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                            - self.patch_size
                        ]
                    pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                    cu_seqlens.append(
                        torch.arange(
                            _offset_pad[i],
                            _offset_pad[i + 1],
                            step=self.patch_size,
                            dtype=torch.int32,
                            device=offset.device,
                        )
                    )
                point[pad_key] = pad
                point[unpad_key] = unpad
                point[cu_seqlens_key] = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])
            else:
                # NOTE(knzo25): needed due to tensorrt reasons
                assert len(offset) == 1

                pad = torch.arange(bincount_pad[0], device=offset.device)
                unpad = torch.arange(offset[0], device=offset.device)
                cu_seqlens = []

                pad[bincount_pad[0] - self.patch_size + (bincount[0] % self.patch_size) : bincount_pad[0]] = pad[
                    bincount_pad[0]
                    - 2 * self.patch_size
                    + (bincount[0] % self.patch_size) : bincount_pad[0]
                    - self.patch_size
                ]

                cu_seqlens.append(
                    torch.arange(
                        0,
                        bincount_pad[0],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

                point[pad_key] = pad
                point[unpad_key] = unpad
                point[cu_seqlens_key] = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=bincount_pad[0])

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            assert offset2bincount(point.offset).min() >= self.patch_size_max  # NOTE(knzo25): assumed for deployment
            self.patch_size = self.patch_size_max

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        export_mode=False,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.export_mode = export_mode

        if export_mode:
            from SparseConvolution.sparse_conv import SubMConv3d
        else:
            from spconv.pytorch import SubMConv3d

        self.cpe = PointSequential(
            SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.ls1 = PointSequential(
            LayerScale(channels, init_values=layer_scale) if layer_scale is not None else nn.Identity()
        )
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            export_mode=self.export_mode,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.ls2 = PointSequential(
            LayerScale(channels, init_values=layer_scale) if layer_scale is not None else nn.Identity()
        )
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.ls1(self.attn(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.ls2(self.mlp(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GridPooling(PointModule):
    """Grid-based pooling for point clouds.

    Unlike m1's ``SerializedPooling`` (which avoids re-serialization by
    right-shifting parent z-order codes), ``GridPooling`` operates on raw
    ``grid_coord`` with an arbitrary stride and therefore must recompute
    serialization data after pooling.

    Serialization is performed **inline** (not via ``point.serialization()``)
    so that in ONNX / TensorRT export mode, ``torch.argsort`` (which maps to
    a TopK op) can be replaced by the custom 1-D ``argsort`` op
    (``autoware::Argsort``).  This follows the same pattern used in m1's
    ``SerializedPooling``.

    Additional ONNX / TensorRT adaptations:

    * ``torch.unique(…, dim=0)`` (2-D) is replaced by a 1-D linearization
      followed by the custom ``unique()`` op that has an ONNX symbolic
      (``autoware::CustomUnique``).
    * ``torch_scatter.segment_csr`` is replaced by the custom ``segment_csr``
      op (``autoware::SegmentCSR``).

    These mirror the patterns established in the m1-base deploy refactor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
        export_mode=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.export_mode = export_mode

        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError("[gird_coord] or [coord, grid_size] should be include in the Point")
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")

        if not self.export_mode:
            # Original path: embed batch into grid_coord with bitwise ops,
            # then use torch.unique(dim=0) for 2-D dedup.
            grid_coord = grid_coord | point.batch.view(-1, 1) << 48

            grid_coord, cluster, counts = torch.unique(
                grid_coord,
                sorted=True,
                return_inverse=True,
                return_counts=True,
                dim=0,
            )
            _, indices = torch.sort(cluster)
        else:
            # Export path: custom unique() only supports 1-D inputs, so we
            # linearize (batch, x, y, z) into a single int64.
            # Layout:  batch * 2^48 + x * 2^32 + y * 2^16 + z
            # (safe because pooled grid coords fit comfortably in 16 bits)
            grid_coord_1d = (
                point.batch.long() * (2**48)
                + grid_coord[:, 0].long() * (2**32)
                + grid_coord[:, 1].long() * (2**16)
                + grid_coord[:, 2].long()
            )
            _, cluster, counts, _ = unique(grid_coord_1d)
            indices = argsort(cluster)

        # index pointer for sorted point, for segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]

        if not self.export_mode:
            # Strip batch from the unique grid_coord returned by torch.unique
            grid_coord = grid_coord & ((1 << 48) - 1)
        else:
            # Recover unique 2-D grid_coord rows via head_indices
            # (no batch was embedded in grid_coord, so no stripping needed)
            grid_coord = grid_coord[head_indices]

        if not self.export_mode:
            import torch_scatter

            point_dict = Dict(
                feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
                coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
                grid_coord=grid_coord,
                batch=point.batch[head_indices],
            )
            if "origin_coord" in point.keys():
                point_dict["origin_coord"] = torch_scatter.segment_csr(
                    point.origin_coord[indices], idx_ptr, reduce="mean"
                )
            if "color" in point.keys():
                point_dict["color"] = torch_scatter.segment_csr(point.color[indices], idx_ptr, reduce="mean")
        else:
            point_dict = Dict(
                feat=segment_csr(self.proj(point.feat)[indices], idx_ptr, self.reduce),
                coord=segment_csr(point.coord[indices], idx_ptr, "mean"),
                grid_coord=grid_coord,
                batch=point.batch[head_indices],
            )
            if "origin_coord" in point.keys():
                point_dict["origin_coord"] = segment_csr(point.origin_coord[indices], idx_ptr, "mean")
            if "color" in point.keys():
                point_dict["color"] = segment_csr(point.color[indices], idx_ptr, "mean")

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
            point_dict["idx_ptr"] = idx_ptr
        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)

        # --- inline serialization -------------------------------------------
        # GridPooling must re-serialize because pooled grid_coord values are
        # new (unlike m1's SerializedPooling which right-shifts parent codes).
        # We inline the logic here instead of calling point.serialization() so
        # that in export mode we can replace torch.argsort (which maps to an
        # ONNX TopK op) with the custom 1-D argsort op, following the same
        # pattern used in m1's SerializedPooling.
        point["order"] = order
        depth = bit_length_tensor(point.grid_coord.max())
        point["serialized_depth"] = depth

        code = torch.stack(
            [encode(point.grid_coord, point.batch, depth, order=o) for o in order]
        )

        if not self.export_mode:
            ser_order = torch.argsort(code)
        else:
            # Custom argsort is 1-D only → apply row-by-row over (k, n).
            ser_order = torch.stack(
                [argsort(code[i]) for i in range(code.shape[0])], dim=0
            )

        inverse = torch.zeros_like(ser_order).scatter_(
            dim=1,
            index=ser_order,
            src=torch.arange(0, code.shape[1], device=ser_order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            ser_order = ser_order[perm]
            inverse = inverse[perm]

        point["serialized_code"] = code
        point["serialized_order"] = ser_order
        point["serialized_inverse"] = inverse
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    """Embedding for m2 (Sonata).

    Unlike m1-base (which uses ``SubMConv3d`` 5×5), m2 uses a plain
    ``nn.Linear`` and therefore does **not** need a ``SparseConvTensor``.
    This means embedding can (and must) run *before* ``sparsify()``.
    No ``export_mode`` flag is required here since ``nn.Linear`` is already
    fully ONNX-compatible.
    """

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        mask_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_channels))
        else:
            self.mask_token = None

    def forward(self, point: Point):
        point = self.stem(point)
        if "mask" in point.keys():
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(point.feat.dtype),
                point.feat,
            )
        return point


@MODELS.register_module("PT-v3m2")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        export_mode=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.enc_mode = enc_mode
        self.freeze_encoder = freeze_encoder
        self.export_mode = export_mode

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        # normalization layer
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        # encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        shuffle_orders=shuffle_orders,
                        export_mode=export_mode,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        layer_scale=layer_scale,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        export_mode=export_mode,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            layer_scale=layer_scale,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            export_mode=export_mode,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Handle SubMConv3d from either spconv or the ONNX-exportable
        # SparseConvolution wrapper — both expose .weight / .bias.
        elif hasattr(module, "weight") and hasattr(module, "bias"):
            name = type(module).__name__
            if "SubMConv3d" in name:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        point = Point(data_dict)
        point = self.embedding(point)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point

    def export_forward(self, data_dict):
        """ONNX-exportable forward pass.

        Differences from ``forward()``:

        1. Serialization data (``serialized_code``, ``serialized_order``,
           ``serialized_inverse``, ``serialized_depth``, ``sparse_shape``)
           are **pre-computed** outside the traced graph and passed in via
           *data_dict*.  This mirrors the m1-base deploy pattern.

        2. ``point["order"]`` is set explicitly so that ``GridPooling``
           (which re-serializes inline for pooled points) knows what
           serialization order strings to use.

        3. Embedding runs *before* ``sparsify()`` — correct for m2 where
           embedding is ``nn.Linear`` (operates on raw ``feat``), unlike m1
           where embedding is ``SubMConv3d`` (needs ``SparseConvTensor``).
        """
        point = Point(data_dict)
        point = self.embedding(point)

        # Inject pre-computed serialization data (replaces serialization() call)
        point["order"] = self.order
        point["serialized_depth"] = data_dict["serialized_depth"]
        point["serialized_code"] = data_dict["serialized_code"]
        point["serialized_order"] = data_dict["serialized_order"]
        point["serialized_inverse"] = data_dict["serialized_inverse"]
        point["sparse_shape"] = data_dict["sparse_shape"]
        point.sparsify()

        point = self.enc(point)

        if not self.enc_mode:
            point = self.dec(point)

        return point
