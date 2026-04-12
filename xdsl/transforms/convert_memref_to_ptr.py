"""
Implements a lowering from operations on memrefs to operations on pointers.

The default lowering of memrefs in MLIR is to a structure that carries the base pointer,
an offset, and strides
(See (memref.extract_strided_metadata)[https://mlir.llvm.org/docs/Dialects/MemRef/#memrefextract_strided_metadata-memrefextractstridedmetadataop]).
If the offset and strides are statically known, they can be encoded in the type, but
they can also be dynamic.

When lowering operations on memrefs to operations on pointers, the offset and stride
information must be lowered to operations that perform the required pointer arithmetic.
In order to simplify the lowering of memory accesses, this pass makes the choice to
lower memrefs to a pointer to the buffer including the dynamic offset.
This means that memory accesses can be a simple dot product of statically known strides
and dynamic indices.
On the other hand, operations that create views of memrefs from other memrefs must lower
to the relevant pointer arithmetic to encode the new inner buffer offset, when possible.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import cast

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, ptr
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.irdl import Any
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa

_index_type = builtin.IndexType()


def get_bytes_offset(
    elements_offset: SSAValue, element_type: Attribute, builder: Builder
) -> SSAValue:
    """
    Returns the offset in bytes given an offset in elements and the element type.
    """
    bytes_per_element_op = builder.insert_op(
        ptr.TypeOffsetOp(element_type, _index_type)
    )
    bytes_offset = builder.insert_op(
        arith.MuliOp(elements_offset, bytes_per_element_op)
    )
    bytes_per_element_op.offset.name_hint = "bytes_per_element"
    bytes_offset.result.name_hint = "scaled_pointer_offset"

    return bytes_offset.result


def get_offset_pointer(
    pointer: SSAValue,
    bytes_offset: SSAValue,
    builder: Builder,
) -> SSAValue:
    """
    Returns the pointer incremented by the given number of bytes.
    """
    target_ptr = builder.insert_op(ptr.PtrAddOp(pointer, bytes_offset))
    target_ptr.result.name_hint = "offset_pointer"
    return target_ptr.result


def get_constant_strides(memref_type: builtin.MemRefType) -> Sequence[int]:
    """
    If the memref has constant strides and offset, returns them, otherwise raises a
    DiagnosticException.
    """
    match memref_type.layout:
        case builtin.NoneAttr():
            strides = builtin.ShapedType.strides_for_shape(memref_type.get_shape())
        case builtin.StridedLayoutAttr():
            strides = memref_type.layout.get_strides()
            if None in strides:
                raise DiagnosticException(
                    f"MemRef {memref_type} with dynamic stride is not yet implemented"
                )
            strides = cast(Sequence[int], strides)
        case _:
            raise DiagnosticException(f"Unsupported layout type {memref_type.layout}")
    return strides


def _to_ssa(v: int | SSAValue, builder: Builder) -> SSAValue:
    # materialize int as arith.constant, pass through ssa values
    match v:
        case int(val):
            return builder.insert_op(
                arith.ConstantOp.from_int_and_width(val, _index_type)
            ).result
        case _:
            return v


def _dim_size(
    memref_val: SSAValue, shape: tuple[int, ...], i: int, builder: Builder
) -> int | SSAValue:
    # return dim size as int if static, ssa value via memref.dim if dynamic
    if shape[i] != builtin.DYNAMIC_INDEX:
        return shape[i]
    dim_idx_val = builder.insert_op(arith.ConstantOp.from_int_and_width(i, _index_type))
    dim_idx_val.result.name_hint = "dim_idx"
    return builder.insert_op(
        memref.DimOp.from_source_and_index(memref_val, dim_idx_val.result)
    ).result


def _mul_factors(
    a: int | SSAValue, b: int | SSAValue, builder: Builder
) -> int | SSAValue:
    # multiply two stride factors, staying as int when both static
    match (a, b):
        case (int(), int()):
            return a * b
        case (1, _):
            return b
        case (_, 1):
            return a
        case _:
            return builder.insert_op(
                arith.MuliOp(_to_ssa(a, builder), _to_ssa(b, builder))
            ).result


def _default_layout_strides(
    memref_val: SSAValue, memref_type: builtin.MemRefType, builder: Builder
) -> list[int | SSAValue]:
    # compute row-major strides for a default-layout (NoneAttr) memref
    shape = memref_type.get_shape()
    rank = len(shape)
    if rank == 0:
        return []
    strides: list[int | SSAValue] = [0] * rank
    strides[rank - 1] = 1
    for i in range(rank - 2, -1, -1):
        strides[i] = _mul_factors(
            strides[i + 1], _dim_size(memref_val, shape, i + 1, builder), builder
        )
    return strides


def get_strides(
    memref_val: SSAValue,
    memref_type: builtin.MemRefType,
    builder: Builder,
) -> list[int | SSAValue]:
    # return strides as int (static) or ssa values (dynamic) per dimension
    match memref_type.layout:
        case builtin.NoneAttr():
            return _default_layout_strides(memref_val, memref_type, builder)
        case builtin.StridedLayoutAttr():
            strides = memref_type.layout.get_strides()
            if None in strides:
                raise DiagnosticException(
                    f"MemRef {memref_type} with dynamic stride is not yet implemented"
                )
            return list(cast(Sequence[int], strides))
        case _:
            raise DiagnosticException(f"Unsupported layout type {memref_type.layout}")


def _apply_stride(
    index: SSAValue, stride: int | SSAValue, builder: Builder
) -> SSAValue:
    # compute index * stride, optimizing stride == 1
    match stride:
        case 1:
            return index
        case int(val):
            assert val > 0, f"Strides must be positive, got {val}"
            stride_op = builder.insert_op(
                arith.ConstantOp.from_int_and_width(val, _index_type)
            )
            stride_op.result.name_hint = "pointer_dim_stride"
            offset_op = builder.insert_op(arith.MuliOp(index, stride_op))
            offset_op.result.name_hint = "pointer_dim_offset"
            return offset_op.result
        case _:
            offset_op = builder.insert_op(arith.MuliOp(index, stride))
            offset_op.result.name_hint = "pointer_dim_offset"
            return offset_op.result


def get_strides_offset(
    indices: Iterable[SSAValue],
    strides: Sequence[int | SSAValue],
    builder: Builder,
) -> SSAValue | None:
    # compute combined index offset from indices and strides
    increments = [
        _apply_stride(idx, s, builder) for idx, s in zip(indices, strides, strict=True)
    ]
    if not increments:
        return None

    def _add(head: SSAValue, inc: SSAValue) -> SSAValue:
        result = builder.insert_op(arith.AddiOp(head, inc))
        result.result.name_hint = "pointer_dim_stride"
        return result.result

    return reduce(_add, increments)


def get_target_ptr(
    target_memref: SSAValue,
    memref_type: memref.MemRefType[Any],
    indices: Iterable[SSAValue],
    builder: Builder,
) -> SSAValue:
    """
    Get operations returning a pointer to an element of a memref referenced by indices.
    """

    memref_ptr = builder.insert_op(ptr.ToPtrOp(target_memref))
    pointer = memref_ptr.res
    pointer.name_hint = target_memref.name_hint

    strides = get_strides(target_memref, memref_type, builder)
    head = get_strides_offset(indices, strides, builder)

    if head is not None:
        offset = get_bytes_offset(head, memref_type.element_type, builder)
        pointer = get_offset_pointer(pointer, offset, builder)

    return pointer


@dataclass
class ConvertStorePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)
        target_ptr = get_target_ptr(op.memref, memref_type, op.indices, rewriter)
        rewriter.replace_op(op, ptr.StoreOp(target_ptr, op.value))


@dataclass
class ConvertLoadPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)
        target_ptr = get_target_ptr(op.memref, memref_type, op.indices, rewriter)
        rewriter.replace_op(op, ptr.LoadOp(target_ptr, memref_type.element_type))


def _resolve_offset(
    static_offset: int, dynamic_offsets: Iterator[SSAValue], builder: Builder
) -> SSAValue:
    # resolve subview offset: dynamic ssa value or materialized constant
    if static_offset == builtin.DYNAMIC_INDEX:
        return next(dynamic_offsets)
    val = builder.insert_op(
        arith.ConstantOp(builtin.IntegerAttr(static_offset, _index_type))
    ).result
    val.name_hint = f"c{static_offset}"
    return val


def _apply_subview_stride(
    stride: int | SSAValue, offset_val: SSAValue, builder: Builder
) -> SSAValue:
    # compute stride * offset for a subview dimension
    match stride:
        case 1:
            return offset_val
        case int(val):
            stride_val = builder.insert_op(
                arith.ConstantOp(builtin.IntegerAttr(val, _index_type))
            ).result
            stride_val.name_hint = f"c{val}"
            increment = builder.insert_op(arith.MuliOp(stride_val, offset_val)).result
        case _:
            increment = builder.insert_op(arith.MuliOp(stride, offset_val)).result
    increment.name_hint = "increment"
    return increment


class ConvertSubviewPattern(RewritePattern):
    """
    Converts the subview to a pointer offset.

    From the subview op documentation:

    > In the absence of rank reductions, the resulting memref type is computed as
    follows:
    ```
    ...
    result_offset = src_offset + dot_product(offset_operands, src_strides)
    ```

    The pointer that the source memref is lowered to is assumed to incorporate the
    `src_offset`, so this lowering just addds the dot product.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        source_type = op.source.type
        assert isa(source_type, builtin.MemRefType)
        result_type = op.result.type

        source_strides = get_strides(op.source, source_type, rewriter)

        pointer = rewriter.insert_op(ptr.ToPtrOp(op.source)).res
        pointer.name_hint = op.source.name_hint

        dynamic_offsets = iter(op.offsets)
        head: SSAValue | None = None
        for stride, static_offset in zip(
            source_strides, op.static_offsets.iter_values(), strict=True
        ):
            offset_val = _resolve_offset(static_offset, dynamic_offsets, rewriter)
            increment = _apply_subview_stride(stride, offset_val, rewriter)
            if head is None:
                head = increment
            else:
                head = rewriter.insert_op(arith.AddiOp(head, increment)).result
                head.name_hint = "subview"

        if head is not None:
            byte_offset = get_bytes_offset(head, result_type.element_type, rewriter)
            pointer = get_offset_pointer(pointer, byte_offset, rewriter)

        rewriter.replace_op(op, ptr.FromPtrOp(pointer, result_type))


@dataclass
class LowerMemRefFuncOpPattern(RewritePattern):
    """
    Rewrites function arguments of MemRefType to PtrType.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        # rewrite function declaration
        new_input_types = [
            ptr.PtrType() if isinstance(arg, builtin.MemRefType) else arg
            for arg in op.function_type.inputs
        ]
        new_output_types = [
            ptr.PtrType() if isinstance(arg, builtin.MemRefType) else arg
            for arg in op.function_type.outputs
        ]
        op.function_type = func.FunctionType.from_lists(
            new_input_types,
            new_output_types,
        )

        if op.is_declaration:
            return

        insert_point = InsertPoint.at_start(op.body.blocks[0])

        # rewrite arguments
        for arg in op.args:
            if not isinstance(arg_type := arg.type, memref.MemRefType):
                continue

            old_type = cast(memref.MemRefType, arg_type)
            arg = rewriter.replace_value_with_new_type(arg, ptr.PtrType())

            if not arg.uses:
                continue

            rewriter.insert_op(
                cast_op := ptr.FromPtrOp(arg, old_type),
                insert_point,
            )
            rewriter.replace_uses_with_if(
                arg,
                cast_op.res,
                lambda x: x.operation is not cast_op,
            )


@dataclass
class LowerMemRefFuncReturnPattern(RewritePattern):
    """
    Rewrites all `memref` arguments to `func.return` into `ptr.PtrType`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter, /):
        if not any(isinstance(arg.type, memref.MemRefType) for arg in op.arguments):
            return

        new_arguments: list[SSAValue] = []

        # insert `memref -> ptr` casts for memref return values
        for argument in op.arguments:
            if isinstance(argument.type, memref.MemRefType):
                rewriter.insert_op(cast_op := ptr.ToPtrOp(argument))
                new_arguments.append(cast_op.res)
                cast_op.res.name_hint = argument.name_hint
            else:
                new_arguments.append(argument)

        rewriter.replace_op(op, func.ReturnOp(*new_arguments))


@dataclass
class LowerMemRefFuncCallPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter, /):
        if not any(
            isinstance(arg.type, memref.MemRefType) for arg in op.arguments
        ) and not any(isinstance(type, memref.MemRefType) for type in op.result_types):
            return

        # rewrite arguments
        new_arguments: list[SSAValue] = []

        # insert `memref -> ptr` casts for memref arguments values, if necessary
        for argument in op.arguments:
            if isinstance(argument.type, memref.MemRefType):
                rewriter.insert_op(cast_op := ptr.ToPtrOp(argument))
                new_arguments.append(cast_op.res)
                cast_op.res.name_hint = argument.name_hint
            else:
                new_arguments.append(argument)

        new_return_types = [
            ptr.PtrType() if isinstance(type, memref.MemRefType) else type
            for type in op.result_types
        ]

        new_ops: list[Operation] = [
            call_op := func.CallOp(op.callee, new_arguments, new_return_types)
        ]
        new_results = list(call_op.results)

        #  insert `ptr -> memref` casts for return values, if necessary
        for i, (new_result, old_result) in enumerate(zip(call_op.results, op.results)):
            new_result.name_hint = old_result.name_hint
            if isa(old_result.type, memref.MemRefType):
                new_ops.append(cast_op := ptr.FromPtrOp(new_result, old_result.type))
                new_results[i] = cast_op.res

        rewriter.replace_op(op, new_ops, new_results)


@dataclass
class ConvertCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CastOp, rewriter: PatternRewriter, /):
        assert isa(op.source.type, memref.MemRefType)
        rewriter.replace_matched_op((), (op.source,))


@dataclass
class ConvertReinterpretCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref.ReinterpretCastOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(
            (
                ptr_cast := ptr.ToPtrOp(op.source),
                builtin.UnrealizedConversionCastOp.get(
                    [ptr_cast.res], [op.result.type]
                ),
            )
        )


@dataclass(frozen=True)
class ConvertMemRefToPtr(ModulePass):
    name = "convert-memref-to-ptr"

    lower_func: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStorePattern(),
                    ConvertLoadPattern(),
                    ConvertSubviewPattern(),
                    ConvertCastOp(),
                    ConvertReinterpretCastOp(),
                ]
            )
        ).rewrite_module(op)

        if self.lower_func:
            PatternRewriteWalker(
                GreedyRewritePatternApplier(
                    [
                        LowerMemRefFuncOpPattern(),
                        LowerMemRefFuncCallPattern(),
                        LowerMemRefFuncReturnPattern(),
                    ]
                )
            ).rewrite_module(op)
