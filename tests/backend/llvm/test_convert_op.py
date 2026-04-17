from typing import Any
from unittest.mock import MagicMock

import pytest
from llvmlite import ir

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.dialects import llvm
from xdsl.dialects.builtin import VectorType, f32, f64, i32
from xdsl.dialects.vector import BitcastOp
from xdsl.ir import Block, SSAValue


def test_convert_indirect_call_raises():
    block = Block(arg_types=[i32])
    arg = block.args[0]
    op = llvm.CallOp("dummy_callee", arg, return_type=i32)

    # simulate indirect call
    op.callee = None

    builder = MagicMock()
    val_map: dict[SSAValue, Any] = {arg: MagicMock()}

    with pytest.raises(NotImplementedError, match="Indirect calls not yet implemented"):
        convert_op(op, builder, val_map)


def test_convert_null():
    op = llvm.NullOp(llvm.LLVMPointerType())
    val_map: dict[SSAValue, Any] = {}

    convert_op(op, MagicMock(), val_map)

    assert str(val_map[op.nullptr]) == "ptr null"


def test_convert_vector_bitcast():
    src_type = VectorType(f32, [4])
    res_type = VectorType(f64, [2])
    block = Block(arg_types=[src_type])
    arg = block.args[0]
    op = BitcastOp(arg, res_type)

    builder = MagicMock()
    src_val = MagicMock()
    val_map: dict[SSAValue, Any] = {arg: src_val}

    convert_op(op, builder, val_map)

    builder.bitcast.assert_called_once()
    bitcast_args = builder.bitcast.call_args.args
    assert bitcast_args[0] is src_val
    assert isinstance(bitcast_args[1], ir.VectorType)
    assert bitcast_args[1].count == 2
    assert isinstance(bitcast_args[1].element, ir.DoubleType)
    assert val_map[op.result] is builder.bitcast.return_value
