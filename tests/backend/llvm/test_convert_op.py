from typing import Any
from unittest.mock import MagicMock

import pytest

from xdsl.backend.llvm.convert_op import convert_op
from xdsl.dialects import llvm, vector
from xdsl.dialects.builtin import VectorType, f32, i32
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


def test_convert_broadcast():
    block = Block(arg_types=[f32])
    arg = block.args[0]
    vec_type = VectorType(f32, [4])
    op = vector.BroadcastOp(arg, vec_type)

    source_val = MagicMock()
    inserted = MagicMock()
    shuffled = MagicMock()

    builder = MagicMock()
    builder.insert_element.return_value = inserted
    builder.shuffle_vector.return_value = shuffled

    val_map: dict[SSAValue, Any] = {arg: source_val}

    convert_op(op, builder, val_map)

    builder.insert_element.assert_called_once()
    builder.shuffle_vector.assert_called_once()
    assert val_map[op.vector] is shuffled
