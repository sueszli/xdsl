from xdsl.dialects import llvm
from xdsl.dialects import arith


def test_return_op_with_value():
    const = arith.ConstantOp.from_int_and_width(42, 32)
    val = const.result

    op = llvm.ReturnOp(val)

    assert op.arg == val
    assert op.operands[0] == val
    assert len(op.operands) == 1


def test_return_op_with_none():
    op_none = llvm.ReturnOp(None)
    assert op_none.arg is None
    assert len(op_none.operands) == 0


def test_return_op_empty():
    op_empty = llvm.ReturnOp()
    assert op_empty.arg is None
    assert len(op_empty.operands) == 0
