
from xdsl.dialects import llvm
from xdsl.dialects import arith
from xdsl.ir import SSAValue
from xdsl.dialects.builtin import i32

def test_return_op():
    # Create an SSA value using a constant op
    const = arith.ConstantOp.from_int_and_width(42, 32)
    val = const.result

    # Create a ReturnOp with the SSA value
    op = llvm.ReturnOp(val)

    assert op.arg == val
    assert op.operands[0] == val
    assert len(op.operands) == 1

    # Create a ReturnOp with None
    op_none = llvm.ReturnOp(None)
    assert op_none.arg is None
    assert len(op_none.operands) == 0

    # Create a ReturnOp with no arguments
    op_empty = llvm.ReturnOp()
    assert op_empty.arg is None
    assert len(op_empty.operands) == 0
