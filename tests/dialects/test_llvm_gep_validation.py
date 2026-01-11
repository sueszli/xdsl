import pytest
from xdsl.dialects import llvm, builtin, arith
from xdsl.dialects.builtin import DenseArrayBase, i64, i32
from xdsl.utils.exceptions import VerifyException

def test_gep_indices_type_validation():
    """
    Test that GEPOp verifies the type of rawConstantIndices.
    It should only allow i32, so i64 should raise a VerifyException.
    """
    size = arith.ConstantOp.from_int_and_width(1, 32)
    opaque_ptr = llvm.AllocaOp(size, builtin.i32)
    ptr_type = llvm.LLVMPointerType()

    # Create a valid GEP first
    gep = llvm.GEPOp.from_mixed_indices(
        opaque_ptr,
        indices=[1],
        pointee_type=builtin.i32,
        result_type=ptr_type,
    )

    # Manually change the rawConstantIndices to use i64
    gep.properties["rawConstantIndices"] = DenseArrayBase.from_list(i64, [1])

    # This should raise VerifyException after the fix.
    with pytest.raises(VerifyException, match="rawConstantIndices"):
        gep.verify()
