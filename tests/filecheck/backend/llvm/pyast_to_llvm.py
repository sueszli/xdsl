# RUN: python %s | filecheck %s

from xdsl.backend.llvm.convert_func_arith_to_llvm import ConvertFuncArithToLLVMPass
from xdsl.context import Context
from xdsl.dialects.arith import AddfOp, Arith
from xdsl.dialects.builtin import Builtin, f64
from xdsl.frontend.pyast.context import PyASTContext

ctx = PyASTContext()
ctx.register_type(float, f64)
ctx.register_function(float.__add__, AddfOp)
ctx.register_dialect(Arith)
ctx.register_dialect(Builtin)


@ctx.parse_program
def float_add(x: float, y: float) -> float:
    return x + y


module = float_add.module

xdsl_ctx = Context()
xdsl_ctx.allow_unregistered = True
ConvertFuncArithToLLVMPass().apply(xdsl_ctx, module)

print(module)
# CHECK:      llvm.func @float_add(%x: f64, %y: f64) -> f64 {
# CHECK-NEXT:   %{{.*}} = llvm.fadd %x, %y : f64
# CHECK-NEXT:   llvm.return %{{.*}} : f64
# CHECK-NEXT: }
