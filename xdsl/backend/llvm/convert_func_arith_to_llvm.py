from xdsl.context import Context
from xdsl.dialects import arith, func, llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.llvm import FastMathAttr, LinkageAttr, LLVMFunctionType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertAddf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddfOp, rewriter: PatternRewriter, /) -> None:
        rewriter.replace_op(
            op, llvm.FAddOp(op.lhs, op.rhs, fast_math=FastMathAttr(op.fastmath.data))
        )


class ConvertReturn(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: func.ReturnOp, rewriter: PatternRewriter, /
    ) -> None:
        if op.arguments:
            rewriter.replace_op(op, llvm.ReturnOp(op.arguments[0]))
        else:
            rewriter.replace_op(op, llvm.ReturnOp())


class ConvertFunc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /) -> None:
        outputs = op.function_type.outputs.data
        rewriter.replace_op(
            op,
            llvm.FuncOp(
                sym_name=op.sym_name.data,
                function_type=LLVMFunctionType(
                    inputs=list(op.function_type.inputs.data),
                    output=outputs[0] if outputs else None,
                ),
                linkage=LinkageAttr("external"),
                body=rewriter.move_region_contents_to_new_regions(op.body),
            ),
        )


class ConvertFuncArithToLLVMPass(ModulePass):
    name = "convert-func-arith-to-llvm"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertAddf(), ConvertReturn()]),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertFunc()]),
            apply_recursively=False,
        ).rewrite_module(op)
