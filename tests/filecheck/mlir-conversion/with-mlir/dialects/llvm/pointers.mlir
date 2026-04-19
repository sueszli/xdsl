// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

builtin.module {
    %null_ptr = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.mlir.zero : !llvm.ptr

    %zero_struct = llvm.mlir.zero : !llvm.struct<(i32, f32)>
    // CHECK: llvm.mlir.zero : !llvm.struct<(i32, f32)>

    %null_addrspace = llvm.mlir.zero : !llvm.ptr<1>
    // CHECK: llvm.mlir.zero : !llvm.ptr<1>
}
