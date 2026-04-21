// RUN: XDSL_ROUNDTRIP
builtin.module {
  %0 = arith.constant 0 : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  %3 = llvm.mlir.zero : !llvm.ptr
  %4 = llvm.alloca %0 x index {alignment = 32 : i64} : (i64) -> !llvm.ptr
  %6 = llvm.getelementptr %4[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %7 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> i32
  %5 = llvm.load %4 : !llvm.ptr -> index
  %8 = llvm.load %4 {alignment = 16 : i64} : !llvm.ptr -> index
  %9 = llvm.load %4 atomic unordered {alignment = 32 : i64} : !llvm.ptr -> index
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = arith.constant 0 : i64
// CHECK-NEXT:    %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
// CHECK-NEXT:    %2 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %3 = llvm.alloca %0 x index {alignment = 32 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %4 = llvm.getelementptr %3[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %5 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %6 = llvm.load %1 : !llvm.ptr -> i32
// CHECK-NEXT:    %7 = llvm.load %3 : !llvm.ptr -> index
// CHECK-NEXT:    %8 = llvm.load %3 {alignment = 16 : i64} : !llvm.ptr -> index
// CHECK-NEXT:    %9 = llvm.load %3 atomic unordered {alignment = 32 : i64} : !llvm.ptr -> index
// CHECK-NEXT:  }
