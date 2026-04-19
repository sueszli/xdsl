// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
  %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr
  %3 = llvm.mlir.zero : !llvm.ptr
  %zero_struct = llvm.mlir.zero : !llvm.struct<(i32, f32)>
  %zero_addrspace = llvm.mlir.zero : !llvm.ptr<1>
  %4 = "llvm.alloca"(%0) {"alignment" = 32 : i64, "elem_type" = i64} : (i64) -> !llvm.ptr
  %6 = "llvm.alloca"(%0) {"alignment" = 32 : i64, "elem_type" = i32} : (i64) -> !llvm.ptr
  %7 = "llvm.getelementptr"(%6, %0) <{"elem_type" = i64, noWrapFlags = 0 : i32, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
  "llvm.store"(%5, %6) {"alignment" = 32 : i64, "nontemporal", "ordering" = 0 : i64, "volatile_"} : (i64, !llvm.ptr) -> ()
  %2 = "llvm.load"(%1) : (!llvm.ptr) -> i32
  %5 = "llvm.load"(%4) : (!llvm.ptr) -> i64
  %8 = "llvm.load"(%4) <{"alignment" = 16 : i64}> : (!llvm.ptr) -> i32
  %9 = "llvm.load"(%4) <{"alignment" = 32 : i64, "ordering" = 1 : i64}> : (!llvm.ptr) -> i32
  %ptr_int = "llvm.ptrtoint"(%1) : (!llvm.ptr) -> i64
}) : () -> ()

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:    %1 = "llvm.inttoptr"(%0) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %2 = "llvm.mlir.zero"() : () -> !llvm.ptr
// CHECK-NEXT:    %3 = "llvm.mlir.zero"() : () -> !llvm.struct<(i32, f32)>
// CHECK-NEXT:    %4 = "llvm.mlir.zero"() : () -> !llvm.ptr<1>
// CHECK-NEXT:    %5 = "llvm.alloca"(%0) <{alignment = 32 : i64, elem_type = i64}> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %6 = "llvm.alloca"(%0) <{alignment = 32 : i64, elem_type = i32}> : (i64) -> !llvm.ptr
// CHECK-NEXT:    %7 = "llvm.getelementptr"(%6, %0) <{elem_type = i64, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:    "llvm.store"(%8, %6) <{alignment = 32 : i64, nontemporal, ordering = 0 : i64, volatile_}> : (i64, !llvm.ptr) -> ()
// CHECK-NEXT:    %9 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
// CHECK-NEXT:    %8 = "llvm.load"(%5) <{ordering = 0 : i64}> : (!llvm.ptr) -> i64
// CHECK-NEXT:    %10 = "llvm.load"(%5) <{alignment = 16 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> i32
// CHECK-NEXT:    %11 = "llvm.load"(%5) <{alignment = 32 : i64, ordering = 1 : i64}> : (!llvm.ptr) -> i32
// CHECK-NEXT:    %12 = "llvm.ptrtoint"(%1) : (!llvm.ptr) -> i64
// CHECK-NEXT:  }) : () -> ()
