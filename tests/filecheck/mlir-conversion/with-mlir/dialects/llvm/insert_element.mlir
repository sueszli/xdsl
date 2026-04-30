// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

%vec, %val, %idx = "test.op"() : () -> (vector<4xf32>, f32, i32)
// CHECK: [[VEC:%\d+]], [[VAL:%\d+]], [[IDX:%\d+]]

%0 = llvm.insertelement %val, %vec[%idx : i32] : vector<4xf32>
// CHECK: llvm.insertelement [[VAL]], [[VEC]]{{\[}}[[IDX]] : i32] : vector<4xf32>
