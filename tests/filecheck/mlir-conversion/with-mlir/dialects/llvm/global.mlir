// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP
builtin.module {
  llvm.mlir.global internal constant @str0("Hello world!") {addr_space = 0 : i32} : !llvm.array<12 x i8>
  llvm.mlir.global internal constant @data(0 : i32) {addr_space = 0 : i32} : i32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("Hello world!") {{.*}}: !llvm.array<12 x i8>
// CHECK-NEXT:   llvm.mlir.global internal constant @data(0 : i32) {{.*}}: i32
// CHECK-NEXT: }
