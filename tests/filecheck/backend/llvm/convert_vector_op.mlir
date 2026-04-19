// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  llvm.func @fma_op_f32(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.fma %arg0, %arg1, %arg2 : vector<4xf32>
    llvm.return %0 : vector<4xf32>
  }

  // CHECK: define <4 x float> @{{"?fma_op_f32"?}}(<4 x float> [[A:%[^ ,)]+]], <4 x float> [[B:%[^ ,)]+]], <4 x float> [[C:%[^ ,)]+]])
  // CHECK:   [[R:%[^ ]+]] = call <4 x float> @{{"?llvm\.fma(\.v4f32)?"?}}(<4 x float> [[A]], <4 x float> [[B]], <4 x float> [[C]])
  // CHECK-NEXT:   ret <4 x float> [[R]]
  // CHECK-NEXT: }

  llvm.func @fma_op_f64(%arg0: vector<2xf64>, %arg1: vector<2xf64>, %arg2: vector<2xf64>) -> vector<2xf64> {
    %0 = vector.fma %arg0, %arg1, %arg2 : vector<2xf64>
    llvm.return %0 : vector<2xf64>
  }

  // CHECK: define <2 x double> @{{"?fma_op_f64"?}}(<2 x double> [[A:%[^ ,)]+]], <2 x double> [[B:%[^ ,)]+]], <2 x double> [[C:%[^ ,)]+]])
  // CHECK:   [[R:%[^ ]+]] = call <2 x double> @{{"?llvm\.fma(\.v2f64)?"?}}(<2 x double> [[A]], <2 x double> [[B]], <2 x double> [[C]])
  // CHECK-NEXT:   ret <2 x double> [[R]]
  // CHECK-NEXT: }

  llvm.func @broadcast_f32(%arg0: f32) -> vector<4xf32> {
    %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
    llvm.return %0 : vector<4xf32>
  }

  // CHECK: define <4 x float> @{{"?broadcast_f32"?}}(float [[A:%[^ ,)]+]])
  // CHECK:   [[INS:%[^ ]+]] = insertelement <4 x float> {{.*}}, float [[A]], i32 0
  // CHECK-NEXT:   [[R:%[^ ]+]] = shufflevector <4 x float> [[INS]], <4 x float> {{.*}}, <4 x i32> {{.*}}
  // CHECK-NEXT:   ret <4 x float> [[R]]
  // CHECK-NEXT: }
}
