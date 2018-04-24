Tensor4ts
---

Tensor library written in TypeScript

Note: this library is compiled and published to npm in es5 as tensor4js

TODO:
* ~~Binary element-wise ops with broadcasting~~ [DONE]
* ~~Element-wise transform~~ [DONE]
* ~~ReduceSum~~, ~~ReduceMax~~, ~~ReduceMin~~ [DONE]
* ~~ReduceMean~~ [DONE]
* ~~Tile~~ [DONE]
* ~~MatMul Standard GEMM~~ [DONE]
* GEMM Optimization (Blocking + Packing)
* WebGL Add
* ~~ArgMin~~[DONE]
* Emscripten / WebAssembly
* Size Optimization
* One Hot
* ~~L1, L2, LInfNorms~~ [DONE]
* Normalization - L1, L2, Max
* Batch Norm



## Rule for different data types

### Below is what it SHOULD have been:

General pairwise ops:

Case 1: Int(x bit) + Int(y bit) -> Int(Max(x,y) bit)
Case 2: Int(x bit) + Float(y bit) -> Float(Max(x * 2, y) bit)
Case 3: Float(x bit) + Float(y bit) -> Float(Max(x, y) bit)

where max X or y = 64 bit

example:
Int8 + Int32 = Int32
Int32 + Float32 = Float64
Float32 + Float64 = Float64
Int64 + Float64 = Float64

Operations with float results

Int(x) -> Float(x*2)
Float(x) -> Float(x)

example:
Int32 -> Float64
sin, exp, log etc.

### HOWEVER, since the most common occurrence is with Int32 and Float32

Rule changes to:

Int(x) + Int(y) -> Int(max(x,y))
Int(x) + Float(y) -> Float(max(x,y))
Float(x) + Float(y) -> Float(max(x,y))