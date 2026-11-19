# UDF ABI

The general value-based device UDF signatures are:

```cpp
// Non-indexed
cudf::errc udf(Output*... outputs, Input... inputs);

// Indexed
cudf::errc udf(cudf::size_type... indices, Output*... outputs, Input... inputs);
```

Output pointers precede input values, and the function returns a status rather than its computed values. Indexed calls place one transform row index or two left/right join row indices before the output pointers. User-data inputs are represented as raw `void*` values. Null-aware calls replace ordinary values with `cuda::std::optional<T>` and output pointers with `cuda::std::optional<T>*`.

A value-based UDF receives already-loaded row values and output locations. An indexed UDF can also express the descriptor-based form through `user_data`.

An **application binary interface (ABI)** specifies how separately compiled code exchanges arguments, results, and errors. libcudf's device UDF ABI is the contract between a row-processing kernel and the device function it calls.

## Why an ABI Is Needed

A kernel and its UDF can be compiled independently, potentially by different language compilers. They still need to agree on parameter order, device value representations, return status, and the exported symbol.

A shared ABI lets libcudf reuse column loading, scalar broadcasting, null-mask handling, and kernel launch code across UDFs.

The language frontend supplies a device function rather than generating a complete column-processing kernel. Explicit symbols also let one module export several UDFs and let those functions call helpers defined in other fragments.


## LTO IR and PTX Compatibility

Both LTO IR and PTX producers must match the kernel's function declaration.

In addition, the LTO IR and PTX fragments must meet their respective CUDA toolchain compatibility requirements.

| Format | Toolchain compatibility requirement |
|---|---|
| **LTO IR** | CUDA guarantees compatibility within a major release, not across major releases. |
| **PTX** | PTX can cross CUDA major-version boundaries, provided the consuming compiler/linker supports its PTX ISA version and the target GPU supports its required features. |

## Numba's ABI

Numba's `abi="numba"` device calling convention already separates the Python result from the function status. For a scalar-returning function:

```python
def magnitude(x, y):
    return sqrt(x * x + y * y)
```

When compiled for two `float32` inputs and a `float32` result, its lowered signature has the following shape:

```cpp
int32_t numba_magnitude(float* result, float x, float y);
```

This has the same parameter shape as a non-indexed, non-null-aware cuDF UDF. The integer return carries status, while the magnitude is written through `result`. See [Numba's device ABI examples](https://nvidia.github.io/numba-cuda/user/cuda_compilation.html#c-and-numba-abi-examples).

| Numba device ABI | cuDF UDF ABI |
|---|---|
| Pointer to the Python return value | Pointer to the output value |
| Lowered Python arguments | Input values |
| Signed 32-bit status | `cudf::errc`, represented by signed 32-bit `cudf::error_int` |
| Compiler-generated device symbol | Explicit `transform_udf.symbol` |

This avoids requiring a Python wrapper that takes an output pointer and returns a status as an ordinary Python result: Numba already supplies those two channels.


## Passing Compiled Fragments from Python

For example, Python can compile a Numba device function to LTO IR and pass its exported symbol to cuDF:

```python
from math import sqrt
import pylibcudf as plc
from numba import cuda, float32
from pylibcudf.udf import LTOBinaryType, LTOFragment, TransformUDF


@cuda.jit(device=True, lto=True)
def magnitude(x, y):
    return sqrt(x * x + y * y)


compiled = magnitude.compile_device((float32, float32), return_type=float32)

lto_ir = bytes(compiled.library.get_ltoir(cc=(7, 5)))

fragment = LTOFragment(LTOBinaryType.LTO_IR, lto_ir)

udf = TransformUDF(compiled.fndesc.llvm_func_name, (fragment,), is_null_aware=False, indexed=False)

result = plc.transform.transform(
    [x_column, y_column],
    udf,
    plc.DataType(plc.TypeId.FLOAT32),
    plc.types.OutputNullability.PRESERVE,
)
```

Other languages can follow the same route without changes to the kernel's column-processing loop.
