UDF ABI
=======

Why an ABI Is Needed
--------------------

An **application binary interface (ABI)** specifies how separately compiled code exchanges arguments, results, and errors. libcudf's device UDF ABI is the contract between a row-processing kernel and the device function it calls.

A kernel and its UDF can be compiled independently, potentially by different language compilers. They still need to agree on parameter order, device value representations, return status, and the exported symbol.

A consistent ABI lets libcudf reuse column loading, scalar broadcasting, null-mask handling, and kernel launch code across UDFs.

The language frontend supplies a device function rather than generating a complete column-processing kernel. Explicit symbols also let one module export several UDFs and let those UDF functions call helpers defined in other fragments.

UDF Signature
-------------

The cuDF's UDF ABI is specified as:

.. code-block:: cpp

   // Non-indexed form
   cudf::errc udf(Output*... outputs, Input... inputs);

   // Indexed form
   cudf::errc udf(cudf::size_type... indices, Output*... outputs, Input... inputs);

It has the following properties:

- The return type is ``cudf::errc``, represented by a signed 32-bit integer.
- Indexed UDFs have logical row indices before the output pointers. e.g. one transform row index or two left/right join row indices.
- The output pointers come before the input values in the parameter list.
- User-data inputs are represented as raw ``void*`` values.
- Null-aware UDFs have ``cuda::std::optional<T>`` inputs and output pointers.


An indexed UDF can also pass user-data inputs through ``user_data`` parameters.

UDF Type Mappings
-----------------

The following table maps column values and UDF arguments to CUDA C++ types.

.. list-table::
   :header-rows: 1

   * - Column type or argument
     - UDF value type ``T`` and usage
   * - ``INT8``
     - ``int8_t``
   * - ``INT16``
     - ``int16_t``
   * - ``INT32``
     - ``int32_t``
   * - ``INT64``
     - ``int64_t``
   * - ``UINT8``
     - ``uint8_t``
   * - ``UINT16``
     - ``uint16_t``
   * - ``UINT32``
     - ``uint32_t``
   * - ``UINT64``
     - ``uint64_t``
   * - ``FLOAT32``
     - ``float``
   * - ``FLOAT64``
     - ``double``
   * - ``BOOL8``
     - ``bool``
   * - ``TIMESTAMP_DAYS``
     - ``cudf::timestamp_D``
   * - ``TIMESTAMP_SECONDS``
     - ``cudf::timestamp_s``
   * - ``TIMESTAMP_MILLISECONDS``
     - ``cudf::timestamp_ms``
   * - ``TIMESTAMP_MICROSECONDS``
     - ``cudf::timestamp_us``
   * - ``TIMESTAMP_NANOSECONDS``
     - ``cudf::timestamp_ns``
   * - ``DURATION_DAYS``
     - ``cudf::duration_D``
   * - ``DURATION_SECONDS``
     - ``cudf::duration_s``
   * - ``DURATION_MILLISECONDS``
     - ``cudf::duration_ms``
   * - ``DURATION_MICROSECONDS``
     - ``cudf::duration_us``
   * - ``DURATION_NANOSECONDS``
     - ``cudf::duration_ns``
   * - ``DECIMAL32``
     - ``numeric::decimal32``
   * - ``DECIMAL64``
     - ``numeric::decimal64``
   * - ``DECIMAL128``
     - ``numeric::decimal128``
   * - ``STRING``
     - ``cudf::string_view``; for outputs, used when offsets are not preallocated.
   * - ``STRING`` output with preallocated offsets
     - ``cudf::mutable_string_view``, describing the writable character range for that row.
   * - ``DICTIONARY32`` input
     - The dictionary key's value type, decoded before calling the UDF. Dictionary outputs are not supported.
   * - ``user_data`` input
     - ``cudf::voidptr_t`` (an alias for ``void*``); never wrapped in ``optional``.
   * - Leading row index
     - ``cudf::size_type``; never wrapped in ``optional``.
   * - ``EMPTY``, ``LIST``, ``STRUCT``
     - No supported column-value mapping for these UDF APIs.

Except where noted in the table, non-null-aware UDF inputs are passed as ``T`` and outputs as ``T*``. For null-aware UDFs, these become ``cuda::std::optional<T>`` and ``cuda::std::optional<T>*``. Decimal values carry both their unscaled integer and scale.

LTO IR and PTX Compatibility
----------------------------

Both LTO IR and PTX producers must match the kernel's function declaration.

In addition, the LTO IR and PTX fragments must meet their respective CUDA toolchain compatibility requirements.

.. list-table::
   :header-rows: 1

   * - Format
     - Toolchain compatibility requirement
   * - **LTO IR**
     - CUDA guarantees compatibility within a major release, not across major releases.
   * - **PTX**
     - PTX can cross CUDA major-version boundaries, provided the consuming compiler/linker supports its PTX ISA version and the target GPU supports its required features.

Numba's ABI
-----------

Numba's ``abi="numba"`` device calling convention already separates the Python result from the function status. For a scalar-returning function:

.. code-block:: python

   def magnitude(x, y):
       return sqrt(x * x + y * y)

When compiled for two ``float32`` inputs and a ``float32`` result, its lowered signature has the following shape:

.. code-block:: cpp

   int32_t numba_magnitude(float* result, float x, float y);

This has the same parameter shape as a non-indexed, non-null-aware cuDF UDF. The integer return carries status, while the magnitude is written through ``result``. See `Numba's device ABI examples <https://nvidia.github.io/numba-cuda/user/cuda_compilation.html#c-and-numba-abi-examples>`__.

.. list-table::
   :header-rows: 1

   * - Numba device ABI
     - cuDF UDF ABI
   * - Pointer to the Python return value
     - Pointer to the output value
   * - Lowered Python arguments
     - Input values
   * - Signed 32-bit status
     - ``cudf::errc``, represented by signed 32-bit ``cudf::error_int``
   * - Compiler-generated device symbol
     - Explicit ``transform_udf.symbol``

Python UDF Interop with cuDF
--------------------------------------

For example, Python can compile a Numba device function to LTO IR and pass its exported symbol to cuDF:

.. code-block:: python

   from math import sqrt

   from numba import cuda, float32

   import pylibcudf as plc
   from pylibcudf.udf import LTOBinaryType, LTOFragment, TransformUDF


   @cuda.jit(device=True, lto=True)
   def magnitude(x, y):
       return sqrt(x * x + y * y)


   compiled = magnitude.compile_device((float32, float32), return_type=float32)

   lto_ir = bytes(compiled.library.get_ltoir(cc=(7, 5)))

   fragment = LTOFragment(LTOBinaryType.LTO_IR, lto_ir)

   udf = TransformUDF(
       compiled.fndesc.llvm_func_name,
       (fragment,),
       is_null_aware=False,
       indexed=False,
   )

   result = plc.transform.transform(
       [x_column, y_column],
       udf,
       plc.DataType(plc.TypeId.FLOAT32),
       plc.types.OutputNullability.PRESERVE,
   )

Other languages can follow the same route without changes to the kernel's column-processing loop.
