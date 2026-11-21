Authoring JIT Kernels
=====================

This guide is for libcudf contributors writing CUDA C++ kernels and device functions.

The kernel author chooses two specialization boundaries:

- **Kernel configuration:** template parameters describe the input/output types and execution policies. Host-side reflection maps runtime metadata to a particular instantiation.
- **UDF behavior:** the kernel declares an external device function with an agreed signature. Separately compiled fragments supply its implementation and any dependencies.

The library-owned kernel TU declares the device-function entry point it calls. User-owned UDF TUs define the computation and any helper functions. The library specializes its kernel independently, then connects the fragments through the UDF ABI:

.. mermaid::

   flowchart TD
       subgraph Library["Kernel Author"]
           KS["Kernel TU"] --> KernelIR
           Types["Input and Output Types"] --> KernelIR
           KernelIR["Kernel LTO IR compiled ahead of time or at runtime"]
           KernelIR --> Link["Link Kernel"]
           Link --> Launch["Launch Kernel"]
       end
       subgraph User["User Path"]
           US["User UDF TUs"] --> UC["Compile ahead of time or at runtime"]
           UC --> UserIR["User UDF fragments containing LTO IR"]
       end
       UserIR --> Link

The following example supports three non-nullable, fixed-width arithmetic inputs and one output.
The example UDF uses three ``FLOAT64`` inputs and a ``FLOAT64`` output. It follows the :doc:`device UDF ABI <udf_abi>`

1. Write the Kernel and UDF Translation Units
---------------------------------------------

Place the kernel in ``cpp/src/sample_ternary_op_kernel/jit/ternary_op_kernel.cu``:

.. code-block:: cpp

   // file: cpp/src/sample_ternary_op_kernel/jit/ternary_op_kernel.cu
   #include <cudf/column/column_device_view_base.cuh>
   #include <cudf/detail/utilities/grid_1d.cuh>
   #include <cudf/errc.hpp>
   #include <cudf/types.hpp>

   #pragma nv_hdrstop

   #include <cudf/detail/kernel_instance.cuh>

   namespace cudf::detail {

   using transform_udf_type = CUDF_TRANSFORM_UDF_TYPE;

   extern "C" __device__ transform_udf_type cudf_transform_udf;

   template <typename X, typename Y, typename Z, typename Output>
   __device__ void ternary_op_kernel(column_device_view_core const* inputs,
                                     mutable_column_device_view_core const* output,
                                     size_type n,
                                     error_int* error_code)
   {
     auto const start  = grid_1d::global_thread_id();
     auto const stride = grid_1d::grid_stride();

     for (auto row = start; row < n; row += stride) {
       auto const error = cudf_transform_udf(
   #if CUDF_TRANSFORM_UDF_INDEXED
         row,
   #endif
         &output->element<Output>(row),
         inputs[0].element<X>(row),
         inputs[1].element<Y>(row),
         inputs[2].element<Z>(row));
       if (error != errc::SUCCESS) { atomicMax(error_code, static_cast<error_int>(error)); }
     }
   }

   extern "C" __global__ void cudf_kernel_entry(column_device_view_core const* inputs,
                                                mutable_column_device_view_core const* output,
                                                size_type n,
                                                error_int* error_code)
   {
     CUDF_KERNEL_INSTANCE(inputs, output, n, error_code);
   }

   }  // namespace cudf::detail

``cudf_kernel_entry`` is the entry-point name required by ``cudf::get_lto_linked_kernel``. C linkage keeps the exported name independent of the kernel's C++ namespace and template parameters. Its pointer arguments let the host launch it without knowing the element types at compile time.

``#pragma nv_hdrstop`` marks the end of the source prefix used to create a **precompiled header (PCH)**. With PCH enabled, NVRTC can reuse the compiler state produced by processing that prefix. The generated ``kernel_instance.cuh`` follows the pragma so each compilation processes its own ``CUDF_KERNEL_INSTANCE``, ``CUDF_TRANSFORM_UDF_TYPE``, and ``CUDF_TRANSFORM_UDF_INDEXED`` definitions.

The user writes a UDF in a separate translation unit. For this example, it can evaluate asymmetric squared loss, weighting rows where ``actual < target`` by ``penalty``:

.. code-block:: sql

   CASE
     WHEN actual >= target THEN POWER(actual - target, 2)
     ELSE penalty * POWER(actual - target, 2)
   END

.. code-block:: cpp

   // file: cpp/src/sample_ternary_op_kernel/jit/asymmetric_squared_loss_udf.cu
   #include <cudf/errc.hpp>

   extern "C" __device__ cudf::errc asymmetric_squared_loss(double* out,
                                                            double actual,
                                                            double target,
                                                            double penalty)
   {
     auto const difference         = actual - target;
     auto const squared_difference = difference * difference;
     *out = actual >= target ? squared_difference : penalty * squared_difference;
     return cudf::errc::SUCCESS;
   }

The kernel declares its fixed dispatch symbol, ``cudf_transform_udf``, while this UDF exports the user-chosen name ``asymmetric_squared_loss``.
A forwarding adapter connects the two names at link time. The UDF assigns its output and returns ``cudf::errc::SUCCESS``; another implementation can return an evaluation error.

2. Embed the Kernel Source and Prepare UDF Fragments
----------------------------------------------------

Register the kernel source on the existing ``cudf_cuda_embed`` target before its ``rtcx_embed`` finalization:

.. code-block:: cmake

   rtcx_embed_includes(
     cudf_cuda_embed
     SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/sample_ternary_op_kernel/jit"
     DEST_DIRECTORY cudf/cpp/src/sample_ternary_op_kernel/jit
     INCLUDE_DIRECTORIES cudf/cpp/src
   )

``DEST_DIRECTORY`` is the virtual directory within the embedded source bundle. The host compiler helper uses the full source identifier ``cudf/cpp/src/sample_ternary_op_kernel/jit/ternary_op_kernel.cu``.

For this example, compile and embed the UDF fragment on the existing ``cudf_fragments`` target before its finalization:

.. code-block:: cmake

   add_fragment(
     cudf_fragments
     FRAGMENT asymmetric_squared_loss_udf
     SOURCE src/sample_ternary_op_kernel/jit/asymmetric_squared_loss_udf.cu
   )

``add_fragment`` is an internal cuDF build helper. Here it compiles only the UDF translation unit into a fatbin containing LTO code. The generated ``<cudf_fragments.hpp>`` header provides access to the embedded LTO bytes.

Embedding the UDF is a packaging choice. Callers can instead compile CUDA C++ with ``cudf::compile_cuda_to_lto_ir`` or provide fragments from another compatible compiler, then describe them with ``cudf::transform_udf``.

3. Reflect, Compile, and Link the Kernel
----------------------------------------

First, provide a reflection function mapping runtime types to the kernel's template arguments:

.. code-block:: cpp

   // file: cpp/src/sample_ternary_op_kernel/ternary_op.cu
   #include "jit/helpers.hpp"
   #include "jit/reflect.hpp"

   #include <cudf/transform.hpp>
   #include <cudf/types.hpp>
   #include <cudf/udf.hpp>
   #include <cudf/utilities/error.hpp>
   #include <cudf/utilities/type_dispatcher.hpp>

   #include <array>
   #include <format>
   #include <string>
   #include <vector>

   std::string reflect_ternary_op_kernel(cudf::data_type x_type,
                                         cudf::data_type y_type,
                                         cudf::data_type z_type,
                                         cudf::data_type output_type)
   {
     return std::format("cudf::detail::ternary_op_kernel<{}, {}, {}, {}>",
                        cudf::type_to_name(x_type),
                        cudf::type_to_name(y_type),
                        cudf::type_to_name(z_type),
                        cudf::type_to_name(output_type));
   }

For the UDF signature, use ``cudf::jit::reflect_udf_signature``. Pass input/output specifications and the requested index count. This example uses logical types (``use_physical_types = false``) to match its typed column accessors. The caller must supply supported arithmetic types and a UDF with the matching signature.

Append the kernel getter to the same file:

.. code-block:: cpp

   // file: cpp/src/sample_ternary_op_kernel/ternary_op.cu

   cudf::kernel get_kernel(cudf::transform_udf const& udf,
                           cudf::data_type x_type,
                           cudf::data_type y_type,
                           cudf::data_type z_type,
                           cudf::data_type output_type)
   {
     CUDF_EXPECTS(udf.is_null_aware == cudf::null_aware::NO,
                  "This example kernel requires a non-null-aware UDF");

     std::array inputs{cudf::transform_input_spec{.type = x_type.id()},
                       cudf::transform_input_spec{.type = y_type.id()},
                       cudf::transform_input_spec{.type = z_type.id()}};
     std::array outputs{cudf::transform_output_spec{.type = output_type.id()}};
     auto const index_count = udf.indexed ? 1 : 0;
     auto const instance    = reflect_ternary_op_kernel(x_type, y_type, z_type, output_type);
     auto const signature =
       cudf::jit::reflect_udf_signature(false, inputs, outputs, false, index_count);
     auto kernel_lto_ir = cudf::jit::get_udf_kernel_fragment(
       "cudf/cpp/src/sample_ternary_op_kernel/jit/ternary_op_kernel.cu",
       instance,
       signature,
       udf.indexed);

     std::vector<rtcx::memory_fragment> fragments{
       {.data = kernel_lto_ir->view(), .type = rtcx::binary_type::LTO_IR}};
     auto adapter = cudf::get_udf_adapter(
       udf.symbol, "cudf_transform_udf", false, inputs, outputs, false, index_count);
     if (adapter) {
       fragments.push_back({.data = adapter->view().data, .type = rtcx::binary_type::LTO_IR});
     }
     for (auto const& fragment : udf.fragments) {
       fragments.push_back(
         {.data = fragment.data, .type = cudf::jit::as_rtcx_binary_type(fragment.type)});
     }

     return cudf::get_lto_linked_kernel(
       "cudf/cpp/src/sample_ternary_op_kernel/jit/ternary_op_kernel.cu", {}, fragments);
   }

``get_udf_kernel_fragment`` specializes `ternary_op_kernel`. For the ``FLOAT64`` inputs and output, it generates the following kernel instance:

.. code-block:: cpp

   // Generated cudf/detail/kernel_instance.cuh
   #define CUDF_KERNEL_INSTANCE       cudf::detail::ternary_op_kernel<double, double, double, double>
   #define CUDF_TRANSFORM_UDF_TYPE    cudf::errc(double*, double, double, double)
   #define CUDF_TRANSFORM_UDF_INDEXED 0

`get_kernel` then links the kernel fragment, a forwarding adapter when needed, and all fragments supplied through ``udf.fragments``.


4. Launch the Kernel
--------------------

Append the following helpers to the host file. ``asymmetric_squared_loss_udf`` packages the embedded UDF fragment.

.. code-block:: cpp

   #include <cudf/column/column_device_view_base.cuh>
   #include <cudf/detail/utilities/grid_1d.cuh>
   #include <cudf/errc.hpp>
   #include <cudf/types.hpp>
   #include <cudf/udf.hpp>

   #include <cuda/stream>

   #include <cudf_fragments.hpp>

   #include <array>
   #include <cstdint>

   cudf::transform_udf asymmetric_squared_loss_udf()
   {
     static std::array const fragments{
       cudf::lto_fragment{.type = cudf::lto_binary_type::FATBIN,
                          .data = cudf_fragments::get(cudf_fragments::asymmetric_squared_loss)}};
     return {.symbol = "asymmetric_squared_loss", .fragments = fragments};
   }

   void launch_ternary_op(cudf::transform_udf const& udf,
                          cudf::column_device_view_core const* device_inputs,
                          cudf::mutable_column_device_view_core const* device_output,
                          cudf::size_type n,
                          cudf::error_int* device_error,
                          cudf::data_type x,
                          cudf::data_type y,
                          cudf::data_type z,
                          cudf::data_type output_type,
                          cuda::stream_ref stream)
   {
     if (n == 0) { return; }

     auto kernel = get_kernel(udf, x, y, z, output_type);

     // Launch the kernel with device-side column views.
     cudf::detail::grid_1d const grid{n, 256};
     kernel.launch_with({static_cast<uint32_t>(grid.num_blocks)},
                        {static_cast<uint32_t>(grid.num_threads_per_block)},
                        0,
                        stream,
                        device_inputs,
                        device_output,
                        n,
                        device_error);
   }

   void asymmetric_squared_loss(cudf::column_device_view_core const* device_inputs,
                                cudf::mutable_column_device_view_core const* device_output,
                                cudf::size_type n,
                                cudf::error_int* device_error,
                                cuda::stream_ref stream)
   {
     launch_ternary_op(asymmetric_squared_loss_udf(),
                       device_inputs,
                       device_output,
                       n,
                       device_error,
                       cudf::data_type{cudf::type_id::FLOAT64},
                       cudf::data_type{cudf::type_id::FLOAT64},
                       cudf::data_type{cudf::type_id::FLOAT64},
                       cudf::data_type{cudf::type_id::FLOAT64},
                       stream);
   }
