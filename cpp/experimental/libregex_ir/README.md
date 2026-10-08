# Regex IR

Regex IR is the C++20 regex compiler and code generator vendored by cuDF.
`regex_ir::compile` parses a regular expression and emits operation-specific
CUDA C++ source. The public header contains only the standalone compiler contract;
cuDF's column wrappers, layout adapter, kernel assembly, and launch policy live
privately in libcudf. All intermediate representations and compiler passes
remain private implementation details.

This directory intentionally contains only the production sources and design
notes:

- `regex_ir.hpp`: public compiler API and generated-code metadata
- `regex_ir_detail.hpp`: private compiler intermediate representations and passes
- `regex_ir.cpp`: parser and compiler passes
- `regex_ir_unicode.hpp`: attributed Unicode predicate tables
- `execution_plan.hpp` / `execution_plan.cpp`: private plan types, executor selection,
  machine construction, and bounded workspace analysis
- `cuda_codegen.hpp` / `cuda_codegen.cpp`: source-generation interface and CUDA emission
- `cuda_abi.hpp`: typed ABI shared by matcher and column wrapper emission
- `executor.cuh`: CUDA implementations of input helpers, executors,
  and regex operations
- `executor.cu`: NVRTC compilation entry point, with the matcher header before the PCH boundary

The implementation includes its private header directly; no implementation
macro changes the declarations exposed by `regex_ir.hpp`.

The private libcudf integration, kernel wrappers, and compiled layout adapter
live in `cpp/src/strings/experimental/regex_jit`.

Both NVRTC entry points include their stable runtime headers before
`#pragma nv_hdrstop`. The column entry point includes `kernels.cuh`, which also
includes the matcher runtime. Workspace sizes and ABI macros stay in generated
code after the boundary, allowing wrapper requests to share the PCH.
