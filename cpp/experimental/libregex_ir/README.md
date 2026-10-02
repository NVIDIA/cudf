# Regex IR

Regex IR is the C++20 regex compiler and code generator vendored by cuDF.
`regex_ir::compile` parses a regular expression and emits operation-specific
NVVM IR. The public header contains only the standalone compiler contract;
cuDF's column wrappers, layout adapter, kernel assembly, and launch policy live
privately in libcudf. All intermediate representations and compiler passes
remain private implementation details.

This directory intentionally contains only the production sources and design
notes:

- `regex_ir.hpp`: public compiler API and generated-code metadata
- `regex_ir_detail.hpp`: private compiler intermediate representations and passes
- `regex_ir.cpp`: compiler, optimizer, and NVVM renderer
- `regex_ir_unicode.hpp`: attributed Unicode predicate tables
- `nvvm/`: sectioned matcher and executor NVVM templates embedded at build time with RTCX

The implementation includes its private header directly; no implementation
macro changes the declarations exposed by `regex_ir.hpp`.

The private libcudf integration, kernel wrappers, and compiled layout adapter
live in `cpp/src/strings/experimental/regex_jit`.
