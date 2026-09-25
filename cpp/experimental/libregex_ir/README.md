# Regex IR

Regex IR is the C++20 regex compiler and code generator vendored by cuDF.
`regex_ir::compile` parses a regular expression and emits operation-specific
NVVM IR. The public `regex_ir::nvvm` helpers construct and assemble the cuDF
kernel modules around that matcher. All intermediate representations and
compiler passes remain private implementation details.

This directory intentionally contains only the production sources and design
notes:

- `regex_ir.hpp`: public compiler and NVVM module-construction API
- `regex_ir.cpp`: compiler, embedded Unicode data, and cuDF kernel-module construction
- `optimization.md`: optimization design notes for agents and developers
