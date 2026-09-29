# Dependencies and distribution

This skill directory contains NVIDIA-authored instructions, evaluation prompts, configuration and an evaluation report. It does not bundle third-party source files, libraries, binaries, containers, codecs or downloaded packages.

The native evaluations reported in BENCHMARK.md used the separately installed components below. Their test programs and comparison checker are not distributed in this package. The instructions guide users to build against their own installed cuDF environment.

| Component | Use in the evaluation | Upstream terms |
| --- | --- | --- |
| cuDF / libcudf | Native column operations in the separately maintained tests | [Apache-2.0](https://github.com/NVIDIA/cudf/blob/main/LICENSE) |
| RMM | Device storage and CUDA stream interfaces | [Apache-2.0](https://github.com/rapidsai/rmm/blob/main/LICENSE) |
| CUDA runtime and toolkit | GPU copies, streams, events and build headers | [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) |
| CMake | Configures and builds the native test program | [BSD-3-Clause and accompanying notices](https://cmake.org/licensing/) |
| Python | Runs the separately maintained comparison checker | [PSF license and accompanying notices](https://docs.python.org/3/license.html) |

A C++20 compiler and its standard library are also required. Their terms depend on the installed toolchain. The September 2026 evaluation used GCC and libstdc++ supplied by the Linux test environment. These tools and their dependencies are not redistributed in this skill directory.

The pandas-only negative prompt asks the agent to return an expression. It does not execute or redistribute pandas. Links to public documentation do not bundle that documentation.

The license files in this directory apply to the NVIDIA-authored skill content. They do not relicense separately installed dependencies. Keep the dependency inventory current when adding code or evaluation tools.
