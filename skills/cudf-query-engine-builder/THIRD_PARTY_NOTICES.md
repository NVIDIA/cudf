# Dependencies and distribution

This directory contains NVIDIA-authored skill instructions, C++ evaluation starters and tests, Python check runners, evaluation configuration and pipeline-generated reports. It does not bundle third-party source files, libraries, binaries, container images, codecs or downloaded packages.

## CPU evaluation environment

The two C++ fixtures simulate queued backend work on the CPU. Each run_checks.py uses only Python's standard library, including Linux/POSIX process and resource controls, to compile the submitted adapter with the image's system g++ or clang++ and execute the tests. The supplied Dockerfile installs g++. The Dockerfile fetches the official python:3.12-slim image and installs g++ with the image's package manager when the evaluation environment is built. The downloaded image and packages are separate runtime dependencies; they are not files in this skill directory.

| Component | Use | Upstream terms |
| --- | --- | --- |
| Python | Runs the shipped compile-and-test scripts | [PSF license and accompanying notices](https://docs.python.org/3/license.html) |
| GCC / g++ | Compiles the C++20 test programs | [GPLv3 license text](https://gcc.gnu.org/onlinedocs/gcc/Copying.html) |
| libstdc++ | C++ standard library used by the compiled tests | [GPLv3 with GCC Runtime Library Exception 3.1](https://gcc.gnu.org/onlinedocs/libstdc++/manual/license.html) |
| Official Python slim image | Provides Python and the Debian-based build environment | [Image source and license information](https://hub.docker.com/_/python); included packages retain their own terms and notices |

The image contains multiple independently licensed components. Its published source and each installed package's copyright files supply the applicable notices. No single license is asserted for the complete image.

## Separately installed native environment

The skill directs users to build against their own cuDF environment. Native GPU evaluation programs and their comparison checkers are maintained separately from this CPU evaluation dataset.

| Component | Use | Upstream terms |
| --- | --- | --- |
| cuDF / libcudf | Native column operations | [Apache-2.0](https://github.com/NVIDIA/cudf/blob/main/LICENSE) |
| RMM | Device storage and CUDA stream interfaces | [Apache-2.0](https://github.com/rapidsai/rmm/blob/main/LICENSE) |
| CUDA runtime and toolkit | GPU copies, streams, events and build headers | [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) |
| CMake | Configures and builds native test programs | [BSD-3-Clause and accompanying notices](https://cmake.org/licensing/) |

A C++20 compiler and its standard library are required for native builds too. Their terms depend on the installed toolchain. These tools and their dependencies are not redistributed in this skill directory.

The pandas-only negative prompt asks the agent to return an expression. It does not execute or redistribute pandas. Links to public documentation do not bundle that documentation.

The license files in this directory apply to the NVIDIA-authored skill content. They do not relicense separately installed dependencies. Keep the dependency inventory current when adding code or evaluation tools.
