# libcudf_streaming

libcudf_streaming is the C++ shared library wheel for cuDF Streaming, providing GPU-accelerated streaming data processing capabilities built on top of libcudf and librapidsmpf.

Most users should not need to install this package directly; it is automatically pulled in as a dependency of `cudf-streaming`.

This package provides `libcudf_streaming.so` and a Python `load_library()` helper to ensure the shared library and its dependencies are properly loaded at runtime.

## Installation

```bash
pip install libcudf-streaming-cu12  # For CUDA 12
pip install libcudf-streaming-cu13  # For CUDA 13
```

## Usage

```python
import libcudf_streaming

# Load the shared library and all dependencies
libcudf_streaming.load_library()
```
