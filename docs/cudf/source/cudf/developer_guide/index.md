# cudf Developer Guide

```{note}
This document assumes familiarity with the
[overall contributing guide](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md).
```

cuDF is a GPU-accelerated, [pandas-like](https://pandas.pydata.org/) DataFrame library.
Under the hood, all of cuDF's functionality relies on the CUDA-accelerated `libcudf` C++ library.
Thus, cuDF's internals are designed to efficiently and robustly map pandas APIs to `libcudf` functions.
The goal of this guide is to provide specific guidance for cuDF Python developers.
It covers the structure of the Python code and discusses best practices.

```{toctree}
:maxdepth: 2

library_design
contributing_guide
documentation
testing
benchmarking
udf_memory_management
```
