# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

pytest.importorskip("numba_cuda_mlir")


def test_mlir_backend_package_importable():
    import cudf.core.udf.mlir_backend as m

    # Module is intentionally empty at this layer, only the docstring
    # is present.
    assert m.__doc__
