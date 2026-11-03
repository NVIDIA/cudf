# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from numba import types
from numba.cuda import config as numba_config

from cudf.core.udf import nrt_utils
from cudf.core.udf.utils import _udf_code_cache, compile_udf


@pytest.fixture(autouse=True)
def clear_udf_cache():
    _udf_code_cache.clear()


def assert_cache_size(size):
    assert _udf_code_cache.currsize == size


def test_first_compile_sets_cache_entry():
    # The first compilation should put an entry in the cache
    compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)


def test_code_cache_same_code_different_function_hit():
    # Compilation of a distinct function with the same code and signature
    # should reuse the cached entry

    compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)

    compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)


def test_code_cache_different_types_miss():
    # Compilation of a distinct function with the same code but different types
    # should create an additional cache entry

    compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)

    compile_udf(lambda x: x + 1, (types.float64,))
    assert_cache_size(2)


def test_code_cache_different_cvars_miss():
    # Compilation of a distinct function with the same types and code as an
    # existing entry but different closure variables should create an
    # additional cache entry

    def gen_closure(y):
        return lambda x: x + y

    compile_udf(gen_closure(1), (types.float32,))
    assert_cache_size(1)

    compile_udf(gen_closure(2), (types.float32,))
    assert_cache_size(2)


def test_lambda_in_loop_code_cached():
    # Compiling a UDF defined in a loop should result in the code cache being
    # reused for each loop iteration after the first. We check for this by
    # ensuring that there is only one entry in the code cache after the loop.

    for i in range(3):
        compile_udf(lambda x: x + 1, (types.float32,))

    assert_cache_size(1)


class FakeModuleSpec:
    def __init__(self, submodule_search_locations):
        self.submodule_search_locations = submodule_search_locations


def _set_libcudf_spec(monkeypatch, spec):
    nrt_utils._get_libcudf_rapids_include_dir.cache_clear()
    monkeypatch.setattr(nrt_utils, "find_spec", lambda name: spec)


def test_get_libcudf_rapids_include_dir_missing(monkeypatch, tmp_path):
    _set_libcudf_spec(monkeypatch, None)
    assert nrt_utils._get_libcudf_rapids_include_dir() is None

    _set_libcudf_spec(monkeypatch, FakeModuleSpec([str(tmp_path)]))
    assert nrt_utils._get_libcudf_rapids_include_dir() is None


def test_get_libcudf_rapids_include_dir_finds_cuda_atomic(
    monkeypatch, tmp_path
):
    include_dir = tmp_path / "include" / "rapids"
    (include_dir / "cuda").mkdir(parents=True)
    (include_dir / "cuda" / "atomic").touch()
    _set_libcudf_spec(monkeypatch, FakeModuleSpec([str(tmp_path)]))

    assert nrt_utils._get_libcudf_rapids_include_dir() == str(include_dir)


def test_nrt_enabled_adds_libcudf_rapids_include_dir(monkeypatch, tmp_path):
    include_dir = tmp_path / "include" / "rapids"
    (include_dir / "cuda").mkdir(parents=True)
    (include_dir / "cuda" / "atomic").touch()
    _set_libcudf_spec(monkeypatch, FakeModuleSpec([str(tmp_path)]))
    monkeypatch.setattr(numba_config, "CUDA_ENABLE_NRT", False)
    monkeypatch.setattr(
        numba_config, "CUDA_NVRTC_EXTRA_SEARCH_PATHS", "/existing"
    )

    with nrt_utils.nrt_enabled():
        assert numba_config.CUDA_ENABLE_NRT is True
        assert numba_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS.split(
            os.pathsep
        ) == ["/existing", str(include_dir)]

    assert numba_config.CUDA_ENABLE_NRT is False
    assert numba_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS == "/existing"


def test_nrt_enabled_does_not_duplicate_libcudf_rapids_include_dir(
    monkeypatch, tmp_path
):
    include_dir = tmp_path / "include" / "rapids"
    (include_dir / "cuda").mkdir(parents=True)
    (include_dir / "cuda" / "atomic").touch()
    _set_libcudf_spec(monkeypatch, FakeModuleSpec([str(tmp_path)]))
    monkeypatch.setattr(numba_config, "CUDA_ENABLE_NRT", False)
    monkeypatch.setattr(
        numba_config,
        "CUDA_NVRTC_EXTRA_SEARCH_PATHS",
        os.pathsep.join(["/existing", str(include_dir)]),
    )

    with nrt_utils.nrt_enabled():
        assert numba_config.CUDA_ENABLE_NRT is True
        assert numba_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS.split(
            os.pathsep
        ) == ["/existing", str(include_dir)]

    assert numba_config.CUDA_ENABLE_NRT is False
    assert numba_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS == os.pathsep.join(
        ["/existing", str(include_dir)]
    )


def test_nrt_enabled_restores_config_when_include_discovery_raises(
    monkeypatch,
):
    def raise_error():
        raise RuntimeError("include discovery failed")

    monkeypatch.setattr(
        nrt_utils, "_get_libcudf_rapids_include_dir", raise_error
    )
    monkeypatch.setattr(numba_config, "CUDA_ENABLE_NRT", False)
    monkeypatch.setattr(
        numba_config, "CUDA_NVRTC_EXTRA_SEARCH_PATHS", "/existing"
    )

    with pytest.raises(RuntimeError, match="include discovery failed"):
        with nrt_utils.nrt_enabled():
            pass

    assert numba_config.CUDA_ENABLE_NRT is False
    assert numba_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS == "/existing"
