# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for versions_compat.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "versions_compat", Path(__file__).parent / "versions_compat.py"
)
versions_compat = importlib.util.module_from_spec(_SPEC)
sys.modules["versions_compat"] = versions_compat
_SPEC.loader.exec_module(versions_compat)


@pytest.fixture
def pyproject(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\ndependencies = ["polars>=1.38,<1.43"]\n')
    return path


def test_minimum_polars_version(pyproject: Path) -> None:
    assert versions_compat.minimum_polars_version(pyproject) == (
        versions_compat.Version("1.38")
    )


def test_minimum_polars_version_missing_dependency(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\ndependencies = ["numpy>=1.26"]\n')
    with pytest.raises(ValueError, match="No 'polars' dependency"):
        versions_compat.minimum_polars_version(path)


def test_minimum_polars_version_no_lower_bound(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\ndependencies = ["polars<1.43"]\n')
    with pytest.raises(ValueError, match="has no '>='"):
        versions_compat.minimum_polars_version(path)


@pytest.mark.parametrize("operator", ["==", "~=", ">"])
def test_minimum_polars_version_accepts_other_floor_operators(
    tmp_path: Path, operator: str
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[project]\ndependencies = ["polars{operator}1.40"]\n')
    assert versions_compat.minimum_polars_version(path) == (
        versions_compat.Version("1.40")
    )


def test_find_stale_flags_none_stale(tmp_path: Path) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_140 = POLARS_VERSION < parse("1.40.0")\n'
    )
    stale = versions_compat.find_stale_flags(
        versions_py, versions_compat.Version("1.38")
    )
    assert stale == []


def test_find_stale_flags_detects_stale_and_fresh(tmp_path: Path) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_136 = POLARS_VERSION < parse("1.36.0")\n'
        'POLARS_VERSION_LT_138 = POLARS_VERSION < parse("1.38.0")\n'
        'POLARS_VERSION_LT_140 = POLARS_VERSION < parse("1.40.0")\n'
    )
    stale = versions_compat.find_stale_flags(
        versions_py, versions_compat.Version("1.38")
    )
    assert [flag.name for flag in stale] == [
        "POLARS_VERSION_LT_136",
        "POLARS_VERSION_LT_138",
    ]
    assert [flag.lineno for flag in stale] == [1, 2]


def test_find_stale_flags_equal_to_minimum_is_stale(tmp_path: Path) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_138 = POLARS_VERSION < parse("1.38")\n'
    )
    stale = versions_compat.find_stale_flags(
        versions_py, versions_compat.Version("1.38")
    )
    assert len(stale) == 1


def test_find_stale_flags_detects_annotated_stale_flag(
    tmp_path: Path,
) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_138: bool = POLARS_VERSION < parse("1.38")\n'
    )
    stale = versions_compat.find_stale_flags(
        versions_py, versions_compat.Version("1.38")
    )
    assert [flag.name for flag in stale] == ["POLARS_VERSION_LT_138"]


def test_find_stale_flags_raises_on_unparseable_annotated_flag(
    tmp_path: Path,
) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_136: bool = parse("1.36.0") > POLARS_VERSION\n'
    )
    with pytest.raises(
        versions_compat.UnparseableFlagError,
        match="POLARS_VERSION_LT_136",
    ):
        versions_compat.find_stale_flags(
            versions_py, versions_compat.Version("1.38")
        )


def test_find_stale_flags_ignores_unrelated_content(tmp_path: Path) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_LOWER_BOUND = parse("1.35")\n'
        "POLARS_VERSION = parse(__version__)\n"
        'POLARS_VERSION_LT_140 = POLARS_VERSION < parse("1.40.0")\n'
        "\n"
        "def _ensure_polars_version() -> None:\n"
        "    if POLARS_VERSION < POLARS_LOWER_BOUND:\n"
        "        raise ImportError\n"
    )
    stale = versions_compat.find_stale_flags(
        versions_py, versions_compat.Version("1.38")
    )
    assert stale == []


def test_find_stale_flags_raises_on_unrecognized_flag_form(
    tmp_path: Path,
) -> None:
    versions_py = tmp_path / "versions.py"
    # A flag in a form FLAG_PATTERN doesn't parse (yoda comparison) must
    # fail loudly rather than be silently treated as non-stale.
    versions_py.write_text(
        'POLARS_VERSION_LT_136 = parse("1.36.0") > POLARS_VERSION\n'
    )
    with pytest.raises(
        versions_compat.UnparseableFlagError,
        match="POLARS_VERSION_LT_136",
    ):
        versions_compat.find_stale_flags(
            versions_py, versions_compat.Version("1.38")
        )


def test_find_stale_flags_raises_on_indented_flag(tmp_path: Path) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        "if True:\n"
        '    POLARS_VERSION_LT_136 = POLARS_VERSION < parse("1.36.0")\n'
    )
    # Leading whitespace is allowed by both patterns, so this one is
    # still recognized (not raised) and correctly flagged as stale.
    stale = versions_compat.find_stale_flags(
        versions_py, versions_compat.Version("1.38")
    )
    assert [flag.name for flag in stale] == ["POLARS_VERSION_LT_136"]


def test_main_clean_exits_zero(pyproject: Path, tmp_path: Path) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_140 = POLARS_VERSION < parse("1.40.0")\n'
    )
    assert (
        versions_compat.main(
            [
                "--pyproject",
                str(pyproject),
                "--versions-file",
                str(versions_py),
            ]
        )
        == 0
    )


def test_main_stale_exits_one(
    pyproject: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_136 = POLARS_VERSION < parse("1.36.0")\n'
    )
    assert (
        versions_compat.main(
            [
                "--pyproject",
                str(pyproject),
                "--versions-file",
                str(versions_py),
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "POLARS_VERSION_LT_136" in out
    assert f"{versions_py}:1" in out


def test_main_unparseable_flag_exits_one_with_error(
    pyproject: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_136 = parse("1.36.0") > POLARS_VERSION\n'
    )
    assert (
        versions_compat.main(
            [
                "--pyproject",
                str(pyproject),
                "--versions-file",
                str(versions_py),
            ]
        )
        == 1
    )
    assert "POLARS_VERSION_LT_136" in capsys.readouterr().err


def test_main_missing_file_exits_one_with_error(
    pyproject: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist.py"
    assert (
        versions_compat.main(
            [
                "--pyproject",
                str(pyproject),
                "--versions-file",
                str(missing),
            ]
        )
        == 1
    )
    assert "Error:" in capsys.readouterr().err


def test_main_missing_project_table_exits_one_with_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[build-system]\nrequires = ["setuptools"]\n')
    versions_py = tmp_path / "versions.py"
    versions_py.write_text(
        'POLARS_VERSION_LT_140 = POLARS_VERSION < parse("1.40.0")\n'
    )
    assert (
        versions_compat.main(
            [
                "--pyproject",
                str(pyproject),
                "--versions-file",
                str(versions_py),
            ]
        )
        == 1
    )
    assert "Error:" in capsys.readouterr().err


def test_main_defaults_to_repo_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no args, main() checks the real repo paths and passes today."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    assert versions_compat.main([]) == 0


def test_flag_pattern_matches_real_versions_file() -> None:
    """
    Regression guard for FLAG_PATTERN itself.

    find_stale_flags() alone can't tell "no stale flags" apart from
    "the regex silently matched nothing", so assert directly that it
    still recognizes every POLARS_VERSION_LT_* assignment that exists
    in the real file today.
    """
    repo_root = Path(__file__).resolve().parents[2]
    versions_py = (
        repo_root / "python/cudf_polars/cudf_polars/utils/versions.py"
    )
    content = versions_py.read_text()
    all_names = {
        m.group("name")
        for m in versions_compat.ASSIGNMENT_PATTERN.finditer(content)
    }
    matched_names = {
        m.group("name") for m in versions_compat.FLAG_PATTERN.finditer(content)
    }
    assert all_names == matched_names
    assert len(all_names) >= 1


def test_current_repo_state_is_clean() -> None:
    """Regression guard: today's cudf_polars versions.py has no stale flags."""
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = repo_root / "python/cudf_polars/pyproject.toml"
    versions_py = (
        repo_root / "python/cudf_polars/cudf_polars/utils/versions.py"
    )
    minimum = versions_compat.minimum_polars_version(pyproject)
    assert versions_compat.find_stale_flags(versions_py, minimum) == []
