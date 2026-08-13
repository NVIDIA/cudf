# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import doctest
import inspect
import io

import numpy as np
import pytest

import cudf

# modules that will be searched for doctests
tests = [
    cudf,
    cudf.core.groupby,
    cudf.core.accessors.string.StringMethods,
    cudf.core.accessors.lists.ListMethods,
    cudf.core.accessors.struct.StructMethods,
    cudf.core.accessors.categorical.CategoricalAccessor,
]


def _name_in_all(parent, name):
    return name in getattr(parent, "__all__", [])


def _is_public_name(parent, name):
    return not name.startswith("_")


def _find_doctests_in_obj(obj, finder=None, criteria=None):
    """Find all doctests in an object.

    Parameters
    ----------
    obj : module or class
        The object to search for docstring examples.
    criteria : callable, optional
        Callable indicating whether to recurse over members of the provided
        object. If not provided, names not defined in the object's ``__all__``
        property are ignored.

    Yields
    ------
    doctest.DocTest
        The next doctest found in the object.
    """
    if finder is None:
        finder = doctest.DocTestFinder()
    if criteria is None:
        criteria = _name_in_all
    for docstring in finder.find(obj):
        if docstring.examples:
            yield docstring
    for name, member in inspect.getmembers(obj):
        # Only recurse over members matching the criteria
        if not criteria(obj, name):
            continue
        # Recurse over the public API of modules (objects defined in the
        # module's __all__)
        if inspect.ismodule(member):
            yield from _find_doctests_in_obj(
                member, finder, criteria=_name_in_all
            )
        # Recurse over the public API of classes (attributes not prefixed with
        # an underscore)
        elif inspect.isclass(member):
            yield from _find_doctests_in_obj(
                member, finder, criteria=_is_public_name
            )
        # Pick up doctests from re-exported functions (whose __module__
        # differs from the parent module, so finder.find(obj) above
        # skips them).
        elif inspect.isfunction(member) or inspect.isbuiltin(member):
            for docstring in finder.find(member):
                if docstring.examples:
                    yield docstring


marks_for_doctests = {
    "register_dataframe_accessor": pytest.mark.filterwarnings(
        "ignore:Attribute .* will be overridden:UserWarning"
    ),
    "register_index_accessor": pytest.mark.filterwarnings(
        "ignore:Attribute .* will be overridden:UserWarning"
    ),
    "register_series_accessor": pytest.mark.filterwarnings(
        "ignore:Attribute .* will be overridden:UserWarning"
    ),
}


def _collect_doctests():
    """Eagerly collect all unique doctests into a list.

    This runs once at import time so the result is an immutable snapshot
    that can be safely shared across pytest-xdist workers without any
    mutable shared state.
    """
    results = []
    for mod in tests:
        for docstring in _find_doctests_in_obj(mod):
            if (mark := marks_for_doctests.get(docstring.name)) is not None:
                doc = pytest.param(docstring, marks=mark)
            else:
                doc = docstring
            results.append(doc)
    return results


_all_doctests = _collect_doctests()


class TestDoctests:
    @pytest.mark.parametrize(
        "docstring",
        _all_doctests,
        ids=lambda docstring: docstring.name,
    )
    def test_docstring(self, docstring, monkeypatch, tmp_path):
        # We ignore differences in whitespace in the doctest output, and enable
        # the use of an ellipsis "..." to match any string in the doctest
        # output. An ellipsis is useful for, e.g., memory addresses or
        # imprecise floating point values.
        monkeypatch.chdir(tmp_path)
        optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        runner = doctest.DocTestRunner(optionflags=optionflags)

        # These global names are pre-defined and can be used in doctests
        # without first importing them.
        globals = dict(
            cudf=cudf,
            np=np,
        )
        docstring.globs = globals

        # Capture stdout and include failing outputs in the traceback.
        doctest_stdout = io.StringIO()
        with contextlib.redirect_stdout(doctest_stdout):
            runner.run(docstring)
            results = runner.summarize()
        assert not results.failed, (
            f"{results.failed} of {results.attempted} doctests failed for "
            f"{docstring.name}:\n{doctest_stdout.getvalue()}"
        )
