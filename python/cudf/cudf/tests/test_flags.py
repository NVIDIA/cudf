# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.fixture(
    params=[(cudf.Series, pd.Series), (cudf.DataFrame, pd.DataFrame)]
)
def frame_or_series(request):
    return request.param


def _make(cu_cls, pd_cls, data=None):
    if data is None:
        data = [1, 2, 3]
    if cu_cls is cudf.Series:
        return cu_cls(data), pd_cls(data)
    return cu_cls({"a": data}), pd_cls({"a": data})


def _assert_flags_eq(cu_obj, pd_obj):
    assert (
        cu_obj.flags.allows_duplicate_labels
        == pd_obj.flags.allows_duplicate_labels
    )


def test_flags_default_allows_duplicates(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    _assert_flags_eq(cu, pd_obj)
    assert_eq(cu, pd_obj)


def test_flags_setter_attribute(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu.flags.allows_duplicate_labels = False
    pd_obj.flags.allows_duplicate_labels = False
    _assert_flags_eq(cu, pd_obj)


def test_flags_getitem_and_setitem(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    assert (
        cu.flags["allows_duplicate_labels"]
        == pd_obj.flags["allows_duplicate_labels"]
    )
    cu.flags["allows_duplicate_labels"] = False
    pd_obj.flags["allows_duplicate_labels"] = False
    _assert_flags_eq(cu, pd_obj)


def test_flags_unknown_key_raises_getitem(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    assert_exceptions_equal(
        lfunc=pd_obj.flags.__getitem__,
        rfunc=cu.flags.__getitem__,
        lfunc_args_and_kwargs=(["nonexistent"],),
        rfunc_args_and_kwargs=(["nonexistent"],),
    )


def test_flags_unknown_key_raises_setitem(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    assert_exceptions_equal(
        lfunc=pd_obj.flags.__setitem__,
        rfunc=cu.flags.__setitem__,
        lfunc_args_and_kwargs=(["nonexistent", True],),
        rfunc_args_and_kwargs=(["nonexistent", True],),
    )


def test_flags_repr(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    assert repr(cu.flags) == repr(pd_obj.flags)
    cu.flags.allows_duplicate_labels = False
    pd_obj.flags.allows_duplicate_labels = False
    assert repr(cu.flags) == repr(pd_obj.flags)


def test_flags_equality(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu_a, pd_a = _make(cu_cls, pd_cls)
    cu_b, pd_b = _make(cu_cls, pd_cls)
    assert (cu_a.flags == cu_b.flags) == (pd_a.flags == pd_b.flags)
    cu_b.flags.allows_duplicate_labels = False
    pd_b.flags.allows_duplicate_labels = False
    assert (cu_a.flags == cu_b.flags) == (pd_a.flags == pd_b.flags)
    assert (cu_a.flags == object()) == (pd_a.flags == object())


def test_flags_setter_casts_to_bool(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu.flags.allows_duplicate_labels = 0
    pd_obj.flags.allows_duplicate_labels = 0
    _assert_flags_eq(cu, pd_obj)
    cu.flags.allows_duplicate_labels = 1
    pd_obj.flags.allows_duplicate_labels = 1
    _assert_flags_eq(cu, pd_obj)


def test_flags_rejects_duplicates_when_disabled():
    def _set_false(df):
        df.flags.allows_duplicate_labels = False

    pdf = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 1, 2])
    gdf = cudf.DataFrame({"a": [1, 2, 3]}, index=[1, 1, 2])
    assert_exceptions_equal(
        lfunc=_set_false,
        rfunc=_set_false,
        lfunc_args_and_kwargs=([pdf],),
        rfunc_args_and_kwargs=([gdf],),
    )


def test_flags_allows_non_unique_when_enabled():
    pdf = pd.DataFrame({"a": [1, 2]}, index=[1, 1])
    gdf = cudf.DataFrame({"a": [1, 2]}, index=[1, 1])
    pdf.flags.allows_duplicate_labels = True
    gdf.flags.allows_duplicate_labels = True
    _assert_flags_eq(gdf, pdf)


def test_flags_weakref_dead_obj_raises():
    # Pandas also raises ValueError once the referenced object is gone.
    def _setter(cls):
        obj = cls({"a": [1]}) if cls is cudf.DataFrame else cls([1])
        flags = obj.flags
        del obj
        flags.allows_duplicate_labels = False

    assert_exceptions_equal(
        lfunc=_setter,
        rfunc=_setter,
        lfunc_args_and_kwargs=([pd.DataFrame],),
        rfunc_args_and_kwargs=([cudf.DataFrame],),
    )


def test_set_flags_returns_new_object(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu_new = cu.set_flags(allows_duplicate_labels=False)
    pd_new = pd_obj.set_flags(allows_duplicate_labels=False)
    assert (cu_new is cu) == (pd_new is pd_obj)
    _assert_flags_eq(cu, pd_obj)
    _assert_flags_eq(cu_new, pd_new)
    assert_eq(cu_new, pd_new)


def test_set_flags_no_change_when_none(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu.flags.allows_duplicate_labels = False
    pd_obj.flags.allows_duplicate_labels = False
    cu_new = cu.set_flags()
    pd_new = pd_obj.set_flags()
    _assert_flags_eq(cu_new, pd_new)


@pytest.mark.filterwarnings(
    "ignore:The copy keyword is deprecated:pandas.errors.Pandas4Warning"
)
def test_set_flags_copy_true_makes_deep_copy():
    # Match pandas' observable behaviour: mutating the copy does not
    # alter the original. ``copy=True`` is deprecated in pandas 3; we
    # still accept it for API parity.
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    gdf = cudf.DataFrame({"a": [1, 2, 3]})
    p_new = pdf.set_flags(copy=True, allows_duplicate_labels=False)
    g_new = gdf.set_flags(copy=True, allows_duplicate_labels=False)
    p_new.loc[0, "a"] = 99
    g_new.loc[0, "a"] = 99
    assert_eq(pdf, gdf)
    assert_eq(p_new, g_new)


def test_copy_preserves_flags(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu = cu.set_flags(allows_duplicate_labels=False)
    pd_obj = pd_obj.set_flags(allows_duplicate_labels=False)
    _assert_flags_eq(cu.copy(), pd_obj.copy())
    _assert_flags_eq(cu.copy(deep=False), pd_obj.copy(deep=False))


@pytest.mark.parametrize("left_flag", [True, False])
@pytest.mark.parametrize("right_flag", [True, False])
def test_binop_flags_anded(frame_or_series, left_flag, right_flag):
    cu_cls, pd_cls = frame_or_series
    cu_l, pd_l = _make(cu_cls, pd_cls)
    cu_r, pd_r = _make(cu_cls, pd_cls)
    cu_l = cu_l.set_flags(allows_duplicate_labels=left_flag)
    cu_r = cu_r.set_flags(allows_duplicate_labels=right_flag)
    pd_l = pd_l.set_flags(allows_duplicate_labels=left_flag)
    pd_r = pd_r.set_flags(allows_duplicate_labels=right_flag)
    cu_out = cu_l + cu_r
    pd_out = pd_l + pd_r
    _assert_flags_eq(cu_out, pd_out)
    assert_eq(cu_out, pd_out)


def test_binop_with_scalar_preserves_flags(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu = cu.set_flags(allows_duplicate_labels=False)
    pd_obj = pd_obj.set_flags(allows_duplicate_labels=False)
    _assert_flags_eq(cu + 1, pd_obj + 1)
    _assert_flags_eq(1 + cu, 1 + pd_obj)
    assert_eq(cu + 1, pd_obj + 1)


def test_binop_attrs_right_overrides_left():
    a_cu = cudf.Series([1])
    b_cu = cudf.Series([1])
    a_pd = pd.Series([1])
    b_pd = pd.Series([1])
    a_cu.attrs = a_pd.attrs = {"a": 1}
    b_cu.attrs = b_pd.attrs = {"b": 2}
    assert (a_cu + b_cu).attrs == (a_pd + b_pd).attrs


def test_binop_attrs_inherit_from_non_empty_side():
    a_cu, a_pd = cudf.Series([1]), pd.Series([1])
    b_cu, b_pd = cudf.Series([1]), pd.Series([1])
    b_cu.attrs = b_pd.attrs = {"x": 1}
    assert (a_cu + b_cu).attrs == (a_pd + b_pd).attrs

    a_cu.attrs = a_pd.attrs = {"y": 2}
    b_cu.attrs = b_pd.attrs = {}
    assert (a_cu + b_cu).attrs == (a_pd + b_pd).attrs


@pytest.mark.parametrize("left_flag", [True, False])
@pytest.mark.parametrize("right_flag", [True, False])
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer", "cross"])
def test_merge_flags_anded(how, left_flag, right_flag):
    gdf_l = cudf.DataFrame({"test": [1]}).set_flags(
        allows_duplicate_labels=left_flag
    )
    gdf_r = cudf.DataFrame({"test": [1]}).set_flags(
        allows_duplicate_labels=right_flag
    )
    pdf_l = pd.DataFrame({"test": [1]}).set_flags(
        allows_duplicate_labels=left_flag
    )
    pdf_r = pd.DataFrame({"test": [1]}).set_flags(
        allows_duplicate_labels=right_flag
    )
    if how == "cross":
        cu_out = gdf_l.merge(gdf_r, how=how)
        pd_out = pdf_l.merge(pdf_r, how=how)
    else:
        cu_out = gdf_l.merge(gdf_r, how=how, on="test")
        pd_out = pdf_l.merge(pdf_r, how=how, on="test")
    _assert_flags_eq(cu_out, pd_out)
    assert_eq(cu_out, pd_out)


def test_merge_propagates_attrs_only_when_equal():
    pdf_l = pd.DataFrame({"test": [1]})
    pdf_r = pd.DataFrame({"test": [1]})
    pdf_l.attrs = pdf_r.attrs = {"a": [1, 2]}
    gdf_l = cudf.DataFrame({"test": [1]})
    gdf_r = cudf.DataFrame({"test": [1]})
    gdf_l.attrs = gdf_r.attrs = {"a": [1, 2]}
    assert (
        gdf_l.merge(gdf_r, how="inner", on="test").attrs
        == pdf_l.merge(pdf_r, how="inner", on="test").attrs
    )


def test_merge_drops_attrs_when_inputs_differ():
    pdf_l = pd.DataFrame({"test": [1]})
    pdf_r = pd.DataFrame({"test": [1]})
    pdf_l.attrs = {"a": 1}
    pdf_r.attrs = {"b": 2}
    gdf_l = cudf.DataFrame({"test": [1]})
    gdf_r = cudf.DataFrame({"test": [1]})
    gdf_l.attrs = {"a": 1}
    gdf_r.attrs = {"b": 2}
    assert (
        gdf_l.merge(gdf_r, how="inner", on="test").attrs
        == pdf_l.merge(pdf_r, how="inner", on="test").attrs
    )


def test_merge_drops_attrs_when_one_side_empty():
    pdf_l = pd.DataFrame({"test": [1]})
    pdf_r = pd.DataFrame({"test": [1]})
    pdf_l.attrs = {"a": 1}
    gdf_l = cudf.DataFrame({"test": [1]})
    gdf_r = cudf.DataFrame({"test": [1]})
    gdf_l.attrs = {"a": 1}
    assert (
        gdf_l.merge(gdf_r, how="inner", on="test").attrs
        == pdf_l.merge(pdf_r, how="inner", on="test").attrs
    )


def test_from_pandas_preserves_flags(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    if cu_cls is cudf.Series:
        src = pd_cls([1, 2]).set_flags(allows_duplicate_labels=False)
    else:
        src = pd_cls({"a": [1, 2]}).set_flags(allows_duplicate_labels=False)
    out = cudf.from_pandas(src)
    _assert_flags_eq(out, src)
    assert_eq(out, src)


def test_to_pandas_preserves_flags(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu = cu.set_flags(allows_duplicate_labels=False)
    pd_obj = pd_obj.set_flags(allows_duplicate_labels=False)
    assert (
        cu.to_pandas().flags.allows_duplicate_labels
        == pd_obj.flags.allows_duplicate_labels
    )
    assert_eq(cu.to_pandas(), pd_obj)


def test_roundtrip_through_pandas_preserves_flags(frame_or_series):
    cu_cls, pd_cls = frame_or_series
    cu, pd_obj = _make(cu_cls, pd_cls)
    cu = cu.set_flags(allows_duplicate_labels=False)
    pd_obj = pd_obj.set_flags(allows_duplicate_labels=False)
    _assert_flags_eq(cudf.from_pandas(cu.to_pandas()), pd_obj)


def test_dataframe_constructor_preserves_flags_from_pandas():
    pdf = pd.DataFrame({"a": [1, 2]}).set_flags(allows_duplicate_labels=False)
    gdf = cudf.DataFrame(pdf)
    _assert_flags_eq(gdf, pdf)
    assert_eq(gdf, pdf)


def test_series_constructor_preserves_flags_from_pandas():
    ps = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    gs = cudf.Series(ps)
    _assert_flags_eq(gs, ps)
    assert_eq(gs, ps)


def test_dataframe_constructor_preserves_flags_from_cudf():
    # Intentional divergence from pandas: cudf preserves flags when
    # constructing from another cudf/pandas frame so that the
    # ``cudf.pandas`` slow->fast round-trip keeps the flag. Pandas'
    # own constructor drops flags (tracked there by ``__finalize__`` on
    # the specific operation, not on the constructor).
    gdf = cudf.DataFrame({"a": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    assert cudf.DataFrame(gdf).flags.allows_duplicate_labels is False


def test_series_constructor_preserves_flags_from_cudf():
    # See note in ``test_dataframe_constructor_preserves_flags_from_cudf``.
    gs = cudf.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    assert cudf.Series(gs).flags.allows_duplicate_labels is False


def test_from_data_accepts_allows_duplicate_labels():
    # cudf-private code path (no pandas equivalent); just verify the
    # flag survives the constructor.
    df = cudf.DataFrame._from_data(
        cudf.DataFrame({"a": [1]})._data,
        allows_duplicate_labels=False,
    )
    assert df.flags.allows_duplicate_labels is False


def test_flags_class_is_exposed():
    # Used by users who want to type-hint or isinstance-check.
    assert isinstance(cudf.DataFrame().flags, pd.Flags)


def test_binop_attrs_deepcopied():
    # Mutating the result's attrs must not bleed back into the source,
    # matching pandas semantics.
    a_cu, a_pd = cudf.Series([1]), pd.Series([1])
    b_cu, b_pd = cudf.Series([2]), pd.Series([2])
    b_cu.attrs = b_pd.attrs = {"nested": [1, 2]}
    (a_cu + b_cu).attrs["nested"].append(3)
    (a_pd + b_pd).attrs["nested"].append(3)
    assert b_cu.attrs == b_pd.attrs


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
    ],
)
def test_binop_flag_propagation_with_operators(op):
    a_cu = cudf.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    b_cu = cudf.Series([3, 4])
    a_pd = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    b_pd = pd.Series([3, 4])
    cu_out = op(a_cu, b_cu)
    pd_out = op(a_pd, b_pd)
    _assert_flags_eq(cu_out, pd_out)
    assert_eq(cu_out, pd_out)


def test_series_rename_preserves_flags():
    ps = pd.Series([0, 1], index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    gs = cudf.Series([0, 1], index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    _assert_flags_eq(gs.rename("renamed"), ps.rename("renamed"))
    assert_eq(gs.rename("renamed"), ps.rename("renamed"))


def test_series_to_frame_preserves_flags():
    ps = pd.Series([1, 2], name="a").set_flags(allows_duplicate_labels=False)
    gs = cudf.Series([1, 2], name="a").set_flags(allows_duplicate_labels=False)
    _assert_flags_eq(gs.to_frame(), ps.to_frame())
    assert_eq(gs.to_frame(), ps.to_frame())


def test_dataframe_getitem_preserves_flags():
    pdf = pd.DataFrame({"A": [1, 2], "B": [3, 4]}).set_flags(
        allows_duplicate_labels=False
    )
    gdf = cudf.DataFrame({"A": [1, 2], "B": [3, 4]}).set_flags(
        allows_duplicate_labels=False
    )
    _assert_flags_eq(gdf[["A"]], pdf[["A"]])
    _assert_flags_eq(gdf["A"], pdf["A"])
    assert_eq(gdf[["A"]], pdf[["A"]])
    assert_eq(gdf["A"], pdf["A"])


def test_dataframe_loc_preserves_flags():
    pdf = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["x", "y"]).set_flags(
        allows_duplicate_labels=False
    )
    gdf = cudf.DataFrame(
        {"A": [1, 2], "B": [3, 4]}, index=["x", "y"]
    ).set_flags(allows_duplicate_labels=False)
    _assert_flags_eq(gdf.loc["x"], pdf.loc["x"])
    _assert_flags_eq(gdf.loc[["x"]], pdf.loc[["x"]])
    _assert_flags_eq(gdf.loc["x", ["A"]], pdf.loc["x", ["A"]])
    assert_eq(gdf.loc["x"], pdf.loc["x"])
    assert_eq(gdf.loc[["x"]], pdf.loc[["x"]])
    assert_eq(gdf.loc["x", ["A"]], pdf.loc["x", ["A"]])


def test_dataframe_iloc_preserves_flags():
    pdf = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
    gdf = cudf.DataFrame({"A": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    _assert_flags_eq(gdf.iloc[[0]], pdf.iloc[[0]])
    assert_eq(gdf.iloc[[0]], pdf.iloc[[0]])


def test_loc_raises_duplicate_label_error():
    pdf = pd.DataFrame({"A": [1, 2]}, index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    gdf = cudf.DataFrame({"A": [1, 2]}, index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    assert_exceptions_equal(
        lfunc=pdf.loc.__getitem__,
        rfunc=gdf.loc.__getitem__,
        lfunc_args_and_kwargs=([["a", "a"]],),
        rfunc_args_and_kwargs=([["a", "a"]],),
    )


def test_iloc_raises_duplicate_label_error():
    pdf = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
    gdf = cudf.DataFrame({"A": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    assert_exceptions_equal(
        lfunc=pdf.iloc.__getitem__,
        rfunc=gdf.iloc.__getitem__,
        lfunc_args_and_kwargs=([[0, 0]],),
        rfunc_args_and_kwargs=([[0, 0]],),
    )


@pytest.mark.parametrize("left_flag", [True, False])
@pytest.mark.parametrize("right_flag", [True, False])
def test_concat_flags_anded(frame_or_series, left_flag, right_flag):
    cu_cls, _pd_cls = frame_or_series
    if cu_cls is cudf.Series:
        cu_a = cudf.Series([1], index=["a"]).set_flags(
            allows_duplicate_labels=left_flag
        )
        cu_b = cudf.Series([2], index=["b"]).set_flags(
            allows_duplicate_labels=right_flag
        )
        pd_a = pd.Series([1], index=["a"]).set_flags(
            allows_duplicate_labels=left_flag
        )
        pd_b = pd.Series([2], index=["b"]).set_flags(
            allows_duplicate_labels=right_flag
        )
    else:
        cu_a = cudf.DataFrame({"x": [1]}, index=["a"]).set_flags(
            allows_duplicate_labels=left_flag
        )
        cu_b = cudf.DataFrame({"x": [2]}, index=["b"]).set_flags(
            allows_duplicate_labels=right_flag
        )
        pd_a = pd.DataFrame({"x": [1]}, index=["a"]).set_flags(
            allows_duplicate_labels=left_flag
        )
        pd_b = pd.DataFrame({"x": [2]}, index=["b"]).set_flags(
            allows_duplicate_labels=right_flag
        )
    cu_out = cudf.concat([cu_a, cu_b])
    pd_out = pd.concat([pd_a, pd_b])
    _assert_flags_eq(cu_out, pd_out)
    assert_eq(cu_out, pd_out)


def test_concat_raises_on_duplicate_labels():
    pd_objs = [
        pd.Series([1], index=["a"]).set_flags(allows_duplicate_labels=False),
        pd.Series([2], index=["a"]).set_flags(allows_duplicate_labels=False),
    ]
    cu_objs = [
        cudf.Series([1], index=["a"]).set_flags(allows_duplicate_labels=False),
        cudf.Series([2], index=["a"]).set_flags(allows_duplicate_labels=False),
    ]
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([pd_objs],),
        rfunc_args_and_kwargs=([cu_objs],),
    )


def test_concat_propagates_attrs_only_when_equal():
    pdf_a = pd.DataFrame({"x": [1]})
    pdf_b = pd.DataFrame({"x": [2]})
    gdf_a = cudf.DataFrame({"x": [1]})
    gdf_b = cudf.DataFrame({"x": [2]})
    pdf_a.attrs = pdf_b.attrs = {"k": 1}
    gdf_a.attrs = gdf_b.attrs = {"k": 1}
    assert cudf.concat([gdf_a, gdf_b]).attrs == pd.concat([pdf_a, pdf_b]).attrs
    pdf_b.attrs = {"k": 2}
    gdf_b.attrs = {"k": 2}
    assert cudf.concat([gdf_a, gdf_b]).attrs == pd.concat([pdf_a, pdf_b]).attrs


def test_set_index_inplace_raises_when_flags_false():
    pdf = pd.DataFrame({"A": [0, 1], "B": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    gdf = cudf.DataFrame({"A": [0, 1], "B": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    assert_exceptions_equal(
        lfunc=pdf.set_index,
        rfunc=gdf.set_index,
        lfunc_args_and_kwargs=(["A"], {"inplace": True}),
        rfunc_args_and_kwargs=(["A"], {"inplace": True}),
    )


def test_reset_index_inplace_raises_when_flags_false():
    pdf = pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    gdf = cudf.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    assert_exceptions_equal(
        lfunc=pdf.reset_index,
        rfunc=gdf.reset_index,
        lfunc_args_and_kwargs=([], {"inplace": True}),
        rfunc_args_and_kwargs=([], {"inplace": True}),
    )


def test_set_index_inplace_ok_when_flags_true():
    pdf = pd.DataFrame({"A": [0, 1], "B": [1, 2]})
    gdf = cudf.DataFrame({"A": [0, 1], "B": [1, 2]})
    pdf.set_index("A", inplace=True)
    gdf.set_index("A", inplace=True)
    assert_eq(gdf, pdf)


def test_reset_index_preserves_flags():
    pdf = pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    gdf = cudf.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
        allows_duplicate_labels=False
    )
    _assert_flags_eq(gdf.reset_index(), pdf.reset_index())
    assert_eq(gdf.reset_index(), pdf.reset_index())
