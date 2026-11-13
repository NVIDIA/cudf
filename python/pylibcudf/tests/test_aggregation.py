# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pylibcudf as plc


def test_repr_name():
    assert repr(plc.aggregation.any()) == "<Aggregation(<Kind.ANY: 7>)>"


def test_nan_semantics_equality():
    unordered = plc.types.NanSemantics.UNORDERED
    ordered = plc.types.NanSemantics.TOTALLY_ORDERED
    for make in (
        plc.aggregation.min,
        plc.aggregation.max,
        plc.aggregation.argmin,
        plc.aggregation.argmax,
    ):
        # DEFAULT is currently UNORDERED
        assert make() == make(unordered)
        assert hash(make()) == hash(make(unordered))
        assert make(ordered) == make(semantics=ordered)
        assert make() != make(ordered)
