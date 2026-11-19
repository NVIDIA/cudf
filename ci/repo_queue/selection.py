# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound explicit tracking selections for both local and scheduled runs."""

MAX_TRACKED_SOURCES = 50


def validate_numbers(numbers):
    if (
        not isinstance(numbers, list)
        or not 1 <= len(numbers) <= MAX_TRACKED_SOURCES
        or any(type(number) is not int or number < 1 for number in numbers)
        or len(set(numbers)) != len(numbers)
    ):
        raise ValueError(
            f"select 1–{MAX_TRACKED_SOURCES} distinct positive issue/PR numbers"
        )
