# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Starter pipeline with an intentionally unsafe join repair.

Keep ``prepare_orders`` compatible with pandas for the CPU smoke reference. The
same operations should run on cuDF DataFrames in the production path.
"""


def prepare_orders(orders, customers):
    """Attach customer segments at one row per order."""
    joined = orders.merge(customers, on="customer_id", how="left")

    # BUG: this hides right-side duplicate keys and arbitrarily retains a segment.
    joined = joined.drop_duplicates(subset=["order_id"], keep="first")
    return joined
