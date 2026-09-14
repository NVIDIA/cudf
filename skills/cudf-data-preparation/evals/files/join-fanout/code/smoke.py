# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CPU semantic smoke test for the backend-compatible preparation function."""

import pandas as pd

from prepare_orders import prepare_orders


def frames():
    orders = pd.DataFrame(
        {
            "order_id": [101, 102, 103],
            "customer_id": [1, 2, 9],
            "amount": [10.0, 20.0, 30.0],
        }
    )
    valid_customers = pd.DataFrame(
        {"customer_id": [1, 2], "segment": ["small", "enterprise"]}
    )
    duplicated_customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 2],
            "segment": ["small", "enterprise", "consumer"],
        }
    )
    return orders, valid_customers, duplicated_customers


def main():
    orders, valid_customers, duplicated_customers = frames()

    valid = prepare_orders(orders, valid_customers)
    assert len(valid) == len(orders)
    assert valid["order_id"].is_unique
    assert valid["amount"].sum() == orders["amount"].sum()
    assert valid["segment"].isna().sum() == 1

    try:
        prepare_orders(orders, duplicated_customers)
    except ValueError as exc:
        assert "customer" in str(exc).lower() or "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate customer keys must fail before the join")

    print("join contract smoke test passed")


if __name__ == "__main__":
    main()
