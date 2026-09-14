# cuDF Validation Patterns

These patterns target the cuDF 26.10 source tree. Check the installed API when using another release. Adapt the assertions to the declared grain and acceptable tolerance; do not copy a check whose assumption does not fit the data.

## Key profile

```python
import cudf


def key_profile(df: cudf.DataFrame, keys: list[str]) -> dict[str, int]:
    """Compute full-table key diagnostics."""
    null_rows = int(df[keys].isna().any(axis=1).sum())
    duplicate_rows = int(df.duplicated(subset=keys, keep=False).sum())
    distinct_keys = int(df[keys].drop_duplicates().shape[0])
    return {
        "rows": len(df),
        "distinct_keys": distinct_keys,
        "null_key_rows": null_rows,
        "duplicate_key_rows": duplicate_rows,
    }
```

Run this on the filtered population actually used in the join. Global uniqueness and cohort-specific uniqueness are different claims.

## Explicit many-to-one join gate

cuDF 26.10 does not implement pandas-style `merge(validate=...)`; calling it with a non-`None` value raises `NotImplementedError`. Check the relationship yourself.

```python
JOIN_KEYS = ["product_id"]

left_profile = key_profile(facts, JOIN_KEYS)
right_profile = key_profile(products, JOIN_KEYS)

if right_profile["null_key_rows"]:
    raise ValueError("dimension has null product keys")
if right_profile["duplicate_key_rows"]:
    raise ValueError("expected one product row per product_id")

left_keys = facts[JOIN_KEYS].drop_duplicates()
right_keys = products[JOIN_KEYS].drop_duplicates()

unmatched_left = left_keys.merge(right_keys, on=JOIN_KEYS, how="leftanti")
unused_right = right_keys.merge(left_keys, on=JOIN_KEYS, how="leftanti")

result = facts.merge(
    products[JOIN_KEYS + ["category", "unit_cost"]],
    on=JOIN_KEYS,
    how="left",
    suffixes=("", "_product"),
)

if len(result) != len(facts):
    raise AssertionError("many-to-one left join changed fact row count")
```

`unused_right` may be acceptable for a dimension table; `unmatched_left` often is not. The contract decides. Preserve examples or counts from both, rather than silently dropping them.

## Reconcile measures before and after a join

A total-only check can hide offsetting group errors. Check both global and declared subgroup totals.

```python
import math


def scalar_sum(df, column: str) -> float:
    return float(df[column].sum())


before = scalar_sum(facts, "quantity")
after = scalar_sum(result, "quantity")
assert math.isclose(before, after, rel_tol=1e-6, abs_tol=1e-6)

before_by_period = (
    facts.groupby("period", sort=True)["quantity"]
    .sum()
    .reset_index(name="quantity_before")
)
after_by_period = (
    result.groupby("period", sort=True)["quantity"]
    .sum()
    .reset_index(name="quantity_after")
)
check = before_by_period.merge(after_by_period, on="period", how="outer")
check["delta"] = check["quantity_after"] - check["quantity_before"]
assert check["delta"].abs().max() <= 1e-6
```

Choose tolerances for the measure and dtype. Currency stored as fixed-point decimal should usually be reconciled exactly in its smallest unit.

## Preserve an explicit population spine

Aggregations cannot emit entities that have no source rows. Start from the required population when zero-activity entities must be present.

```python
activity = (
    events.groupby(["account_id"], sort=False)
    .agg({"amount": "sum", "event_id": "count"})
    .reset_index()
    .rename(columns={"event_id": "event_count"})
)

output = population[["account_id"]].merge(
    activity,
    on="account_id",
    how="left",
)
output["event_count"] = output["event_count"].fillna(0)

# Fill amount only if the contract says no event means zero amount.
output["amount"] = output["amount"].fillna(0)

assert len(output) == population["account_id"].nunique()
assert not output.duplicated(subset=["account_id"]).any()
```

Do not apply this fill when a missing amount means unknown rather than no activity.

## Point-in-time filter

Use both event/effective time and availability time where applicable.

```python
import pandas as pd

origin = pd.Timestamp("2026-04-01T00:00:00")
eligible = facts[
    (facts["event_time"] < origin)
    & (facts["available_at"] < origin)
]

assert (eligible["event_time"] < origin).all()
assert (eligible["available_at"] < origin).all()
```

The strict comparison is an example, not a universal rule. If records available exactly at the origin are permitted, use `<=` and test that boundary explicitly.

## Deterministic sequence logic

```python
ordered = events.sort_values(
    ["entity_id", "event_time", "event_id"],
    ascending=[True, True, True],
)
ordered["previous_value"] = ordered.groupby("entity_id")["value"].shift(1)
```

`event_id` is a tie-breaker. Without a complete ordering key, tied timestamps do not define a unique prior row.

## CPU/GPU semantic comparison

Use a compact representative fixture for operations whose pandas and cuDF semantics may differ. Compare keys and values after deterministic sorting rather than trusting incidental row order.

```python
import pandas as pd
from pandas.testing import assert_frame_equal

expected = pandas_transform(input_pdf).sort_values(OUTPUT_KEYS).reset_index(drop=True)
actual = (
    cudf_transform(cudf.from_pandas(input_pdf))
    .sort_values(OUTPUT_KEYS)
    .reset_index(drop=True)
    .to_pandas(nullable=True)
)

assert_frame_equal(
    actual,
    expected,
    check_dtype=True,
    rtol=1e-6,
    atol=1e-8,
)
```

Exercise duplicate keys, nulls, empty groups, tied timestamps, unmatched rows, and threshold boundaries—not only the happy path.
