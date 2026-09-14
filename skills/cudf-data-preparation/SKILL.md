---
name: cudf-data-preparation
version: "26.10.00"
description: Use to validate cuDF join cardinality, keys, null semantics, time cutoffs, aggregation grain, and reconciliation. Do NOT use for pandas migration.
license: Apache-2.0
metadata:
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - cudf
    - data-preparation
    - data-quality
    - joins
    - etl
---

# cuDF Data Preparation

## Purpose

Build cuDF transformations whose output can be trusted as analytical input. This skill focuses on **data meaning and validation**: stable keys, row grain, join cardinality, missingness, units, time boundaries, and reconciled aggregates.

For a general pandas-to-GPU migration, API-coverage question, or memory/performance tuning task, use [`accelerated-computing-cudf`](../accelerated-computing-cudf/SKILL.md). The two skills may be composed when a migration also needs correctness gates.

## Prerequisites

- A supported NVIDIA GPU environment with a version-matched cuDF installation; consult the RAPIDS installation guide for current CUDA, driver, OS, and Python requirements.
- Access to the input snapshot and enough domain context to define output grain, keys, units, null meaning, and time cutoff.
- dask-cuDF and Dask CUDA only when an out-of-core or multi-GPU path is required. No API key is required by cuDF itself.

## When to use this skill

Use it when the user asks to:

- prepare feature, graph, reporting, simulation, or optimization inputs with cuDF;
- review a join, filter, groupby, window, reshape, or imputation pipeline for correctness;
- prevent fanout, dropped entities, duplicate keys, grain changes, or unit mismatches;
- enforce an as-of cutoff or distinguish event time from data-availability time;
- produce a validated, keyed table with an audit summary.

Do not use it for an ordinary lookup, for model fitting, or merely because a table exists. Do not turn an unavailable value or undefined mapping into an unlabeled default.

## Instructions

## 1. Write the output contract first

Before changing code, identify:

| Contract item | Question to answer |
|---|---|
| Row grain | What does one output row represent? |
| Keys | Which columns uniquely and stably identify that row? |
| Population | Which entities and periods must be present? |
| Measures | What are their definitions, units, valid ranges, and null meanings? |
| Time | Which event time, effective time, availability time, and snapshot apply? |
| Sources | Which input columns and transformations produce each output field? |
| Acceptance | Which counts, totals, rates, and tolerances must reconcile? |

Keep business identifiers as columns. Never use row position, a transient cuDF index, a graph-renumbered ID, or a model matrix row number as the durable key.

If the contract is ambiguous in a way that changes rows or values, ask a focused question. Otherwise state the interpretation and make it testable.

## 2. Inspect before transforming

Profile the exact input snapshot, not only a sample:

- row count and distinct entity count;
- null count by required column;
- duplicate count at the declared key;
- category/domain counts and unexpected values;
- numeric range, non-finite count, and unit consistency;
- minimum/maximum relevant timestamps;
- mapping coverage between source and output taxonomies.

A sample is useful for understanding shape, but it does not prove full-population uniqueness or completeness. Preserve source paths or table revisions and the transformation parameters needed to rerun the result.

## 3. Build transformations at the declared grain

Prefer explicit column expressions, filters, `groupby`, `merge`, reshape, and window operations. Keep operations on GPU when supported, but never trade semantic correctness for device residency.

### Joins

For every join:

1. State the intended cardinality: one-to-one, many-to-one, one-to-many, or many-to-many.
2. Check key nulls and uniqueness on the relevant side **before** joining.
3. Measure unmatched keys in both directions when population coverage matters.
4. Run the join with explicit keys, type, and suffixes.
5. Reconcile output rows, entities, and invariant totals.

In the 26.10 source tree, `DataFrame.merge(..., validate=...)` accepts the parameter but raises `NotImplementedError`; enforce cardinality explicitly. See [validation patterns](references/validation-patterns.md) for runnable checks.

Never fix fanout with `drop_duplicates()` unless duplicate equivalence and the retained-row rule are part of the data contract. Never let a join silently change measure grain.

### Grouping and ordering

Use explicit grouping keys and aggregations. cuDF `groupby` defaults to `sort=False`, so result order is not a correctness guarantee. Sort only where downstream behavior requires deterministic order, and include tie-break columns when row order is meaningful.

For `shift`, rolling, cumulative, first/last, or sequence logic, sort by the partition keys and the full ordering key first. Test tied timestamps and boundary rows.

### Nulls and categories

Distinguish:

- zero from missing;
- unknown from not applicable;
- a missing row from a present row with a null value;
- source null from a null introduced by an outer/left join.

Impute only under an explicit rule. Fit learned imputation or encoding on the allowed training population, not on future or held-out rows. Preserve nullable dtypes when they carry meaning; do not substitute magic sentinels just to simplify code.

### Time boundaries

An event timestamp does not prove the row was available at an earlier analysis or prediction time. Use availability/ingestion/revision timestamps when the question is point-in-time. Define same-time inclusion explicitly (`<` versus `<=`) and test just below, at, and above the cutoff.

For mutable plans or dimensions, select the version effective and available at the requested origin. Do not join every historical fact to today's dimension value unless that is the intended question.

## 4. Validate the materialized output

Run checks on the actual result that will be handed off or written:

- exact output key uniqueness and required-population coverage;
- expected schema, dtypes, units, and null policy;
- row/entity counts before and after each cardinality-changing step;
- totals and rates reconstructed independently from source columns;
- group-level checks so aggregate agreement cannot hide offsetting errors;
- range, finite-value, and domain checks;
- cutoff and ordering invariants;
- a small CPU reference for semantics-sensitive operations when practical.

Use tolerances derived from dtype and business meaning. Do not round first and then claim equality. Hashes can establish byte identity, not semantic correctness.

## Examples

### Minimal join review example

```python
keys = ["customer_id"]

# many orders to one customer
assert customers[keys].isna().sum().sum() == 0
assert not customers.duplicated(subset=keys).any()

prepared = orders.merge(
    customers[keys + ["segment"]],
    on=keys,
    how="left",
)

assert len(prepared) == len(orders)  # valid after right-side uniqueness check
assert prepared["order_id"].nunique() == orders["order_id"].nunique()
```

This is only a local invariant. Also inspect unmatched customer keys and reconcile material measures. The reference file includes a fuller pattern.

## 5. Scale without changing the contract

Start with single-GPU cuDF when the working set and intermediates fit. Enable spilling before creating cuDF objects when needed. Use dask-cuDF only when the workload requires out-of-core or multi-GPU execution, and validate partition-boundary behavior, shuffles, and the final materialization separately.

Do not call `.compute()` on a distributed collection unless the result fits on one GPU. Push projections and filters before joins and shuffles. Performance claims require an end-to-end measurement at representative scale; a small fixture proves correctness, not speedup.

## Limitations

- This guide does not provide credentials, authorize source access, or define business meanings that are absent from the data contract.
- cuDF implements a pandas-like API, not every pandas behavior; use the installed API and a targeted parity fixture for semantics-sensitive operations.
- Spilling and dask-cuDF can extend capacity but do not guarantee an allocation will succeed or preserve an invalid partitioning strategy.
- Validation can show that code satisfies declared invariants; it cannot prove an incorrect or incomplete contract is the right one.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Row count jumps after a left join | Duplicate right-side keys | Profile the filtered join domain; resolve or model one-to-many explicitly |
| Totals agree but subgroup rates differ | Grain or denominator mismatch | Recompute numerator and denominator per declared group, then aggregate |
| Missing entities disappear | Inner join or edge-only population | Compare required keys before/after; use an explicit population spine |
| Time-based metric is too good | Future/revised data leaked across cutoff | Audit availability time and fit scope, not only event date |
| CPU and GPU rows differ in order | Order was implicit | Sort by complete deterministic keys only if order is contractual |
| GPU OOM | Large intermediates or exposed buffers | Project/filter earlier, enable spilling before allocation, or move to dask-cuDF |

## Deliverable

Return:

1. the cuDF transformation or reviewed code;
2. output grain, keys, schema, units, snapshot, and cutoff;
3. source-to-output mapping and explicit assumptions;
4. validation results with counts, unmatched keys, duplicates, nulls, and reconciliations;
5. runtime path (cuDF, `cudf.pandas`, or dask-cuDF), fallbacks/boundaries, and version;
6. unresolved data gaps and their consequence.

A downstream model should reject a missing critical field or accept a clearly labeled hypothetical value; it should not receive fabricated completeness.

## References

- [cuDF validation patterns](references/validation-patterns.md)
- [cuDF API documentation](https://docs.rapids.ai/api/cudf/stable/)
- [dask-cuDF documentation](https://docs.rapids.ai/api/dask-cudf/stable/)
