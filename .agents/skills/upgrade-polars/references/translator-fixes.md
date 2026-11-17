# Step 3 detail: adapting the translator and expressions

Run the code against the new polars and fix in this order.

1. **IR version gate.** `Translator.translate_ir` in `cudf_polars/dsl/translate.py` rejects IR versions
   it has not been validated against. The error message prints the version of the polars you are *adopting*
   (for example `(16, 0)`), which is a version you must now support, **not** the value to put in the gate. Once the
   code works against it, set the gate to the next minor, `>= (16, 1)`. Keep the `# pragma: no cover` on the raise.
2. **Enum drift.** `tests/dsl/test_serialization.py::test_from_polars_all_names` compares polars' function
   name enums with ours exactly. For each new member add `Name.X = auto()` (alphabetical) in
   `dsl/expressions/{boolean,string,struct,temporal}.py`. If it is unsupported, ensure it raises
   `NotImplementedError` so it falls back to CPU, and add a translation-raises test. Track unimplemented
   new features in a drafted issue.
3. **Changed IR/option payloads.** When a node gains or reshapes a field (for example `MergeSorted.key`
   becoming a list, a new `HConcat` slot, `Sort.slice` becoming a 3-tuple), update the translate function and
   the IR class. When an IR class signature changes, grep every constructor call (translator, `streaming/`,
   tests). `ci/check_cudf_polars_ir.py` validates `do_evaluate` signatures.
4. **New fallbacks.** If a new option cannot be honoured on GPU, raise `NotImplementedError` from the translator
   (or the node constructor) with a clear message. Add `assert_ir_translation_raises(q, engine, NotImplementedError)`
   tests; where the *upstream* test cannot pass on GPU, this is the preferred fix.
5. **Semantic changes.** Fix when small (with a unit test). If it needs libcudf work or a redesign, fall back to CPU
   and draft an issue.
6. **Pass the stream.** Pass `stream=df.stream` to every pylibcudf call you add or touch (reviewers sweep for it).
   Prefer `Column.astype` over hand-rolled cast checks, and drop `hasattr` guards for APIs that now always exist.
7. **Imports.** Use `from polars import _plr as plrs` for the private Rust bindings. mypy needs the overrides already in
   the root `pyproject.toml` for `polars._plr`.

8. **Polars constant-folds literals into scans.** In 2.1 `select(pl.lit(<struct/enum>))` becomes a `DataFrameScan`, so the translator
   never sees a `Literal` and a test that expects `NotImplementedError` from that query stops raising. Keep the raise test but build
   the query with `with_columns` (or `when/then`), which still reaches the `Literal`, and add a small `assert_gpu_result_equal` test for
   the folded `select` form. Check what a changed test now exercises (print `translator.errors`), not just that it passes.
9. **Option payloads can change shape.** 2.1 added a join type whose `options[0]` is a 2-tuple (`("Left", "cross_and_filter")`) where
   code unpacked a 3-tuple. Look at the kind of the payload before unpacking, and raise `NotImplementedError` for kinds you do not support.
10. **A polars panic cannot be caught.** If a visitor call panics (for example `get_schema()` on a `BinaryOffset` column, which polars
    maps to `unimplemented!()`), the exception is a `BaseException` and the visitor's arena lock is poisoned, so neither a catch nor the CPU
    fallback works. Do not add a try/except. Xfail the affected tests with a reason and draft an issue for `pola-rs/polars`.
11. **Optimizer rewrites that hide behind a visitor limit.** 2.1 lowers `x.sort_by(k).first()` in `group_by` to `min_by` with a "total
    order" that the Python visitor refuses to expose, so such queries fall back to CPU. Where the rewrite is skipped (for example
    `maintain_order=True`) keep the GPU path covered by passing that option in tests, and add a canary test that the default form raises.
12. **Version-specific code.** The supported range spans two polars minors, but CI measures coverage on the newest. Code reachable
    only on the older version gets `# pragma: no cover; polars 2.0 only` on its `if`/`raise` line; code now unreachable on the newer
    version one gets a note naming that version. Re-run the coverage check after every polars-induced plan change.
13. **mypy.** `DataType.polars_type` is the generic `pl.DataType`; narrow it with `assert isinstance(t, pl.Decimal)` before reading
    `.precision`/`.scale`. Run `mypy --config-file=pyproject.toml cudf_polars tests` from `python/cudf_polars` and ignore the
    `import-untyped`/`_expr_nodes` noise that comes from not having the hook's isolated environment.

Keep comments about polars behaviour short and link the polars issue/PR when there is one.
