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
   tests). `ci/check_cudf_polars_ir.py` validates `do_evaluate` signatures. Payload shapes can also vary by kind (a join type whose
   `options[0]` is a 2-tuple where others are 3-tuples): check the kind before unpacking, and raise `NotImplementedError` for kinds you
   do not support.
4. **New fallbacks.** If a new option cannot be honoured on GPU, raise `NotImplementedError` from the translator
   (or the node constructor) with a clear message. Add `assert_ir_translation_raises(q, engine, NotImplementedError)`
   tests; where the *upstream* test cannot pass on GPU, this is the preferred fix.
5. **Semantic changes.** Fix when small (with a unit test). If it needs libcudf work or a redesign, fall back to CPU
   and draft an issue.
6. **Pass the stream.** Pass `stream=df.stream` to every pylibcudf call you add or touch, and prefer `Column.astype` over hand-rolled casts.
7. **Imports.** Use `from polars import _plr as plrs` for the private Rust bindings. mypy needs the overrides already in
   the root `pyproject.toml` for `polars._plr`.
8. **Optimizer changes can hide a code path.** Polars may constant-fold a query (`select(pl.lit(...))` becoming a `DataFrameScan`) or
   rewrite it into a node the visitor does not expose (`sort_by(k).first()` in `group_by` becoming `min_by`). A test that expected a
   raise, or GPU execution, then silently tests something else. Rebuild the query so it still reaches the path (`with_columns`,
   `maintain_order=True`), add a test for the new form, and print `translator.errors` to check what a changed test exercises.
9. **A polars panic cannot be caught.** A panicking visitor call (for example `unimplemented!()`) raises a `BaseException` and poisons
   the visitor's arena lock, so neither a catch nor the CPU fallback works. Do not add a try/except. Xfail the affected tests with a
   reason and draft an issue for `pola-rs/polars`.
10. **Version-specific code.** See "Coverage" in `unit-tests-oldest-and-coverage.md` for the pragma rules, and re-run the coverage
    check after every polars-induced plan change.
11. **mypy.** `DataType.polars_type` is the generic `pl.DataType`; narrow it with `assert isinstance(t, pl.Decimal)` before reading
    `.precision`/`.scale`. Run `mypy --config-file=pyproject.toml cudf_polars tests` from `python/cudf_polars` and ignore the
    `import-untyped`/`_expr_nodes` noise that comes from not having the hook's isolated environment.

Keep comments about polars behaviour short and link the polars issue/PR when there is one.
