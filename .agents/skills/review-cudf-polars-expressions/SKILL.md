---
name: review-cudf-polars-expressions
description: Use when implementing or reviewing support for Polars Expressions in cudf-polars
---

# Polars Expression implementation in cudf-polars

This rule describes guidelines and implementation patterns for supporting Polars' Expression Python APIs in cudf-polars. A successful implementation:

1. Runs entirely on the GPU through cudf-polars and does not fall back to Polars on the CPU.
2. If an expression cannot be supported or only partially supported (e.g. a parameter is unsupported), an error is raised during IR translation and not during runtime.
3. The expression is supported for all Polars versions that cudf-polars supports specified in `python/cudf_polars/pyproject.toml`
4. Passes all `pre-commit` checks.

## Prerequisites

1. Identify a suitable cudf-polars development environment, if not provided by the user, and ensure the Polars version is the latest supported version by cudf-polars.
2. Identify a directory that contains the Polars source code to use as a reference for implementations. If it is not provided by the user or one cannot be identified, clone it to a temporary location. Ensure the Polars Git source tree is checked-out to the same Polars version installed in the development environment.

For example, to clone Polars matching the Polars version in a fictional, conda development environment named "cudf-dev"
```bash
POLARS_VERSION=$(conda run -n cudf-dev python -c "import polars as pl; print(pl.__version__)")
git clone https://github.com/pola-rs/polars.git --single-branch --branch py-$POLARS_VERSION /tmp/polars
```

If you are unable to clone Polars, the Polars repository is located at https://github.com/pola-rs/polars.

## Step 1: Understand the Polars expression implementation and behavior.

First, review the Polars implementation of the expression and understand its behavior without cudf-polars.

* The Python APIs for expressions are mostly defined in the `py-polars/src/polars/expr` directory in a Polars repository.
* Expressions are eventually exposed to cudf-polars in Rust in the `crates/polars-python/src/lazyframe/visitor/expr_nodes.rs` file in a Polars repository

Next, further understand the runtime behavior of the expressions by:

* Reviewing relevant tests in the `py-polars/tests` directory that use the expression.
* Generating examples locally with the expression(s)
    1. Ensure examples demonstrate a wide variety of behaviors given differing input data, including all applicable data types, missing values, and values that may introduce edge cases like 0 for numeric data.
    2. Ensure examples demonstrate when exceptions are raised given invalid input arguments. Understand if the exceptions are raised before expressions are evaluated and therefore might not reach cudf-polars versus exceptions that are raised during expression evaluation runtime.

## Step 2: Understand the current Polars expression behavior with cudf-polars.

Now, review if the Polars expression is currently supported and correct with cudf-polars.

Run the same examples generated in Step 1 with `pl.GPUEngine(executor="streaming", raise_on_fail=True)` passed to the `engine` argument of `collect` so failures do not fall back to CPU Polars. If the example data was relatively small, this should test the single partition path of cudf-polars.

Additionally run another variation of the Step 1 examples with `pl.GPUEngine(executor="streaming", raise_on_fail=True, executor_options={"max_rows_per_partition": 2})` passed to the `engine` argument of `collect`. With a large enough input data for the examples, this configuration would test the multiple partition path of cudf-polars.

Note the following failure and fallback cases:

1. A failure might occur because an expression isn't exposed in Polars through `crates/polars-python/src/lazyframe/visitor/expr_nodes.rs`. A fix therefore would be needed upstream in Polars in order to proceed with the next steps.
2. A failure might occur because an expression isn't supported in cudf-polars.
3. An expression might not be implemented when run in the multiple partition path of cudf-polars and might fall back to the single partition path.

## Step 3: Implement the Polars expression in cudf-polars.

### Step 3a. Single Partition Implementation

Start with scoping an implementation for the single partition path for cudf-polars. This ensures that the multiple partition path can fall back to this implementation.

1. An expression implementation should belong in the `python/cudf_polars/cudf_polars/dsl/expressions` directory.
2. Review the `pylibcudf` API to find the appropriate function or functions needed for the implementation.
    1. Additionally review the `libcudf` public API as a function may be appropriate but not exposed through `pylibcudf`. If needed, expose this API through `pylibcudf` as part of the implementation. `pylibcudf` will need to be rebuilt from source in order to test the implementation.
    2. Minimize the amount of kernel launches by using specific `pylibcudf` functions where possible. Evaluate using `pylibcudf.expressions` where sensible for combining several operations.
    3. Do not use any `pylibcudf` methods that are deprecated.
    4. Implement any short-circuiting opportunities by checking Column properties like the `null_count` or `size` that minimizes `pylibcudf` calls.
    5. Do not use APIs that convert the data CPU objects like `to_arrow` or `to_pylist` unless absolutely necessary.
    6. Ensure `pylibcudf` APIs calls are passed a stream object so they do not use their default stream argument.
    7. Follow any additional guidance in `python/REVIEW_GUIDELINES.md`
3. When raising an exception to match Polars or to note that some functionality is not supported, prefer raising these exceptions in the `__init__` method instead of the `do_evaluate` methods of the `Expr` subclasses.
4. Ensure the `is_pointwise` on the `Expr` subclass correctly reflects if the expression is a pointwise operation.

### Step 3b. Multiple Partition Implementation

Next, if the expression is non-pointwise, add a multiple partition implementation in `python/cudf_polars/cudf_polars/streaming/expressions.py`

TODO: Add more guidance on a multiple partition implementation.

## Step 4: Test the Polars expression implementation in cudf-polars

Finally, add unit tests to an existing or new file in the `python/cudf_polars/tests/expressions` directory to test the implementation.

1. The added unit tests should cover all the lines added in the implementation. A CI job validates that there is 100% code coverage.
2. The unit tests should use the `engine` fixture from `python/cudf_polars/tests/conftest.py` to test all applicable cudf-polars engine types including single and multiple partition execution.
3. Review the existing unit tests in `python/cudf_polars/tests` to check if existing tests used this expressions. Unit tests may have existed that asserted that this expression was not supported.
4. The unit test should use `pytest.mark.skipif` with a boolean variable from `python/cudf_polars/cudf_polars/utils/versions.py` if a unit test exercises a Polars expression or an argument that doesn't exist since a particular Polars version within the cudf-polars support window defined in `python/cudf_polars/pyproject.toml`.
5. When using `assert_gpu_result_equal` for expressions that return floats, consider specifying `check_exact=False` if necessary.

When running the unit tests:

1. First, `git stash` the new implementation (i.e. all changes not in `python/cudf_polars/tests`) and only run the newly added tests to validate they fail without the new implementation.
2. Next, `git stash apply` the stashed files and run *all* unit tests in `python/cudf_polars/tests`. Address failures from the newly added tests or existing tests that fail because they use the Polars expressions being implemented.
