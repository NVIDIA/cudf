# Contributing Guide

This document focuses on a high-level overview of best practices in cuDF.

## Directory structure and file naming

cuDF generally presents the same importable modules and subpackages as pandas.
All Cython code is contained in `python/cudf/cudf/_lib`.

## Code style

cuDF employs a number of linters through [`pre-commit`](https://pre-commit.com/) to ensure consistent style across the code base.
These linting checks must all pass when submitting a pull request, and the
`.pre-commit-config.yaml` file at the root of the repo contains configurations for all linting tools.

Linter configurations are primarily stored in `pyproject.toml`, shared among other Python projects, and extended with cudf specific configurations in `python/cudf/pyproject.toml`

For more information on how to use pre-commit hooks, see the code formatting section of the
[overall contributing guide](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#using-pre-commit-hooks).

## Deprecating and removing code

cuDF follows the policy of deprecating code for one release prior to removal.
For example, if we decide to remove an API during the 22.08 release cycle,
it will be marked as deprecated in the 22.08 release and removed in the 22.10 release.
Note that if it is explicitly mentioned in a comment (like `# Do not remove until..`),
do not enforce the deprecation by removing the affected code until the condition in the comment is met.

When implementing a deprecation:

- Remove and replace all internal usage of the deprecated APIs in cuDF
- Update the documentation with a Sphinx `deprecated` directive describing the deprecation. For example:
- Use `warnings.warn` with a `FutureWarning` and a message describing the deprecation. The deprecation message should:
    - Consist of a single line with no newline characters
    - Indicate a replacement API(s), if any
    - NOT specify a future version when the deprecation will occur.
- Add a unit test that validates that the warning raises

A mock example of a deprecation:

```python
import warnings

def foo(self):
    """
    Return a result from foo

    .. deprecated:: 23.08
        `foo` is deprecated and will be removed in a future version of cudf.
    """
    warnings.warn(
        "`Series.foo` is deprecated and will be removed in a future version of cudf. "
        "Use `Series.new_foo` instead.",
        FutureWarning
    )
```

When enforcing a deprecation:

- Remove the API implementation
- Remove the associated tests in `python/cudf/cudf/tests`
- Remove references in documentation in `docs/cudf`


## `pandas` compatibility

cuDF API signatures and behaviors should align with the [pandas API](https://pandas.pydata.org/docs/reference/index.html). While cuDF
may support a range of pandas versions, API signatures and behaviors should always align with the latest supported pandas version.

Occasionally, cuDF APIs may deviate from pandas behavior. Common reasons include:

- Performance: Match pandas behavior would incur exorbitant runtime or memory costs. Deviations due to performance should be agreed upon by cuDF developers.
- Data type representations: cuDF does not support the full type system of pandas and vice versa, commonly encountered with the `object` or nested types.
- Exception messages: The exception type raised in cuDF should match pandas, but the error messages do not need to exactly align.
- Warnings: cuDF should generally match warnings raised in APIs that mirror pandas, but some warnings might not be applicable due to intentional differences between both libraries.

Intentional deviations should be documented in the [pandas comparison](./documentation.md#comparing-to-pandas).

If it is not possible to match a pandas API, an entire API or a specific component of an API, at all, it should raise a `NotImplementedError`.

### Catching warnings from dependencies

If a cuDF API raises a warning from a cuDF dependency and cannot be reasonably addressed in the API, use [`warnings.catch_warnings`](https://docs.python.org/3/library/warnings.html#warnings.catch_warnings) to suppress the warning from the users.
