# Add fixed-width list inputs and outputs to transforms

## Description

CUDA and LTO transforms can now consume list rows as `cuda::std::span<T const>` and write list rows through `cuda::std::span<T>*`. This lets a UDF decode variable-length binary records directly from a `LIST<UINT8>` input, or encode variable-length results into a list child allocated from supplied offsets.

List outputs specify their fixed-width child type through `transform_output::list_element_type`. The existing output offsets argument accepts list offsets as well as string offsets; `transform_output_spec::has_offsets` describes both representations. List offsets must be non-null INT32 values with `row_count + 1` entries, starting at zero and nondecreasing. They are validated before allocating the child.

Variable-length output uses two transform passes: measure each row's length, scan the lengths into offsets, then run the encoding UDF with those offsets. The second UDF writes into its allocated row span without changing the span's pointer or size. The resulting list owns its child and retains the supplied offsets. Decimal children use integer coefficients in the UDF and retain their scale in the column metadata.

Parent nulls, sliced columns, scalar inputs, and mixed list/string/fixed-width outputs are supported. Null-aware UDFs receive optional spans. Child elements must be fixed-width and contain no nulls; nested lists, string children, and PTX list transforms are rejected. Reserved ranges for null output rows are initialized and retained; the sizing pass should give null rows zero length for canonical output.

The shared transform path also avoids assigning disengaged optional outputs and reading undefined string views or mask padding while handling nulls. API documentation and the pylibcudf Cython declaration reflect the added child type and generalized offsets argument.

## Validation

- Eight CUDA list cases in the existing transform integration tests cover decimal coefficients, size/scan/encode, slices and scalars, parent nulls, empty/all-null input, mixed outputs, malformed offsets, and unsupported children.
- Rebuilt libcudf and both transform test binaries against `upstream/main` (`dfc5fa9030`), using the RAPIDS-pinned CCCL revision. `TRANSFORM_TEST`: 220 tests passed; `TRANSFORM_LTO_TEST`: 6 tests passed.
- Compute Sanitizer memcheck and initcheck: all eight `ListOperationTest.*` cases passed with zero errors.
- Cython translation of the transform bindings passed. Clang-format and `git diff --check` passed.

## Checklist

- [x] I am familiar with the [Contributing Guidelines](https://github.com/NVIDIA/cudf/blob/HEAD/CONTRIBUTING.md).
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
