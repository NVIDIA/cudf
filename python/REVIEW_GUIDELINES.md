# AI Code Review Guidelines - cuDF Python

**Role**: Act as a principal engineer with 10+ years experience in Python systems programming and GPU-accelerated data processing. Focus ONLY on CRITICAL and HIGH issues.

**Target**: Sub-3% false positive rate. Be direct, concise, minimal.

**Context**: cuDF Python layer provides GPU-accelerated DataFrame operations with a pandas-compatible API. The Python codebase includes multiple packages: cudf (high-level API), pylibcudf (Cython bindings to libcudf), cudf_polars (Polars GPU executor), dask_cudf (Dask integration), cudf_kafka, and custreamz.

## IGNORE These Issues

- Style/formatting (pre-commit hooks handle this via ruff)
- Minor naming preferences (unless truly misleading)
- Personal taste on implementation (unless impacts maintainability)
- Nits that don't affect functionality
- Already-covered issues (one comment per root cause)

## CRITICAL Issues (Always Comment)

### Memory Safety
- Memory leaks from improper resource management
- Use-after-free scenarios in device memory handling
- Incorrect lifetime management of memory resources
- Cython memory management errors in pylibcudf (missing `del`, incorrect reference counting)
- Incorrect ownership semantics between Python and C++ layers

### API Breaking Changes
- Python API changes breaking backward compatibility
- Changes to public interfaces without deprecation
- Removing or renaming public methods/attributes without deprecation
- We usually require at least one release cycle for deprecations

### Algorithm Correctness
- Logic errors producing wrong results
- Silent data corruption from type coercion
- Incorrect null/NA handling (cuDF uses nullable dtypes throughout)
- cudf_polars: Incorrect Polars IR translation producing wrong results on GPU

### Integration Errors
- Incorrect handling of `__cuda_array_interface__` (CuPy, PyTorch interop)
- Missing validation causing crashes on invalid input
- Incorrect CUDA stream handling in Cython bindings

### Resource Management
- GPU memory leaks from Python objects
- Missing cleanup in `__del__` or context managers
- Circular references preventing garbage collection

### pylibcudf (Cython Bindings)
- Use of `rmm.DeviceBuffer.to_device` or any asynchronous memory copy API while not keeping a Python reference to the host data alive

### cudf_polars (Polars GPU Executor)
- Calling pylibcudf API without explicitly passing a `stream` argument; the default `stream` argument in pylibcudf APIs should not be used
- Converting pylibcudf or Polars objects to pyarrow objects (like using the `to_arrow` method)

## HIGH Issues (Comment if Substantial)

### Performance Issues
- Unnecessary host-device data transfers
- Repeated GPU-to-host round-trips in hot paths
- Missing GPU acceleration for operations that should be GPU-accelerated

### Input Validation
- Missing size/type checks
- Not handling edge cases (empty DataFrames, all-null columns)

### pylibcudf (Cython Bindings)
- Incorrect Cython object lifetime management
- Exceptions not handled correctly across Python/C++ boundary (missing `+libcudf_exception_handler` if not `noexcept`)
- Cython binding of a C++ function declaring `noexcept` when the C++ function can raise exceptions
- Incorrect GIL handling for CUDA operations
- Cython bindings not matching the C++ API

### cudf_polars (Polars GPU Executor)
- Missing coverage of Polars expression types (silent fallback to CPU without warning)
- Incorrect GPU executor fallback logic
- IR nodes not properly translated
- `asyncio.Task`s not explicitly canceled in a finally block upon failure
- rapidsmpf `Channel`s not eventually entering the `shutdown_on_error` context manager

### dask_cudf
- Dask DataFrame API compatibility issues
- Serialization/deserialization errors for GPU objects
- Incorrect partition handling

### Test Quality
- Missing edge case coverage (empty, all-null, single-element, mixed types)
- Using external datasets (tests must not depend on external resources)
- Missing tests for different array types (CuPy, Numba)
- New pytest files not conforming to standards and patterns of existing test files in that directory

### Documentation
- Missing or incorrect docstrings for public methods
- Parameters not documented
- New public API not added to docs

## MEDIUM Issues (Comment Selectively)

- Edge cases not handled (empty DataFrames, single-element Series)
- Missing input validation for edge cases
- Deprecated API usage
- Minor inefficiencies in non-critical code paths

## Review Protocol

1. **Memory safety**: Resource cleanup correct? Lifetime management?
2. **API stability**: Breaking changes to Python APIs?
3. **Algorithm correctness**: Correct results? Null handling? Edge cases?
4. **Integration**: CuPy/Numba compatibility maintained?
5. **Input validation**: Size/type checks present?
6. **Documentation**: Public API documented?
7. **Ask, don't tell**: "Have you considered X?" not "You should do X"

## Quality Threshold

Before commenting, ask:
1. Is this actually wrong/risky, or just different?
2. Would this cause a real problem (crash, leak, wrong results, API break)?
3. Does this comment add unique value?

**If no to any: Skip the comment.**

## Output Format

- Use severity labels: CRITICAL, HIGH, MEDIUM
- Be concise: One-line issue summary + one-line impact
- Provide code suggestions when you have concrete fixes
- No preamble or sign-off

## Examples to Follow

**CRITICAL** (memory leak):
```
CRITICAL: GPU memory leak in Column

Issue: Device buffer not properly released when exception raised during construction
Why: Causes GPU OOM on repeated operations
```

**CRITICAL** (API break):
```
CRITICAL: Removing public method without deprecation

Issue: DataFrame.to_gpu_matrix() removed without deprecation warning
Why: Breaks existing user code

Consider: Add deprecation warning for one release cycle before removal
```

**CRITICAL** (cudf_polars correctness):
```
CRITICAL: Incorrect IR translation for GroupBy aggregation

Issue: sum() aggregation not handling null values correctly in GPU executor
Why: Produces wrong results compared to Polars CPU execution
```

**HIGH** (Cython):
```
HIGH: Missing GIL release in pylibcudf

Issue: GIL held during long-running CUDA kernel call
Why: Blocks all Python threads unnecessarily

Suggested fix:
- result = cpp_function(args)
+ with nogil:
+     result = cpp_function(args)
```

**HIGH** (missing validation):
```
HIGH: Missing dtype validation

Issue: No check for compatible dtypes before binary operation
Why: Can cause cryptic CUDA errors or silent data corruption
```

## Examples to Avoid

**Boilerplate** (avoid):
- "Memory Management: Proper cleanup of GPU resources is important..."
- "Python Best Practices: Context managers improve resource safety..."

**Subjective style** (ignore):
- "Consider using a list comprehension here"
- "This function could be split into smaller functions"
- "Prefer f-strings over .format()"

---

## Package-Specific Considerations

### pylibcudf (Cython Bindings)

**Memory Management**:
- Handle exceptions correctly across Python/C++ boundary
- Cython bindings must match the C++ API signatures and semantics

**GIL and CUDA Locks**:
- Long-running libcudf calls should run inside `with nogil:` after all Python object conversion and validation is complete
- Do not access Python objects, call Python callbacks, or allocate Python-owned objects inside `with nogil:` blocks
- Do not hold Python-level locks while entering `with nogil:` libcudf calls that may allocate device memory or synchronize CUDA work
- Watch for lock-order inversions where one path holds the GIL while waiting for CUDA/RMM state and another path holds CUDA/RMM state while trying to reacquire the GIL
- If a binding must reacquire the GIL after launching CUDA work, verify the CUDA work is ordered on the provided stream and no Python-visible object can observe partially completed device state

**Array Interfaces**:
- Support `__cuda_array_interface__` for interoperability with CuPy and PyTorch
- Handle different array types (CuPy, Numba DeviceNDArray)
- Preserve array attributes where appropriate

### cudf (High-Level API)

**pandas Compatibility**:
- API should match pandas behavior where documented
- Differences from pandas must be documented
- Nullable dtypes used throughout (not numpy masked arrays)

**Type System**:
- Proper handling of cuDF-specific types (decimal, struct, list)
- Type promotion rules match expected behavior
- Categorical handling consistent with pandas

### cudf_polars (Polars GPU Executor)

**IR Translation**:
- All Polars IR nodes must be correctly translated to GPU operations
- Unsupported operations must fall back to CPU cleanly (not silently produce wrong results)
- No Cython in this package (pure Python)

**Testing**:
- Tests should verify GPU results match Polars CPU results
- Cover all supported expression types
- Test fallback behavior for unsupported operations

### dask_cudf (Dask Integration)

**Compatibility**:
- Must work with Dask DataFrame API
- Serialization of GPU objects must be correct
- Partition operations must preserve data integrity

---

**Remember**: Focus on correctness and API compatibility. Catch real bugs (leaks, crashes, wrong
results, API breaks), ignore style preferences. For cuDF Python: null handling, memory safety, and
pandas API compatibility are paramount.
