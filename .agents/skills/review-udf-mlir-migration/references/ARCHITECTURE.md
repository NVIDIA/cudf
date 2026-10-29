# Architecture map

This map is conceptual. It names modules and symbols only as orientation;
locate current paths by searching source rather than relying on the names here,
which evolve.

## Extension registration

The MLIR backend's recurring shape is:

- a frontend type in a `*_typing.py` module;
- typing templates registered with `typing_registry`;
- a physical model registered with `register_model`;
- operation and cast implementations registered with `lowering_registry` and
  `lower_cast`;
- import-time registration, either through decorators/direct calls or through
  a module `_register()` helper.

Use types and templates from `numba_cuda_mlir` and its vendored Numba namespace.
Do not mix stock Numba classes into signatures or registrations.

## Masked and NA

Logical model:

```text
MaskedType(T) -> { T value, i1 valid }
NAType        -> payload-free sentinel
```

Review together:

- constructor typing and lowering;
- `.value`/`.valid` attribute typing and extraction;
- `NAType` and `MaskedType.unify`;
- scalar/NA/Masked casts;
- binary and unary result typing;
- validity propagation;
- datetime/timedelta unit-aware delegation;
- truth semantics;
- membership;
- `pack_return`.

Important details:

- `Masked op Masked`: validity is normally `lhs.valid AND rhs.valid`.
- `Masked op scalar`: preserve Masked validity.
- an invalid payload is poison/undefined and must not be inspected by tests;
- comparison returns a Masked boolean when nullability remains;
- truth testing returns a plain boolean according to the API's invalid-value
  semantics;
- datetime-like operations must reuse unit-aware scalar lowering rather than
  blindly applying an integer operation to raw payloads.

## Strings

Current incremental design:

```text
mlir_string -> { ptr meminfo, ptr data, i64 nbytes }
```

`MLIRStringType` is not a NumPy-internal scalar and is pointer-marshalled to
kernels. `ManagedStrArrayWrapper` preserves the intended extension type instead
of allowing CUDA Array Interface inference to reduce it to a byte array.

Review strings by operation class:

### Borrowed/read-only

Examples: `len`, comparisons, `contains`, `find`, `rfind`, `count`,
`startswith`, `endswith`, and character predicates.

- input `meminfo` may be null;
- no allocation should be introduced;
- literals should become views without unnecessary owning-string allocation;
- nullable variants must preserve/AND validity;
- `len` is a character count, not blindly `nbytes`; test multi-byte UTF-8.

### Allocating/string-producing

Examples: concatenation, case conversion, replacement, and strip variants.

- allocate variable-sized storage;
- construct a non-null owner;
- preserve returned ownership across output assignment;
- release non-escaping temporaries;
- materialize or transfer final results into a cuDF column;
- release temporary result owners afterward.

### Legacy production representation

The first port retained `StringView`, `UDFString`, and `ManagedUDFString` plus
C++ shims. These types remain active in the legacy row/groupby backend while
the incremental MLIR design replaces them with `mlir_string`. The project
deliberately stopped making the new-backend PR sequence teach this three-type
design only to delete it later. Do not mistake "legacy in the MLIR design" for
"already removed from production."

## Row apply

Conceptual flow:

1. infer column and UDF types;
2. build a row record/schema;
3. marshal data and mask arguments;
4. generate a kernel that loads one row per logical thread;
5. create Masked/string extension values;
6. inline or call the user function;
7. normalize the return;
8. scatter payload and validity;
9. build the output column and clean managed temporaries.

Check both type-inference compilation and execution compilation. Transitional
branches sometimes used different compilers for these phases.

## Groupby apply and cooperative arrays

Conceptual flow:

1. grouped rows are contiguous and described by offsets;
2. one CUDA block handles one group;
3. `group["x"]` creates a borrowed cooperative view;
4. elementwise operations may allocate a managed block-shared result;
5. reductions use block-wide shuffle/shared-memory/atomic algorithms;
6. one thread writes a block-uniform result.

Review:

- allocation by one thread;
- shared pointer and owner broadcast;
- barriers before consumers and before lifetime end;
- empty-group semantics;
- shared-memory sizing, especially multi-stage reductions such as variance/std;
- dtype promotion and reduction accumulator type;
- ownership of intermediate arrays;
- whether compiler releases include the required cooperative/linking support.

Groupby NRT and arena allocation have historically been later than basic
cooperative reductions. Confirm landed state rather than assuming every
prototype feature is production.

## Rolling apply

Rolling UDFs have a production numba-cuda-mlir path in `rolling_utils.py`.
Treat it as its own vertical slice. Review:

- return-type inference and execution compilation both use the intended backend;
- window start/stop bounds and indexing;
- `min_periods` and output-validity construction;
- empty-input behavior;
- kernel cache keys and reuse;
- dtype/return-type conversion and output allocation.
- preservation of the current public rejection of nullable inputs with
  `NotImplementedError`, unless the PR deliberately changes and tests that API.

Do not infer that row/groupby support proves rolling support, or vice versa.

## Compilation and linking

Search current numba-cuda-mlir for the APIs actually used by the PR:

- extension typing/lowering/model registries;
- compiler entry points and requested output, such as
  `compiler.compile(..., output="ltoir")`;
- library/link inputs and post-load callbacks;
- NRT function emission or object linking;
- runtime setup globals;
- argument handler hooks.

Review symbol ownership:

- every required device symbol is defined;
- no runtime is linked twice;
- destructors remain callable across module boundaries;
- post-load setup targets the loaded module/library;
- architecture/toolkit support matches cuDF's supported matrix.

## Test layers

Typical focused tests live under:

`python/cudf/cudf/tests/private_objects/mlir_backend/`

Interpret them by layer:

- `test_*_typing.py`: signatures, attributes, unification, rejection;
- `test_*_lowering.py`: direct tiny kernels and runtime values;
- marshalling helpers: ABI and host/device boundary;
- dataframe/series/groupby apply tests: production integration;
- NRT stats tests: allocation/refcount accounting;
- Compute Sanitizer: illegal access/use-after-free evidence.

For focused tests, verify the intended extension registration is loaded. A
generic template can accidentally steal an operation while leaving the test
green.
