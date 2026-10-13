# Keep owners alive when a native call throws

Read this when a native operation may return or throw while GPU work still uses engine-owned storage. First establish the engine's rules for input recycling, result lifetime, borrowed streams and allocator ownership. A successful return from a C++ call is not proof that its GPU work has finished.

A stream view, table view or resource reference describes access. It does not by itself retain the object or storage it refers to. Keep the owning objects alive for their actual dependent uses, including result views that alias input memory. Check the installed version's public headers and the engine interface rather than relying on a type's name.

## A host allocation can fail after work was submitted

Suppose the engine lends a buffer to an adapter. The adapter queues a copy, calls a native operation, then allocates a host object to describe its result. The native call or that final allocation throws. Ordinary stack unwinding can release the input owner while the queued copy is still reading it. Waiting only on the successful-return path misses this failure.

Establish the ownership needed for cleanup before the first submission. Keep input storage, the stream owner and allocator owners alive across every potentially throwing native call and result-wrapper construction. Account for both allocations in the adapter and allocations inside called libraries.

The following is control-flow pseudocode, not a cuDF API or an engine-independent implementation:

```text
lease = retain required owners before submitting work
try:
    result = submit native work using the lease
    attach the required ownership to the result
    return result under the engine's completion contract
catch original_error:
    completion = wait for every submitted use covered by the lease
    if completion is established:
        release temporary ownership and propagate original_error
    otherwise:
        preserve the required owners through the engine's failed-context policy
        report the completion failure as well as original_error
```

The failure path must not depend on a new allocation that can fail while trying to save the owners. If the engine needs a deferred-retention record, establish that capacity before submission. Do not invent a failed-context handler or silently continue after a synchronization error; use the engine's documented policy, or identify the missing recovery boundary.

## Choose completion and destruction rules deliberately

For a synchronous POC, completing relevant work before returning or unwinding can be an acceptable implementation. For an asynchronous result, transfer ownership and a valid readiness dependency to the result or engine. Preserve owners until their last dependent use completes. If the engine requires the whole context stream to drain before recycling an input, satisfy that stronger rule.

An event recorded only after a native call returns cannot cover an exception thrown before that event is recorded. A stream wait can cover earlier work on that stream, but work on other streams needs the corresponding dependencies. Destroying a stream is not a substitute for waiting: CUDA permits destruction to return while queued work remains. Check synchronization status; it can report an earlier asynchronous error. [CUDA stream management](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__STREAM.html).

Ensure dependent allocations are destroyed before their allocator owner, and preserve any stream owner needed by deallocation. Check destruction order explicitly for result members and exception guards. A destructor that cannot throw still needs a defined policy for failure to establish completion. Resource references do not remove the caller's ownership obligation. [cuDF resource lifetime guidance](https://docs.nvidia.com/cudf/latest/libcudf/api_docs/memory_resource/).

## Validate the engine's required failure behavior

Use a controlled recoverable allocation failure after submission, while the relevant work is demonstrably pending. Check that required owners are not released early and that successful results still match the independent reference. If the interface promises recovery, check a subsequent valid call. Also exercise a failure before submission and the normal path, so the test can distinguish a cleanup defect from an invalid setup.

Accept a valid synchronous solution or a valid asynchronous solution when the engine allows both. Distinguish host-allocation failures from GPU out-of-memory, lost-device and other fatal-context failures; a test of one does not establish recovery from the others. Report exactly which error path was exercised. The native operation must still perform the requested GPU work and return its result.
