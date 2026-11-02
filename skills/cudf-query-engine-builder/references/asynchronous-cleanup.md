# Complete native work before releasing its owners

Read this before implementing a path that uses asynchronous copies, borrowed events, multiple streams, engine-owned device buffers, or an owning result. Establish the caller's rules for input recycling, host-readable output, result destruction and failed CUDA contexts from its code and tests.

A stream view, table view or resource reference does not keep its owner alive. Retain the objects that own input storage, native intermediates, output storage, the stream and the allocator for every dependent use. Check the installed public headers for the types and APIs being used.

## Choose what return means

A GPU-to-GPU handoff may return pending work when the caller receives both its owners and a readiness dependency. The consuming stream must wait on that dependency before using the buffers.

A function that promises host-readable output must finish its device-to-host copies before returning. `cudaStreamWaitEvent(consumer, ready, 0)` orders future work on the consumer stream; it does not wait for host completion. Check `cudaStreamSynchronize(consumer)` or a completion event that covers the copies before exposing the host buffers. Keep borrowed events unchanged and preserve their owner's lifetime. [CUDA stream management](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__STREAM.html).

Apply the caller's completion rule to empty output too. If the caller requires its producer event to be complete at return, a zero-row branch must establish that completion even though it has no bytes to copy. Avoid an early return before the common completion step. Do not impose this synchronous rule on an interface that explicitly permits pending empty results.

## Put retained owners outside the throwing scope

A native call can enqueue GPU work and then throw because a host allocation fails. Later output allocations, container growth and result-wrapper construction can also throw. Keeping owners only in a successful result does not protect these paths.

In C++, an object declared inside a `try` block is destroyed before that block's `catch` begins. A wait in the catch is too late if the needed owner was already destroyed. Declare the owner bundle and any partially constructed output owners before the protected `try`, or use a correctly ordered scope guard that runs before those owners are destroyed. Prepare owner-container capacity and failure-retention storage before submission when practical. If work is already pending on entry, first retain its existing owners through a nonthrowing transfer or an existing caller lease; an allocation while constructing the protection must not open another release path.

This structural C++ example uses engine-specific placeholder types and operations. It shows the required scopes, not a cuDF API or a reusable implementation:

```cpp
RetainedOwners retained = take_existing_owners_noexcept();
// retained outlives the try block and catch. It also holds partial outputs.
try {
    // This call may enqueue work before throwing. Do not arm cleanup afterward.
    submit_native_work(retained);
    construct_result_in(retained);  // This can allocate and throw too.
    establish_required_completion_or_result_dependency(retained);
    return transfer_complete_result_ownership(retained);
} catch (...) {
    // All required owners are still alive here.
    auto status = cudaStreamSynchronize(retained.stream());
    if (status != cudaSuccess) {
        // Engine policy must retain these owners without a new allocation,
        // preserve both errors, and stop normal unwinding until safe.
        enter_engine_failed_context_policy(status, retained);
    }
    throw;  // Propagate the original error after completion is established.
}
```

The failed-context operation above is a required engine policy, not a supplied helper. It must not return to ordinary unwinding while owners can still be used. If the engine has no such policy, identify the missing recovery boundary instead of inventing one. A checked wait may report an earlier asynchronous error; a no-throw wait does not by itself prove safe cleanup. For multiple streams, complete every dependent use or first join those uses onto the stream being waited on.

An event recorded after a native call returns cannot protect an exception thrown before the record. Stream destruction can return while queued work remains, so destroying a stream is not a completion mechanism. A valid synchronous POC may wait on success and exception paths. An asynchronous result must retain owners and a usable completion dependency through its final consumer.

## Complete downloads before returning host buffers

When the host-output interface permits it, allocate all output buffers before submitting the first download. This removes the avoidable failure window in which the first copy is pending and allocation of a later output fails. It does not remove the need to protect other exceptions.

The following body belongs inside the protected scope above, with `output` and the native-result owners retained outside that scope:

```cpp
check_cuda(cudaStreamWaitEvent(download_stream, borrowed_ready, 0));
if (row_count != 0) {
    allocate_all_host_outputs(output);
    enqueue_all_downloads(native_result, output, download_stream);
}
check_cuda(cudaStreamSynchronize(download_stream));  // Also for empty output.
return output;
```

The unconditional completion point applies when this interface promises ready output and completed producer work. On an allocation or copy error, the outer catch must keep both host and device owners alive while already submitted work completes. Resetting output owners before waiting reverses the required order.

## Check the actual allocator requirement

Use the engine's stream and requested resource for the allocations covered by the API contract. Read the installed parameter documentation: an `mr` parameter documented for returned memory is not a guarantee that every internal temporary uses that resource. Treat counts of temporary or default-resource allocations as observations unless the engine explicitly restricts them. Do not reject a valid result solely because an internal temporary used another supported resource.

Retain the allocator owner until every allocation made from it has been destroyed. Retain the stream owner needed by stream-ordered deallocation. In a result object, declaration order matters because members are destroyed in reverse order: declare the context owner before the dependent buffers. cuDF's memory-resource interfaces distinguish returned and temporary allocations; check the version in use rather than changing a process-wide default as a workaround. [cuDF resource guidance](https://docs.nvidia.com/cudf/latest/libcudf/api_docs/memory_resource/).

## Test the required handoffs and failures

Exercise a delayed producer and a delayed final download, so a missing wait cannot pass because work happened to finish early. Test zero-row output under the same caller readiness rule as nonempty output. For successful calls, compare actual returned values with the named independent reference.

Inject a recoverable allocation failure after native submission while work is demonstrably pending. Cover later output allocations and result-owner construction, including library allocations when the test mechanism supports them. Observe whether an owner is released before its dependent use finishes. Check a subsequent valid call when the interface promises recovery. Also run a failure before submission to distinguish cleanup defects from a broken setup.

Accept synchronous or asynchronous implementations when the caller allows them. Keep host allocation failure, device out-of-memory and fatal CUDA errors separate. A passing controlled host-allocation test does not establish recovery from the other failures. Report what was exercised and what remains untested.
