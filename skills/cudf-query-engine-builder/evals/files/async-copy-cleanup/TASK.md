# Repair the engine adapter

The user of this adapter is a query-engine engineer implementing a first backend proof of concept. Complete `adapter.hpp`, compile it, and run the supplied checks. Do not change `sim_backend.hpp`, `test_adapter.cpp`, or `run_checks.py`. Preserve the public function signatures and returned values. You may change the implementation and result members in `adapter.hpp`.

This fixture is a deterministic CPU simulation of the host integration boundary. It does not compile or execute CUDA or cuDF. A passing result establishes only the ownership, ordering, result selection and failure behavior exercised here. Native compilation, GPU execution and comparison against the engine's CPU result remain separate acceptance requirements.

The operation selects non-null signed 64-bit values strictly greater than `cutoff`, then returns their sum and count. Test inputs cannot overflow signed 64-bit arithmetic. An empty or all-null selection returns sum 0 and count 0.

## Caller requirements

`execute` is synchronous. A successful call must finish the requested backend upload and aggregation, return that operation's answer, and release temporary buffers without invalid uses. The caller permits completion waits before return or error propagation. CPU fallback is not permitted for this entry point.

The caller transfers an `Input` value and may retain no other reference to its objects. The backend can throw `std::bad_alloc` while allocating the staging buffer, allocating the aggregate output, or constructing the final result wrapper. Propagate the original allocation failure and leave the engine able to execute the next valid call. The submitted operation order in the starter models an existing engine boundary: upload, allocate aggregate output, submit aggregation, construct result wrapper. Preserve that order.

`sim_backend.hpp` defines the available interface. Its queued operations deliberately borrow their buffers and contexts. Queueing work does not transfer ownership. Its audit detects early buffer release, use of an expired allocation resource, abandoned stream work, and pending uses that remain after the call.

## Run

```sh
# Run CPU ownership checks inside the isolated evaluation environment.
python3 run_checks.py
```

Run this command in the disposable Linux evaluation environment provided for this task. The runner needs Python 3 and a system C++20 compiler (`/usr/bin/g++` or `/usr/bin/clang++`). It compiles the adapter with the fixed test caller in a private temporary directory, supplies a restricted child environment and bounds time, memory and output. The runner itself is not a sandbox; the execution environment must restrict filesystem/network access and keep credentials out of reach of submitted code. It prints JSON with one result for each condition and exits 0 only when every condition passes.

The supplied checks cover nullable and empty inputs, a failure before submission, output allocation failure after submission, result-wrapper failure, and a subsequent valid call after each failure. Report your changes, the exact command, its observed result, and the native acceptance checks still needed.
