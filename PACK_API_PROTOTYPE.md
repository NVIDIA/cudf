# Prepared non-chunked pack prototype

## Overview

| Item | Current position |
| --- | --- |
| Primary uses | Transient shuffle and spilling within a running job |
| Existing API | `cudf::pack()` and `cudf::unpack()` remain unchanged |
| Prototype API | Additive API under `cudf::experimental` |
| Representations | Uncompressed and per-region mixtures of nvCOMP Cascaded, Zstd, Snappy, or raw bytes |
| Compression unit | Independent physical column regions: data, validity, offsets, and string characters |
| Destinations | Caller-owned device-accessible memory, including mapped pinned-host memory |
| Lifetime | Job-local; temporary spill files are not durable storage |
| Compatibility assumption | Producer and consumer use compatible software in the same deployment |
| Explicitly excluded | At-rest persistence, cross-version compatibility, and chunked pack APIs |
| Remote branch | `prototype/nonchunked-pack-api` |

## API at a glance

| Stage | API or type | Purpose | Important behavior |
| --- | --- | --- | --- |
| Select behavior | `pack_options` | Choose compression, uniform codec options, and compressed-output policy | Defaults to `none`; supports libcudf-owned `automatic` and uniform explicit codecs; compressed output can be `compact` or `reserved` |
| Configure expert policy | `make_pack_plan_builder()` / `pack_plan_builder::regions()` | Discover the physical regions once and edit their codec settings before finalizing the plan | Region descriptions are immutable; codec, chunk size, minimum savings, and Cascaded parameters are mutable per region |
| Prepare | `prepare_pack(input, options, stream, temp_mr)` | Discover buffers, compute layout, and retain reusable state | Accepts a `table_view` for fused pack-plus-compress or ordinary uncompressed `packed_columns` for late compression; bound to the input and stream |
| Inspect capacity | `pack_plan::sizes()` | Report metadata size, destination capacity, alignment, and uncompressed size | Capacity is exact when uncompressed and the sum of aligned per-region nvCOMP upper bounds when compressed |
| Allocate | Caller-owned storage | Let shuffle or spill code choose the destination | Must be device-accessible and satisfy the reported capacity and alignment |
| Execute | `pack_into(plan, destination)` | Pack into caller-owned memory | Returns metadata, representation, output policy, and bytes to retain |
| Borrow | `unpack_view(packed_data_view)` | Reconstruct without copying | Available only for uncompressed payloads; returned view borrows both buffers |
| Own | `materialize(packed_data_view, stream, mr)` | Produce an owning cuDF table | Supports every implemented representation |

## Use-case examples

These examples are limited to the shuffle and transient-spill cases raised in the discussions. `allocate_mapped_pinned()` represents the caller's mapped pinned-host allocator; the returned storage must be device-accessible.

| Discussed use case | Recommended mode | Why |
| --- | --- | --- |
| Spill directly to host while minimizing temporary device memory | Uncompressed into mapped pinned host | Exact size is known during planning and no full-size output allocation is needed on the device |
| Enqueue a compressed spill without waiting for final compressed sizes | Cascaded, Zstd, or Snappy with `reserved` output | Caller allocates the reported bound once; `pack_into()` does not query final frame sizes |
| Send a compressed Spark shuffle block using only the bytes that must be transferred | Compressed with `compact` output | Returns the retained prefix size required for network framing, at the cost of per-region size synchronization |
| Receive a shuffle block or restore a spill into an owning GPU table | `materialize()` | Decompresses every region and returns a table independent of the transport buffers |
| Keep an uncompressed shuffle block device-resident and reconstruct without copying | `unpack_view()` | Returns a borrowing view over the packed destination |
| Emit the same prepared batch into another caller-owned destination | Reuse the same `pack_plan` | Reuses table discovery, region classification, layout, metadata, and codec setup while the input is unchanged |
| Decide to compress after ordinary packing has completed | `prepare_pack(packed_columns, options)` | Borrows the existing packed allocation as the codec input and avoids another packing copy |
| Let libcudf choose an appropriate codec for every region | `pack_compression::automatic` | Uses a throughput-oriented built-in policy and retains compact regions uncompressed when compression misses the configured savings threshold |
| Apply engine-specific prior knowledge | `make_pack_plan_builder()` | Expert callers inspect all regions, edit each `pack_region_options`, and consume the builder with `build()`; individual regions may force a codec, remain raw, or retain `automatic` |

### Example-to-request traceability

The buildable examples are in `cpp/examples/pack/pack_example.cu`. The table distinguishes behavior directly requested by a consumer from prototype design choices used to satisfy that request.

| Buildable example | Request or issue it demonstrates | Relationship | Sources |
| --- | --- | --- | --- |
| `direct_uncompressed_spill()` | Obtain the exact packed size, reserve memory first, and pack directly into caller-selected mapped pinned-host memory | Direct implementation of RP-1 through RP-5 | [cuDF #21321: packed table buffer size](https://github.com/NVIDIA/cudf/issues/21321), [pinned-host destination request](https://nvidia.slack.com/archives/C0773FR630B/p1759775752732259?thread_ts=1759773356.773689&cid=C0773FR630B), [direct-to-pinned discussion](https://nvidia.slack.com/archives/C06CUH0CHF1/p1788868412810599?thread_ts=1788541452.851109&cid=C06CUH0CHF1), [cuDF PR #23088: spill reservation alignment](https://github.com/NVIDIA/cudf/pull/23088) |
| `direct_uncompressed_spill()` restore half | Reconstruct an owning GPU table from a host-resident spill rather than return a view tied to transport storage | Direct response to the owning-reconstruction and spill/receive requirements, RP-17 and TX-1/TX-2 | [compressed spill/exchange use](https://nvidia.slack.com/archives/C045NBR9YKT/p1789574277074999?thread_ts=1789574099.680079&cid=C045NBR9YKT), [cuDF #20966: table buffers and metadata iterator](https://github.com/NVIDIA/cudf/issues/20966), [current view-only behavior](https://nvidia.slack.com/archives/CDTANRCTT/p1770066613741579?thread_ts=1770066280.180409&cid=CDTANRCTT) |
| `asynchronous_reserved_spill()` | Submit typed-region compression without waiting to discover final compressed sizes | `reserved` is a prototype design response to the requested optional compressed spill path and its unknown-size/asynchrony tradeoff; the exact enum and reserved-slot layout were not prescribed in the thread | [compressed spill/exchange request](https://nvidia.slack.com/archives/C045NBR9YKT/p1789574277074999?thread_ts=1789574099.680079&cid=C045NBR9YKT), [optional compressed-pack proposal](https://nvidia.slack.com/archives/C045NBR9YKT/p1789659501402019?thread_ts=1789574099.680079&cid=C045NBR9YKT), [type-homogeneous regions](https://nvidia.slack.com/archives/C045NBR9YKT/p1789654697615489?thread_ts=1789574099.680079&cid=C045NBR9YKT) |
| `compact_shuffle_block()` | Compress a complete transient shuffle/exchange payload and transfer only the actual retained bytes | Directly demonstrates the requested compressed spill/exchange use; `compact` is the prototype policy chosen when transport framing needs the actual byte count | [source Cascade-Next thread](https://nvidia.slack.com/archives/C045NBR9YKT/p1789574099680079), [compressed spill/exchange use](https://nvidia.slack.com/archives/C045NBR9YKT/p1789574277074999?thread_ts=1789574099.680079&cid=C045NBR9YKT), [codec-sensitive typed regions](https://nvidia.slack.com/archives/C045NBR9YKT/p1789654697615489?thread_ts=1789574099.680079&cid=C045NBR9YKT) |
| Retry in `compact_shuffle_block()` | Reuse prepared buffer discovery instead of traversing the same table hierarchy again | Direct implementation of RP-7; the retry is illustrative, while repeated execution into caller-owned destinations is the underlying capability | [prepared-state discussion](https://nvidia.slack.com/archives/C01CW5L51QC/p1776277366225799), [cuDF #21321](https://github.com/NVIDIA/cudf/issues/21321) |
| `device_resident_zero_copy()` | Preserve the current metadata-only, non-owning unpack behavior when an uncompressed payload remains device-resident | Direct implementation of RP-13 through RP-16 | [current unpack behavior](https://nvidia.slack.com/archives/CDTANRCTT/p1770066613741579?thread_ts=1770066280.180409&cid=CDTANRCTT), [lifetime requirement](https://nvidia.slack.com/archives/CDTANRCTT/p1770066793417679?thread_ts=1770066280.180409&cid=CDTANRCTT), [stream-ordering requirement](https://nvidia.slack.com/archives/CDTANRCTT/p1770068345445029?thread_ts=1770066280.180409&cid=CDTANRCTT) |
| `compress_existing_shuffle_block()` | Make a late compression decision after exchange has already received ordinary `cudf::packed_columns` | Direct implementation of the Velox exchange requirement | [exact Slack request](https://nvidia.slack.com/archives/C0773FR630B/p1790105847001459?thread_ts=1790102556.016479&cid=C0773FR630B) |
| `automatic_compressed_spill()` | Make good per-region codec choices without requiring shuffle/spill callers to understand physical packed regions | Implements the primary automatic-selection path discussed after the per-region request | [per-region codec request](https://nvidia.slack.com/archives/C0773FR630B/p1790120481822889?thread_ts=1790102556.016479&cid=C0773FR630B), [caller-policy discussion](https://nvidia.slack.com/archives/C0773FR630B/p1790125129881049?thread_ts=1790102556.016479&cid=C0773FR630B) |
| `expert_region_selection()` | Allow specialized callers to override individual physical regions using column identity, role, type, and size | Implements the rich manual interface while retaining automatic selection as the normal production path | [caller-policy discussion](https://nvidia.slack.com/archives/C0773FR630B/p1790125129881049?thread_ts=1790102556.016479&cid=C0773FR630B) |

The examples intentionally do not claim to demonstrate bounded chunked Spark spilling. That request is covered by SP-1 through SP-11 in the requirements document and remains a separate API track, beginning with the [strict memory-bound request](https://nvidia.slack.com/archives/C045NBR9YKT/p1789660770301469?thread_ts=1789574099.680079&cid=C045NBR9YKT) and [cuDF #21874](https://github.com/NVIDIA/cudf/issues/21874).

### Direct uncompressed spill to mapped pinned host

This is the low-device-memory path. Planning provides the exact host allocation size.

```cpp
auto const stream = cudf::get_default_stream();
auto plan         = cudf::experimental::prepare_pack(input, stream);

auto host_payload = allocate_mapped_pinned(plan.sizes().payload_bytes);
auto result       = cudf::experimental::pack_into(
  plan, cudf::device_span<uint8_t>{host_payload.device_data(), host_payload.size()});

// Wait before the CPU spill manager reads, transfers, or reuses the host allocation.
cudaStreamSynchronize(stream.get());
spill_store.put(result.metadata,
                std::span<uint8_t const>{host_payload.data(), result.payload_bytes},
                result.compression);
```

Restoring that transient spill into an owning GPU table uses the same host allocation directly if it remains mapped and device-accessible:

```cpp
auto packed = cudf::experimental::packed_data_view{
  saved_metadata,
  cudf::device_span<uint8_t const>{host_payload.device_data(), saved_payload_bytes},
  saved_compression};

auto restored = cudf::experimental::materialize(packed, stream);
```

### Automatic and expert per-region codec selection

Compression remains opt-in: default-constructed `pack_options` produce the ordinary uncompressed representation. The recommended compressed path lets libcudf select each physical region independently:

```cpp
auto options        = cudf::experimental::pack_options{};
options.compression = cudf::experimental::pack_compression::automatic;

auto plan = cudf::experimental::prepare_pack(input, options, stream);
```

The initial throughput-oriented policy leaves regions smaller than `automatic_min_region_bytes` raw, selects Snappy for string-character regions, selects Cascaded for other typed regions, and—in compact mode—retains a region raw when compression saves fewer than `automatic_min_savings_bytes`. The defaults are 4 KiB and 256 bytes respectively. They are prototype policy rather than a permanent API guarantee and should be tuned with representative workloads. Reserved mode cannot inspect final frame sizes without sacrificing its asynchronous contract, so its automatic decision uses type, role, and size but does not apply the post-compression savings fallback.

Specialized callers use a two-stage builder. Region discovery happens once; descriptions are immutable while the complete codec configuration remains editable until `build()` finalizes destination capacity and compressor state:

```cpp
auto builder = cudf::experimental::make_pack_plan_builder(input, options, stream);
for (auto& region : builder.regions()) {
  if (region.info.kind == cudf::experimental::pack_region_kind::validity) {
    region.options.codec = cudf::experimental::pack_compression::none;
  } else if (region.info.column_index == 0) {
    region.options.codec               = cudf::experimental::pack_compression::cascaded;
    region.options.cascaded_num_RLEs   = 1;
    region.options.cascaded_num_deltas = 2;
  }
}
auto plan = std::move(builder).build();
```

`pack_region_info` exposes the stable region index within the plan, owning top-level column index, physical role (`data`, `validity`, `offsets`, or `string_characters`), logical/native type, and uncompressed extent. `pack_region_options` controls the codec, nvCOMP chunk size, minimum automatic savings, and Cascaded transforms for that region. A concrete codec is forced; `none` stores raw bytes; `automatic` delegates that region to libcudf's policy. The same builder interface is available for an existing ordinary `packed_columns` allocation.

### Asynchronous compressed spill with reserved capacity

This matches a spill manager that prefers avoiding a final-size synchronization and can retain the upper-bound host allocation. The returned payload length is the complete reserved capacity, not the sum of actual frame sizes.

```cpp
auto options        = cudf::experimental::pack_options{};
options.compression = cudf::experimental::pack_compression::cascaded;
options.output_mode = cudf::experimental::compressed_output_mode::reserved;

auto plan         = cudf::experimental::prepare_pack(input, options, stream);
auto host_payload = allocate_mapped_pinned(plan.sizes().payload_bytes);
auto result       = cudf::experimental::pack_into(
  plan, cudf::device_span<uint8_t>{host_payload.device_data(), host_payload.size()});

// pack_into() has enqueued all typed-region frames without querying their final sizes.
// The allocation must remain alive until the stream completes.
record_completion_event(stream);
spill_store.put_when_ready(result.metadata,
                           host_payload,
                           result.payload_bytes,
                           result.compression);
```

### Compact compressed Spark shuffle block

This matches a shuffle sender that must know how many payload bytes to frame and transfer. `compact` is the default, but it is shown explicitly here because the synchronization tradeoff is intentional.

```cpp
auto options        = cudf::experimental::pack_options{};
options.compression = cudf::experimental::pack_compression::cascaded;
options.output_mode = cudf::experimental::compressed_output_mode::compact;

auto plan         = cudf::experimental::prepare_pack(input, options, stream);
auto host_payload = allocate_mapped_pinned(plan.sizes().payload_bytes);
auto result       = cudf::experimental::pack_into(
  plan, cudf::device_span<uint8_t>{host_payload.device_data(), host_payload.size()});

shuffle_writer.send(result.metadata,
                    std::span<uint8_t const>{host_payload.data(), result.payload_bytes},
                    result.compression);
```

The receiver retains the metadata, compression identifier, and exactly `payload_bytes` bytes. It can materialize directly from a device buffer or mapped pinned-host receive buffer:

```cpp
auto received = cudf::experimental::packed_data_view{
  received_metadata,
  cudf::device_span<uint8_t const>{received_payload, received_payload_bytes},
  received_compression};

auto table = cudf::experimental::materialize(received, stream);
```

### Device-resident uncompressed shuffle block

When neither compression nor host spilling is needed, the packed data can remain in caller-owned device memory and be exposed without reconstruction or allocation:

```cpp
auto plan = cudf::experimental::prepare_pack(input, stream);
rmm::device_buffer payload(plan.sizes().payload_bytes, stream);
auto result = cudf::experimental::pack_into(
  plan,
  cudf::device_span<uint8_t>{static_cast<uint8_t*>(payload.data()), payload.size()});

auto packed = cudf::experimental::packed_data_view{
  result.metadata,
  cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(payload.data()),
                                   result.payload_bytes}};
auto borrowed = cudf::experimental::unpack_view(packed);
```

`borrowed` must not outlive `result.metadata` or `payload`. Use `materialize()` instead when the reconstructed table must own its buffers.

### Reusing a plan for the same batch

This is the repeated-destination case from the planning requirement. It does not permit replacing the source table; it avoids rediscovering the same table when the same batch must be emitted again.

```cpp
auto plan = cudf::experimental::prepare_pack(input, options, stream);

auto first_result = cudf::experimental::pack_into(plan, first_destination);
auto retry_result = cudf::experimental::pack_into(plan, retry_destination);
```

Both executions use the stream captured by `plan`. The source table must remain alive and unchanged until both operations complete.

## Size and output model

| Representation | `sizes().payload_bytes` | `pack_result::payload_bytes` | Synchronization at end of `pack_into()` | Intermediate storage | Zero-copy view |
| --- | --- | --- | --- | --- | --- |
| Uncompressed | Exact destination size | Exact bytes written | None added by size reporting | None beyond planning scratch | Yes |
| Automatic/mixed, compact | Sum of aligned per-region raw sizes or selected-codec upper bounds | Compact sequence of independently tagged raw or compressed regions | Yes for selected codecs; required to apply the savings fallback and place the next frame | Direct source regions when already canonical; full staging fallback for transformed regions | No |
| Automatic/mixed, reserved | Sum of aligned per-region raw sizes or selected-codec upper bounds | Full reserved capacity | No; type/role/size selection only, without post-compression fallback | Direct source regions when already canonical; full staging fallback for transformed regions | No |
| Cascaded, compact | Combined regional nvCOMP upper bound | Compact sequence of typed-region frames | Yes, once per region to obtain each frame size | Direct source regions when already canonical; full staging fallback for transformed regions | No |
| Cascaded, reserved | Required nvCOMP upper bound | Full reserved capacity | No | Direct source regions when already canonical; full staging fallback for transformed regions | No |
| Zstd, compact | Combined regional nvCOMP upper bound | Compact sequence of region frames | Yes, once per region to obtain each frame size | Direct source regions when already canonical; full staging fallback for transformed regions | No |
| Zstd, reserved | Required nvCOMP upper bound | Full reserved capacity | No | Direct source regions when already canonical; full staging fallback for transformed regions | No |
| Snappy, compact | Combined regional nvCOMP upper bound | Compact sequence of region frames | Yes, once per region to obtain each frame size | Direct source regions when already canonical; full staging fallback for transformed regions | No |
| Snappy, reserved | Required nvCOMP upper bound | Full reserved capacity | No | Direct source regions when already canonical; full staging fallback for transformed regions | No |

## Non-chunked requirements tracker

| Area | Requirement | Status | Evidence or remaining gap |
| --- | --- | --- | --- |
| Compatibility | Preserve existing `pack()` and `unpack()` behavior | Met | Prototype is additive; the existing test suite continues to pass |
| API | Use one execution shape for compressed and uncompressed packing | Met | All modes use `prepare_pack()` and `pack_into()` |
| Planning | Avoid repeating table discovery and layout traversal | Met | `pack_plan` retains contiguous-layout state and metadata |
| Planning | Reuse a plan with multiple destinations | Met | Dedicated reuse tests pass for uncompressed, Cascaded, Zstd, and Snappy |
| Input form | Compress an existing uncompressed `cudf::packed_columns` after a caller makes a late compression decision | Met | The overload borrows the existing device allocation as the regional compression source; compact and reserved round trips cover Cascaded, Zstd, and Snappy ([Slack request](https://nvidia.slack.com/archives/C0773FR630B/p1790105847001459?thread_ts=1790102556.016479&cid=C0773FR630B)) |
| Destination | Write into caller-owned device memory | Met | Tested for uncompressed, Cascaded, Zstd, and Snappy |
| Destination | Write directly into mapped pinned-host memory | Met | Round trips pass for uncompressed, Cascaded, Zstd, and Snappy |
| Sizing | Report exact uncompressed size before allocation | Met | `payload_bytes` is exact for `none` |
| Sizing | Bound compressed storage before execution | Met | Planning reports nvCOMP's required upper-bound capacity |
| Sizing | Return actual compressed size | Met | `pack_into()` returns the actual byte count |
| Output policy | Let callers choose upper-bound/reserved or compact output | Met | `compressed_output_mode::{reserved, compact}` selects asynchronous retained capacity or synchronized actual prefix |
| Unpack | Provide zero-copy uncompressed reconstruction | Met | `unpack_view()` returns a borrowing `table_view` |
| Unpack | Provide owning reconstruction | Met | `materialize()` handles all implemented representations |
| Compression | Preserve uncompressed packing as the default | Met | Default-constructed `pack_options` use `pack_compression::none` |
| Compression | Provide a good automatic per-region default | Met for prototype policy | `automatic` selects Cascaded for sufficiently large typed/offset/validity regions, Snappy for string characters, leaves small regions raw, and applies a configurable compact-output savings fallback |
| Compression | Permit uniform explicit codec selection | Met | `pack_options` selects `cascaded`, `zstd`, or `snappy` for every region |
| Compression | Permit expert per-region codec selection | Met | `pack_plan_builder::regions()` exposes immutable region identity plus mutable codec, chunk-size, savings, and Cascaded settings; mixed raw/Cascaded/Zstd/Snappy round trips pass for both `table_view` and existing `packed_columns` inputs ([Slack request](https://nvidia.slack.com/archives/C0773FR630B/p1790120481822889?thread_ts=1790102556.016479&cid=C0773FR630B), [policy discussion](https://nvidia.slack.com/archives/C0773FR630B/p1790125129881049?thread_ts=1790102556.016479&cid=C0773FR630B)) |
| Compression | Support nvCOMP Cascaded | Met | Device and mapped pinned-host round trips pass |
| Compression | Support Zstd and Snappy | Met | Device and mapped pinned-host round trips pass for both codecs |
| Compression | Support Cascade-Next | Unmet | Installed nvCOMP 5.3 exposes Cascaded but no distinct Cascade-Next API |
| Compression | Compress per column or native-typed region | Met | Each physical column buffer is an independent frame; Cascaded uses signed/unsigned native widths for supported types, with byte fallback for unsupported 128-bit or structural regions |
| Pack memory | Avoid a full uncompressed device staging buffer while compressing | Partial | Canonical source regions are compressed directly, eliminating staging for the measured unsliced fixed-width workload. A full staging fallback remains when any region requires offset rebasing, validity-bit shifting, or padding normalization. |
| Restore memory | Avoid a full uncompressed device staging buffer while decompressing | Met | `materialize()` allocates the final column hierarchy and decompresses each region directly into its owning data or validity buffer |
| Types | Cover nested, sliced, dictionary, empty, and zero-column tables | Met | Every existing table-shape case now round trips through Cascaded, Zstd, and Snappy as well as uncompressed packing |
| Validation | Fail safely on invalid runtime input | Met for agreed scope | Size, alignment, metadata, codec/header mismatch, truncated payload, and nvCOMP failures are checked; persistence-grade integrity is out of scope |
| Asynchrony | Avoid synchronizing compressed execution | Met when requested | `reserved` returns after enqueueing compression without querying final size; `compact` intentionally synchronizes for the actual prefix size |
| Performance | Meet or improve current pack/copy performance and peak memory | Unmet | Cascaded reduces the measured 64 MiB payload about 4x and restores faster than legacy, but its combined pack-and-restore time remains about 46-55% slower; Zstd and Snappy are slower still |
| Other use | Provide a CPU/Kudo-compatible representation | Separate API | Different ownership and reconstruction requirements |
| Other use | Enumerate and reconstruct native table buffers directly | Separate API | Not the contiguous regular-pack representation |

### Late compression of an existing packed representation

Velox exchange currently decides whether to compress only after it has received an existing uncompressed `cudf::packed_columns`. The API therefore supports both fused `table_view` pack-plus-compress and late compression of an ordinary packed allocation. ([exact Slack request](https://nvidia.slack.com/archives/C0773FR630B/p1790105847001459?thread_ts=1790102556.016479&cid=C0773FR630B))

An additive overload is the smallest extension:

```cpp
pack_plan prepare_pack(
    cudf::packed_columns const& input,
    pack_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);
```

For this overload, planning creates a metadata-only `table_view` over the ordinary packed allocation to identify typed physical regions. It does not copy or repack the payload. `pack_into()` applies the selected encoding directly from the existing packed device allocation into the caller-owned destination. The input metadata and device allocation must remain alive until submitted work completes. The overload rejects `pack_compression::none`; it is specifically a late-compression entry point, not a general re-encoding API.

This overlaps with the fused path after region discovery: both should use the same codec adapters, output layouts, result metadata, and restore APIs. The only difference is whether physical regions originate from a `table_view` or from an existing packed allocation.

## Runtime contract

| Object or operation | Contract |
| --- | --- |
| `pack_plan` | Bound to the input table and preparation stream |
| Source table | Buffers remain alive and unchanged until all plan work completes |
| Destination | Remains alive until work on the plan's stream completes |
| Reuse | Calls using one plan are ordered on its captured stream |
| `unpack_view()` | Returned view must not outlive its metadata or payload |
| `materialize()` | Returned table owns its data independently of packed buffers |
| Compressed planning | Borrows canonical physical source regions directly. If any region requires normalization, the current implementation falls back to one full uncompressed staging buffer. |
| Compressed execution | `compact` queries each frame's actual size to place the next frame; `reserved` launches every frame into a planned slot without querying final sizes |
| Compressed metadata | A host-side region directory wraps the existing pack metadata and records each region's actual codec (including raw), logical type, validity role, uncompressed extent, and retained payload extent |
| Runtime validation | Must reject invalid sizes, alignment, codec identifiers, metadata bounds, and nvCOMP failures safely |
| Integrity | Persistence-grade checksums and recovery are not required; transport checksums belong to the shuffle layer when needed |

## Out-of-scope tracker

| Item | Reason |
| --- | --- |
| Bounded sequential chunk iterator | Covered by the separate chunked-pack design |
| Independently executable parallel chunks | Covered by the separate parallel/chunked design |
| Durable at-rest format | Shuffle and spill data is transient |
| Cross-version wire compatibility | Producer and consumer are assumed to use a compatible deployment |
| Persistence-grade corruption recovery | Not required for transient data; basic safe failure remains required |

## Verification

| Check | Result |
| --- | --- |
| Focused target | `COPYING_TEST` builds successfully |
| Complete suite | 43 enabled `PackUnpackTest` tests pass |
| Existing disabled test | One pre-existing test remains disabled |
| Uncompressed destination | Device and mapped pinned-host round trips pass |
| Cascaded | Device and mapped pinned-host round trips pass |
| Zstd | Device and mapped pinned-host round trips pass |
| Snappy | Device and mapped pinned-host round trips pass |
| Automatic selection | Mixed Cascaded/Snappy output and forced raw fallback round trips pass |
| Expert selection | Mixed raw/Cascaded/Zstd/Snappy output round trips pass and exposes expected column, role, type, and size metadata |
| Reuse | Repeated execution into different destinations passes for every codec |
| Complex tables | Compressed and uncompressed round trips pass for fixed-width, strings, lists, structs, nested, sliced, dictionary, empty, zero-column, and long-offset cases |
| Typed regions | A mixed nullable `int16`/`int64`/`float32`/string test verifies distinct typed data, validity, character, and offset regions before round trip |
| Reserved regions | Multi-region reserved-output round trips pass for Cascaded, Zstd, and Snappy |
| Output policy | Compact and reserved compressed results pass for every codec |
| Error handling | Undersized and misaligned destinations, codec/header mismatch, and truncated compressed input are rejected |
| Benchmark target | `PACK_NVBENCH` builds and compares legacy, prepared uncompressed, automatic, Cascaded, Zstd, and Snappy paths, including direct legacy `unpack()` versus prepared `unpack_view()` |
| Buildable example | `pack_example` builds and runs automatic and expert-selection paths successfully |
| Formatting | `git diff --check` passes |

Test command:

```bash
cpp/build/gtests/COPYING_TEST --gtest_filter=PackUnpackTest.\*
```

### Pack, compression, and restore benchmark

The `PACK_NVBENCH` target now contains six benchmarks:

| Benchmark | Timed operation |
| --- | --- |
| `pack_to_pinned_host` | Legacy `pack()` plus D2H, or prepared `pack_into()` directly into mapped pinned-host memory |
| `pack_to_device` | Legacy `pack()` or prepared `pack_into()` into device memory; includes compression when selected |
| `encode_existing_pack_to_device` | Encode an already-created ordinary `packed_columns` allocation without timing its initial pack |
| `restore_from_pinned_host` | Legacy H2D plus `unpack()` plus an owning table copy, or prepared `materialize()` directly from mapped pinned-host memory |
| `restore_from_device` | Construct an owning table from an already device-resident packed payload |
| `device_unpack_view` | Legacy `unpack()` versus prepared uncompressed `unpack_view()` over an already device-resident payload; both return borrowing views and do not copy column data |

The restore comparison intentionally produces an owning table in every case. Timing only legacy `unpack()` would compare a borrowing metadata view against compressed decompression and would not represent shuffle receive or spill restoration.

Test configuration: one A100 80 GB PCIe; 64 MiB across four `int32` columns; 20 samples for compact results; cardinality 16 and the generator's high-cardinality setting. Network or disk transfer time is not included.

The benchmark implementation label for the new no-compression path is `prepared-uncompressed`. It uses `prepare_pack()` plus `pack_into()` for packing and either `unpack_view()` for a borrowing view or `materialize()` for an owning restore.

#### Compact output

| Path | Cardinality | Pack to host | Restore owning table | Combined | Retained payload | Uncompressed / retained | Pack peak | Restore peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy `pack()`/`unpack()` path | 16 | 3.182 ms | 2.724 ms | 5.906 ms | 64.000 MiB | 1.00x | 64.01 MiB | 128.00 MiB |
| Prepared uncompressed | 16 | 18.051 ms | 2.644 ms | 20.695 ms | 64.000 MiB | 1.00x | 0.006 MiB | 64.00 MiB |
| Cascaded typed regions | 16 | 7.350 ms | 1.380 ms | 8.730 ms | 15.846 MiB | 4.04x | 64.00 MiB | 64.00 MiB |
| Zstd regions | 16 | 51.984 ms | 19.186 ms | 71.170 ms | 7.861 MiB | 8.14x | 64.00 MiB | 64.00 MiB |
| Snappy regions | 16 | 20.045 ms | 2.130 ms | 22.175 ms | 9.816 MiB | 6.52x | 64.00 MiB | 64.00 MiB |
| Legacy `pack()`/`unpack()` path | High | 3.163 ms | 2.710 ms | 5.873 ms | 64.000 MiB | 1.00x | 64.01 MiB | 128.00 MiB |
| Prepared uncompressed | High | 18.912 ms | 2.635 ms | 21.547 ms | 64.000 MiB | 1.00x | 0.006 MiB | 64.00 MiB |
| Cascaded typed regions | High | 7.775 ms | 1.330 ms | 9.105 ms | 16.290 MiB | 3.93x | 64.00 MiB | 64.00 MiB |
| Zstd regions | High | 49.086 ms | 16.799 ms | 65.885 ms | 15.681 MiB | 4.08x | 64.00 MiB | 64.00 MiB |
| Snappy regions | High | 26.382 ms | 3.310 ms | 29.692 ms | 25.798 MiB | 2.48x | 64.00 MiB | 64.00 MiB |

#### Direct uncompressed unpack

This isolates the non-owning unpack operation from host transfer and owning-table construction. It uses 100 samples. Because both calls only parse host metadata and build a `table_view`, CPU time is the relevant measurement; their reported GPU timings are harness overhead rather than data movement.

| API | Cardinality | CPU time | Relative to legacy | Copies or device allocation |
| --- | ---: | ---: | ---: | --- |
| Legacy `unpack()` | 16 | 21.379 us | 1.00x | None |
| Prepared `unpack_view()` | 16 | 20.475 us | 0.96x | None |
| Legacy `unpack()` | High | 20.425 us | 1.00x | None |
| Prepared `unpack_view()` | High | 20.544 us | 1.01x | None |

The difference is within normal benchmark noise. For device-resident uncompressed payloads, the prepared API retains the legacy metadata-only, zero-copy unpack behavior.

#### Device-resident pack and owning restore

This removes host transfer from both halves of the operation. The table reports GPU time over 20 samples. `Restore` deliberately constructs an independent owning table for every representation; applications that can borrow the uncompressed packed allocation should use the approximately 20 us `unpack_view()` result above instead.

| Path | Cardinality | Pack to device | Restore owning table | Combined | Retained payload | Pack peak | Restore peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy `pack()`/`unpack()` path | 16 | 0.361 ms | 0.104 ms | 0.465 ms | 64.000 MiB | 64.01 MiB | 64.00 MiB |
| Prepared uncompressed | 16 | 0.187 ms | 0.104 ms | 0.291 ms | 64.000 MiB | 64.01 MiB | 64.00 MiB |
| Automatic per-region policy | 16 | 1.316 ms | Not remeasured | — | 15.846 MiB | 64.03 MiB | — |
| Cascaded typed regions | 16 | 1.291 ms | 1.132 ms | 2.423 ms | 15.846 MiB | 64.03 MiB | 64.00 MiB |
| Zstd regions | 16 | 48.921 ms | 16.872 ms | 65.793 ms | 7.861 MiB | 64.04 MiB | 64.00 MiB |
| Snappy regions | 16 | 16.482 ms | 2.249 ms | 18.731 ms | 9.816 MiB | 74.73 MiB | 64.00 MiB |
| Legacy `pack()`/`unpack()` path | High | 0.367 ms | 0.106 ms | 0.473 ms | 64.000 MiB | 64.01 MiB | 64.00 MiB |
| Prepared uncompressed | High | 0.187 ms | 0.104 ms | 0.291 ms | 64.000 MiB | 64.01 MiB | 64.00 MiB |
| Automatic per-region policy | High | 1.326 ms | Not remeasured | — | 16.290 MiB | 64.03 MiB | — |
| Cascaded typed regions | High | 1.331 ms | 1.080 ms | 2.411 ms | 16.290 MiB | 64.03 MiB | 64.00 MiB |
| Zstd regions | High | 42.889 ms | 13.602 ms | 56.491 ms | 15.681 MiB | 64.04 MiB | 64.00 MiB |
| Snappy regions | High | 16.278 ms | 3.218 ms | 19.496 ms | 25.798 MiB | 74.73 MiB | 64.00 MiB |

For device-resident data, prepared uncompressed is about 8.3x faster end to end than Cascaded and about 1.6x faster than the legacy owning path on this workload. Compression only becomes competitive when its smaller retained allocation or a later host, network, or storage transfer has enough value to repay the codec cost.

On this fixed-width workload, the automatic policy selects Cascaded for every data region. Its measured device-pack time and retained size are effectively identical to explicitly selecting Cascaded, so the policy and per-region dispatch add no measurable cost here. The mixed-type automatic behavior is covered separately by the unit tests and example.

The compressed `table_view` path now reads canonical physical regions directly from the input columns. Compared with the former full-staging implementation, this reduces pack peak memory from 128.03 MiB to 64.03 MiB for Cascaded, from 128.04 MiB to 64.04 MiB for Zstd, and from 138.72 MiB to 74.73 MiB for Snappy. Cascaded pack time improves by 17% (1.560 to 1.291 ms at cardinality 16, and 1.597 to 1.331 ms at high cardinality). Zstd and Snappy improve by 1–2% because codec time dominates their former packing copy. These measurements use unsliced, non-nullable fixed-width columns; inputs requiring normalization still take the staging fallback.

#### Reserved output

Reserved mode was measured with five samples. These retained sizes are allocation bounds, not actual compressed sizes. Because the benchmark waits for GPU completion to measure throughput, it does not capture reserved mode's host-submission/asynchrony benefit.

| Codec | Reserved capacity | Pack, cardinality 16 / high | Restore, cardinality 16 / high |
| --- | ---: | ---: | ---: |
| Cascaded | 64.024 MiB | 7.063 / 7.273 ms | 1.226 / 1.328 ms |
| Zstd | 64.032 MiB | 51.700 / 49.354 ms | 17.132 / 13.995 ms |
| Snappy | 74.719 MiB | 20.572 / 26.251 ms | 2.065 / 3.272 ms |

#### Interpretation

| Observation | Consequence |
| --- | --- |
| Cascaded gives roughly 4x reduction and 1.2 ms restoration | It is the strongest current candidate for transient shuffle/spill, but pack latency still leaves the combined path slower than legacy before transfer savings are counted |
| Zstd gives the smallest compact payload for low-cardinality data | Its approximately 52 ms pack cost is too high for the primary transient-data path in this implementation |
| Snappy restores quickly but region compression is slow | It does not currently justify itself against Cascaded on either measured distribution |
| Direct uncompressed host output uses almost no additional device memory | Its mapped-host write path is much slower than legacy device packing plus D2H |
| Compact output materially reduces retained and transferable bytes | Reserved output preserves asynchrony but retaining codec upper bounds eliminates the storage/transfer reduction unless the caller later compacts it |
| Compressed restore now peaks at 64 MiB | `materialize()` allocates the final column hierarchy and decompresses each packed region directly into its owning buffer; the former full-table staging allocation is gone |

The benchmark does not yet cover strings, nullable columns, nested schemas, different batch sizes, PCIe/network transfer time, or concurrent shuffle streams. Those are necessary before choosing a default codec or output policy.

## Next work

| Priority | Work | Tracker rows addressed |
| --- | --- | --- |
| 1 | Add a caller-selectable host-transfer strategy: direct mapped-host output versus device staging plus D2H | Performance versus peak-memory tradeoff |
| 2 | Batch or parallelize region execution and choose codecs by region type/size | Per-region compression latency and effectiveness |
| 3 | Replace the all-or-nothing fallback with selective scratch only for regions requiring offset rebasing, validity-bit shifting, or padding normalization | Eliminate full staging for sliced and nested inputs as well |
| 4 | Extend benchmarks to mixed, nullable, string, and nested tables plus representative shuffle block sizes | Workload coverage |
| 5 | Revisit Cascade-Next when nvCOMP exposes a distinct supported API | Cascade-Next requirement |
