/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Keep the embedded copy independent of tracing callbacks supplied by other loaded libraries.
extern "C" {
void ZSTD_trace_compress_begin() __attribute__((weak, visibility("hidden")));
void ZSTD_trace_compress_end() __attribute__((weak, visibility("hidden")));
void ZSTD_trace_decompress_begin() __attribute__((weak, visibility("hidden")));
void ZSTD_trace_decompress_end() __attribute__((weak, visibility("hidden")));
}

namespace {
using hook = void (*)();
// [[gnu::used]] preserves this otherwise-unused array so its weak symbol references reach the
// linker, which merges their hidden visibility with Zstd's tracing references.
[[gnu::used]] hook const disabled_hooks[] = {ZSTD_trace_compress_begin,
                                             ZSTD_trace_compress_end,
                                             ZSTD_trace_decompress_begin,
                                             ZSTD_trace_decompress_end};
}  // namespace
