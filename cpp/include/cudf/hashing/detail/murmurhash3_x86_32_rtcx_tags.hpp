/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cudf::hashing::detail::rtcx_murmur {

struct fragment_tag_entry_int32 {};
struct fragment_tag_entry_generic {};
struct fragment_tag_hasher_int32 {};
struct fragment_tag_entry_string {};
struct fragment_tag_hasher_string {};
struct fragment_tag_dictionary_string_entry_dictionary_string {};
struct fragment_tag_dictionary_string_hasher_dictionary_string {};

}  // namespace cudf::hashing::detail::rtcx_murmur
