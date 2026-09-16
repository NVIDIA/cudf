/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "insert.cuh"

namespace cudf::groupby {

template streaming_groupby::impl::batch_insert_result
streaming_groupby::impl::probe_and_insert_impl<false>(table_view const& batch_keys,
                                                      cuda::stream_ref stream);

}  // namespace cudf::groupby
