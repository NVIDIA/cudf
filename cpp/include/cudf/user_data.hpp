/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {

/// @brief Opaque pointer type for user-defined data.
using voidptr_t = void*;

/**
 * @brief A non-owning pointer to user-defined data passed to a UDF.
 *
 * A user-data input maps to a `void*` parameter at its specified input position in the UDF
 * signature. The same pointer is passed for every row.
 *
 * The caller owns the referenced storage and must keep it accessible to device code until
 * all operations using it have completed on the execution stream.
 */
struct user_data {
  voidptr_t ptr = nullptr;  ///< User-defined data pointer; may be null.
};

}  // namespace CUDF_EXPORT cudf
