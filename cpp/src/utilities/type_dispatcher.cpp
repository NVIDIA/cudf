/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {

std::string type_to_name(data_type type)
{
  if (type.id() == type_id::EMPTY) { return "EMPTY"; }
  return type_dispatcher(type, type_to_name_impl{});
}

}  // namespace cudf
