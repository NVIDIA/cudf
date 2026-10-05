/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
//
dictionary_column_view::dictionary_column_view(column_view const& dictionary_column)
  : column_view(dictionary_column)
{
  CUDF_EXPECTS(type().id() == type_id::DICTIONARY32,
               "dictionary_column_view only supports DICTIONARY type");
  if (size() > 0) CUDF_EXPECTS(num_children() == 2, "dictionary column has no children");
}

column_view dictionary_column_view::parent() const noexcept
{ return static_cast<column_view>(*this); }

column_view dictionary_column_view::indices() const noexcept { return child(0); }

column_view dictionary_column_view::get_indices_annotated() const noexcept
{ return {indices().type(), size(), indices().head(), null_mask(), null_count(), offset()}; }

column_view dictionary_column_view::keys() const noexcept { return child(1); }

size_type dictionary_column_view::keys_size() const noexcept
{ return (size() == 0) ? 0 : keys().size(); }

data_type dictionary_column_view::keys_type() const noexcept
{ return (size() == 0) ? data_type{type_id::EMPTY} : keys().type(); }

}  // namespace cudf
