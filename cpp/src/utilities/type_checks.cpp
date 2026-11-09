/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>

namespace cudf {
namespace {

struct columns_equal_fn {
  template <typename T>
  bool operator()(column_view const& lhs, column_view const& rhs)
  {
    return lhs.type() == rhs.type();
  }
};

template <>
bool columns_equal_fn::operator()<dictionary32>(column_view const& lhs, column_view const& rhs)
{
  if (not cudf::is_dictionary(rhs.type())) { return false; }
  auto const kidx = dictionary_column_view::keys_column_index;
  return lhs.num_children() > 0 and rhs.num_children() > 0
           ? lhs.child(kidx).type() == rhs.child(kidx).type()
           : lhs.is_empty() and rhs.is_empty();
}

template <>
bool columns_equal_fn::operator()<list_view>(column_view const& lhs, column_view const& rhs)
{
  if (rhs.type().id() != type_id::LIST) { return false; }
  auto const& ci = lists_column_view::child_column_index;
  return have_same_types(lhs.child(ci), rhs.child(ci));
}

template <>
bool columns_equal_fn::operator()<struct_view>(column_view const& lhs, column_view const& rhs)
{
  if (rhs.type().id() != type_id::STRUCT) { return false; }
  return std::equal(lhs.child_begin(),
                    lhs.child_end(),
                    rhs.child_begin(),
                    rhs.child_end(),
                    [](auto const& lhs, auto const& rhs) { return have_same_types(lhs, rhs); });
}

struct column_scalar_equal_fn {
  template <typename T>
  bool operator()(column_view const& col, scalar const& slr)
  {
    return col.type() == slr.type();
  }
};

template <>
bool column_scalar_equal_fn::operator()<dictionary32>(column_view const& col, scalar const& slr)
{
  // It is not possible to have a scalar dictionary, so compare the dictionary
  // column keys type to the scalar type.
  auto col_keys = cudf::dictionary_column_view(col).keys();
  return have_same_types(col_keys, slr);
}

template <>
bool column_scalar_equal_fn::operator()<list_view>(column_view const& col, scalar const& slr)
{
  if (slr.type().id() != type_id::LIST) { return false; }
  auto const& ci      = lists_column_view::child_column_index;
  auto const list_slr = static_cast<list_scalar const*>(&slr);
  return have_same_types(col.child(ci), list_slr->view());
}

template <>
bool column_scalar_equal_fn::operator()<struct_view>(column_view const& col, scalar const& slr)
{
  if (slr.type().id() != type_id::STRUCT) { return false; }
  auto const struct_slr = static_cast<struct_scalar const*>(&slr);
  auto const slr_tbl    = struct_slr->view();
  return std::equal(col.child_begin(),
                    col.child_end(),
                    slr_tbl.begin(),
                    slr_tbl.end(),
                    [](auto const& lhs, auto const& rhs) { return have_same_types(lhs, rhs); });
}

struct scalars_equal_fn {
  template <typename T>
  bool operator()(scalar const& lhs, scalar const& rhs)
  {
    return lhs.type() == rhs.type();
  }
};

template <>
bool scalars_equal_fn::operator()<list_view>(scalar const& lhs, scalar const& rhs)
{
  if (rhs.type().id() != type_id::LIST) { return false; }
  auto const list_lhs = static_cast<list_scalar const*>(&lhs);
  auto const list_rhs = static_cast<list_scalar const*>(&rhs);
  return have_same_types(list_lhs->view(), list_rhs->view());
}

template <>
bool scalars_equal_fn::operator()<struct_view>(scalar const& lhs, scalar const& rhs)
{
  if (rhs.type().id() != type_id::STRUCT) { return false; }
  auto const tbl_lhs = static_cast<struct_scalar const*>(&lhs)->view();
  auto const tbl_rhs = static_cast<struct_scalar const*>(&rhs)->view();
  return have_same_types(tbl_lhs, tbl_rhs);
}

};  // namespace

// Implementation note: avoid using double dispatch for this function
// as it increases code paths to NxN for N types.
bool have_same_types(column_view const& lhs, column_view const& rhs)
{
  return type_dispatcher(lhs.type(), columns_equal_fn{}, lhs, rhs);
}

bool have_same_types(column_view const& lhs, scalar const& rhs)
{
  return type_dispatcher(lhs.type(), column_scalar_equal_fn{}, lhs, rhs);
}

bool have_same_types(scalar const& lhs, column_view const& rhs)
{
  return have_same_types(rhs, lhs);
}

bool have_same_types(scalar const& lhs, scalar const& rhs)
{
  return type_dispatcher(lhs.type(), scalars_equal_fn{}, lhs, rhs);
}

bool have_same_types(table_view const& lhs, table_view const& rhs)
{
  return std::ranges::equal(lhs, rhs, [](column_view const& lcol, column_view const& rcol) {
    return have_same_types(lcol, rcol);
  });
}

bool column_types_equivalent(column_view const& lhs, column_view const& rhs)
{
  // Check if the columns have fixed point types. This is the only case where
  // type equality and equivalence differ.
  if (cudf::is_fixed_point(lhs.type())) { return lhs.type().id() == rhs.type().id(); }
  return have_same_types(lhs, rhs);
}

}  // namespace cudf
