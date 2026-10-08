/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace test {

template <typename T, typename SourceElementT>
class lists_column_wrapper;

/**
 * @brief True when `Iterator` can be dereferenced and incremented.
 */
template <typename Iterator>
concept iterator_like = requires(Iterator i) {
  *i;
  ++i;
};

/**
 * @brief True when `Iterator` yields values convertible to `bool` and is not a string.
 *
 * Excludes string-like iterators (such as `char const*`) so that string values are not mistaken
 * for validity.
 */
template <typename Iterator>
concept validity_iterator =
  iterator_like<Iterator> && !std::is_convertible_v<Iterator, std::string_view> &&
  requires(Iterator i) {
    static_cast<bool>(*i);
    requires(!std::is_convertible_v<decltype(*i), std::string_view>);
  };

/**
 * @brief Host-side recursive initializer tree for constructing list columns with an
 * explicit stream and memory resources at every nesting level.
 *
 * Example:
 * @code{.cpp}
 * using LCW = cudf::test::lists_column_wrapper<int>;
 * LCW col{{{1, 2}, {3}}, stream, mr};
 * @endcode
 *
 * Leaf and nested constructors accept the existing validity iterators
 * (`valids`, `null_at(...)`, etc.) and materialize them into owned storage.
 *
 * @tparam T Host leaf element type (e.g. `int32_t` or `std::string`)
 */
template <typename T>
class lists_column_initializer {
 public:
  /**
   * @brief Host leaf element type.
   */
  using value_type = T;

  /**
   * @brief Construct an empty leaf. Avoids ambiguity between the leaf and nested
   * empty `initializer_list` constructors.
   */
  lists_column_initializer() = default;

  /**
   * @brief Construct a leaf from scalar values.
   *
   * @param values Leaf element values
   */
  template <typename Element>
  lists_column_initializer(std::initializer_list<Element> values)
    requires(std::is_convertible_v<Element, T>)
    : values_(values.begin(), values.end())
  {
  }

  /**
   * @brief Construct a leaf from two or more scalar values.
   *
   * @param first First leaf element value
   * @param rest Remaining leaf element values
   */
  template <typename First, typename... Rest>
  lists_column_initializer(First first, Rest... rest)
    requires(sizeof...(Rest) > 0 && std::is_convertible_v<First, T> &&
             (std::is_convertible_v<Rest, T> && ...))
    : values_{static_cast<T>(first), static_cast<T>(rest)...}
  {
  }

  /**
   * @brief Construct a leaf from an iterator range of values.
   *
   * @tparam InputIterator Iterator whose elements are convertible to `T`
   * @param begin Beginning of the leaf values
   * @param end End of the leaf values
   */
  template <iterator_like InputIterator>
  lists_column_initializer(InputIterator begin, std::type_identity_t<InputIterator> end)
    requires(std::is_constructible_v<T, std::iter_reference_t<InputIterator>>)
    : values_(begin, end)
  {
  }

  /**
   * @brief Construct a leaf or nested node with validity.
   *
   * @tparam ValidityIterator Iterator convertible to `bool`
   * @param values Leaf values or child initializers
   * @param v Validity iterator over the values or child rows
   */
  template <validity_iterator ValidityIterator>
  lists_column_initializer(lists_column_initializer values, ValidityIterator v)
    : lists_column_initializer(std::move(values).with_validity(v))
  {
  }

  /**
   * @brief Construct a leaf or nested node with validity from an initializer list.
   *
   * @tparam Validity Element type convertible to `bool`
   * @param values Leaf values or child initializers
   * @param validity Validity of each value or child row
   */
  template <typename Validity>
  lists_column_initializer(lists_column_initializer values,
                           std::initializer_list<Validity> validity)
    requires(validity_iterator<Validity const*>)
    : lists_column_initializer(std::move(values), validity.begin())
  {
  }

  /**
   * @brief Construct a nested node from child initializers.
   *
   * Scalar values deduce the leaf constructor's element type. Brace-enclosed
   * child lists cannot deduce that type and instead use this overload's default
   * `NestedInit`, preserving their nesting.
   *
   * @param children Child list initializers
   */
  template <typename NestedInit = lists_column_initializer>
  lists_column_initializer(std::initializer_list<NestedInit> children)
    requires(std::is_same_v<NestedInit, lists_column_initializer>)
    : children_{children.begin(), children.end()}, nested_{true}
  {
  }

  /**
   * @brief True if this node holds nested child initializers rather than leaf values.
   * @return Whether this node is nested
   */
  [[nodiscard]] bool nested() const { return nested_; }
  /**
   * @brief True if this row is valid (non-null) in its parent list.
   * @return Whether this row is valid
   */
  [[nodiscard]] bool valid() const { return valid_; }
  /**
   * @brief Leaf element values when `nested()` is false.
   * @return Reference to the leaf values
   */
  [[nodiscard]] auto const& values() const { return values_; }
  /**
   * @brief Per-element validity for leaf values; empty when all leaf values are valid.
   * @return Reference to the leaf validity mask
   */
  [[nodiscard]] auto const& value_validity() const { return value_validity_; }
  /**
   * @brief True if validity was explicitly provided.
   * @return Whether this initializer has explicit validity
   */
  [[nodiscard]] bool has_validity() const { return has_validity_; }
  /**
   * @brief Child initializers when `nested()` is true.
   * @return Reference to the child initializers
   */
  [[nodiscard]] auto const& children() const { return children_; }

 private:
  template <validity_iterator ValidityIterator>
  lists_column_initializer with_validity(ValidityIterator validity) &&
  {
    has_validity_ = true;
    if (nested_) {
      for (auto& child : children_) {
        child.valid_ = static_cast<bool>(*validity++);
      }
    } else {
      value_validity_.clear();
      value_validity_.reserve(values_.size());
      for (std::size_t i = 0; i < values_.size(); ++i) {
        value_validity_.push_back(static_cast<bool>(*validity++));
      }
    }
    return std::move(*this);
  }

  template <typename, typename>
  friend class lists_column_wrapper;

  std::vector<T> values_;
  std::vector<bool> value_validity_;
  std::vector<lists_column_initializer> children_;
  bool nested_{false};
  bool valid_{true};
  bool has_validity_{false};
};

}  // namespace test
}  // namespace CUDF_EXPORT cudf
