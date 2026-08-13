/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/string_parsing.hpp"
#include "io/utilities/trie.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <cstddef>
#include <string>
#include <vector>

using cudf::io::parse_options;
using cudf::io::detail::infer_data_type;

// Base test fixture for tests
struct TypeInference : public cudf::test::BaseFixture {};

TEST_F(TypeInference, Basic)
{
  auto const stream = cudf::get_default_stream();

  auto options       = parse_options{',', '\n', '\"'};
  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data      = R"json([42,52,5])json";
  auto d_data           = cudf::make_string_scalar(data);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_data);

  auto const string_offset   = std::vector<cudf::size_type>{1, 4, 7};
  auto const string_length   = std::vector<cudf::size_type>{2, 2, 1};
  auto const d_string_offset = cudf::detail::make_device_uvector_async(
    string_offset, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const d_string_length = cudf::detail::make_device_uvector_async(
    string_length, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_col_strings = cuda::make_zip_iterator(
    cuda::std::make_tuple(d_string_offset.begin(), d_string_length.begin()));

  auto res_type =
    infer_data_type(options.json_view(),
                    {d_string_scalar.data(), static_cast<std::size_t>(d_string_scalar.size())},
                    d_col_strings,
                    string_offset.size(),
                    stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::INT64});
}

TEST_F(TypeInference, Null)
{
  auto const stream = cudf::get_default_stream();

  auto options       = parse_options{',', '\n', '\"'};
  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data      = R"json([52,5])json";
  auto d_data           = cudf::make_string_scalar(data);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_data);

  auto const string_offset   = std::vector<cudf::size_type>{1, 1, 4};
  auto const string_length   = std::vector<cudf::size_type>{0, 2, 1};
  auto const d_string_offset = cudf::detail::make_device_uvector_async(
    string_offset, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const d_string_length = cudf::detail::make_device_uvector_async(
    string_length, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_col_strings = cuda::make_zip_iterator(
    cuda::std::make_tuple(d_string_offset.begin(), d_string_length.begin()));

  auto res_type =
    infer_data_type(options.json_view(),
                    {d_string_scalar.data(), static_cast<std::size_t>(d_string_scalar.size())},
                    d_col_strings,
                    string_offset.size(),
                    stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::INT64});
}

TEST_F(TypeInference, AllNull)
{
  auto const stream = cudf::get_default_stream();

  auto options       = parse_options{',', '\n', '\"'};
  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data      = R"json([null])json";
  auto d_data           = cudf::make_string_scalar(data);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_data);

  auto const string_offset   = std::vector<cudf::size_type>{1, 1, 1};
  auto const string_length   = std::vector<cudf::size_type>{0, 0, 4};
  auto const d_string_offset = cudf::detail::make_device_uvector_async(
    string_offset, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const d_string_length = cudf::detail::make_device_uvector_async(
    string_length, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_col_strings = cuda::make_zip_iterator(
    cuda::std::make_tuple(d_string_offset.begin(), d_string_length.begin()));

  auto res_type =
    infer_data_type(options.json_view(),
                    {d_string_scalar.data(), static_cast<std::size_t>(d_string_scalar.size())},
                    d_col_strings,
                    string_offset.size(),
                    stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::INT8});  // INT8 if all nulls
}

TEST_F(TypeInference, String)
{
  auto const stream = cudf::get_default_stream();

  auto options       = parse_options{',', '\n', '\"'};
  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data      = R"json(["1990","8","25"])json";
  auto d_data           = cudf::make_string_scalar(data);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_data);

  auto const string_offset   = std::vector<cudf::size_type>{1, 8, 12};
  auto const string_length   = std::vector<cudf::size_type>{6, 3, 4};
  auto const d_string_offset = cudf::detail::make_device_uvector_async(
    string_offset, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const d_string_length = cudf::detail::make_device_uvector_async(
    string_length, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_col_strings = cuda::make_zip_iterator(
    cuda::std::make_tuple(d_string_offset.begin(), d_string_length.begin()));

  auto res_type =
    infer_data_type(options.json_view(),
                    {d_string_scalar.data(), static_cast<std::size_t>(d_string_scalar.size())},
                    d_col_strings,
                    string_offset.size(),
                    stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::STRING});
}

TEST_F(TypeInference, Bool)
{
  auto const stream = cudf::get_default_stream();

  auto options       = parse_options{',', '\n', '\"'};
  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data      = R"json([true,false,false])json";
  auto d_data           = cudf::make_string_scalar(data);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_data);

  auto const string_offset   = std::vector<cudf::size_type>{1, 6, 12};
  auto const string_length   = std::vector<cudf::size_type>{4, 5, 5};
  auto const d_string_offset = cudf::detail::make_device_uvector_async(
    string_offset, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const d_string_length = cudf::detail::make_device_uvector_async(
    string_length, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_col_strings = cuda::make_zip_iterator(
    cuda::std::make_tuple(d_string_offset.begin(), d_string_length.begin()));

  auto res_type =
    infer_data_type(options.json_view(),
                    {d_string_scalar.data(), static_cast<std::size_t>(d_string_scalar.size())},
                    d_col_strings,
                    string_offset.size(),
                    stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::BOOL8});
}

TEST_F(TypeInference, Timestamp)
{
  auto const stream = cudf::get_default_stream();

  auto options       = parse_options{',', '\n', '\"'};
  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data      = R"json([1970/2/5,1970/8/25])json";
  auto d_data           = cudf::make_string_scalar(data);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_data);

  auto const string_offset   = std::vector<cudf::size_type>{1, 10};
  auto const string_length   = std::vector<cudf::size_type>{8, 9};
  auto const d_string_offset = cudf::detail::make_device_uvector_async(
    string_offset, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const d_string_length = cudf::detail::make_device_uvector_async(
    string_length, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_col_strings = cuda::make_zip_iterator(
    cuda::std::make_tuple(d_string_offset.begin(), d_string_length.begin()));

  auto res_type =
    infer_data_type(options.json_view(),
                    {d_string_scalar.data(), static_cast<std::size_t>(d_string_scalar.size())},
                    d_col_strings,
                    string_offset.size(),
                    stream);

  // All data time (quoted and unquoted) is inferred as string for now
  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::STRING});
}

TEST_F(TypeInference, InvalidInput)
{
  auto const stream = cudf::get_default_stream();

  auto options       = parse_options{',', '\n', '\"'};
  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data      = R"json([1,2,3,a,5])json";
  auto d_data           = cudf::make_string_scalar(data);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_data);

  auto const string_offset   = std::vector<cudf::size_type>{1, 3, 5, 7, 9};
  auto const string_length   = std::vector<cudf::size_type>{1, 1, 1, 1, 1};
  auto const d_string_offset = cudf::detail::make_device_uvector_async(
    string_offset, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const d_string_length = cudf::detail::make_device_uvector_async(
    string_length, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_col_strings = cuda::make_zip_iterator(
    cuda::std::make_tuple(d_string_offset.begin(), d_string_length.begin()));

  auto res_type =
    infer_data_type(options.json_view(),
                    {d_string_scalar.data(), static_cast<std::size_t>(d_string_scalar.size())},
                    d_col_strings,
                    string_offset.size(),
                    stream);

  // Invalid input is inferred as string for now
  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::STRING});
}

CUDF_TEST_PROGRAM_MAIN()
