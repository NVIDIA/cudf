/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/span>

struct range32 {
  int32_t begin{};
  int32_t end{};
};

struct url_ranges {
  range32 protocol;
  range32 host;
  range32 port;
  range32 path;
  range32 query;
  range32 fragment;
};

__device__ bool is_ascii_alpha(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

__device__ bool is_ascii_digit(char c) { return c >= '0' && c <= '9'; }

__device__ bool is_hex_digit(char c)
{
  return is_ascii_digit(c) || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
}

// Parses the first valid URL candidate and records byte ranges for all six components.
__device__ bool parse_url(cudf::string_view input, url_ranges* out)
{
  auto n              = input.size_bytes();
  *out                = {};
  auto is_scheme_char = [](char c) {
    return is_ascii_alpha(c) || is_ascii_digit(c) || c == '+' || c == '-' || c == '.';
  };
  auto is_unreserved = [](char c) {
    return is_ascii_alpha(c) || is_ascii_digit(c) || c == '-' || c == '.' || c == '_' || c == '~';
  };
  auto is_sub_delim = [](char c) {
    return c == '!' || c == '$' || c == '&' || c == '\'' || c == '(' || c == ')' || c == '*' ||
           c == '+' || c == ',' || c == ';' || c == '=';
  };
  auto is_gen_delim = [](char c) {
    return c == ':' || c == '/' || c == '?' || c == '#' || c == '[' || c == ']' || c == '@';
  };
  auto is_context_delimiter = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '"' || c == '<' || c == '>';
  };

  auto scheme_end = n;
  for (auto i = 1; i + 2 < n; ++i) {
    if (input.data()[i] == ':' && input.data()[i + 1] == '/' && input.data()[i + 2] == '/') {
      scheme_end = i;
      break;
    }
  }
  if (scheme_end == n) { return false; }

  auto url_begin = scheme_end;
  while (url_begin > 0 && is_scheme_char(input.data()[url_begin - 1])) {
    --url_begin;
  }
  if (url_begin == scheme_end || !is_ascii_alpha(input.data()[url_begin])) { return false; }

  auto url_end = n;
  for (auto i = scheme_end + 3; i < n; ++i) {
    if (is_context_delimiter(input.data()[i])) {
      url_end = i;
      break;
    }
  }
  for (auto i = url_begin; i < url_end; ++i) {
    auto c = input.data()[i];
    if (c == '%') {
      if (i + 2 >= url_end || !is_hex_digit(input.data()[i + 1]) ||
          !is_hex_digit(input.data()[i + 2])) {
        return false;
      }
      i += 2;
    } else if (!is_unreserved(c) && !is_sub_delim(c) && !is_gen_delim(c)) {
      return false;
    }
  }

  auto hash = url_end;
  for (auto i = scheme_end + 3; i < url_end; ++i) {
    if (input.data()[i] == '#') {
      hash = i;
      break;
    }
  }
  auto question = hash;
  for (auto i = scheme_end + 3; i < hash; ++i) {
    if (input.data()[i] == '?') {
      question = i;
      break;
    }
  }
  auto base_end = question < hash ? question : hash;
  out->protocol = {url_begin, scheme_end};
  if (question < hash) { out->query = {question + 1, hash}; }
  if (hash < url_end) { out->fragment = {hash + 1, url_end}; }

  auto authority_begin = scheme_end + 3;
  auto authority_end   = base_end;
  for (auto i = authority_begin; i < base_end; ++i) {
    if (input.data()[i] == '/') {
      authority_end = i;
      break;
    }
  }
  out->path = {authority_end, base_end};

  auto host_begin = authority_begin;
  for (auto i = authority_begin; i < authority_end; ++i) {
    if (input.data()[i] == '@') { host_begin = i + 1; }
  }

  if (host_begin < authority_end && input.data()[host_begin] == '[') {
    auto close = authority_end;
    for (auto i = host_begin + 1; i < authority_end; ++i) {
      if (input.data()[i] == ']') {
        close = i;
        break;
      }
    }
    if (close == authority_end) { return false; }
    out->host = {host_begin, close + 1};
    if (close + 1 < authority_end) {
      if (input.data()[close + 1] != ':') { return false; }
      out->port = {close + 2, authority_end};
    }
  } else {
    auto colon = authority_end;
    for (auto i = host_begin; i < authority_end; ++i) {
      if (input.data()[i] == ':') { colon = i; }
    }
    out->host = {host_begin, colon};
    if (colon < authority_end) { out->port = {colon + 1, authority_end}; }
  }

  for (auto i = out->port.begin; i < out->port.end; ++i) {
    if (!is_ascii_digit(input.data()[i])) { return false; }
  }
  return true;
}

// Computes exact output byte counts for the six URL component columns.
__device__ int compute_url_component_sizes(int32_t* protocol_size,
                                           int32_t* host_size,
                                           int32_t* port_size,
                                           int32_t* path_size,
                                           int32_t* query_size,
                                           int32_t* fragment_size,
                                           cudf::string_view input)
{
  *protocol_size = 0;
  *host_size     = 0;
  *port_size     = 0;
  *path_size     = 0;
  *query_size    = 0;
  *fragment_size = 0;
  url_ranges ranges;
  if (!parse_url(input, &ranges)) { return 0; }
  *protocol_size = ranges.protocol.end - ranges.protocol.begin;
  *host_size     = ranges.host.end - ranges.host.begin;
  *port_size     = ranges.port.end - ranges.port.begin;
  *path_size     = ranges.path.end - ranges.path.begin;
  *query_size    = ranges.query.end - ranges.query.begin;
  *fragment_size = ranges.fragment.end - ranges.fragment.begin;
  return 0;
}

// Copies the six parsed URL components into their preallocated string buffers.
__device__ int write_url_components(cuda::std::span<char>* protocol,
                                    cuda::std::span<char>* host,
                                    cuda::std::span<char>* port,
                                    cuda::std::span<char>* path,
                                    cuda::std::span<char>* query,
                                    cuda::std::span<char>* fragment,
                                    cudf::string_view input)
{
  url_ranges ranges;
  if (!parse_url(input, &ranges)) { return 0; }
  cuda::std::span<char>* outputs[] = {protocol, host, port, path, query, fragment};
  range32 components[]             = {
    ranges.protocol, ranges.host, ranges.port, ranges.path, ranges.query, ranges.fragment};
  for (auto component = 0; component < 6; ++component) {
    auto range = components[component];
    auto size  = range.end - range.begin;
    if (size > 0) { memcpy(outputs[component]->data(), input.data() + range.begin, size); }
  }
  return 0;
}

#ifdef UDF_COMPUTE_SIZES
// Exposes the sizing pass through the transform LTO ABI.
extern "C" __device__ int transform(int32_t* protocol_size,
                                    int32_t* host_size,
                                    int32_t* port_size,
                                    int32_t* path_size,
                                    int32_t* query_size,
                                    int32_t* fragment_size,
                                    cudf::string_view input)
{
  return compute_url_component_sizes(
    protocol_size, host_size, port_size, path_size, query_size, fragment_size, input);
}
#else
#ifdef UDF_WRITE_OUTPUT
// Exposes the component-writing pass through the transform LTO ABI.
extern "C" __device__ int transform(cuda::std::span<char>* protocol,
                                    cuda::std::span<char>* host,
                                    cuda::std::span<char>* port,
                                    cuda::std::span<char>* path,
                                    cuda::std::span<char>* query,
                                    cuda::std::span<char>* fragment,
                                    cudf::string_view input)
{
  return write_url_components(protocol, host, port, path, query, fragment, input);
}
#else
#error "Must define either UDF_COMPUTE_SIZES or UDF_WRITE_OUTPUT"
#endif
#endif
