/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace CUDF_EXPORT cudf {

/**
 * @brief A CUDA UDF containing the source code and the expression name to be used as the UDF.
 */
struct cuda_udf {
  char const* source = "";   ///< Null-terminated string-view containing the body of the UDF
  std::string expression{};  ///< symbol name of the UDF to be used for linking
  std::span<char const*>
    include_names{};  ///< Null-terminated strings containing the names of the included files to be
                      ///< provided for the UDF compilation
  std::span<char const*> includes{};  ///< Null-terminated strings containing the contents of the
  ///< included files to be provided for the UDF compilation

  /**
   * @brief Construct a CUDA UDF with the given source code and expression name.
   * @param source The source code of the UDF, a null-terminated C string.
   * @param expression The expression name of the UDF.
   */
  cuda_udf(char const* source, std::string_view expression) : source{source}, expression{expression}
  {
  }

  /**
   * @brief Construct a CUDA UDF with the given source code and expression name.
   * @param source The source code of the UDF, a null-terminated C string.
   * @param expression The expression name of the UDF.
   * @param include_names The names of the included files to be provided for the `source`
   * translation unit.
   * @param includes The contents of the included files to be provided for the `source` translation
   * unit.
   */
  cuda_udf(char const* source,
           std::string_view expression,
           std::span<char const*> include_names,
           std::span<char const*> includes)
    : source{source}, expression{expression}, include_names{include_names}, includes{includes}
  {
  }
};

/**
 * @brief Binary Fragment Types for compiled CUDA programs.
 */
enum class fragment_type : uint8_t {
  LTO_IR,  //< LTO-IR binary
  FATBIN,  //< FATBIN binary
  PTX      //< PTX source
};

/**
 * @brief The pre-compiled LTO UDF containing the binary fragments and the symbol name to be
 * used as the UDF.
 */
struct lto_udf {
  std::vector<std::span<uint8_t const>> fragments{};  ///< Binary fragments
  fragment_type type{};                               ///< The type of the LTO binary fragments
  std::string symbol{};  ///< Symbol name of the UDF to be used for linking

  /**
   * @brief Construct a new LTO UDF with the given binary fragment, type, and symbol name.
   * @param fragment The binary fragment of the LTO UDF.
   * @param type The type of the LTO binary fragment.
   * @param symbol The symbol name of the LTO UDF to be used for linking.
   */
  lto_udf(std::span<uint8_t const> fragment, fragment_type type, std::string_view symbol)
    : fragments{fragment}, type{type}, symbol{symbol}
  {
  }

  /**
   * @brief Construct a new LTO UDF with the given binary fragment, type, and symbol name.
   * @param fragment The binary fragment of the LTO UDF.
   * @param type The type of the LTO binary fragment.
   * @param symbol The symbol name of the LTO UDF to be used for linking.
   */
  lto_udf(std::string_view fragment, fragment_type type, std::string_view symbol)
    : fragments{std::span<uint8_t const>{reinterpret_cast<uint8_t const*>(fragment.data()),
                                         fragment.size()}},
      type{type},
      symbol{symbol}
  {
  }

  /**
   * @brief Construct a new LTO UDF with the given binary fragments, type, and symbol name.
   * @param fragments The binary fragments of the LTO UDF.
   * @param type The type of the LTO binary fragments.
   * @param symbol The symbol name of the LTO UDF to be used for linking.
   */
  lto_udf(std::span<std::span<uint8_t const> const> fragments,
          fragment_type type,
          std::string_view symbol)
    : fragments{fragments.begin(), fragments.end()}, type{type}, symbol{symbol}
  {
  }

  /**
   * @brief Construct a new PTX UDF with the given binary fragment and symbol name.
   * @param fragment The binary fragment of the PTX UDF.
   * @param symbol The symbol name of the PTX UDF to be used for linking.
   * @return The constructed PTX UDF.
   */
  static lto_udf ptx(std::span<uint8_t const> fragment, std::string_view symbol)
  {
    return lto_udf{fragment, fragment_type::PTX, symbol};
  }

  /**
   * @brief Construct a new PTX UDF with the given binary fragment and symbol name.
   * @param fragment The binary fragment of the PTX UDF.
   * @param symbol The symbol name of the PTX UDF to be used for linking.
   * @return The constructed PTX UDF.
   */
  static lto_udf ptx(std::string_view fragment, std::string_view symbol)
  {
    return lto_udf{fragment, fragment_type::PTX, symbol};
  }

  /**
   * @brief Construct a new FATBIN UDF with the given binary fragment and symbol name.
   * @param fragment The binary fragment of the FATBIN UDF.
   * @param symbol The symbol name of the FATBIN UDF to be used for linking.
   * @return The constructed FATBIN UDF.
   */
  static lto_udf fatbin(std::span<uint8_t const> fragment, std::string_view symbol)
  {
    return lto_udf{fragment, fragment_type::FATBIN, symbol};
  }

  /**
   * @brief Construct a new FATBIN UDF with the given binary fragments and symbol name.
   * @param fragments The binary fragments of the FATBIN UDF.
   * @param symbol The symbol name of the FATBIN UDF to be used for linking.
   * @return The constructed FATBIN UDF.
   */
  static lto_udf fatbin(std::span<std::span<uint8_t const> const> fragments,
                        std::string_view symbol)
  {
    return lto_udf{fragments, fragment_type::FATBIN, symbol};
  }

  /**
   * @brief Construct a new LTO IR UDF with the given binary fragment and symbol name.
   * @param fragment The binary fragment of the LTO IR UDF.
   * @param symbol The symbol name of the LTO IR UDF to be used for linking.
   * @return The constructed LTO IR UDF.
   */
  static lto_udf lto_ir(std::span<uint8_t const> fragment, std::string_view symbol)
  {
    return lto_udf{fragment, fragment_type::LTO_IR, symbol};
  }

  /**
   * @brief Construct a new LTO IR UDF with the given binary fragments and symbol name.
   * @param fragments The binary fragments of the LTO IR UDF.
   * @param symbol The symbol name of the LTO IR UDF to be used for linking.
   * @return The constructed LTO IR UDF.
   */
  static lto_udf lto_ir(std::span<std::span<uint8_t const> const> fragments,
                        std::string_view symbol)
  {
    return lto_udf{fragments, fragment_type::LTO_IR, symbol};
  }
};

/**
 * @brief A UDF containing either a CUDA UDF or an LTO UDF.
 */
using udf = std::variant<cuda_udf, lto_udf>;

}  // namespace CUDF_EXPORT cudf
