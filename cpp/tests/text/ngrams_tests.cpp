/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/generate_ngrams.hpp>

#include <cuda/iterator>

struct TextGenerateNgramsTest : public cudf::test::BaseFixture {};

TEST_F(TextGenerateNgramsTest, Ngrams)
{
  cudf::test::strings_column_wrapper strings{"the", "fox", "jumped", "over", "thé", "dog"};
  cudf::strings_column_view strings_view(strings);
  auto const separator = cudf::string_scalar("_");

  {
    cudf::test::strings_column_wrapper expected{
      "the_fox", "fox_jumped", "jumped_over", "over_thé", "thé_dog"};
    auto const results = nvtext::generate_ngrams(strings_view, 2, separator);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    cudf::test::strings_column_wrapper expected{
      "the_fox_jumped", "fox_jumped_over", "jumped_over_thé", "over_thé_dog"};
    auto const results = nvtext::generate_ngrams(strings_view, 3, separator);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  {
    LCW expected({LCW({"th", "he"}),
                  LCW({"fo", "ox"}),
                  LCW({"ju", "um", "mp", "pe", "ed"}),
                  LCW({"ov", "ve", "er"}),
                  LCW({"th", "hé"}),
                  LCW({"do", "og"})});
    auto const results = nvtext::generate_character_ngrams(strings_view, 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    LCW expected({LCW({"the"}),
                  LCW({"fox"}),
                  LCW({"jum", "ump", "mpe", "ped"}),
                  LCW({"ove", "ver"}),
                  LCW({"thé"}),
                  LCW({"dog"})});
    auto const results = nvtext::generate_character_ngrams(strings_view, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextGenerateNgramsTest, NgramsWithNulls)
{
  auto validity = cudf::test::iterators::null_at(5);
  cudf::test::strings_column_wrapper input({"the", "fox", "", "jumped", "over", "", "the", "dog"},
                                           validity);
  auto const separator = cudf::string_scalar("_");

  cudf::strings_column_view sv(input);
  {
    auto const results = nvtext::generate_ngrams(sv, 3, separator);
    cudf::test::strings_column_wrapper expected{
      "the_fox_jumped", "fox_jumped_over", "jumped_over_the", "over_the_dog"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
    LCW expected({LCW({"the"}),
                  LCW({"fox"}),
                  LCW{},
                  LCW({"jum", "ump", "mpe", "ped"}),
                  LCW({"ove", "ver"}),
                  LCW{},
                  LCW({"the"}),
                  LCW({"dog"})});
    auto const results = nvtext::generate_character_ngrams(sv, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextGenerateNgramsTest, Empty)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto const separator = cudf::string_scalar("_");

  auto results =
    nvtext::generate_ngrams(cudf::strings_column_view(zero_size_strings_column), 2, separator);
  cudf::test::expect_column_empty(results->view());
  results = nvtext::generate_character_ngrams(cudf::strings_column_view(zero_size_strings_column));
  cudf::test::expect_column_empty(results->view());
}

TEST_F(TextGenerateNgramsTest, Errors)
{
  cudf::test::strings_column_wrapper strings{""};
  auto const separator = cudf::string_scalar("_");
  // invalid parameter value
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings), 1, separator),
               std::invalid_argument);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings), 1),
               std::invalid_argument);
  auto const invalid_separator = cudf::string_scalar("", false);
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings), 2, invalid_separator),
               std::invalid_argument);
  // not enough strings to generate ngrams
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings), 3, separator),
               cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings), 3),
               cudf::logic_error);

  cudf::test::strings_column_wrapper strings_no_tokens({"", "", "", ""}, {1, 0, 1, 0});
  EXPECT_THROW(nvtext::generate_ngrams(cudf::strings_column_view(strings_no_tokens), 2, separator),
               cudf::logic_error);
  EXPECT_THROW(nvtext::generate_character_ngrams(cudf::strings_column_view(strings_no_tokens)),
               cudf::logic_error);
}

TEST_F(TextGenerateNgramsTest, NgramsHash)
{
  auto input =
    cudf::test::strings_column_wrapper({"the quick brown fox", "jumped over the lazy dog."});

  auto view    = cudf::strings_column_view(input);
  auto results = nvtext::hash_character_ngrams(view);

  using LCW = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW expected({LCW{2169381797u, 3924065905u, 1634753325u, 3766025829u,  771291085u,
                    2286480985u, 2815102125u, 2383213292u, 1587939873u, 3417728802u,
                     741580288u, 1721912990u, 3322339040u, 2530504717u, 1448945146u},
                LCW{3542029734u, 2351937583u, 2373822151u, 2610417165u, 1303810911u,
                    2541942822u, 1736466351u, 3466558519u,  408633648u, 1698719372u,
                     620653030u,   16851044u,  608863326u,  948572753u, 3672211877u,
                    4097451013u, 1444462157u, 3762829398u,  743082018u, 2953783152u,
                    2319357747u}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::hash_character_ngrams(view, 10, 10);
  // clang-format off
  LCW expected2({LCW{2818025299u, 4026424618u, 578054337u, 2107870805u, 3942221995u,
                     2802685757u, 2686450821u, 584898501u, 2206824201u, 487979059u},
                 LCW{1154048732u, 3209682333u, 3246563372u, 3789750511u, 1287153502u,
                     3759561568u, 1092423314u,  339538635u, 4265577390u,  879551618u,
                     4222824617u, 1774528854u, 1028254379u,  485918316u,  879142987u, 3619248543u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
}

TEST_F(TextGenerateNgramsTest, NgramsHashErrors)
{
  auto input = cudf::test::strings_column_wrapper({"1", "2", "3"});
  auto view  = cudf::strings_column_view(input);

  // invalid parameter value
  EXPECT_THROW(nvtext::hash_character_ngrams(view, 1), std::invalid_argument);
  // strings not long enough to generate ngrams
  EXPECT_THROW(nvtext::hash_character_ngrams(view), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
