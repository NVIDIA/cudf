/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/minhash.hpp>

#include <cuda/iterator>

#include <vector>

struct MinHashTest : public cudf::test::BaseFixture {};

TEST_F(MinHashTest, Permuted)
{
  auto input =
    cudf::test::strings_column_wrapper({"doc 1",
                                        "this is doc 2",
                                        "doc 3",
                                        "d",
                                        "The quick brown fox jumpéd over the lazy brown dog.",
                                        "line six",
                                        "line seven",
                                        "line eight",
                                        "line nine",
                                        "line ten"});

  auto view = cudf::strings_column_view(input);

  auto first   = cuda::counting_iterator<uint32_t>{10};
  auto params  = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results = nvtext::minhash(view, 0, cudf::column_view(params), cudf::column_view(params), 4);

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{1392101586u,  394869177u,  811528444u},
    LCW32{ 211415830u,  187088503u,  130291444u},
    LCW32{2098117052u,  394869177u,  799753544u},
    LCW32{2264583304u, 2920538364u, 3576493424u},
    LCW32{ 253327882u,   41747273u,  302030804u},
    LCW32{2109809594u, 1017470651u,  326988172u},
    LCW32{1303819864u,  850676747u,  147107852u},
    LCW32{ 736021564u,  720812292u, 1405158760u},
    LCW32{ 902780242u,  134064807u, 1613944636u},
    LCW32{ 547084870u, 1748895564u,  656501844u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 =
    nvtext::minhash64(view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({
    LCW64{ 827364888116975697ul, 1601854279692781452ul,  70500662054893256ul},
    LCW64{  18312093741021833ul,  133793446674258329ul,  21974512489226198ul},
    LCW64{  22474244732520567ul, 1638811775655358395ul, 949306297364502264ul},
    LCW64{1332357434996402861ul, 2157346081260151330ul, 676491718310205848ul},
    LCW64{  65816830624808020ul,   43323600380520789ul,  63511816333816345ul},
    LCW64{ 629657184954525200ul,   49741036507643002ul,  97466271004074331ul},
    LCW64{ 301611977846331113ul,  101188874709594830ul,  97466271004074331ul},
    LCW64{ 121498891461700668ul,  171065800427907402ul,  97466271004074331ul},
    LCW64{  54617739511834072ul,  231454301607238929ul,  97466271004074331ul},
    LCW64{ 576418665851990314ul,  231454301607238929ul,  97466271004074331ul}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, PermutedWide)
{
  std::string const small(2 << 10, 'x');  // below wide_string_threshold
  std::string const wide(2 << 19, 'y');   // above wide_string_threshold
  auto input = cudf::test::strings_column_wrapper({small, wide});
  auto view  = cudf::strings_column_view(input);

  auto first   = cuda::counting_iterator<uint32_t>{20};
  auto params  = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results = nvtext::minhash(view, 0, cudf::column_view(params), cudf::column_view(params), 4);

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{1731998032u,  315359380u, 3193688024u},
    LCW32{1293098788u, 2860992281u,  133918478u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 =
    nvtext::minhash64(view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
   LCW64 expected64({
     LCW64{1818322427062143853ul, 641024893347719371ul, 1769570368846988848ul},
     LCW64{1389920339306667795ul, 421787002125838902ul, 1759496674158703968ul}
   });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, PermutedManyParameters)
{
  std::string const small(2 << 10, 'x');
  std::string const wide(2 << 19, 'y');
  auto input = cudf::test::strings_column_wrapper({small, wide});
  auto view  = cudf::strings_column_view(input);

  auto first = cuda::counting_iterator<uint32_t>{20};
  // more than params_per_thread
  auto params  = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 31);
  auto results = nvtext::minhash(view, 0, cudf::column_view(params), cudf::column_view(params), 4);

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{1731998032u,  315359380u, 3193688024u, 1777049372u,  360410720u, 3238739364u, 1822100712u,  405462060u,
          3283790704u, 1867152052u,  450513400u, 3328842044u, 1912203392u,  495564740u, 3373893384u, 1957254732u,
           540616080u, 3418944724u, 2002306072u,  585667420u, 3463996064u, 2047357412u,  630718760u, 3509047404u,
          2092408752u,  675770100u, 3554098744u, 2137460092u,  720821440u, 3599150084u, 2182511432u},
    LCW32{1293098788u, 2860992281u,  133918478u, 1701811971u, 3269705464u,  542631661u, 2110525154u, 3678418647u,
           951344844u, 2519238337u, 4087131830u, 1360058027u, 2927951520u,  200877717u, 1768771210u, 3336664703u,
           609590900u, 2177484393u, 3745377886u, 1018304083u, 2586197576u, 4154091069u, 1427017266u, 2994910759u,
           267836956u, 1835730449u, 3403623942u,  676550139u, 2244443632u, 3812337125u, 1085263322u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // more than params_per_thread
  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 31);
  auto results64 =
    nvtext::minhash64(view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
   LCW64 expected64({
     LCW64{1818322427062143853,  641024893347719371, 1769570368846988848, 592272835132564366,
           1720818310631833835,  543520776917409353, 1672066252416678822, 494768718702254348,
           1623314194201523817,  446016660487099335, 1574562135986368804, 397264602271944322,
           1525810077771213799,  348512544056789317, 1477058019556058786, 299760485841634304,
           1428305961340903773,  251008427626479291, 1379553903125748768, 202256369411324286,
           1330801844910593755,  153504311196169273, 1282049786695438742, 104752252981014268,
           1233297728480283737,   56000194765859255, 1184545670265128724,   7248136550704242,
           1135793612049973719, 2264339087549243188, 1087041553834818706},
     LCW64{1389920339306667795,  421787002125838902, 1759496674158703968,  791363336977875075,
           2129073009010740141, 1160939671829911248,  192806334649082363, 1530516006681947421,
            562382669501118536, 1900092341533983602,  931959004353154709, 2269668676386019775,
           1301535339205190882,  333402002024361997, 1671111674057227055,  702978336876398170,
           2040688008909263228, 1072554671728434343,  104421334547605450, 1442131006580470516,
            473997669399641631, 1811707341432506689,  843574004251677804, 2181283676284542862,
           1213150339103713977,  245017001922885084, 1582726673955750150,  614593336774921257,
           1952303008807786323,  984169671626957438,   16036334446128545}
   });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, EmptyTest)
{
  auto input   = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto view    = cudf::strings_column_view(input->view());
  auto params  = cudf::test::fixed_width_column_wrapper<uint32_t>({1, 2, 3});
  auto results = nvtext::minhash(view, 0, cudf::column_view(params), cudf::column_view(params), 4);
  EXPECT_EQ(results->size(), 0);
  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t>({1, 2, 3});
  results = nvtext::minhash64(view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);
  EXPECT_EQ(results->size(), 0);

  auto empty = cudf::test::lists_column_wrapper<cudf::string_view>();
  auto lview = cudf::lists_column_view(empty);
  results =
    nvtext::minhash_ngrams(lview, 4, 0, cudf::column_view(params), cudf::column_view(params));
  EXPECT_EQ(results->size(), 0);
  results =
    nvtext::minhash64_ngrams(lview, 4, 0, cudf::column_view(params64), cudf::column_view(params64));
  EXPECT_EQ(results->size(), 0);
}

TEST_F(MinHashTest, ErrorsTest)
{
  auto input = cudf::test::strings_column_wrapper({"this string intentionally left blank"});
  auto view  = cudf::strings_column_view(input);
  auto empty = cudf::test::fixed_width_column_wrapper<uint32_t>();
  auto eview = cudf::column_view(empty);
  EXPECT_THROW(nvtext::minhash(view, 0, eview, eview, 0), std::invalid_argument);
  auto empty64 = cudf::test::fixed_width_column_wrapper<uint64_t>();
  auto eview64 = cudf::column_view(empty64);
  EXPECT_THROW(nvtext::minhash64(view, 0, eview64, eview64, 0), std::invalid_argument);
  EXPECT_THROW(nvtext::minhash(view, 0, eview, eview, 4), std::invalid_argument);
  EXPECT_THROW(nvtext::minhash64(view, 0, eview64, eview64, 4), std::invalid_argument);

  auto empty_list = cudf::test::lists_column_wrapper<cudf::string_view>();
  auto lview      = cudf::lists_column_view(empty_list);
  EXPECT_THROW(nvtext::minhash_ngrams(lview, 0, 0, eview, eview), std::invalid_argument);
  EXPECT_THROW(nvtext::minhash64_ngrams(lview, 0, 0, eview64, eview64), std::invalid_argument);
  EXPECT_THROW(nvtext::minhash_ngrams(lview, 4, 0, eview, eview), std::invalid_argument);
  EXPECT_THROW(nvtext::minhash64_ngrams(lview, 4, 0, eview64, eview64), std::invalid_argument);

  std::vector<std::string> h_input(50000, "");
  input = cudf::test::strings_column_wrapper(h_input.begin(), h_input.end());
  view  = cudf::strings_column_view(input);

  auto const zeroes = cuda::constant_iterator<uint32_t>(0);
  auto params       = cudf::test::fixed_width_column_wrapper<uint32_t>(zeroes, zeroes + 50000);
  auto pview        = cudf::column_view(params);
  EXPECT_THROW(nvtext::minhash(view, 0, pview, pview, 4), std::overflow_error);
  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t>(zeroes, zeroes + 50000);
  auto pview64  = cudf::column_view(params64);
  EXPECT_THROW(nvtext::minhash64(view, 0, pview64, pview64, 4), std::overflow_error);

  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{static_cast<cudf::size_type>(h_input.size() + 1)});
  auto input_ngrams =
    cudf::make_lists_column(h_input.size(), offsets.release(), input.release(), 0, {});
  lview = cudf::lists_column_view(input_ngrams->view());
  EXPECT_THROW(nvtext::minhash_ngrams(lview, 4, 0, pview, pview), std::overflow_error);
  EXPECT_THROW(nvtext::minhash64_ngrams(lview, 4, 0, pview64, pview64), std::overflow_error);
}

TEST_F(MinHashTest, Ngrams)
{
  using LCWS = cudf::test::lists_column_wrapper<cudf::string_view>;
  auto input =
    LCWS({LCWS{"The", "quick", "brown", "fox", "jumpéd", "over", "the", "lazy", "brown", "dog."},
          LCWS{"The", "quick", "brown", "fox", "jumpéd", "over", "the", "lazy", "", "dog."},
          LCWS{"short", "row"}});

  auto view = cudf::lists_column_view(input);

  auto first  = cuda::counting_iterator<uint32_t>{10};
  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results =
    nvtext::minhash_ngrams(view, 4, 0, cudf::column_view(params), cudf::column_view(params));
  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{ 230924604u,   55492793u, 963436400u},
    LCW32{ 230924604u,  367515795u, 963436400u},
    LCW32{2380648568u, 1330223236u, 279797904u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 =
    nvtext::minhash64_ngrams(view, 4, 0, cudf::column_view(params64), cudf::column_view(params64));
  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({
    LCW64{ 208926840193078200ul, 576399628675212695ul,  312927673584437419ul},
    LCW64{ 677038498284219393ul, 326338087730412201ul,  298455901014050223ul},
    LCW64{1493265692486268500ul, 720255058049417768ul, 2253087432826260995ul}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, NgramsWide)
{
  auto many     = std::vector<char const*>(1024, "hello");
  auto str_data = cudf::test::strings_column_wrapper(many.begin(), many.end());
  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t, uint64_t>({0ul, many.size() / 2, many.size()});
  auto input = cudf::make_lists_column(2, offsets.release(), str_data.release(), 0, {});

  auto view = cudf::lists_column_view(input->view());

  auto first  = cuda::counting_iterator<uint32_t>{10};
  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results =
    nvtext::minhash_ngrams(view, 4, 0, cudf::column_view(params), cudf::column_view(params));
  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{ 571536396u, 2346676954u, 4121817512u},
    LCW32{ 571536396u, 2346676954u, 4121817512u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 =
    nvtext::minhash64_ngrams(view, 4, 0, cudf::column_view(params64), cudf::column_view(params64));
  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({
    LCW64{ 1947142336021414174ul, 1219519365938078011ul, 491896395854741840ul},
    LCW64{ 1947142336021414174ul, 1219519365938078011ul, 491896395854741840ul}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, NgramsSliced)
{
  using LCWS = cudf::test::lists_column_wrapper<cudf::string_view>;
  auto input =
    LCWS({LCWS{"ignored", "row"},
          LCWS{"The", "quick", "brown", "fox", "jumpéd", "over", "the", "lazy", "brown", "dog."},
          LCWS{"The", "quick", "brown", "fox", "jumpéd", "over", "the", "lazy", "", "dog."},
          LCWS{"short", "row"},
          LCWS{"ignored", "row"}});

  auto view  = cudf::lists_column_view(cudf::slice(input, {1, 4}).front());
  auto first = cuda::counting_iterator<uint32_t>{10};

  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results =
    nvtext::minhash_ngrams(view, 4, 0, cudf::column_view(params), cudf::column_view(params));

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{ 230924604u,   55492793u, 963436400u},
    LCW32{ 230924604u,  367515795u, 963436400u},
    LCW32{2380648568u, 1330223236u, 279797904u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 =
    nvtext::minhash64_ngrams(view, 4, 0, cudf::column_view(params64), cudf::column_view(params64));
  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({
    LCW64{ 208926840193078200ul, 576399628675212695ul,  312927673584437419ul},
    LCW64{ 677038498284219393ul, 326338087730412201ul,  298455901014050223ul},
    LCW64{1493265692486268500ul, 720255058049417768ul, 2253087432826260995ul}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}
