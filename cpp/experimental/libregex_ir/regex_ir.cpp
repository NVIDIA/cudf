/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define REGEX_IR_IMPLEMENTATION
#include <regex_ir.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <format>
#include <iterator>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

// parser and automata IR

namespace regex_ir {
namespace {

/*
 * Derived from NVIDIA RAPIDS cuDF's Apache-2.0-licensed char_flags.h and the
 * Unicode character database shipped with Python. The generated tables are
 * embedded here so this implementation has no generated-file dependency.
 */

struct unicode_data_range {
  std::uint32_t first;
  std::uint32_t last;
};

inline constexpr unicode_data_range unicode_word_ranges[] = {
  {0x30U, 0x39U},     {0x41U, 0x5aU},     {0x61U, 0x7aU},     {0xaaU, 0xaaU},
  {0xb2U, 0xb3U},     {0xb5U, 0xb5U},     {0xb9U, 0xbaU},     {0xbcU, 0xbeU},
  {0xc0U, 0xd6U},     {0xd8U, 0xf6U},     {0xf8U, 0x2c1U},    {0x2c6U, 0x2d1U},
  {0x2e0U, 0x2e4U},   {0x2ecU, 0x2ecU},   {0x2eeU, 0x2eeU},   {0x370U, 0x374U},
  {0x376U, 0x377U},   {0x37aU, 0x37dU},   {0x37fU, 0x37fU},   {0x386U, 0x386U},
  {0x388U, 0x38aU},   {0x38cU, 0x38cU},   {0x38eU, 0x3a1U},   {0x3a3U, 0x3f5U},
  {0x3f7U, 0x481U},   {0x48aU, 0x52fU},   {0x531U, 0x556U},   {0x559U, 0x559U},
  {0x561U, 0x587U},   {0x5d0U, 0x5eaU},   {0x5f0U, 0x5f2U},   {0x620U, 0x64aU},
  {0x660U, 0x669U},   {0x66eU, 0x66fU},   {0x671U, 0x6d3U},   {0x6d5U, 0x6d5U},
  {0x6e5U, 0x6e6U},   {0x6eeU, 0x6fcU},   {0x6ffU, 0x6ffU},   {0x710U, 0x710U},
  {0x712U, 0x72fU},   {0x74dU, 0x7a5U},   {0x7b1U, 0x7b1U},   {0x7c0U, 0x7eaU},
  {0x7f4U, 0x7f5U},   {0x7faU, 0x7faU},   {0x800U, 0x815U},   {0x81aU, 0x81aU},
  {0x824U, 0x824U},   {0x828U, 0x828U},   {0x840U, 0x858U},   {0x8a0U, 0x8b4U},
  {0x8b6U, 0x8bdU},   {0x904U, 0x939U},   {0x93dU, 0x93dU},   {0x950U, 0x950U},
  {0x958U, 0x961U},   {0x966U, 0x96fU},   {0x971U, 0x980U},   {0x985U, 0x98cU},
  {0x98fU, 0x990U},   {0x993U, 0x9a8U},   {0x9aaU, 0x9b0U},   {0x9b2U, 0x9b2U},
  {0x9b6U, 0x9b9U},   {0x9bdU, 0x9bdU},   {0x9ceU, 0x9ceU},   {0x9dcU, 0x9ddU},
  {0x9dfU, 0x9e1U},   {0x9e6U, 0x9f1U},   {0x9f4U, 0x9f9U},   {0xa05U, 0xa0aU},
  {0xa0fU, 0xa10U},   {0xa13U, 0xa28U},   {0xa2aU, 0xa30U},   {0xa32U, 0xa33U},
  {0xa35U, 0xa36U},   {0xa38U, 0xa39U},   {0xa59U, 0xa5cU},   {0xa5eU, 0xa5eU},
  {0xa66U, 0xa6fU},   {0xa72U, 0xa74U},   {0xa85U, 0xa8dU},   {0xa8fU, 0xa91U},
  {0xa93U, 0xaa8U},   {0xaaaU, 0xab0U},   {0xab2U, 0xab3U},   {0xab5U, 0xab9U},
  {0xabdU, 0xabdU},   {0xad0U, 0xad0U},   {0xae0U, 0xae1U},   {0xae6U, 0xaefU},
  {0xaf9U, 0xaf9U},   {0xb05U, 0xb0cU},   {0xb0fU, 0xb10U},   {0xb13U, 0xb28U},
  {0xb2aU, 0xb30U},   {0xb32U, 0xb33U},   {0xb35U, 0xb39U},   {0xb3dU, 0xb3dU},
  {0xb5cU, 0xb5dU},   {0xb5fU, 0xb61U},   {0xb66U, 0xb6fU},   {0xb71U, 0xb77U},
  {0xb83U, 0xb83U},   {0xb85U, 0xb8aU},   {0xb8eU, 0xb90U},   {0xb92U, 0xb95U},
  {0xb99U, 0xb9aU},   {0xb9cU, 0xb9cU},   {0xb9eU, 0xb9fU},   {0xba3U, 0xba4U},
  {0xba8U, 0xbaaU},   {0xbaeU, 0xbb9U},   {0xbd0U, 0xbd0U},   {0xbe6U, 0xbf2U},
  {0xc05U, 0xc0cU},   {0xc0eU, 0xc10U},   {0xc12U, 0xc28U},   {0xc2aU, 0xc39U},
  {0xc3dU, 0xc3dU},   {0xc58U, 0xc5aU},   {0xc60U, 0xc61U},   {0xc66U, 0xc6fU},
  {0xc78U, 0xc7eU},   {0xc80U, 0xc80U},   {0xc85U, 0xc8cU},   {0xc8eU, 0xc90U},
  {0xc92U, 0xca8U},   {0xcaaU, 0xcb3U},   {0xcb5U, 0xcb9U},   {0xcbdU, 0xcbdU},
  {0xcdeU, 0xcdeU},   {0xce0U, 0xce1U},   {0xce6U, 0xcefU},   {0xcf1U, 0xcf2U},
  {0xd05U, 0xd0cU},   {0xd0eU, 0xd10U},   {0xd12U, 0xd3aU},   {0xd3dU, 0xd3dU},
  {0xd4eU, 0xd4eU},   {0xd54U, 0xd56U},   {0xd58U, 0xd61U},   {0xd66U, 0xd78U},
  {0xd7aU, 0xd7fU},   {0xd85U, 0xd96U},   {0xd9aU, 0xdb1U},   {0xdb3U, 0xdbbU},
  {0xdbdU, 0xdbdU},   {0xdc0U, 0xdc6U},   {0xde6U, 0xdefU},   {0xe01U, 0xe30U},
  {0xe32U, 0xe33U},   {0xe40U, 0xe46U},   {0xe50U, 0xe59U},   {0xe81U, 0xe82U},
  {0xe84U, 0xe84U},   {0xe87U, 0xe88U},   {0xe8aU, 0xe8aU},   {0xe8dU, 0xe8dU},
  {0xe94U, 0xe97U},   {0xe99U, 0xe9fU},   {0xea1U, 0xea3U},   {0xea5U, 0xea5U},
  {0xea7U, 0xea7U},   {0xeaaU, 0xeabU},   {0xeadU, 0xeb0U},   {0xeb2U, 0xeb3U},
  {0xebdU, 0xebdU},   {0xec0U, 0xec4U},   {0xec6U, 0xec6U},   {0xed0U, 0xed9U},
  {0xedcU, 0xedfU},   {0xf00U, 0xf00U},   {0xf20U, 0xf33U},   {0xf40U, 0xf47U},
  {0xf49U, 0xf6cU},   {0xf88U, 0xf8cU},   {0x1000U, 0x102aU}, {0x103fU, 0x1049U},
  {0x1050U, 0x1055U}, {0x105aU, 0x105dU}, {0x1061U, 0x1061U}, {0x1065U, 0x1066U},
  {0x106eU, 0x1070U}, {0x1075U, 0x1081U}, {0x108eU, 0x108eU}, {0x1090U, 0x1099U},
  {0x10a0U, 0x10c5U}, {0x10c7U, 0x10c7U}, {0x10cdU, 0x10cdU}, {0x10d0U, 0x10faU},
  {0x10fcU, 0x1248U}, {0x124aU, 0x124dU}, {0x1250U, 0x1256U}, {0x1258U, 0x1258U},
  {0x125aU, 0x125dU}, {0x1260U, 0x1288U}, {0x128aU, 0x128dU}, {0x1290U, 0x12b0U},
  {0x12b2U, 0x12b5U}, {0x12b8U, 0x12beU}, {0x12c0U, 0x12c0U}, {0x12c2U, 0x12c5U},
  {0x12c8U, 0x12d6U}, {0x12d8U, 0x1310U}, {0x1312U, 0x1315U}, {0x1318U, 0x135aU},
  {0x1369U, 0x137cU}, {0x1380U, 0x138fU}, {0x13a0U, 0x13f5U}, {0x13f8U, 0x13fdU},
  {0x1401U, 0x166cU}, {0x166fU, 0x167fU}, {0x1681U, 0x169aU}, {0x16a0U, 0x16eaU},
  {0x16eeU, 0x16f8U}, {0x1700U, 0x170cU}, {0x170eU, 0x1711U}, {0x1720U, 0x1731U},
  {0x1740U, 0x1751U}, {0x1760U, 0x176cU}, {0x176eU, 0x1770U}, {0x1780U, 0x17b3U},
  {0x17d7U, 0x17d7U}, {0x17dcU, 0x17dcU}, {0x17e0U, 0x17e9U}, {0x17f0U, 0x17f9U},
  {0x1810U, 0x1819U}, {0x1820U, 0x1877U}, {0x1880U, 0x1884U}, {0x1887U, 0x18a8U},
  {0x18aaU, 0x18aaU}, {0x18b0U, 0x18f5U}, {0x1900U, 0x191eU}, {0x1946U, 0x196dU},
  {0x1970U, 0x1974U}, {0x1980U, 0x19abU}, {0x19b0U, 0x19c9U}, {0x19d0U, 0x19daU},
  {0x1a00U, 0x1a16U}, {0x1a20U, 0x1a54U}, {0x1a80U, 0x1a89U}, {0x1a90U, 0x1a99U},
  {0x1aa7U, 0x1aa7U}, {0x1b05U, 0x1b33U}, {0x1b45U, 0x1b4bU}, {0x1b50U, 0x1b59U},
  {0x1b83U, 0x1ba0U}, {0x1baeU, 0x1be5U}, {0x1c00U, 0x1c23U}, {0x1c40U, 0x1c49U},
  {0x1c4dU, 0x1c7dU}, {0x1c80U, 0x1c88U}, {0x1c90U, 0x1cbaU}, {0x1cbdU, 0x1cbfU},
  {0x1ce9U, 0x1cecU}, {0x1ceeU, 0x1cf1U}, {0x1cf5U, 0x1cf6U}, {0x1d00U, 0x1dbfU},
  {0x1e00U, 0x1f15U}, {0x1f18U, 0x1f1dU}, {0x1f20U, 0x1f45U}, {0x1f48U, 0x1f4dU},
  {0x1f50U, 0x1f57U}, {0x1f59U, 0x1f59U}, {0x1f5bU, 0x1f5bU}, {0x1f5dU, 0x1f5dU},
  {0x1f5fU, 0x1f7dU}, {0x1f80U, 0x1fb4U}, {0x1fb6U, 0x1fbcU}, {0x1fbeU, 0x1fbeU},
  {0x1fc2U, 0x1fc4U}, {0x1fc6U, 0x1fccU}, {0x1fd0U, 0x1fd3U}, {0x1fd6U, 0x1fdbU},
  {0x1fe0U, 0x1fecU}, {0x1ff2U, 0x1ff4U}, {0x1ff6U, 0x1ffcU}, {0x2070U, 0x2071U},
  {0x2074U, 0x2079U}, {0x207fU, 0x2089U}, {0x2090U, 0x209cU}, {0x2102U, 0x2102U},
  {0x2107U, 0x2107U}, {0x210aU, 0x2113U}, {0x2115U, 0x2115U}, {0x2119U, 0x211dU},
  {0x2124U, 0x2124U}, {0x2126U, 0x2126U}, {0x2128U, 0x2128U}, {0x212aU, 0x212dU},
  {0x212fU, 0x2139U}, {0x213cU, 0x213fU}, {0x2145U, 0x2149U}, {0x214eU, 0x214eU},
  {0x2150U, 0x2189U}, {0x2460U, 0x249bU}, {0x24eaU, 0x24ffU}, {0x2776U, 0x2793U},
  {0x2c00U, 0x2c2eU}, {0x2c30U, 0x2c5eU}, {0x2c60U, 0x2ce4U}, {0x2cebU, 0x2ceeU},
  {0x2cf2U, 0x2cf3U}, {0x2cfdU, 0x2cfdU}, {0x2d00U, 0x2d25U}, {0x2d27U, 0x2d27U},
  {0x2d2dU, 0x2d2dU}, {0x2d30U, 0x2d67U}, {0x2d6fU, 0x2d6fU}, {0x2d80U, 0x2d96U},
  {0x2da0U, 0x2da6U}, {0x2da8U, 0x2daeU}, {0x2db0U, 0x2db6U}, {0x2db8U, 0x2dbeU},
  {0x2dc0U, 0x2dc6U}, {0x2dc8U, 0x2dceU}, {0x2dd0U, 0x2dd6U}, {0x2dd8U, 0x2ddeU},
  {0x2e2fU, 0x2e2fU}, {0x3005U, 0x3007U}, {0x3021U, 0x3029U}, {0x3031U, 0x3035U},
  {0x3038U, 0x303cU}, {0x3041U, 0x3096U}, {0x309dU, 0x309fU}, {0x30a1U, 0x30faU},
  {0x30fcU, 0x30ffU}, {0x3105U, 0x312dU}, {0x3131U, 0x318eU}, {0x3192U, 0x3195U},
  {0x31a0U, 0x31baU}, {0x31f0U, 0x31ffU}, {0x3220U, 0x3229U}, {0x3248U, 0x324fU},
  {0x3251U, 0x325fU}, {0x3280U, 0x3289U}, {0x32b1U, 0x32bfU}, {0x3400U, 0x4db5U},
  {0x4e00U, 0x9fd5U}, {0xa000U, 0xa48cU}, {0xa4d0U, 0xa4fdU}, {0xa500U, 0xa60cU},
  {0xa610U, 0xa62bU}, {0xa640U, 0xa66eU}, {0xa67fU, 0xa69dU}, {0xa6a0U, 0xa6efU},
  {0xa717U, 0xa71fU}, {0xa722U, 0xa788U}, {0xa78bU, 0xa7aeU}, {0xa7b0U, 0xa7bfU},
  {0xa7c2U, 0xa7c6U}, {0xa7f7U, 0xa801U}, {0xa803U, 0xa805U}, {0xa807U, 0xa80aU},
  {0xa80cU, 0xa822U}, {0xa830U, 0xa835U}, {0xa840U, 0xa873U}, {0xa882U, 0xa8b3U},
  {0xa8d0U, 0xa8d9U}, {0xa8f2U, 0xa8f7U}, {0xa8fbU, 0xa8fbU}, {0xa8fdU, 0xa8fdU},
  {0xa900U, 0xa925U}, {0xa930U, 0xa946U}, {0xa960U, 0xa97cU}, {0xa984U, 0xa9b2U},
  {0xa9cfU, 0xa9d9U}, {0xa9e0U, 0xa9e4U}, {0xa9e6U, 0xa9feU}, {0xaa00U, 0xaa28U},
  {0xaa40U, 0xaa42U}, {0xaa44U, 0xaa4bU}, {0xaa50U, 0xaa59U}, {0xaa60U, 0xaa76U},
  {0xaa7aU, 0xaa7aU}, {0xaa7eU, 0xaaafU}, {0xaab1U, 0xaab1U}, {0xaab5U, 0xaab6U},
  {0xaab9U, 0xaabdU}, {0xaac0U, 0xaac0U}, {0xaac2U, 0xaac2U}, {0xaadbU, 0xaaddU},
  {0xaae0U, 0xaaeaU}, {0xaaf2U, 0xaaf4U}, {0xab01U, 0xab06U}, {0xab09U, 0xab0eU},
  {0xab11U, 0xab16U}, {0xab20U, 0xab26U}, {0xab28U, 0xab2eU}, {0xab30U, 0xab5aU},
  {0xab5cU, 0xab65U}, {0xab70U, 0xabe2U}, {0xabf0U, 0xabf9U}, {0xac00U, 0xd7a3U},
  {0xd7b0U, 0xd7c6U}, {0xd7cbU, 0xd7fbU}, {0xf900U, 0xfa6dU}, {0xfa70U, 0xfad9U},
  {0xfb00U, 0xfb06U}, {0xfb09U, 0xfb0dU}, {0xfb13U, 0xfb17U}, {0xfb1dU, 0xfb1dU},
  {0xfb1fU, 0xfb28U}, {0xfb2aU, 0xfb36U}, {0xfb38U, 0xfb3cU}, {0xfb3eU, 0xfb3eU},
  {0xfb40U, 0xfb41U}, {0xfb43U, 0xfb44U}, {0xfb46U, 0xfbb1U}, {0xfbd3U, 0xfd3dU},
  {0xfd50U, 0xfd8fU}, {0xfd92U, 0xfdc7U}, {0xfdf0U, 0xfdfbU}, {0xfe70U, 0xfe74U},
  {0xfe76U, 0xfefcU}, {0xff10U, 0xff19U}, {0xff21U, 0xff3aU}, {0xff41U, 0xff5aU},
  {0xff66U, 0xffbeU}, {0xffc2U, 0xffc7U}, {0xffcaU, 0xffcfU}, {0xffd2U, 0xffd7U},
  {0xffdaU, 0xffdcU},
};

inline constexpr unicode_data_range unicode_digit_ranges[] = {
  {0x30U, 0x39U},     {0xb2U, 0xb3U},     {0xb9U, 0xb9U},     {0x660U, 0x669U},
  {0x6f0U, 0x6f9U},   {0x7c0U, 0x7c9U},   {0x966U, 0x96fU},   {0x9e6U, 0x9efU},
  {0xa66U, 0xa6fU},   {0xae6U, 0xaefU},   {0xb66U, 0xb6fU},   {0xbe6U, 0xbefU},
  {0xc66U, 0xc6fU},   {0xce6U, 0xcefU},   {0xd66U, 0xd6fU},   {0xde6U, 0xdefU},
  {0xe50U, 0xe59U},   {0xed0U, 0xed9U},   {0xf20U, 0xf29U},   {0x1040U, 0x1049U},
  {0x1090U, 0x1099U}, {0x1369U, 0x1371U}, {0x17e0U, 0x17e9U}, {0x1810U, 0x1819U},
  {0x1946U, 0x194fU}, {0x19d0U, 0x19daU}, {0x1a80U, 0x1a89U}, {0x1a90U, 0x1a99U},
  {0x1b50U, 0x1b59U}, {0x1bb0U, 0x1bb9U}, {0x1c40U, 0x1c49U}, {0x1c50U, 0x1c59U},
  {0x2070U, 0x2070U}, {0x2074U, 0x2079U}, {0x2080U, 0x2089U}, {0x2460U, 0x2468U},
  {0x2474U, 0x247cU}, {0x2488U, 0x2490U}, {0x24eaU, 0x24eaU}, {0x24f5U, 0x24fdU},
  {0x24ffU, 0x24ffU}, {0x2776U, 0x277eU}, {0x2780U, 0x2788U}, {0x278aU, 0x2792U},
  {0xa620U, 0xa629U}, {0xa8d0U, 0xa8d9U}, {0xa900U, 0xa909U}, {0xa9d0U, 0xa9d9U},
  {0xa9f0U, 0xa9f9U}, {0xaa50U, 0xaa59U}, {0xabf0U, 0xabf9U}, {0xff10U, 0xff19U},
};

inline constexpr unicode_data_range unicode_space_ranges[] = {
  {0x9U, 0xdU},
  {0x1cU, 0x20U},
  {0x85U, 0x85U},
  {0xa0U, 0xa0U},
  {0x1680U, 0x1680U},
  {0x2000U, 0x200aU},
  {0x2028U, 0x2029U},
  {0x202fU, 0x202fU},
  {0x205fU, 0x205fU},
  {0x3000U, 0x3000U},
};

inline constexpr unicode_data_range unicode_math_symbol_ranges[] = {
  {0x00002bU, 0x00002bU}, {0x00003cU, 0x00003eU}, {0x00007cU, 0x00007cU}, {0x00007eU, 0x00007eU},
  {0x0000acU, 0x0000acU}, {0x0000b1U, 0x0000b1U}, {0x0000d7U, 0x0000d7U}, {0x0000f7U, 0x0000f7U},
  {0x0003f6U, 0x0003f6U}, {0x000606U, 0x000608U}, {0x002044U, 0x002044U}, {0x002052U, 0x002052U},
  {0x00207aU, 0x00207cU}, {0x00208aU, 0x00208cU}, {0x002118U, 0x002118U}, {0x002140U, 0x002144U},
  {0x00214bU, 0x00214bU}, {0x002190U, 0x002194U}, {0x00219aU, 0x00219bU}, {0x0021a0U, 0x0021a0U},
  {0x0021a3U, 0x0021a3U}, {0x0021a6U, 0x0021a6U}, {0x0021aeU, 0x0021aeU}, {0x0021ceU, 0x0021cfU},
  {0x0021d2U, 0x0021d2U}, {0x0021d4U, 0x0021d4U}, {0x0021f4U, 0x0022ffU}, {0x002320U, 0x002321U},
  {0x00237cU, 0x00237cU}, {0x00239bU, 0x0023b3U}, {0x0023dcU, 0x0023e1U}, {0x0025b7U, 0x0025b7U},
  {0x0025c1U, 0x0025c1U}, {0x0025f8U, 0x0025ffU}, {0x00266fU, 0x00266fU}, {0x0027c0U, 0x0027c4U},
  {0x0027c7U, 0x0027e5U}, {0x0027f0U, 0x0027ffU}, {0x002900U, 0x002982U}, {0x002999U, 0x0029d7U},
  {0x0029dcU, 0x0029fbU}, {0x0029feU, 0x002affU}, {0x002b30U, 0x002b44U}, {0x002b47U, 0x002b4cU},
  {0x00fb29U, 0x00fb29U}, {0x00fe62U, 0x00fe62U}, {0x00fe64U, 0x00fe66U}, {0x00ff0bU, 0x00ff0bU},
  {0x00ff1cU, 0x00ff1eU}, {0x00ff5cU, 0x00ff5cU}, {0x00ff5eU, 0x00ff5eU}, {0x00ffe2U, 0x00ffe2U},
  {0x00ffe9U, 0x00ffecU}, {0x01d6c1U, 0x01d6c1U}, {0x01d6dbU, 0x01d6dbU}, {0x01d6fbU, 0x01d6fbU},
  {0x01d715U, 0x01d715U}, {0x01d735U, 0x01d735U}, {0x01d74fU, 0x01d74fU}, {0x01d76fU, 0x01d76fU},
  {0x01d789U, 0x01d789U}, {0x01d7a9U, 0x01d7a9U}, {0x01d7c3U, 0x01d7c3U}, {0x01eef0U, 0x01eef1U},
};

struct parse_failure : std::exception {};

enum class node_kind : std::uint8_t {
  EMPTY       = 0,
  PREDICATE   = 1,
  CONCATENATE = 2,
  ALTERNATE   = 3,
  REPEAT      = 4,
  GROUP       = 5,
  ASSERTION   = 6,
};

struct node {
  node_kind kind                              = node_kind::EMPTY;
  source_span source                          = source_span{};
  character_predicate predicate               = character_predicate{};
  assertion_kind assertion                    = assertion_kind::BEGIN_INPUT;
  std::vector<std::unique_ptr<node>> children = std::vector<std::unique_ptr<node>>{};
  std::uint32_t minimum                       = 0;
  std::uint32_t maximum                       = 0;
  bool greedy                                 = true;
  std::uint32_t capture_index                 = 0;
  bool capturing                              = false;
};

bool can_consume_character(node const& value)
{
  switch (value.kind) {
    case node_kind::PREDICATE: return true;
    case node_kind::EMPTY:
    case node_kind::ASSERTION: return false;
    case node_kind::GROUP: return can_consume_character(*value.children.front());
    case node_kind::CONCATENATE:
    case node_kind::ALTERNATE:
      return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
        return can_consume_character(*child);
      });
    case node_kind::REPEAT: return can_consume_character(*value.children.front());
  }
  return false;
}

bool is_unconditional_empty(node const& value)
{
  switch (value.kind) {
    case node_kind::EMPTY: return true;
    case node_kind::GROUP: return is_unconditional_empty(*value.children.front());
    case node_kind::CONCATENATE:
    case node_kind::ALTERNATE:
      return std::all_of(value.children.begin(), value.children.end(), [](auto& child) {
        return is_unconditional_empty(*child);
      });
    case node_kind::PREDICATE:
    case node_kind::REPEAT:
    case node_kind::ASSERTION: return false;
  }
  return false;
}

bool can_match_empty(node const& value)
{
  switch (value.kind) {
    case node_kind::EMPTY:
    case node_kind::ASSERTION: return true;
    case node_kind::PREDICATE: return false;
    case node_kind::GROUP: return can_match_empty(*value.children.front());
    case node_kind::CONCATENATE:
      return std::all_of(value.children.begin(), value.children.end(), [](auto& child) {
        return can_match_empty(*child);
      });
    case node_kind::ALTERNATE:
      return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
        return can_match_empty(*child);
      });
    case node_kind::REPEAT: return value.minimum == 0 || can_match_empty(*value.children.front());
  }
  return false;
}

bool contains_capture(node const& value)
{
  if (value.kind == node_kind::GROUP && value.capturing) return true;
  return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
    return contains_capture(*child);
  });
}

void normalize_ranges(character_predicate& predicate)
{
  if (predicate.ranges.empty()) { return; }

  std::sort(predicate.ranges.begin(), predicate.ranges.end(), [](auto& lhs, auto& rhs) {
    return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.last < rhs.last);
  });

  std::vector<codepoint_range> merged;
  for (auto range : predicate.ranges) {
    if (merged.empty() || static_cast<std::uint64_t>(range.first) >
                            static_cast<std::uint64_t>(merged.back().last) + 1U) {
      merged.push_back(range);
    } else if (range.last > merged.back().last) {
      merged.back().last = range.last;
    }
  }

  predicate.ranges = std::move(merged);
}

template <std::size_t Size>
void append_unicode_ranges(character_predicate& predicate, unicode_data_range const (&ranges)[Size])
{
  predicate.ranges.reserve(predicate.ranges.size() + Size);
  for (unicode_data_range range : ranges) {
    predicate.ranges.push_back(
      {static_cast<char32_t>(range.first), static_cast<char32_t>(range.last)});
  }
}

std::vector<codepoint_range> complement_ranges(std::vector<codepoint_range> ranges)
{
  character_predicate normalized;
  normalized.ranges = std::move(ranges);
  normalize_ranges(normalized);

  std::vector<codepoint_range> result;
  char32_t begin = U'\0';
  for (codepoint_range range : normalized.ranges) {
    if (begin < range.first) result.push_back({begin, static_cast<char32_t>(range.first - 1)});
    if (range.last == static_cast<char32_t>(0x10FFFF)) return result;
    begin = static_cast<char32_t>(range.last + 1);
  }

  result.push_back({begin, static_cast<char32_t>(0x10FFFF)});
  return result;
}

void remove_codepoint(std::vector<codepoint_range>& ranges, char32_t value)
{
  std::vector<codepoint_range> result;
  result.reserve(ranges.size() + 1);

  for (codepoint_range range : ranges) {
    if (value < range.first || value > range.last) {
      result.push_back(range);
      continue;
    }
    if (range.first < value) result.push_back({range.first, static_cast<char32_t>(value - 1)});
    if (value < range.last) result.push_back({static_cast<char32_t>(value + 1), range.last});
  }

  ranges = std::move(result);
}

bool append_posix_class(character_predicate& predicate, std::string_view name)
{
  if (name == "alpha") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'A', U'Z'}, {U'a', U'z'}});
  } else if (name == "alnum") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'0', U'9'}, {U'A', U'Z'}, {U'a', U'z'}});
  } else if (name == "digit") {
    predicate.ranges.push_back({U'0', U'9'});
  } else if (name == "xdigit") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'0', U'9'}, {U'A', U'F'}, {U'a', U'f'}});
  } else if (name == "space") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'\t', U'\r'}, {U' ', U' '}});
  } else if (name == "word") {
    predicate.ranges.insert(predicate.ranges.end(),
                            {{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}});
  } else if (name == "punct") {
    predicate.ranges.insert(predicate.ranges.end(),
                            {{U'!', U'/'}, {U':', U'@'}, {U'[', U'`'}, {U'{', U'~'}});
  } else {
    return false;
  }
  return true;
}

char32_t swap_case(char32_t value)
{
  static std::locale locale{"C.UTF-8"};
  auto wide = static_cast<wchar_t>(value);
  return static_cast<char32_t>(std::isupper(wide, locale) ? std::tolower(wide, locale)
                                                          : std::toupper(wide, locale));
}

void add_case_pair(character_predicate& predicate, char32_t first, char32_t last)
{
  predicate.ranges.push_back({first, last});
  auto swapped_first = swap_case(first);
  auto swapped_last  = swap_case(last);
  if (swapped_first <= swapped_last) predicate.ranges.push_back({swapped_first, swapped_last});
}

class parser {
 public:
  parser(std::string_view pattern, compile_options const& options)
    : pattern_(pattern), options_(options)
  {
  }

  std::unique_ptr<node> parse()
  {
    if (pattern_.size() > options_.limits.max_pattern_bytes) {
      fail(
        diagnostic_code::RESOURCE_LIMIT, {0, pattern_.size()}, "pattern exceeds max_pattern_bytes");
    }

    auto expression = parse_alternation();
    if (position_ != pattern_.size()) {
      fail(diagnostic_code::UNEXPECTED_TOKEN, {position_, 1}, "unexpected token");
    }

    return expression;
  }

  std::vector<diagnostic> diagnostics = std::vector<diagnostic>{};
  std::uint32_t capture_count         = 0;

 private:
  [[noreturn]] void fail(diagnostic_code code, source_span span, std::string message)
  {
    diagnostics.push_back({code, span, std::move(message)});
    throw parse_failure{};
  }

  [[nodiscard]] bool at_end() const noexcept { return position_ >= pattern_.size(); }

  [[nodiscard]] char peek() const noexcept { return at_end() ? '\0' : pattern_[position_]; }

  char take()
  {
    if (at_end()) {
      fail(diagnostic_code::UNEXPECTED_END, {position_, 0}, "unexpected end of pattern");
    }
    return pattern_[position_++];
  }

  bool consume(char value)
  {
    if (peek() != value) { return false; }
    ++position_;
    return true;
  }

  std::unique_ptr<node> make(node_kind kind, std::size_t start)
  {
    auto result    = std::make_unique<node>();
    result->kind   = kind;
    result->source = {start, position_ - start};
    return result;
  }

  std::unique_ptr<node> parse_alternation()
  {
    auto lhs = parse_concatenation();
    while (consume('|')) {
      auto separator = position_ - 1;
      auto rhs       = parse_concatenation();
      auto alternate = make(node_kind::ALTERNATE, lhs->source.offset);
      if (lhs->kind == node_kind::EMPTY && rhs->kind == node_kind::EMPTY) {
        fail(diagnostic_code::UNEXPECTED_TOKEN, {separator, 1}, "empty alternation");
      }
      alternate->children.push_back(std::move(lhs));
      alternate->children.push_back(std::move(rhs));
      alternate->source.length = position_ - alternate->source.offset;
      lhs                      = std::move(alternate);
    }
    return lhs;
  }

  std::unique_ptr<node> parse_concatenation()
  {
    auto start = position_;
    std::vector<std::unique_ptr<node>> children;
    while (!at_end() && peek() != ')' && peek() != '|') {
      children.push_back(parse_quantified());
    }
    if (children.empty()) { return make(node_kind::EMPTY, start); }
    if (children.size() == 1) { return std::move(children.front()); }
    auto result      = make(node_kind::CONCATENATE, start);
    result->children = std::move(children);
    return result;
  }

  std::uint32_t parse_decimal()
  {
    std::uint64_t value{};
    std::size_t count{};
    while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
      value = value * 10U + static_cast<unsigned>(take() - '0');
      ++count;
      if (value > options_.limits.max_repeat) {
        fail(diagnostic_code::RESOURCE_LIMIT,
             {position_ - count, count},
             "repeat bound exceeds max_repeat");
      }
    }
    if (count == 0) {
      fail(diagnostic_code::INVALID_QUANTIFIER, {position_, 0}, "expected repeat bound");
    }
    return static_cast<std::uint32_t>(value);
  }

  std::unique_ptr<node> parse_quantified()
  {
    auto atom = parse_atom();
    if (at_end()) { return atom; }

    auto start = atom->source.offset;
    std::uint32_t minimum{};
    std::uint32_t maximum{};
    bool quantified = true;
    if (consume('*')) {
      minimum = 0;
      maximum = unbounded_repeat;
    } else if (consume('+')) {
      minimum = 1;
      maximum = unbounded_repeat;
    } else if (consume('?')) {
      minimum = 0;
      maximum = 1;
    } else if (peek() == '{' && position_ + 1 < pattern_.size() &&
               std::isdigit(static_cast<unsigned char>(pattern_[position_ + 1])) != 0) {
      ++position_;
      minimum = parse_decimal();
      maximum = minimum;
      if (consume(',')) { maximum = peek() == '}' ? unbounded_repeat : parse_decimal(); }
      if (!consume('}')) {
        fail(diagnostic_code::INVALID_QUANTIFIER, {position_, 0}, "unterminated quantifier");
      }
      if (maximum != unbounded_repeat && maximum < minimum) {
        fail(diagnostic_code::INVALID_QUANTIFIER,
             {start, position_ - start},
             "repeat maximum is smaller than minimum");
      }
    } else {
      quantified = false;
    }

    if (!quantified) { return atom; }
    auto greedy = !consume('?');
    if (peek() == '*' || peek() == '+' || peek() == '?' || peek() == '{') {
      fail(diagnostic_code::INVALID_QUANTIFIER, {position_, 1}, "multiple repeat operators");
    }
    if (is_unconditional_empty(*atom) && !contains_capture(*atom)) {
      // repeating an unconditional empty expression is still empty
      atom->source.length = position_ - start;
      return atom;
    }
    if (!can_consume_character(*atom)) {
      fail(diagnostic_code::INVALID_QUANTIFIER,
           {start, position_ - start},
           "zero-width assertions cannot be repeated");
    }
    auto result = make(node_kind::REPEAT, start);
    result->children.push_back(std::move(atom));
    result->minimum       = minimum;
    result->maximum       = maximum;
    result->greedy        = greedy;
    result->source.length = position_ - start;
    return result;
  }

  char32_t decode_literal(std::size_t& length)
  {
    if (options_.characters == character_mode::BYTES) {
      length = 1;
      return static_cast<unsigned char>(pattern_[position_]);
    }
    auto first = static_cast<unsigned char>(pattern_[position_]);
    if (first < 0x80U) {
      length = 1;
      return first;
    }
    std::size_t count = first < 0xE0U ? 2 : (first < 0xF0U ? 3 : 4);
    if (position_ + count > pattern_.size() || first < 0xC2U || first > 0xF4U) {
      fail(diagnostic_code::INVALID_ESCAPE, {position_, 1}, "invalid UTF-8 in pattern");
    }
    char32_t value = first & (count == 2 ? 0x1FU : (count == 3 ? 0x0FU : 0x07U));
    for (std::size_t index = 1; index < count; ++index) {
      auto next = static_cast<unsigned char>(pattern_[position_ + index]);
      if ((next & 0xC0U) != 0x80U) {
        fail(diagnostic_code::INVALID_ESCAPE, {position_, count}, "invalid UTF-8 in pattern");
      }
      value = static_cast<char32_t>((value << 6U) | (next & 0x3FU));
    }
    if ((count == 3 && value < 0x800U) || (count == 4 && value < 0x10000U) || value > 0x10FFFFU ||
        (value >= 0xD800U && value <= 0xDFFFU)) {
      fail(diagnostic_code::INVALID_ESCAPE, {position_, count}, "invalid UTF-8 scalar value");
    }
    length = count;
    return value;
  }

  char32_t parse_hex(std::size_t digits, std::size_t escape_start)
  {
    char32_t value{};
    for (std::size_t index = 0; index < digits; ++index) {
      if (at_end()) {
        fail(diagnostic_code::INVALID_ESCAPE,
             {escape_start, position_ - escape_start},
             "truncated hexadecimal escape");
      }
      char const digit = take();
      value <<= 4U;
      if (digit >= '0' && digit <= '9') {
        value |= static_cast<char32_t>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        value |= static_cast<char32_t>(digit - 'a' + 10);
      } else if (digit >= 'A' && digit <= 'F') {
        value |= static_cast<char32_t>(digit - 'A' + 10);
      } else {
        fail(diagnostic_code::INVALID_ESCAPE, {position_ - 1, 1}, "invalid hexadecimal digit");
      }
    }
    return value;
  }

  char32_t parse_octal(char first)
  {
    auto value         = static_cast<char32_t>(first - '0');
    std::size_t digits = 1;
    while (digits < 3 && peek() >= '0' && peek() <= '7') {
      value = static_cast<char32_t>((value << 3U) | static_cast<char32_t>(take() - '0'));
      ++digits;
    }
    return value;
  }

  character_predicate predefined(char value)
  {
    character_predicate result;
    auto append_ascii = [&](char kind) {
      if (kind == 'd') result.ranges = {{U'0', U'9'}};
      if (kind == 'w') { result.ranges = {{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}}; }
      if (kind == 's') result.ranges = {{U'\t', U' '}};
    };
    auto append_unicode = [&](char kind) {
      if (kind == 'd') append_unicode_ranges(result, unicode_digit_ranges);
      if (kind == 'w') {
        append_unicode_ranges(result, unicode_word_ranges);
        result.ranges.push_back({U'_', U'_'});
      }
      if (kind == 's') append_unicode_ranges(result, unicode_space_ranges);
    };
    auto base = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
    if (options_.ascii_classes) {
      append_ascii(base);
    } else {
      append_unicode(base);
    }
    normalize_ranges(result);
    bool negative = value == 'D' || value == 'W' || value == 'S';
    if (negative) {
      result.ranges = complement_ranges(std::move(result.ranges));
      if (!options_.ascii_classes && (value == 'D' || value == 'W')) {
        remove_codepoint(result.ranges, U'\n');
      }
    }
    switch (value) {
      case 'd':
      case 'D':
        result.recognized = value == 'd' ? predicate_class::DIGIT : predicate_class::NOT_DIGIT;
        break;
      case 'w':
      case 'W':
        result.recognized = value == 'w' ? predicate_class::WORD : predicate_class::NOT_WORD;
        break;
      case 's':
      case 'S':
        result.recognized = value == 's' ? predicate_class::SPACE : predicate_class::NOT_SPACE;
        break;
      default: break;
    }
    return result;
  }

  char32_t escaped_literal(char value, std::size_t start)
  {
    switch (value) {
      case 'a': return U'\a';
      case 'b': return U'\b';
      case 'n': return U'\n';
      case 'r': return U'\r';
      case 't': return U'\t';
      case 'f': return U'\f';
      case 'v': return U'\v';
      case 'x': return parse_hex(2, start);
      case 'u': return parse_hex(4, start);
      default: return static_cast<unsigned char>(value);
    }
  }

  std::unique_ptr<node> parse_escape(bool in_class)
  {
    auto start = position_ - 1;
    if (at_end()) { fail(diagnostic_code::INVALID_ESCAPE, {start, 1}, "trailing backslash"); }
    char const value = take();
    if (!in_class &&
        (value == 'b' || value == 'B' || value == 'A' || value == 'Z' || value == 'z')) {
      auto result = make(node_kind::ASSERTION, start);
      if (value == 'b') result->assertion = assertion_kind::WORD_BOUNDARY;
      if (value == 'B') result->assertion = assertion_kind::NOT_WORD_BOUNDARY;
      if (value == 'A') result->assertion = assertion_kind::BEGIN_INPUT;
      if (value == 'Z' || value == 'z') result->assertion = assertion_kind::END_INPUT;
      return result;
    }
    auto result = make(node_kind::PREDICATE, start);
    if (value == 'p' || value == 'P') {
      if (!consume('{')) {
        fail(diagnostic_code::INVALID_ESCAPE, {start, position_ - start}, "missing property name");
      }
      auto property_begin = position_;
      while (!at_end() && peek() != '}')
        static_cast<void>(take());
      if (!consume('}')) {
        fail(diagnostic_code::INVALID_ESCAPE,
             {start, position_ - start},
             "unterminated Unicode property");
      }
      auto property = pattern_.substr(property_begin, position_ - property_begin - 1U);
      if (property != "Sm") {
        fail(diagnostic_code::UNSUPPORTED_FEATURE,
             {start, position_ - start},
             "unsupported Unicode property");
      }
      append_unicode_ranges(result->predicate, unicode_math_symbol_ranges);
      if (value == 'P') {
        result->predicate.ranges = complement_ranges(std::move(result->predicate.ranges));
      }
      normalize_ranges(result->predicate);
    } else if (value == 'd' || value == 'D' || value == 'w' || value == 'W' || value == 's' ||
               value == 'S') {
      result->predicate = predefined(value);
    } else {
      if (std::isalpha(static_cast<unsigned char>(value)) != 0 && value != 'a' && value != 'b' &&
          value != 'f' && value != 'n' && value != 'r' && value != 't' && value != 'u' &&
          value != 'v' && value != 'x') {
        fail(diagnostic_code::INVALID_ESCAPE, {start, 2}, "unknown alphabetic escape");
      }
      bool three_digit_octal = value >= '0' && value <= '7' && position_ + 1 < pattern_.size() &&
                               pattern_[position_] >= '0' && pattern_[position_] <= '7' &&
                               pattern_[position_ + 1] >= '0' && pattern_[position_ + 1] <= '7';
      if (value >= '1' && value <= '9' && !three_digit_octal) {
        fail(diagnostic_code::UNSUPPORTED_FEATURE, {start, 2}, "backreferences are not supported");
      }
      auto literal = three_digit_octal ? parse_octal(value) : escaped_literal(value, start);
      if (options_.case_insensitive) {
        add_case_pair(result->predicate, literal, literal);
      } else {
        result->predicate.ranges.push_back({literal, literal});
      }
      normalize_ranges(result->predicate);
    }
    result->source.length = position_ - start;
    return result;
  }

  std::unique_ptr<node> parse_class()
  {
    auto start                = position_ - 1;
    auto result               = make(node_kind::PREDICATE, start);
    result->predicate.negated = consume('^');
    bool first                = true;
    bool closed               = false;
    while (!at_end()) {
      if (peek() == ']' && !first) {
        ++position_;
        closed = true;
        break;
      }
      first = false;
      if (peek() == '[' && position_ + 1U < pattern_.size() && pattern_[position_ + 1U] == ':') {
        auto class_start = position_;
        position_ += 2U;
        auto name_start = position_;
        while (position_ + 1U < pattern_.size() &&
               !(pattern_[position_] == ':' && pattern_[position_ + 1U] == ']')) {
          ++position_;
        }
        if (position_ + 1U >= pattern_.size()) {
          fail(diagnostic_code::INVALID_CHARACTER_CLASS,
               {class_start, position_ - class_start},
               "unterminated POSIX character class");
        }
        auto name = pattern_.substr(name_start, position_ - name_start);
        position_ += 2U;
        if (!append_posix_class(result->predicate, name)) {
          fail(diagnostic_code::UNSUPPORTED_FEATURE,
               {class_start, position_ - class_start},
               "unsupported POSIX character class");
        }
        continue;
      }
      char32_t lower{};
      if (consume('\\')) {
        auto escaped = parse_escape(true);
        if (escaped->predicate.ranges.size() != 1 ||
            escaped->predicate.ranges.front().first != escaped->predicate.ranges.front().last) {
          result->predicate.ranges.insert(result->predicate.ranges.end(),
                                          escaped->predicate.ranges.begin(),
                                          escaped->predicate.ranges.end());
          continue;
        }
        lower = escaped->predicate.ranges.front().first;
      } else {
        std::size_t length{};
        lower = decode_literal(length);
        position_ += length;
      }

      char32_t upper = lower;
      if (peek() == '-' && position_ + 1 < pattern_.size() && pattern_[position_ + 1] != ']') {
        ++position_;
        if (consume('\\')) {
          auto escaped = parse_escape(true);
          if (!escaped->predicate.is_singleton()) {
            fail(diagnostic_code::INVALID_CHARACTER_CLASS,
                 {position_, 1},
                 "range endpoint must be a literal");
          }
          upper = escaped->predicate.singleton();
        } else {
          std::size_t length{};
          upper = decode_literal(length);
          position_ += length;
        }
        if (upper < lower) {
          fail(diagnostic_code::INVALID_CHARACTER_CLASS,
               {start, position_ - start},
               "descending character range");
        }
      }
      if (options_.case_insensitive) {
        add_case_pair(result->predicate, lower, upper);
      } else {
        result->predicate.ranges.push_back({lower, upper});
      }
    }
    if (!closed) {
      fail(diagnostic_code::INVALID_CHARACTER_CLASS,
           {start, position_ - start},
           "unterminated character class");
    }
    if (result->predicate.ranges.empty()) {
      fail(diagnostic_code::INVALID_CHARACTER_CLASS,
           {start, position_ - start},
           "empty character class");
    }
    normalize_ranges(result->predicate);
    result->source.length = position_ - start;
    return result;
  }

  std::unique_ptr<node> parse_atom()
  {
    auto start       = position_;
    char const value = take();
    if (value == '(') {
      bool capturing = true;
      if (consume('?')) {
        if (consume(':')) {
          capturing = false;
        } else {
          fail(diagnostic_code::UNSUPPORTED_FEATURE,
               {start, position_ - start + 1},
               "lookaround and inline group extensions are not supported");
        }
      }
      if (++depth_ > options_.limits.max_nesting) {
        fail(diagnostic_code::RESOURCE_LIMIT, {start, 1}, "group nesting exceeds max_nesting");
      }
      std::uint32_t capture{};
      if (capturing) {
        if (capture_count >= options_.limits.max_captures) {
          fail(diagnostic_code::RESOURCE_LIMIT, {start, 1}, "capture count exceeds max_captures");
        }
        capture = ++capture_count;
      }
      auto child = parse_alternation();
      if (!consume(')')) {
        fail(
          diagnostic_code::UNMATCHED_PARENTHESIS, {start, position_ - start}, "unterminated group");
      }
      --depth_;
      auto group = make(node_kind::GROUP, start);
      group->children.push_back(std::move(child));
      group->capturing     = capturing;
      group->capture_index = capture;
      return group;
    }
    if (value == '[') { return parse_class(); }
    if (value == '\\') { return parse_escape(false); }
    if (value == '.') {
      auto result                        = make(node_kind::PREDICATE, start);
      result->predicate.recognized       = predicate_class::ANY;
      result->predicate.matches_newline  = options_.dot_all;
      result->predicate.extended_newline = options_.extended_newline;
      return result;
    }
    if (value == '^' || value == '$') {
      auto result       = make(node_kind::ASSERTION, start);
      result->assertion = value == '^' ? assertion_kind::BEGIN_LINE : assertion_kind::END_LINE;
      return result;
    }
    if (value == ')' || value == '|' || value == '*' || value == '+' || value == '?' ||
        (value == '{' && std::isdigit(static_cast<unsigned char>(peek())) != 0)) {
      fail(diagnostic_code::UNEXPECTED_TOKEN, {start, 1}, "unexpected metacharacter");
    }

    --position_;
    std::size_t length{};
    auto literal = decode_literal(length);
    position_ += length;
    auto result = make(node_kind::PREDICATE, start);
    if (options_.case_insensitive) {
      add_case_pair(result->predicate, literal, literal);
    } else {
      result->predicate.ranges.push_back({literal, literal});
    }
    normalize_ranges(result->predicate);
    result->source.length = length;
    return result;
  }

  std::string_view pattern_;
  compile_options const& options_;
  std::size_t position_ = 0;
  std::size_t depth_    = 0;
};

struct patch_reference {
  state_id state   = invalid_state;
  std::size_t edge = 0;
};
struct fragment {
  state_id start                    = invalid_state;
  std::vector<patch_reference> outs = std::vector<patch_reference>{};
};

class thompson_builder {
 public:
  thompson_builder(std::string_view pattern, compile_options const& options)
  {
    ir.pattern = std::string(pattern);
    ir.options = options;
  }

  automata_ir ir                      = automata_ir{};
  std::vector<diagnostic> diagnostics = std::vector<diagnostic>{};

  fragment build(node const& expression)
  {
    switch (expression.kind) {
      case node_kind::EMPTY: return make_empty(expression.source);
      case node_kind::PREDICATE: return make_predicate(expression);
      case node_kind::ASSERTION: return make_assertion(expression);
      case node_kind::GROUP: return make_group(expression);
      case node_kind::CONCATENATE: return make_concatenate(expression);
      case node_kind::ALTERNATE:
        ir.has_alternation = true;
        return make_alternate(expression);
      case node_kind::REPEAT:
        ir.has_lazy_quantifier = ir.has_lazy_quantifier || !expression.greedy;
        return make_repeat(expression);
    }
    return {};
  }

  void finish(fragment expression, std::uint32_t captures)
  {
    auto accept = add_state(automata_state_kind::ACCEPT, {ir.pattern.size(), 0});
    patch(expression.outs, accept);
    auto entry = add_state(automata_state_kind::JUMP, {0, 0});
    add_edge(entry, expression.start, 0);
    ir.entry         = entry;
    ir.accept        = accept;
    ir.capture_count = captures;
  }

 private:
  state_id add_state(automata_state_kind kind, source_span source)
  {
    if (ir.states.size() >= ir.options.limits.max_states) {
      diagnostics.push_back(
        {diagnostic_code::RESOURCE_LIMIT, source, "state count exceeds max_states"});
      throw parse_failure{};
    }
    auto id = static_cast<state_id>(ir.states.size());
    automata_state state;
    state.id     = id;
    state.kind   = kind;
    state.source = source;
    ir.states.push_back(std::move(state));
    return id;
  }

  std::size_t add_edge(state_id from, state_id to, std::uint32_t priority)
  {
    if (transition_count_ >= ir.options.limits.max_transitions) {
      diagnostics.push_back({diagnostic_code::RESOURCE_LIMIT,
                             ir.states[from].source,
                             "transition count exceeds limit"});
      throw parse_failure{};
    }
    ++transition_count_;
    ir.states[from].edges.push_back({to, priority});
    return ir.states[from].edges.size() - 1;
  }

  patch_reference add_open_edge(state_id from, std::uint32_t priority)
  {
    return {from, add_edge(from, invalid_state, priority)};
  }

  void patch(std::vector<patch_reference> const& references, state_id target)
  {
    // open edges let Thompson fragments be joined without rebuilding either fragment
    for (auto reference : references) {
      ir.states[reference.state].edges[reference.edge].target = target;
    }
  }

  fragment make_empty(source_span span)
  {
    auto state = add_state(automata_state_kind::JUMP, span);
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_predicate(node const& expression)
  {
    auto state                 = add_state(automata_state_kind::CONSUME, expression.source);
    ir.states[state].predicate = expression.predicate;
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_assertion(node const& expression)
  {
    auto state                 = add_state(automata_state_kind::ASSERTION, expression.source);
    ir.states[state].assertion = expression.assertion;
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_group(node const& expression)
  {
    if (!expression.capturing) return build(*expression.children.front());

    auto begin                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[begin].capture       = capture_action::BEGIN;
    ir.states[begin].capture_index = expression.capture_index;

    auto inner = build(*expression.children.front());
    add_edge(begin, inner.start, 0);

    auto end                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[end].capture       = capture_action::END;
    ir.states[end].capture_index = expression.capture_index;
    patch(inner.outs, end);
    return {begin, {add_open_edge(end, 0)}};
  }

  fragment concatenate(fragment left, fragment right)
  {
    patch(left.outs, right.start);
    return {left.start, std::move(right.outs)};
  }

  fragment make_concatenate(node const& expression)
  {
    if (expression.children.empty()) return make_empty(expression.source);
    auto result = build(*expression.children.front());
    for (std::size_t index = 1; index < expression.children.size(); ++index) {
      result = concatenate(std::move(result), build(*expression.children[index]));
    }
    return result;
  }

  fragment alternate(fragment left, fragment right, source_span span)
  {
    auto branch = add_state(automata_state_kind::BRANCH, span);
    add_edge(branch, left.start, 0);
    add_edge(branch, right.start, 1);
    left.outs.insert(left.outs.end(),
                     std::make_move_iterator(right.outs.begin()),
                     std::make_move_iterator(right.outs.end()));
    return {branch, std::move(left.outs)};
  }

  fragment make_alternate(node const& expression)
  {
    if (expression.children.empty()) return make_empty(expression.source);
    auto result = build(*expression.children.front());
    for (std::size_t index = 1; index < expression.children.size(); ++index) {
      result = alternate(std::move(result), build(*expression.children[index]), expression.source);
    }
    return result;
  }

  fragment optional(node const& expression, source_span span, bool greedy)
  {
    auto inner  = build(expression);
    auto branch = add_state(automata_state_kind::BRANCH, span);
    // lower priorities are attempted first, so swapping them implements lazy quantifiers
    auto take_priority = greedy ? 0U : 1U;
    auto exit_priority = greedy ? 1U : 0U;
    add_edge(branch, inner.start, take_priority);
    inner.outs.push_back(add_open_edge(branch, exit_priority));
    return {branch, std::move(inner.outs)};
  }

  fragment star(node const& expression, source_span span, bool greedy)
  {
    auto inner           = build(expression);
    auto branch          = add_state(automata_state_kind::BRANCH, span);
    auto repeat_priority = greedy ? 0U : 1U;
    auto exit_priority   = greedy ? 1U : 0U;
    add_edge(branch, inner.start, repeat_priority);
    patch(inner.outs, branch);
    return {branch, {add_open_edge(branch, exit_priority)}};
  }

  fragment make_repeat(node const& expression)
  {
    auto& repeated = *expression.children.front();
    if (expression.maximum == unbounded_repeat && can_match_empty(repeated)) {
      diagnostics.push_back({diagnostic_code::INVALID_QUANTIFIER,
                             expression.source,
                             "unbounded repetition of a nullable expression is not supported"});
      throw parse_failure{};
    }
    std::optional<fragment> result;
    auto append = [&](fragment next) {
      if (result) {
        *result = concatenate(std::move(*result), std::move(next));
      } else {
        result = std::move(next);
      }
    };

    for (std::uint32_t count = 0; count < expression.minimum; ++count) {
      append(build(repeated));
    }

    if (expression.maximum == unbounded_repeat) {
      append(star(repeated, expression.source, expression.greedy));
    } else {
      for (std::uint32_t count = expression.minimum; count < expression.maximum; ++count) {
        append(optional(repeated, expression.source, expression.greedy));
      }
    }

    return result ? std::move(*result) : make_empty(expression.source);
  }

  std::size_t transition_count_ = 0;
};

}  // namespace

bool character_predicate::matches(char32_t value) const noexcept
{
  if (recognized == predicate_class::ANY) {
    if (matches_newline) return true;
    if (!extended_newline) return value != U'\n';
    return value != U'\n' && value != U'\r' && value != static_cast<char32_t>(0x85) &&
           value != static_cast<char32_t>(0x2028) && value != static_cast<char32_t>(0x2029);
  }
  bool contained = false;
  for (auto range : ranges) {
    if (value >= range.first && value <= range.last) {
      contained = true;
      break;
    }
  }
  return negated ? !contained : contained;
}

bool character_predicate::is_singleton() const noexcept
{
  return !negated && recognized == predicate_class::NONE && ranges.size() == 1 &&
         ranges.front().first == ranges.front().last;
}

char32_t character_predicate::singleton() const noexcept
{
  return is_singleton() ? ranges.front().first : U'\0';
}

std::vector<diagnostic> verify(automata_ir const& ir)
{
  std::vector<diagnostic> result;
  auto invalid = [&](source_span span, std::string message) {
    result.push_back({diagnostic_code::INVALID_AUTOMATA_IR, span, std::move(message)});
  };

  if (ir.entry >= ir.states.size()) invalid({}, "entry state is invalid");
  if (ir.accept >= ir.states.size()) invalid({}, "accept state is invalid");

  for (std::size_t index = 0; index < ir.states.size(); ++index) {
    auto& state = ir.states[index];
    if (state.id != index) invalid(state.source, "state ID does not match storage index");

    for (auto edge : state.edges) {
      if (edge.target >= ir.states.size()) invalid(state.source, "edge target is invalid");
    }

    if (state.kind == automata_state_kind::ACCEPT && !state.edges.empty()) {
      invalid(state.source, "accept state has outgoing edges");
    }
    if (state.kind == automata_state_kind::CONSUME && state.edges.size() != 1) {
      invalid(state.source, "consume state must have one edge");
    }
    if ((state.kind == automata_state_kind::JUMP || state.kind == automata_state_kind::ASSERTION ||
         state.kind == automata_state_kind::CAPTURE) &&
        state.edges.size() != 1) {
      invalid(state.source, "linear epsilon state must have one edge");
    }
    if (state.kind == automata_state_kind::BRANCH && state.edges.size() < 2) {
      invalid(state.source, "branch state must have at least two edges");
    }
    if (state.kind == automata_state_kind::CAPTURE &&
        (state.capture_index == 0 || state.capture_index > ir.capture_count)) {
      invalid(state.source, "capture index is out of range");
    }
  }

  return result;
}

automata_result compile_automata(std::string_view pattern, compile_options const& options)
{
  parser parse_pattern(pattern, options);
  std::unique_ptr<node> expression;
  try {
    expression = parse_pattern.parse();
  } catch (parse_failure const&) {
    return {std::nullopt, std::move(parse_pattern.diagnostics)};
  }

  thompson_builder builder(pattern, options);
  try {
    auto fragment = builder.build(*expression);
    builder.finish(std::move(fragment), parse_pattern.capture_count);
  } catch (parse_failure const&) {
    auto diagnostics = std::move(builder.diagnostics);
    diagnostics.insert(
      diagnostics.end(), parse_pattern.diagnostics.begin(), parse_pattern.diagnostics.end());
    return {std::nullopt, std::move(diagnostics)};
  }

  auto diagnostics = verify(builder.ir);
  if (!diagnostics.empty()) { return {std::nullopt, std::move(diagnostics)}; }
  return {std::move(builder.ir), {}};
}

}  // namespace regex_ir

namespace regex_ir::nvvm {
namespace {

void replace_all(std::string& text, std::string_view from, std::string_view to)
{
  for (auto position = text.find(from); position != std::string::npos;
       position      = text.find(from, position + to.size())) {
    text.replace(position, from.size(), to);
  }
}

std::string common_nvvm(bool offset64, bool pairs)
{
  std::string result = R"NVVM(
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

define internal i32 @libregex_ir_row_index() alwaysinline nounwind readnone {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %width
  %row = add i32 %base, %thread
  ret i32 %row
}

define internal i1 @libregex_ir_is_valid(i32* %mask, i32 %row) alwaysinline nounwind readonly {
entry:
  %all_valid = icmp eq i32* %mask, null
  br i1 %all_valid, label %yes, label %check
check:
  %word_index = lshr i32 %row, 5
  %word_ptr = getelementptr i32, i32* %mask, i32 %word_index
  %word = load i32, i32* %word_ptr, align 4
  %bit_index = and i32 %row, 31
  %shifted = lshr i32 %word, %bit_index
  %bit = and i32 %shifted, 1
  %valid = icmp ne i32 %bit, 0
  ret i1 %valid
yes:
  ret i1 true
}

define internal i64 @libregex_ir_load_offset(i8* %offsets, i32 %index) alwaysinline nounwind readonly {
entry:
  %typed = bitcast i8* %offsets to @OFFSET_TYPE@*
  %ptr = getelementptr @OFFSET_TYPE@, @OFFSET_TYPE@* %typed, i32 %index
  %raw = load @OFFSET_TYPE@, @OFFSET_TYPE@* %ptr, align @OFFSET_ALIGN@
  %value = @OFFSET_EXTEND@ @OFFSET_TYPE@ %raw to i64
  ret i64 %value
}

define internal i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %position) alwaysinline nounwind readonly {
entry:
  %at_end = icmp uge i64 %position, %size
  br i1 %at_end, label %done, label %read
read:
  %ptr = getelementptr i8, i8* %data, i64 %position
  %byte = load i8, i8* %ptr, align 1
  %unsigned = zext i8 %byte to i32
  %ascii = icmp ult i32 %unsigned, 128
  %two_test = and i32 %unsigned, 224
  %is_two = icmp eq i32 %two_test, 192
  %three_test = and i32 %unsigned, 240
  %is_three = icmp eq i32 %three_test, 224
  %wide = select i1 %is_three, i64 3, i64 4
  %non_ascii = select i1 %is_two, i64 2, i64 %wide
  %width = select i1 %ascii, i64 1, i64 %non_ascii
  %advanced = add i64 %position, %width
  %clamped_test = icmp ult i64 %advanced, %size
  %clamped = select i1 %clamped_test, i64 %advanced, i64 %size
  ret i64 %clamped
done:
  ret i64 %size
}
)NVVM";
  replace_all(result, "@OFFSET_TYPE@", offset64 ? "i64" : "i32");
  replace_all(result, "@OFFSET_ALIGN@", offset64 ? "8" : "4");
  replace_all(result, "@OFFSET_EXTEND@", offset64 ? "add i64 0," : "sext");
  if (offset64) {
    replace_all(result, "%value = add i64 0, i64 %raw to i64", "%value = add i64 %raw, 0");
  }
  if (pairs) {
    result += R"NVVM(
%libregex_ir_pair = type { i8*, i32 }

define internal void @libregex_ir_write_pair(%libregex_ir_pair* %output, i64 %index, i8* %data, i8* %empty, i64 %begin, i64 %end, i1 %present) alwaysinline nounwind {
entry:
  %size64 = sub i64 %end, %begin
  %size = trunc i64 %size64 to i32
  %data_ptr = getelementptr i8, i8* %data, i64 %begin
  %is_empty = icmp eq i64 %size64, 0
  %empty_ptr = select i1 %is_empty, i8* %empty, i8* %data_ptr
  %pointer = select i1 %present, i8* %empty_ptr, i8* null
  %stored_size = select i1 %present, i32 %size, i32 0
  %pair_ptr = getelementptr %libregex_ir_pair, %libregex_ir_pair* %output, i64 %index
  %pointer_ptr = getelementptr %libregex_ir_pair, %libregex_ir_pair* %pair_ptr, i32 0, i32 0
  %size_ptr = getelementptr %libregex_ir_pair, %libregex_ir_pair* %pair_ptr, i32 0, i32 1
  store i8* %pointer, i8** %pointer_ptr, align 8
  store i32 %stored_size, i32* %size_ptr, align 4
  ret void
}
)NVVM";
  }
  return result;
}

void require_kernel_name(std::string_view kernel_name)
{
  auto first_is_valid = [](unsigned char character) {
    return std::isalpha(character) != 0 || character == '_';
  };
  auto rest_is_valid = [&](unsigned char character) {
    return first_is_valid(character) || std::isdigit(character) != 0;
  };
  if (kernel_name.empty() || !first_is_valid(static_cast<unsigned char>(kernel_name.front())) ||
      !std::all_of(kernel_name.begin() + 1, kernel_name.end(), [&](char character) {
        return rest_is_valid(static_cast<unsigned char>(character));
      })) {
    throw std::invalid_argument("kernel_name must be a valid source identifier");
  }
  if (kernel_name.starts_with("llvm.") || kernel_name.starts_with("nvvm.")) {
    throw std::invalid_argument("kernel_name uses a reserved identifier");
  }
}

std::string annotate_kernel(std::string module,
                            std::string_view signature,
                            std::string_view kernel_name)
{
  require_kernel_name(kernel_name);
  replace_all(module, "@KERNEL_ENTRY@", std::format("@{}", kernel_name));
  module += std::format(
    "\n@llvm.used = appending global [1 x i8*] [i8* bitcast (void ({})* "
    "@{} to i8*)], section \"llvm.metadata\"\n",
    signature,
    kernel_name);
  module += "\n!nvvm.annotations = !{!900000}\n";
  module +=
    std::format("!900000 = !{{void ({})* @{}, !\"kernel\", i32 1}}\n", signature, kernel_name);
  return module;
}

}  // namespace

std::string assemble(std::string matcher, std::string wrapper)
{
  auto metadata = matcher.find("\n!nvvmir.version");
  if (metadata == std::string::npos) {
    throw std::invalid_argument("generated regex NVVM IR has no version metadata");
  }
  matcher.insert(metadata, std::move(wrapper));
  return matcher;
}

std::string make_warp_literal_contains_kernel(bool offset64,
                                              std::string_view literal,
                                              std::string_view kernel_name)
{
  if (literal.empty()) { throw std::invalid_argument("literal must not be empty"); }
  auto comparisons = std::string{};
  auto matched     = std::string{};
  auto offset      = std::size_t{1};
  auto index       = std::size_t{0};
  while (offset < literal.size()) {
    auto remaining      = literal.size() - offset;
    auto width          = remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < width; ++byte) {
      value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal[offset + byte]))
               << (byte * 8U);
    }
    auto bits = width * 8U;
    std::format_to(std::back_inserter(comparisons),
                   "  %literal_ptr_{0} = getelementptr i8, i8* %data, i64 %literal_offset_{0}\n",
                   index);
    if (width == 1U) {
      std::format_to(std::back_inserter(comparisons),
                     "  %literal_chunk_{0} = load i8, i8* %literal_ptr_{0}, align 1\n",
                     index);
    } else {
      std::format_to(std::back_inserter(comparisons),
                     "  %literal_chunk_ptr_{0} = bitcast i8* %literal_ptr_{0} to i{1}*\n"
                     "  %literal_chunk_{0} = load i{1}, i{1}* %literal_chunk_ptr_{0}, align 1\n",
                     index,
                     bits);
    }
    std::format_to(std::back_inserter(comparisons),
                   "  %literal_equal_{0} = icmp eq i{1} %literal_chunk_{0}, {2}\n",
                   index,
                   bits,
                   value);
    if (matched.empty()) {
      matched = std::format("%literal_equal_{}", index);
    } else {
      std::format_to(std::back_inserter(comparisons),
                     "  %literal_through_{0} = and i1 {1}, %literal_equal_{0}\n",
                     index,
                     matched);
      matched = std::format("%literal_through_{}", index);
    }
    offset += width;
    ++index;
  }

  auto offsets = std::string{};
  offset       = 1;
  for (std::size_t chunk = 0; chunk < index; ++chunk) {
    std::format_to(std::back_inserter(offsets),
                   "  %literal_offset_{0} = add i64 %position, {1}\n",
                   chunk,
                   offset);
    auto remaining = literal.size() - offset;
    offset += remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
  }

  auto verify = literal.size() == 1U
                  ? std::string{"  br i1 %first_equal, label %local_yes, label %inner_continue\n"}
                  : std::format(
                      R"NVVM(  br i1 %first_equal, label %verify, label %inner_continue
verify:
{0}{1}  br i1 {2}, label %local_yes, label %inner_continue
)NVVM",
                      offsets,
                      comparisons,
                      matched);

  auto result = common_nvvm(offset64, false);
  result += std::format(
    R"NVVM(
declare i1 @llvm.nvvm.vote.any.sync(i32, i1) nounwind convergent

define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {{
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %width
  %global = add i32 %base, %thread
  %lane = and i32 %thread, 31
  %row = lshr i32 %global, 5
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %lane64 = zext i32 %lane to i64
  %lane_base = mul nuw i64 %lane64, 4
  br label %search
search:
  %round = phi i64 [ 0, %work ], [ %next_round, %continue ]
  %first_round = icmp eq i64 %round, 0
  %round_lane = select i1 %first_round, i64 %lane64, i64 %lane_base
  %candidate_limit = select i1 %first_round, i64 1, i64 4
  %lane_position = add i64 %round, %round_lane
  br label %inner
inner:
  %candidate_offset = phi i64 [ 0, %search ], [ %next_candidate, %inner_continue ]
  %position = add i64 %lane_position, %candidate_offset
  %candidate_end = add i64 %position, {0}
  %candidate = icmp ule i64 %candidate_end, %size
  br i1 %candidate, label %compare, label %inner_done
compare:
  %first_ptr = getelementptr i8, i8* %data, i64 %position
  %first = load i8, i8* %first_ptr, align 1
  %first_equal = icmp eq i8 %first, {1}
{2}inner_continue:
  %next_candidate = add nuw i64 %candidate_offset, 1
  %more_candidates = icmp ult i64 %next_candidate, %candidate_limit
  br i1 %more_candidates, label %inner, label %inner_done
local_yes:
  br label %inner_done
inner_done:
  %local_match = phi i1 [ true, %local_yes ], [ false, %inner ], [ false, %inner_continue ]
  br label %vote
vote:
  %warp_match = call i1 @llvm.nvvm.vote.any.sync(i32 -1, i1 %local_match)
  br i1 %warp_match, label %store_true, label %continue
continue:
  %later_round = add i64 %round, 128
  %next_round = select i1 %first_round, i64 32, i64 %later_round
  %round_possible = icmp ult i64 %next_round, %size
  br i1 %round_possible, label %search, label %store_false
store_true:
  br label %store
store_false:
  br label %store
store:
  %value = phi i8 [ 1, %store_true ], [ 0, %store_false ]
  %lane_zero = icmp eq i32 %lane, 0
  %write = and i1 %lane_zero, %valid
  br i1 %write, label %write_value, label %done
write_value:
  %out = getelementptr i8, i8* %output, i32 %row
  store i8 %value, i8* %out, align 1
  br label %done
done:
  ret void
}}
)NVVM",
    literal.size(),
    static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
    verify);
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i8*", kernel_name);
}

std::string make_fixed_kernel(bool offset64,
                              regex_ir::operation_kind operation,
                              std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false);
  if (operation == regex_ir::operation_kind::CONTAINS) {
    result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %match, label %store_false
match:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size)
  %value = zext i1 %matched to i8
  br label %store
store_false:
  br label %store
store:
  %result = phi i8 [ %value, %match ], [ 0, %store_false ]
  %out = getelementptr i8, i8* %output, i32 %row
  store i8 %result, i8* %out, align 1
  br label %done
done:
  ret void
}
)NVVM";
  } else if (operation == regex_ir::operation_kind::COUNT) {
    result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %count, label %store_zero
count:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %count64 = call i64 @regex_ir_execute(i8* %data, i64 %size)
  %value = trunc i64 %count64 to i32
  br label %store
store_zero:
  br label %store
store:
  %result = phi i32 [ %value, %count ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %result, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  } else {
    result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %span = alloca [2 x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %find, label %store_missing
find:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %span_ptr = getelementptr [2 x i64], [2 x i64]* %span, i32 0, i32 0
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64* %span_ptr)
  br i1 %matched, label %convert, label %store_missing
convert:
  %match_begin = load i64, i64* %span_ptr, align 8
  br label %utf8_loop
utf8_loop:
  %position = phi i64 [ 0, %convert ], [ %advanced, %utf8_loop ]
  %characters = phi i32 [ 0, %convert ], [ %next_characters, %utf8_loop ]
  %at_match = icmp uge i64 %position, %match_begin
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %position)
  %next_characters = add i32 %characters, 1
  br i1 %at_match, label %store_found, label %utf8_loop
store_found:
  br label %store
store_missing:
  br label %store
store:
  %result = phi i32 [ %characters, %store_found ], [ -1, %store_missing ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %result, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  }
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i8*", kernel_name);
}

std::string make_capture_kernel(bool offset64,
                                std::int32_t capture_slots,
                                std::int32_t first_group,
                                std::int32_t output_groups,
                                bool column_major,
                                std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, true);
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %match, label %output_begin
match:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 0, i64* %capture_ptr)
  br label %output_begin
output_begin:
  %row_data = phi i8* [ %data, %match ], [ %chars, %work ]
  %row_matched = phi i1 [ %matched, %match ], [ false, %work ]
  %typed_output = bitcast i8* %output to %libregex_ir_pair*
  br label %group_loop
group_loop:
  %group = phi i32 [ 0, %output_begin ], [ %next_group, %group_done ]
  br i1 %row_matched, label %group_found, label %group_missing
group_found:
  %capture_group = add i32 %group, @FIRST_GROUP@
  %capture_plus_whole = add i32 %capture_group, 1
  %capture_slot = shl i32 %capture_plus_whole, 1
  %capture_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_slot
  %capture_end_slot = add i32 %capture_slot, 1
  %capture_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_end_slot
  %capture_begin = load i64, i64* %capture_begin_ptr, align 8
  %capture_end = load i64, i64* %capture_end_ptr, align 8
  %has_begin = icmp sge i64 %capture_begin, 0
  %has_end = icmp sge i64 %capture_end, 0
  %present = and i1 %has_begin, %has_end
  br label %group_write
group_missing:
  br label %group_write
group_write:
  %stored_begin = phi i64 [ %capture_begin, %group_found ], [ 0, %group_missing ]
  %stored_end = phi i64 [ %capture_end, %group_found ], [ 0, %group_missing ]
  %stored_present = phi i1 [ %present, %group_found ], [ false, %group_missing ]
  %group64 = sext i32 %group to i64
  @PAIR_INDEX@
  call void @libregex_ir_write_pair(%libregex_ir_pair* %typed_output, i64 %pair_index, i8* %row_data, i8* %offsets, i64 %stored_begin, i64 %stored_end, i1 %stored_present)
  br label %group_done
group_done:
  %next_group = add i32 %group, 1
  %finished = icmp eq i32 %next_group, @OUTPUT_GROUPS@
  br i1 %finished, label %done, label %group_loop
done:
  ret void
}
)NVVM";
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@FIRST_GROUP@", std::to_string(first_group));
  replace_all(result, "@OUTPUT_GROUPS@", std::to_string(output_groups));
  replace_all(result,
              "@PAIR_INDEX@",
              column_major ? "%group_base = mul i64 %group64, %rows64\n  %row64 = sext i32 %row to "
                             "i64\n  %pair_index = add i64 %group_base, %row64"
                           : "%pair_index = sext i32 %row to i64");
  if (column_major) {
    replace_all(result,
                "%typed_output = bitcast i8* %output to %libregex_ir_pair*",
                "%typed_output = bitcast i8* %output to %libregex_ir_pair*\n  %rows64 = sext i32 "
                "%rows to i64");
  }
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i8*", kernel_name);
}

std::string capture_cache_helpers();

std::string make_enumeration_size_kernel(bool offset64,
                                         std::int32_t capture_slots,
                                         std::int32_t multiplier,
                                         bool require_match,
                                         bool cache,
                                         std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false);
  if (cache) { result += capture_cache_helpers(); }
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_validity) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %setup, label %store_invalid
setup:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  br label %match_loop
match_loop:
  %search = phi i64 [ 0, %setup ], [ %match_end, %continue_nonempty ], [ %advanced, %continue_empty ]
  %count = phi i64 [ 0, %setup ], [ %next_count, %continue_nonempty ], [ %next_count, %continue_empty ]
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search, i64* %capture_ptr)
  br i1 %matched, label %found, label %store_count
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %next_count = add i64 %count, 1
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %match_loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %store_after_match, label %continue_empty
continue_empty:
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %match_loop
store_after_match:
  br label %store
store_count:
  br label %store
store_invalid:
  br label %store
store:
  %matches = phi i64 [ %next_count, %store_after_match ], [ %count, %store_count ], [ 0, %store_invalid ]
  %scaled = mul i64 %matches, @MULTIPLIER@
  %stored_count = trunc i64 %scaled to i32
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %stored_count, i32* %out, align 4
  %has_match = icmp ne i64 %matches, 0
  %row_valid = @ROW_VALID@
  %valid_byte = zext i1 %row_valid to i8
  %valid_out = getelementptr i8, i8* %output_validity, i32 %row
  store i8 %valid_byte, i8* %valid_out, align 1
  br label %done
done:
  ret void
}
)NVVM";
  if (cache) {
    replace_all(result,
                "i8* %output, i8* %output_validity) nounwind {",
                "i8* %output, i8* %output_validity, i8* %cache_buffer, i32 %capacity, i8* "
                "%overflow) nounwind {");
    replace_all(
      result,
      "  %match_end = load i64, i64* %match_end_ptr, align 8\n  %next_count =",
      std::format(
        "  %match_end = load i64, i64* %match_end_ptr, align 8\n  %typed_cache = bitcast i8* "
        "%cache_buffer to i64*\n  %overflow_ptr = getelementptr i8, i8* %overflow, i32 %row\n  "
        "call void @libregex_ir_cache_captures(i64* %capture_ptr, i64* %typed_cache, i32 %row, i32 "
        "%capacity, i32 {}, i64 %count, i8* %overflow_ptr)\n  %next_count =",
        capture_slots));
  }
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@MULTIPLIER@", std::to_string(multiplier));
  replace_all(
    result, "@ROW_VALID@", require_match ? "and i1 %valid, %has_match" : "and i1 %valid, true");
  return annotate_kernel(std::move(result),
                         cache ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i32, i8*"
                               : "i8*, i8*, i32*, i32, i32, i8*, i8*",
                         kernel_name);
}

std::string make_enumeration_emit_kernel(bool offset64,
                                         std::int32_t capture_slots,
                                         std::int32_t groups,
                                         bool findall,
                                         bool overflow_only,
                                         std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, true);
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %setup, label %done
setup:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %typed_offsets = bitcast i8* %output_offsets to i32*
  %row_output_ptr = getelementptr i32, i32* %typed_offsets, i32 %row
  %row_output = load i32, i32* %row_output_ptr, align 4
  %row_output64 = sext i32 %row_output to i64
  %typed_output = bitcast i8* %output to %libregex_ir_pair*
  br label %match_loop
match_loop:
  %search = phi i64 [ 0, %setup ], [ %match_end, %continue_nonempty ], [ %advanced, %continue_empty ]
  %output_index = phi i64 [ 0, %setup ], [ %next_output_index, %continue_nonempty ], [ %next_output_index, %continue_empty ]
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search, i64* %capture_ptr)
  br i1 %matched, label %found, label %done
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  @WRITE_MATCH@
after_write:
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %match_loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %done, label %continue_empty
continue_empty:
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %match_loop
done:
  ret void
}
)NVVM";
  if (overflow_only) {
    replace_all(result,
                "i8* %output, i8* %output_offsets) nounwind {",
                "i8* %output, i8* %output_offsets, i8* %overflow) nounwind {");
    replace_all(
      result,
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  br i1 %valid, "
      "label %setup, label %done",
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  %overflow_ptr = "
      "getelementptr i8, i8* %overflow, i32 %row\n  %overflow_value = load i8, i8* %overflow_ptr, "
      "align 1\n  %did_overflow = icmp ne i8 %overflow_value, 0\n  %selected = and i1 %valid, "
      "%did_overflow\n  br i1 %selected, label %setup, label %done");
  }
  auto write_findall =
    groups == 0
      ? R"NVVM(%selected_begin = add i64 %match_begin, 0
  %selected_end = add i64 %match_end, 0
  %present = icmp sge i64 %selected_begin, 0
  %pair_index = add i64 %row_output64, %output_index
  call void @libregex_ir_write_pair(%libregex_ir_pair* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %selected_begin, i64 %selected_end, i1 %present)
  %next_output_index = add i64 %output_index, 1
  br label %after_write)NVVM"
      : R"NVVM(%selected_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 2
  %selected_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 3
  %selected_begin = load i64, i64* %selected_begin_ptr, align 8
  %selected_end = load i64, i64* %selected_end_ptr, align 8
  %has_begin = icmp sge i64 %selected_begin, 0
  %has_end = icmp sge i64 %selected_end, 0
  %present = and i1 %has_begin, %has_end
  %pair_index = add i64 %row_output64, %output_index
  call void @libregex_ir_write_pair(%libregex_ir_pair* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %selected_begin, i64 %selected_end, i1 %present)
  %next_output_index = add i64 %output_index, 1
  br label %after_write)NVVM";
  auto write_extract = R"NVVM(br label %group_loop
group_loop:
  %group = phi i32 [ 0, %found ], [ %next_group, %group_write ]
  %capture_plus_whole = add i32 %group, 1
  %capture_slot = shl i32 %capture_plus_whole, 1
  %capture_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_slot
  %capture_end_slot = add i32 %capture_slot, 1
  %capture_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 %capture_end_slot
  %capture_begin = load i64, i64* %capture_begin_ptr, align 8
  %capture_end = load i64, i64* %capture_end_ptr, align 8
  %has_begin = icmp sge i64 %capture_begin, 0
  %has_end = icmp sge i64 %capture_end, 0
  %present = and i1 %has_begin, %has_end
  %group64 = sext i32 %group to i64
  %group_output = add i64 %output_index, %group64
  %pair_index = add i64 %row_output64, %group_output
  call void @libregex_ir_write_pair(%libregex_ir_pair* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %capture_begin, i64 %capture_end, i1 %present)
  br label %group_write
group_write:
  %next_group = add i32 %group, 1
  %groups_done = icmp eq i32 %next_group, @GROUPS@
  br i1 %groups_done, label %groups_finished, label %group_loop
groups_finished:
  %next_output_index = add i64 %output_index, @GROUPS@
  br label %after_write)NVVM";
  replace_all(result, "@WRITE_MATCH@", findall ? write_findall : write_extract);
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@GROUPS@", std::to_string(groups));
  return annotate_kernel(std::move(result),
                         overflow_only ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*"
                                       : "i8*, i8*, i32*, i32, i32, i8*, i8*",
                         kernel_name);
}

namespace {

std::string llvm_bytes(std::string_view value)
{
  std::string result;
  result.reserve(value.size() * 3);
  for (auto character : value) {
    result += std::format("\\{:02X}", static_cast<unsigned char>(character));
  }
  return result;
}

}  // namespace

std::string capture_cache_helpers()
{
  return R"NVVM(
define internal void @libregex_ir_cache_captures(i64* %captures, i64* %cache, i32 %row, i32 %capacity, i32 %values, i64 %match, i8* %overflow) nounwind {
entry:
  %missing = icmp eq i64* %cache, null
  br i1 %missing, label %done, label %check_capacity
check_capacity:
  %capacity64 = zext i32 %capacity to i64
  %fits = icmp ult i64 %match, %capacity64
  br i1 %fits, label %copy_setup, label %mark_overflow
mark_overflow:
  store i8 1, i8* %overflow, align 1
  br label %done
copy_setup:
  %row64 = zext i32 %row to i64
  %values64 = zext i32 %values to i64
  %row_matches = mul i64 %row64, %capacity64
  %record = add i64 %row_matches, %match
  %base = mul i64 %record, %values64
  br label %copy_loop
copy_loop:
  %slot = phi i32 [ 0, %copy_setup ], [ %next_slot, %copy_loop ]
  %slot64 = zext i32 %slot to i64
  %source = getelementptr i64, i64* %captures, i64 %slot64
  %value = load i64, i64* %source, align 8
  %index = add i64 %base, %slot64
  %destination = getelementptr i64, i64* %cache, i64 %index
  store i64 %value, i64* %destination, align 8
  %next_slot = add i32 %slot, 1
  %finished = icmp eq i32 %next_slot, %values
  br i1 %finished, label %done, label %copy_loop
done:
  ret void
}
)NVVM";
}

std::string make_limited_replace_kernel(bool offset64,
                                        bool emit,
                                        bool output_offset64,
                                        bool cache,
                                        std::span<replacement_piece const> replacement,
                                        std::int32_t capture_slots,
                                        std::int32_t max_replace_count,
                                        std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false);
  if (cache) { result += capture_cache_helpers(); }
  for (std::size_t index = 0; index < replacement.size(); ++index) {
    auto& literal = replacement[index].literal;
    if (!literal.empty()) {
      result +=
        std::format("\n@libregex_ir_replacement_{0} = private constant [{1} x i8] c\"{2}\"\n",
                    index,
                    literal.size(),
                    llvm_bytes(literal));
    }
  }
  result += R"NVVM(
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

define internal i64 @libregex_ir_append_range(i8* %source, i64 %begin, i64 %end, i8* %output, i64 %cursor) alwaysinline nounwind {
entry:
  %length = sub i64 %end, %begin
  %next_cursor = add i64 %cursor, %length
  %output_missing = icmp eq i8* %output, null
  %empty = icmp eq i64 %length, 0
  %skip = or i1 %output_missing, %empty
  br i1 %skip, label %done, label %copy
copy:
  %source_ptr = getelementptr i8, i8* %source, i64 %begin
  %output_ptr = getelementptr i8, i8* %output, i64 %cursor
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %output_ptr, i8* align 1 %source_ptr, i64 %length, i1 false)
  br label %done
done:
  ret i64 %next_cursor
}
)NVVM";

  std::string steps;
  auto cursor = std::string{"%cursor_unmatched"};
  for (std::size_t index = 0; index < replacement.size(); ++index) {
    auto& piece      = replacement[index];
    auto next_cursor = std::format("%cursor_piece_{}", index);
    if (piece.capture.has_value()) {
      auto slot = static_cast<std::int64_t>(*piece.capture) * 2;
      steps += std::format(
        R"NVVM(  %capture_begin_ptr_{0} = getelementptr i64, i64* %captures, i64 {2}
  %capture_end_ptr_{0} = getelementptr i64, i64* %captures, i64 {3}
  %capture_begin_{0} = load i64, i64* %capture_begin_ptr_{0}, align 8
  %capture_end_{0} = load i64, i64* %capture_end_ptr_{0}, align 8
  %capture_has_begin_{0} = icmp sge i64 %capture_begin_{0}, 0
  %capture_has_end_{0} = icmp sge i64 %capture_end_{0}, 0
  %capture_present_{0} = and i1 %capture_has_begin_{0}, %capture_has_end_{0}
  %capture_selected_begin_{0} = select i1 %capture_present_{0}, i64 %capture_begin_{0}, i64 0
  %capture_selected_end_{0} = select i1 %capture_present_{0}, i64 %capture_end_{0}, i64 0
  {4} = call i64 @libregex_ir_append_range(i8* %data, i64 %capture_selected_begin_{0}, i64 %capture_selected_end_{0}, i8* %output, i64 {5})
)NVVM",
        index,
        capture_slots,
        slot,
        slot + 1,
        next_cursor,
        cursor);
    } else if (!piece.literal.empty()) {
      steps += std::format(
        R"NVVM(  %literal_{0} = getelementptr [{1} x i8], [{1} x i8]* @libregex_ir_replacement_{0}, i32 0, i32 0
  {2} = call i64 @libregex_ir_append_range(i8* %literal_{0}, i64 0, i64 {1}, i8* %output, i64 {3})
)NVVM",
        index,
        piece.literal.size(),
        next_cursor,
        cursor);
    } else {
      continue;
    }
    cursor = std::move(next_cursor);
  }
  steps += std::format("  %replacement_cursor = add i64 {}, 0\n", cursor);

  result += R"NVVM(
define internal i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output) nounwind {
entry:
  %capture_array = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %captures = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %capture_array, i32 0, i32 0
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %match_end, %continue_nonempty ], [ %advanced_start, %continue_empty ]
  %copied = phi i64 [ 0, %entry ], [ %match_end, %continue_nonempty ], [ %match_end, %continue_empty ]
  %cursor = phi i64 [ 0, %entry ], [ %replacement_cursor, %continue_nonempty ], [ %replacement_cursor, %continue_empty ]
  %replacement_count = phi i64 [ 0, %entry ], [ %next_replacement_count, %continue_nonempty ], [ %next_replacement_count, %continue_empty ]
  %limit_reached = @LIMIT_REACHED@
  br i1 %limit_reached, label %finish_limit, label %search
search:
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search_start, i64* %captures)
  br i1 %matched, label %found, label %no_match
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %capture_array, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %capture_array, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %cursor_unmatched = call i64 @libregex_ir_append_range(i8* %data, i64 %copied, i64 %match_begin, i8* %output, i64 %cursor)
@REPLACEMENT_STEPS@  %next_replacement_count = add i64 %replacement_count, 1
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %finish_empty, label %continue_empty
continue_empty:
  %advanced_start = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %loop
finish_limit:
  br label %finish
no_match:
  br label %finish
finish_empty:
  br label %finish
finish:
  %tail_begin = phi i64 [ %copied, %finish_limit ], [ %copied, %no_match ], [ %match_end, %finish_empty ]
  %tail_cursor = phi i64 [ %cursor, %finish_limit ], [ %cursor, %no_match ], [ %replacement_cursor, %finish_empty ]
  %final_cursor = call i64 @libregex_ir_append_range(i8* %data, i64 %tail_begin, i64 %size, i8* %output, i64 %tail_cursor)
  ret i64 %final_cursor
}
)NVVM";
  if (cache) {
    replace_all(
      result,
      "define internal i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output) "
      "nounwind {",
      "define internal i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output, i64* "
      "%cache, i32 %row, i32 %capacity, i8* %overflow, i32* %count_out) nounwind {");
    replace_all(
      result,
      "  %match_end = load i64, i64* %match_end_ptr, align 8\n  %cursor_unmatched =",
      std::format(
        "  %match_end = load i64, i64* %match_end_ptr, align 8\n  call void "
        "@libregex_ir_cache_captures(i64* %captures, i64* %cache, i32 %row, i32 %capacity, i32 {}, "
        "i64 %replacement_count, i8* %overflow)\n  %cursor_unmatched =",
        capture_slots));
    replace_all(
      result,
      "  %final_cursor = call i64 @libregex_ir_append_range(i8* %data, i64 %tail_begin, i64 %size, "
      "i8* %output, i64 %tail_cursor)\n  ret i64 %final_cursor",
      "  %final_count = phi i64 [ %replacement_count, %finish_limit ], [ %replacement_count, "
      "%no_match ], [ %next_replacement_count, %finish_empty ]\n  %stored_count = trunc i64 "
      "%final_count to i32\n  store i32 %stored_count, i32* %count_out, align 4\n  %final_cursor = "
      "call i64 @libregex_ir_append_range(i8* %data, i64 %tail_begin, i64 %size, i8* %output, i64 "
      "%tail_cursor)\n  ret i64 %final_cursor");
    result += R"NVVM(
define internal i64 @libregex_ir_replace_cached(i8* %data, i64 %size, i8* %output, i64* %cache, i32 %row, i32 %capacity, i32 %match_count) nounwind {
entry:
  %row64 = zext i32 %row to i64
  %capacity64 = zext i32 %capacity to i64
  %slots64 = zext i32 @CAPTURE_SLOTS@ to i64
  %row_records = mul i64 %row64, %capacity64
  %row_base = mul i64 %row_records, %slots64
  br label %loop
loop:
  %match = phi i32 [ 0, %entry ], [ %next_match, %copy_match ]
  %copied = phi i64 [ 0, %entry ], [ %match_end, %copy_match ]
  %cursor = phi i64 [ 0, %entry ], [ %replacement_cursor, %copy_match ]
  %finished = icmp eq i32 %match, %match_count
  br i1 %finished, label %finish, label %copy_match
copy_match:
  %match64 = zext i32 %match to i64
  %record_offset = mul i64 %match64, %slots64
  %capture_offset = add i64 %row_base, %record_offset
  %captures = getelementptr i64, i64* %cache, i64 %capture_offset
  %match_begin_ptr = getelementptr i64, i64* %captures, i64 0
  %match_end_ptr = getelementptr i64, i64* %captures, i64 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %cursor_unmatched = call i64 @libregex_ir_append_range(i8* %data, i64 %copied, i64 %match_begin, i8* %output, i64 %cursor)
@REPLACEMENT_STEPS@  %next_match = add i32 %match, 1
  br label %loop
finish:
  %final_cursor = call i64 @libregex_ir_append_range(i8* %data, i64 %copied, i64 %size, i8* %output, i64 %cursor)
  ret i64 %final_cursor
}
)NVVM";
  }
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(capture_slots));
  replace_all(result, "@REPLACEMENT_STEPS@", steps);
  replace_all(result,
              "@LIMIT_REACHED@",
              std::format("icmp uge i64 %replacement_count, {}", max_replace_count));

  auto output_offset_type   = output_offset64 ? std::string{"i64"} : std::string{"i32"};
  auto output_offset_align  = output_offset64 ? std::string{"8"} : std::string{"4"};
  auto extend_output_offset = output_offset64
                                ? std::string{"%output_offset64 = add i64 %output_offset, 0"}
                                : std::string{"%output_offset64 = sext i32 %output_offset to i64"};

  result += emit ? R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %done
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_output_offsets = bitcast i8* %output_offsets to @OUTPUT_OFFSET_TYPE@*
  %output_offset_ptr = getelementptr @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %typed_output_offsets, i32 %row
  %output_offset = load @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %output_offset_ptr, align @OUTPUT_OFFSET_ALIGN@
  @EXTEND_OUTPUT_OFFSET@
  %output_ptr = getelementptr i8, i8* %output, i64 %output_offset64
  %written = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output_ptr)
  br label %done
done:
  ret void
}
)NVVM"
                 : R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i32* %size_overflow) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %store_zero
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %result_size = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* null)
  %too_large = icmp ugt i64 %result_size, 2147483647
  br i1 %too_large, label %mark_overflow, label %size_fits
mark_overflow:
  %overflow_old = atomicrmw max i32* %size_overflow, i32 1 monotonic
  br label %size_fits
size_fits:
  %stored_size = trunc i64 %result_size to i32
  br label %store
store_zero:
  br label %store
store:
  %value = phi i32 [ %stored_size, %size_fits ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %value, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  if (cache) {
    auto kernel_position = result.rfind("define void @KERNEL_ENTRY@");
    result.resize(kernel_position);
    result +=
      emit
        ? R"NVVM(define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets, i8* %cache_buffer, i32 %capacity, i8* %overflow, i8* %match_counts) nounwind {
entry:
  %dummy_count = alloca i32, align 4
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %done
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_output_offsets = bitcast i8* %output_offsets to @OUTPUT_OFFSET_TYPE@*
  %output_offset_ptr = getelementptr @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %typed_output_offsets, i32 %row
  %output_offset = load @OUTPUT_OFFSET_TYPE@, @OUTPUT_OFFSET_TYPE@* %output_offset_ptr, align @OUTPUT_OFFSET_ALIGN@
  @EXTEND_OUTPUT_OFFSET@
  %output_ptr = getelementptr i8, i8* %output, i64 %output_offset64
  %overflow_ptr = getelementptr i8, i8* %overflow, i32 %row
  %row_overflow = load i8, i8* %overflow_ptr, align 1
  %did_overflow = icmp ne i8 %row_overflow, 0
  br i1 %did_overflow, label %rematch, label %cached
rematch:
  %rematched = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* %output_ptr, i64* null, i32 %row, i32 %capacity, i8* %overflow_ptr, i32* %dummy_count)
  br label %done
cached:
  %typed_cache = bitcast i8* %cache_buffer to i64*
  %typed_counts = bitcast i8* %match_counts to i32*
  %count_ptr = getelementptr i32, i32* %typed_counts, i32 %row
  %match_count = load i32, i32* %count_ptr, align 4
  %written = call i64 @libregex_ir_replace_cached(i8* %data, i64 %size, i8* %output_ptr, i64* %typed_cache, i32 %row, i32 %capacity, i32 %match_count)
  br label %done
done:
  ret void
}
)NVVM"
        : R"NVVM(define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %cache_buffer, i32 %capacity, i8* %overflow, i8* %match_counts, i32* %size_overflow) nounwind {
entry:
  %typed_counts = bitcast i8* %match_counts to i32*
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %store_zero
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_cache = bitcast i8* %cache_buffer to i64*
  %overflow_ptr = getelementptr i8, i8* %overflow, i32 %row
  %count_ptr = getelementptr i32, i32* %typed_counts, i32 %row
  %result_size = call i64 @libregex_ir_replace_execute(i8* %data, i64 %size, i8* null, i64* %typed_cache, i32 %row, i32 %capacity, i8* %overflow_ptr, i32* %count_ptr)
  %too_large = icmp ugt i64 %result_size, 2147483647
  br i1 %too_large, label %mark_overflow, label %size_fits
mark_overflow:
  %overflow_old = atomicrmw max i32* %size_overflow, i32 1 monotonic
  br label %size_fits
size_fits:
  %stored_size = trunc i64 %result_size to i32
  br label %store
store_zero:
  %invalid_count_ptr = getelementptr i32, i32* %typed_counts, i32 %row
  store i32 0, i32* %invalid_count_ptr, align 4
  br label %store
store:
  %value = phi i32 [ %stored_size, %size_fits ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %value, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  }
  replace_all(result, "@OUTPUT_OFFSET_TYPE@", output_offset_type);
  replace_all(result, "@OUTPUT_OFFSET_ALIGN@", output_offset_align);
  replace_all(result, "@EXTEND_OUTPUT_OFFSET@", extend_output_offset);
  return annotate_kernel(
    std::move(result),
    cache ? (emit ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i32, i8*, i8*"
                  : "i8*, i8*, i32*, i32, i32, i8*, i8*, i32, i8*, i8*, i32*")
          : (emit ? "i8*, i8*, i32*, i32, i32, i8*, i8*" : "i8*, i8*, i32*, i32, i32, i8*, i32*"),
    kernel_name);
}

std::string encode_replacement(std::span<replacement_piece const> replacement)
{
  std::string result;
  for (auto& piece : replacement) {
    if (piece.capture.has_value()) {
      result += std::format("${{{}}}", *piece.capture);
      continue;
    }
    for (auto character : piece.literal) {
      result.push_back(character);
      if (character == '$') { result.push_back('$'); }
    }
  }
  return result;
}

std::string make_replace_kernel(bool offset64,
                                bool emit,
                                bool output_offset64,
                                std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false);
  result += emit ? R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %output_offsets) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %done
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_output_offsets = bitcast i8* %output_offsets to @OUTPUT_TYPE@*
  %output_offset_ptr = getelementptr @OUTPUT_TYPE@, @OUTPUT_TYPE@* %typed_output_offsets, i32 %row
  %raw_output_offset = load @OUTPUT_TYPE@, @OUTPUT_TYPE@* %output_offset_ptr, align @OUTPUT_ALIGN@
  %output_offset = @OUTPUT_EXTEND@ @OUTPUT_TYPE@ %raw_output_offset to i64
  %output_ptr = getelementptr i8, i8* %output, i64 %output_offset
  %written = call i64 @regex_ir_execute(i8* %data, i64 %size, i8* %output_ptr)
  br label %done
done:
  ret void
}
)NVVM"
                 : R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %replace, label %store_zero
replace:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %result_size = call i64 @regex_ir_execute(i8* %data, i64 %size, i8* null)
  %stored_size = trunc i64 %result_size to i32
  br label %store
store_zero:
  br label %store
store:
  %value = phi i32 [ %stored_size, %replace ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %value, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  if (emit) {
    replace_all(result, "@OUTPUT_TYPE@", output_offset64 ? "i64" : "i32");
    replace_all(result, "@OUTPUT_ALIGN@", output_offset64 ? "8" : "4");
    replace_all(result, "@OUTPUT_EXTEND@", output_offset64 ? "add i64 0," : "sext");
    if (output_offset64) {
      replace_all(result,
                  "%output_offset = add i64 0, i64 %raw_output_offset to i64",
                  "%output_offset = add i64 %raw_output_offset, 0");
    }
  }
  return annotate_kernel(
    std::move(result),
    emit ? "i8*, i8*, i32*, i32, i32, i8*, i8*" : "i8*, i8*, i32*, i32, i32, i8*",
    kernel_name);
}

std::string make_split_size_kernel(bool offset64,
                                   std::int32_t maxsplit,
                                   bool cache,
                                   std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false);
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %split, label %store_zero
split:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %count64 = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* null, i64 @MAXSPLIT@, i64 -1, i8* null)
  %count = trunc i64 %count64 to i32
  br label %store
store_zero:
  br label %store
store:
  %stored_count = phi i32 [ %count, %split ], [ 0, %store_zero ]
  %typed_output = bitcast i8* %output to i32*
  %out = getelementptr i32, i32* %typed_output, i32 %row
  store i32 %stored_count, i32* %out, align 4
  br label %done
done:
  ret void
}
)NVVM";
  if (cache) {
    replace_all(result,
                "i8* %output) nounwind {",
                "i8* %output, i8* %cache_buffer, i32 %capacity, i8* %overflow) nounwind {");
    replace_all(
      result,
      "  %count64 = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* null, "
      "i64 @MAXSPLIT@, i64 -1, i8* null)",
      "  %typed_cache = bitcast i8* %cache_buffer to i64*\n  %row64 = zext i32 %row to i64\n  "
      "%fields_per_row = add i32 %capacity, 1\n  %fields_per_row64 = zext i32 %fields_per_row "
      "to i64\n  %row_fields = mul i64 %row64, %fields_per_row64\n  %row_base = shl i64 "
      "%row_fields, 1\n  %row_spans = getelementptr i64, i64* %typed_cache, i64 %row_base\n  "
      "%overflow_ptr = getelementptr i8, i8* %overflow, i32 %row\n  %count64 = call i64 "
      "@libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* %row_spans, i64 "
      "@MAXSPLIT@, i64 %fields_per_row64, i8* %overflow_ptr)");
  }
  replace_all(result, "@MAXSPLIT@", std::to_string(maxsplit));
  return annotate_kernel(
    std::move(result),
    cache ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i32, i8*" : "i8*, i8*, i32*, i32, i32, i8*",
    kernel_name);
}

std::string make_split_emit_kernel(bool offset64,
                                   bool reverse,
                                   std::int32_t maxsplit,
                                   bool overflow_only,
                                   std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, true);
  result += R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i8* %output, i8* %effective_offsets, i8* %full_offsets, i64* %spans) nounwind {
entry:
  %row = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %row, %rows
  br i1 %in_bounds, label %work, label %done
work:
  %physical = add i32 %row_offset, %row
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %setup, label %done
setup:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %typed_effective_offsets = bitcast i8* %effective_offsets to i32*
  %effective_begin_ptr = getelementptr i32, i32* %typed_effective_offsets, i32 %row
  %effective_next = add i32 %row, 1
  %effective_end_ptr = getelementptr i32, i32* %typed_effective_offsets, i32 %effective_next
  %effective_begin = load i32, i32* %effective_begin_ptr, align 4
  %effective_end = load i32, i32* %effective_end_ptr, align 4
  %effective_count = sub i32 %effective_end, %effective_begin
  %typed_full_offsets = bitcast i8* %full_offsets to i32*
  %full_begin_ptr = getelementptr i32, i32* %typed_full_offsets, i32 %row
  %full_next = add i32 %row, 1
  %full_end_ptr = getelementptr i32, i32* %typed_full_offsets, i32 %full_next
  %full_begin = load i32, i32* %full_begin_ptr, align 4
  %full_end = load i32, i32* %full_end_ptr, align 4
  %full_count = sub i32 %full_end, %full_begin
  %full_begin64 = sext i32 %full_begin to i64
  %span_base_index = shl i64 %full_begin64, 1
  %row_spans = getelementptr i64, i64* %spans, i64 %span_base_index
  %written_fields = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* %row_spans, i64 @MAXSPLIT@, i64 -1, i8* null)
  %typed_output = bitcast i8* %output to %libregex_ir_pair*
  %effective_begin64 = sext i32 %effective_begin to i64
  %truncated = icmp sgt i32 %full_count, %effective_count
  br label %token_loop
token_loop:
  %token = phi i32 [ 0, %setup ], [ %next_token, %token_loop ]
  @SOURCE_INDEX@
  %source64 = sext i32 %source to i64
  %source_base = shl i64 %source64, 1
  %source_end_index = add i64 %source_base, 1
  %source_begin_ptr = getelementptr i64, i64* %row_spans, i64 %source_base
  %source_end_ptr = getelementptr i64, i64* %row_spans, i64 %source_end_index
  %source_begin = load i64, i64* %source_begin_ptr, align 8
  %source_end = load i64, i64* %source_end_ptr, align 8
  @SELECT_SPAN@
  %token64 = sext i32 %token to i64
  %pair_index = add i64 %effective_begin64, %token64
  call void @libregex_ir_write_pair(%libregex_ir_pair* %typed_output, i64 %pair_index, i8* %data, i8* %offsets, i64 %selected_begin, i64 %selected_end, i1 true)
  %next_token = add i32 %token, 1
  %finished = icmp eq i32 %next_token, %effective_count
  br i1 %finished, label %done, label %token_loop
done:
  ret void
}
)NVVM";
  if (overflow_only) {
    replace_all(
      result,
      "i8* %effective_offsets, i8* %full_offsets, i64* %spans) nounwind {",
      "i8* %effective_offsets, i8* %full_offsets, i64* %spans, i8* %overflow) nounwind {");
    replace_all(
      result,
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  br i1 %valid, "
      "label %setup, label %done",
      "  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)\n  %overflow_ptr = "
      "getelementptr i8, i8* %overflow, i32 %row\n  %overflow_value = load i8, i8* "
      "%overflow_ptr, align 1\n  %did_overflow = icmp ne i8 %overflow_value, 0\n  %selected = "
      "and i1 %valid, %did_overflow\n  br i1 %selected, label %setup, label %done");
  }
  replace_all(result, "@MAXSPLIT@", std::to_string(reverse ? -1 : maxsplit));
  if (reverse) {
    replace_all(result,
                "@SOURCE_INDEX@",
                R"NVVM(%removed = sub i32 %full_count, %effective_count
  %is_first = icmp eq i32 %token, 0
  %shifted = add i32 %removed, %token
  %truncated_source = select i1 %is_first, i32 0, i32 %shifted
  %source = select i1 %truncated, i32 %truncated_source, i32 %token)NVVM");
    replace_all(result,
                "@SELECT_SPAN@",
                R"NVVM(%merged_field = sub i32 %full_count, %effective_count
  %merged_field64 = sext i32 %merged_field to i64
  %merged_field_base = shl i64 %merged_field64, 1
  %merged_end_index = add i64 %merged_field_base, 1
  %merged_end_ptr = getelementptr i64, i64* %row_spans, i64 %merged_end_index
  %merged_end = load i64, i64* %merged_end_ptr, align 8
  %is_first_selected = and i1 %truncated, %is_first
  %selected_begin = add i64 %source_begin, 0
  %selected_end = select i1 %is_first_selected, i64 %merged_end, i64 %source_end)NVVM");
  } else {
    replace_all(result, "@SOURCE_INDEX@", "%source = add i32 %token, 0");
    replace_all(result,
                "@SELECT_SPAN@",
                R"NVVM(%last_token = sub i32 %effective_count, 1
  %is_last = icmp eq i32 %token, %last_token
  %merge_tail = and i1 %truncated, %is_last
  %selected_begin = add i64 %source_begin, 0
  %selected_end = select i1 %merge_tail, i64 %size, i64 %source_end)NVVM");
  }
  return annotate_kernel(std::move(result),
                         overflow_only ? "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i64*, i8*"
                                       : "i8*, i8*, i32*, i32, i32, i8*, i8*, i8*, i64*",
                         kernel_name);
}

std::string make_span_cache_sample_kernel(bool offset64,
                                          bool split,
                                          std::int32_t capture_slots,
                                          std::int32_t match_limit,
                                          std::string_view kernel_name)
{
  auto result = common_nvvm(offset64, false);
  result += split ? R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i32 %samples, i32 %capacity, i8* %statistics) nounwind {
entry:
  %sample = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %sample, %samples
  br i1 %in_bounds, label %select, label %done
select:
  %sample64 = zext i32 %sample to i64
  %rows64 = zext i32 %rows to i64
  %samples64 = zext i32 %samples to i64
  %scaled = mul i64 %sample64, %rows64
  %selected64 = udiv i64 %scaled, %samples64
  %selected = trunc i64 %selected64 to i32
  %physical = add i32 %row_offset, %selected
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %sample_row, label %done
sample_row:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %fields = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* null, i64 @MATCH_LIMIT@, i64 -1, i8* null)
  %matches = sub i64 %fields, 1
  br label %record
record:
  %capacity64 = zext i32 %capacity to i64
  %overflow = icmp ugt i64 %matches, %capacity64
  %cap_plus_one = add i64 %capacity64, 1
  %sampled_matches = select i1 %overflow, i64 %cap_plus_one, i64 %matches
  %stats = bitcast i8* %statistics to i64*
  %valid_ptr = getelementptr i64, i64* %stats, i64 0
  %bytes_ptr = getelementptr i64, i64* %stats, i64 1
  %matches_ptr = getelementptr i64, i64* %stats, i64 2
  %overflow_ptr = getelementptr i64, i64* %stats, i64 3
  %valid_old = atomicrmw add i64* %valid_ptr, i64 1 monotonic
  %bytes_old = atomicrmw add i64* %bytes_ptr, i64 %size monotonic
  %matches_old = atomicrmw add i64* %matches_ptr, i64 %sampled_matches monotonic
  %overflow64 = zext i1 %overflow to i64
  %overflow_old = atomicrmw add i64* %overflow_ptr, i64 %overflow64 monotonic
  br label %done
done:
  ret void
}
)NVVM"
                  : R"NVVM(
define void @KERNEL_ENTRY@(i8* %chars, i8* %offsets, i32* %validity, i32 %row_offset, i32 %rows, i32 %samples, i32 %capacity, i8* %statistics) nounwind {
entry:
  %captures = alloca [@CAPTURE_SLOTS@ x i64], align 8
  %sample = call i32 @libregex_ir_row_index()
  %in_bounds = icmp slt i32 %sample, %samples
  br i1 %in_bounds, label %select, label %done
select:
  %sample64 = zext i32 %sample to i64
  %rows64 = zext i32 %rows to i64
  %samples64 = zext i32 %samples to i64
  %scaled = mul i64 %sample64, %rows64
  %selected64 = udiv i64 %scaled, %samples64
  %selected = trunc i64 %selected64 to i32
  %physical = add i32 %row_offset, %selected
  %valid = call i1 @libregex_ir_is_valid(i32* %validity, i32 %physical)
  br i1 %valid, label %sample_row, label %done
sample_row:
  %begin = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %physical)
  %next = add i32 %physical, 1
  %end = call i64 @libregex_ir_load_offset(i8* %offsets, i32 %next)
  %size = sub i64 %end, %begin
  %data = getelementptr i8, i8* %chars, i64 %begin
  %capture_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  br label %match_loop
match_loop:
  %search = phi i64 [ 0, %sample_row ], [ %match_end, %continue_nonempty ], [ %advanced, %continue_empty ]
  %count = phi i64 [ 0, %sample_row ], [ %next_count, %continue_nonempty ], [ %next_count, %continue_empty ]
  %semantic_limited = icmp sge i64 @MATCH_LIMIT@, 0
  %at_semantic_limit = icmp eq i64 %count, @MATCH_LIMIT@
  %semantic_done = and i1 %semantic_limited, %at_semantic_limit
  br i1 %semantic_done, label %record, label %search_match
search_match:
  %matched = call i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search, i64* %capture_ptr)
  br i1 %matched, label %found, label %record
found:
  %match_begin_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 0
  %match_end_ptr = getelementptr [@CAPTURE_SLOTS@ x i64], [@CAPTURE_SLOTS@ x i64]* %captures, i32 0, i32 1
  %match_begin = load i64, i64* %match_begin_ptr, align 8
  %match_end = load i64, i64* %match_end_ptr, align 8
  %next_count = add i64 %count, 1
  %capacity64_found = zext i32 %capacity to i64
  %overflow_found = icmp ugt i64 %next_count, %capacity64_found
  br i1 %overflow_found, label %record_overflow, label %check_empty
check_empty:
  %nonempty = icmp ne i64 %match_begin, %match_end
  br i1 %nonempty, label %continue_nonempty, label %empty
continue_nonempty:
  br label %match_loop
empty:
  %empty_at_end = icmp eq i64 %match_end, %size
  br i1 %empty_at_end, label %record_after_match, label %continue_empty
continue_empty:
  %advanced = call i64 @libregex_ir_advance_utf8(i8* %data, i64 %size, i64 %match_end)
  br label %match_loop
record_after_match:
  br label %record
record_overflow:
  br label %record
record:
  %matches = phi i64 [ %count, %match_loop ], [ %count, %search_match ], [ %next_count, %record_after_match ], [ %next_count, %record_overflow ]
  %capacity64 = zext i32 %capacity to i64
  %overflow = icmp ugt i64 %matches, %capacity64
  %stats = bitcast i8* %statistics to i64*
  %valid_ptr = getelementptr i64, i64* %stats, i64 0
  %bytes_ptr = getelementptr i64, i64* %stats, i64 1
  %matches_ptr = getelementptr i64, i64* %stats, i64 2
  %overflow_ptr = getelementptr i64, i64* %stats, i64 3
  %valid_old = atomicrmw add i64* %valid_ptr, i64 1 monotonic
  %bytes_old = atomicrmw add i64* %bytes_ptr, i64 %size monotonic
  %matches_old = atomicrmw add i64* %matches_ptr, i64 %matches monotonic
  %overflow64 = zext i1 %overflow to i64
  %overflow_old = atomicrmw add i64* %overflow_ptr, i64 %overflow64 monotonic
  br label %done
done:
  ret void
}
)NVVM";
  replace_all(result, "@CAPTURE_SLOTS@", std::to_string(std::max(capture_slots, 2)));
  replace_all(result, "@MATCH_LIMIT@", std::to_string(match_limit));
  return annotate_kernel(std::move(result), "i8*, i8*, i32*, i32, i32, i32, i32, i8*", kernel_name);
}

}  // namespace regex_ir::nvvm

// instruction IR lowering

namespace regex_ir {
namespace {

std::vector<diagnostic> parse_replacement(std::string const& replacement,
                                          std::uint32_t capture_count,
                                          std::vector<replacement_token>& output)
{
  std::vector<diagnostic> diagnostics;
  std::string literal;
  auto flush_literal = [&] {
    if (!literal.empty()) {
      output.push_back({replacement_token::kind::LITERAL, std::move(literal), 0});
      literal.clear();
    }
  };

  for (std::size_t position = 0; position < replacement.size();) {
    if (replacement[position] != '$') {
      literal.push_back(replacement[position++]);
      continue;
    }
    auto start = position++;
    if (position < replacement.size() && replacement[position] == '$') {
      literal.push_back('$');
      ++position;
      continue;
    }
    auto braced = position < replacement.size() && replacement[position] == '{';
    if (braced) { ++position; }
    if (position == replacement.size() ||
        std::isdigit(static_cast<unsigned char>(replacement[position])) == 0) {
      diagnostics.push_back({diagnostic_code::INVALID_REPLACEMENT,
                             {start, position - start},
                             "dollar must be followed by a capture number or dollar"});
      return diagnostics;
    }
    std::uint64_t capture{};
    while (position < replacement.size() &&
           std::isdigit(static_cast<unsigned char>(replacement[position])) != 0) {
      capture = capture * 10U + static_cast<unsigned>(replacement[position++] - '0');
      if (capture > capture_count) {
        diagnostics.push_back({diagnostic_code::INVALID_REPLACEMENT,
                               {start, position - start},
                               "replacement capture is out of range"});
        return diagnostics;
      }
    }
    if (braced) {
      if (position == replacement.size() || replacement[position] != '}') {
        diagnostics.push_back({diagnostic_code::INVALID_REPLACEMENT,
                               {start, position - start},
                               "braced replacement capture is not terminated"});
        return diagnostics;
      }
      ++position;
    }
    flush_literal();
    output.push_back({replacement_token::kind::CAPTURE, {}, static_cast<std::uint32_t>(capture)});
  }
  flush_literal();
  return diagnostics;
}

}  // namespace

std::vector<diagnostic> verify(instruction_ir const& ir)
{
  std::vector<diagnostic> result;
  auto invalid = [&](source_span span, std::string message) {
    result.push_back({diagnostic_code::INVALID_INSTRUCTION_IR, span, std::move(message)});
  };

  if (ir.entry >= ir.blocks.size()) invalid({}, "entry block is invalid");
  if (ir.accept >= ir.blocks.size()) invalid({}, "accept block is invalid");

  for (std::size_t index = 0; index < ir.blocks.size(); ++index) {
    auto& block = ir.blocks[index];
    if (block.id != index) invalid(block.source, "block ID does not match storage index");

    for (auto edge : block.successors) {
      if (edge.target >= ir.blocks.size()) invalid(block.source, "successor target is invalid");
    }

    bool accepting{};
    std::size_t character_tests{};
    std::size_t advances{};
    for (auto& item : block.instructions) {
      if (std::holds_alternative<emit_accept>(item)) accepting = true;
      if (std::holds_alternative<match_character>(item) ||
          std::holds_alternative<match_literal>(item)) {
        ++character_tests;
      }
      if (std::holds_alternative<advance_cursor>(item)) ++advances;
      if (auto* capture = std::get_if<write_capture>(&item);
          capture != nullptr &&
          (capture->capture_index == 0 || capture->capture_index > ir.capture_count)) {
        invalid(block.source, "capture write index is out of range");
      }
    }
    if (accepting && !block.successors.empty()) {
      invalid(block.source, "accept block has successors");
    }
    if (character_tests > 1) invalid(block.source, "block has multiple character tests");
    if (advances > 1) invalid(block.source, "block advances more than once");
  }

  return result;
}

instruction_result lower(automata_ir const& automata, operation const& selected)
{
  auto diagnostics = verify(automata);
  if (!diagnostics.empty()) { return {std::nullopt, std::move(diagnostics)}; }

  instruction_ir result;
  result.pattern            = automata.pattern;
  result.options            = automata.options;
  result.selected_operation = selected;
  switch (selected.kind) {
    case operation_kind::MATCHES:
      result.control = {false, true, true, true, result_shape::BOOLEAN};
      break;
    case operation_kind::CONTAINS:
      result.control = {true, false, true, true, result_shape::BOOLEAN};
      break;
    case operation_kind::FIND:
      result.control = {true, false, true, true, result_shape::MATCH_SPAN};
      break;
    case operation_kind::FIND_ALL:
      result.control = {true, false, false, true, result_shape::MATCH_SPAN};
      break;
    case operation_kind::COUNT:
      result.control = {true, false, false, true, result_shape::MATCH_COUNT};
      break;
    case operation_kind::EXTRACT:
      result.control = {true, false, true, true, result_shape::CAPTURES};
      break;
    case operation_kind::REPLACE:
      result.control = {true, false, false, true, result_shape::REPLACEMENT};
      break;
    case operation_kind::SPLIT:
      result.control = {true, false, false, true, result_shape::SPLIT_FIELDS};
      break;
  }
  result.entry               = automata.entry;
  result.accept              = automata.accept;
  result.capture_count       = automata.capture_count;
  result.has_alternation     = automata.has_alternation;
  result.has_lazy_quantifier = automata.has_lazy_quantifier;
  result.blocks.reserve(automata.states.size());

  for (auto& state : automata.states) {
    instruction_block block;
    block.id     = state.id;
    block.source = state.source;
    block.successors.reserve(state.edges.size());
    for (auto edge : state.edges)
      block.successors.push_back({edge.target, edge.priority});

    switch (state.kind) {
      case automata_state_kind::JUMP:
      case automata_state_kind::BRANCH: break;
      case automata_state_kind::CONSUME:
        block.instructions.emplace_back(can_peek{1});
        block.instructions.emplace_back(read_character{});
        block.instructions.emplace_back(match_character{state.predicate});
        block.instructions.emplace_back(advance_cursor{1});
        break;
      case automata_state_kind::ASSERTION:
        block.instructions.emplace_back(test_assertion{state.assertion});
        break;
      case automata_state_kind::CAPTURE:
        block.instructions.emplace_back(write_capture{state.capture, state.capture_index});
        break;
      case automata_state_kind::ACCEPT: block.instructions.emplace_back(emit_accept{}); break;
    }
    result.blocks.push_back(std::move(block));
  }

  if (selected.kind == operation_kind::REPLACE) {
    auto replacement_diagnostics =
      parse_replacement(selected.replacement, result.capture_count, result.replacement);
    diagnostics.insert(diagnostics.end(),
                       std::make_move_iterator(replacement_diagnostics.begin()),
                       std::make_move_iterator(replacement_diagnostics.end()));
  }

  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};
  diagnostics = verify(result);
  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};
  return {std::move(result), {}};
}

instruction_result compile_instruction_ir(std::string_view pattern,
                                          operation const& selected,
                                          compile_options const& options,
                                          optimization_options const& optimization)
{
  auto automata = compile_automata(pattern, options);
  if (!automata) return {std::nullopt, std::move(automata.diagnostics)};
  auto instructions = lower(*automata.value, selected);
  if (!instructions) return instructions;
  return optimize(std::move(*instructions.value), optimization);
}

}  // namespace regex_ir

// instruction IR optimization

namespace regex_ir {
namespace {

std::optional<char32_t> singleton(instruction_block const& block)
{
  for (auto& item : block.instructions) {
    if (auto* match = std::get_if<match_character>(&item);
        match != nullptr && match->predicate.is_singleton()) {
      return match->predicate.singleton();
    }
  }
  return std::nullopt;
}

void strip_captures(instruction_ir& ir)
{
  std::vector<bool> observed(ir.capture_count + 1U, false);
  if (ir.selected_operation.kind == operation_kind::EXTRACT) {
    std::fill(observed.begin(), observed.end(), true);
  } else if (ir.selected_operation.kind == operation_kind::REPLACE) {
    for (auto& token : ir.replacement) {
      if (token.type == replacement_token::kind::CAPTURE) { observed[token.capture_index] = true; }
    }
  }
  for (auto& block : ir.blocks) {
    block.instructions.erase(std::remove_if(block.instructions.begin(),
                                            block.instructions.end(),
                                            [&](instruction const& item) {
                                              auto* capture = std::get_if<write_capture>(&item);
                                              return capture != nullptr &&
                                                     !observed[capture->capture_index];
                                            }),
                             block.instructions.end());
  }
}

block_id resolve_empty(instruction_ir const& ir, block_id start)
{
  // stop at cycles because nullable repetition can produce an all-empty component
  std::unordered_set<block_id> visited;
  auto current = start;
  while (current < ir.blocks.size() && visited.insert(current).second) {
    auto& block = ir.blocks[current];
    if (!block.instructions.empty() || block.successors.size() != 1) break;
    current = block.successors.front().target;
  }
  return current;
}

void fold_empty_jumps(instruction_ir& ir)
{
  ir.entry = resolve_empty(ir, ir.entry);
  for (auto& block : ir.blocks) {
    for (auto& edge : block.successors)
      edge.target = resolve_empty(ir, edge.target);
  }
}

void fuse_literals(instruction_ir& ir, std::size_t limit)
{
  if (limit < 2) return;
  std::vector<std::size_t> incoming(ir.blocks.size());
  for (auto& block : ir.blocks) {
    for (auto edge : block.successors) {
      if (edge.target < incoming.size()) ++incoming[edge.target];
    }
  }

  for (auto& block : ir.blocks) {
    auto first = singleton(block);
    if (!first || block.successors.size() != 1) continue;

    std::u32string value{*first};
    auto next = block.successors.front().target;
    std::unordered_set<block_id> visited{block.id};
    // a single incoming edge makes it safe to consume the candidate into this block
    while (value.size() < limit && next < ir.blocks.size() && incoming[next] == 1 &&
           visited.insert(next).second) {
      auto& candidate = ir.blocks[next];
      auto character  = singleton(candidate);
      if (!character || candidate.successors.size() != 1) break;
      value.push_back(*character);
      next = candidate.successors.front().target;
    }
    if (value.size() < 2) continue;

    block.instructions.clear();
    block.instructions.emplace_back(can_peek{static_cast<std::uint32_t>(value.size())});
    block.instructions.emplace_back(match_literal{std::move(value)});
    block.instructions.emplace_back(advance_cursor{
      static_cast<std::uint32_t>(std::get<match_literal>(block.instructions[1]).value.size())});
    block.successors = {{next, 0}};
  }
}

void remove_unreachable(instruction_ir& ir)
{
  if (ir.entry >= ir.blocks.size()) return;
  std::vector<bool> reachable(ir.blocks.size());
  std::vector<block_id> work{ir.entry};
  reachable[ir.entry] = true;
  while (!work.empty()) {
    auto current = work.back();
    work.pop_back();
    for (auto edge : ir.blocks[current].successors) {
      if (edge.target < reachable.size() && !reachable[edge.target]) {
        reachable[edge.target] = true;
        work.push_back(edge.target);
      }
    }
  }

  std::vector<block_id> remap(ir.blocks.size(), invalid_block);
  std::vector<instruction_block> blocks;
  blocks.reserve(ir.blocks.size());
  for (std::size_t old = 0; old < ir.blocks.size(); ++old) {
    if (!reachable[old]) continue;
    remap[old] = static_cast<block_id>(blocks.size());
    auto block = std::move(ir.blocks[old]);
    block.id   = static_cast<block_id>(blocks.size());
    blocks.push_back(std::move(block));
  }
  // rewrite dense IDs only after every old-to-new mapping has been established
  for (auto& block : blocks) {
    for (auto& edge : block.successors)
      edge.target = remap[edge.target];
  }
  ir.entry  = remap[ir.entry];
  ir.accept = remap[ir.accept];
  ir.blocks = std::move(blocks);
}

}  // namespace

instruction_result optimize(instruction_ir ir, optimization_options const& options)
{
  auto diagnostics = verify(ir);
  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};

  if (options.strip_unobserved_captures) strip_captures(ir);
  if (options.fold_epsilon_jumps) fold_empty_jumps(ir);
  if (options.fuse_literals) fuse_literals(ir, options.literal_fusion_limit);
  if (options.fold_epsilon_jumps) fold_empty_jumps(ir);
  if (options.remove_unreachable) remove_unreachable(ir);

  diagnostics = verify(ir);
  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};
  return {std::move(ir), {}};
}

}  // namespace regex_ir

// device IR generation

namespace regex_ir {
namespace {

class source_buffer {
 public:
  template <typename... Args>
  void emit(std::format_string<Args...> format, Args&&... args)
  {
    std::format_to(std::back_inserter(value_), format, std::forward<Args>(args)...);
    value_ += '\n';
  }

  void blank() { value_ += '\n'; }
  [[nodiscard]] std::string take() { return std::move(value_); }

 private:
  std::string value_ = "";
};

void require_identifier(std::string_view value, std::string_view field)
{
  auto first_is_valid = [](unsigned char character) {
    return std::isalpha(character) != 0 || character == '_';
  };
  auto rest_is_valid = [&](unsigned char character) {
    return first_is_valid(character) || std::isdigit(character) != 0;
  };
  if (value.empty() || !first_is_valid(static_cast<unsigned char>(value.front())) ||
      !std::all_of(value.begin() + 1, value.end(), [&](char character) {
        return rest_is_valid(static_cast<unsigned char>(character));
      })) {
    throw std::invalid_argument(std::format("{} must be a valid source identifier", field));
  }
  if (value.starts_with("llvm.") || value.starts_with("nvvm.")) {
    throw std::invalid_argument(std::format("{} uses a reserved identifier", field));
  }
}

void require_codegen_ir(instruction_ir const& ir)
{
  auto diagnostics = verify(ir);
  if (!diagnostics.empty()) throw std::invalid_argument("cannot generate code from invalid IR");
}

std::string nvvm_symbol(std::string_view prefix, std::string_view suffix)
{
  return std::format("{}_{}", prefix, suffix);
}

struct deterministic_nfa_node {
  character_predicate predicate           = character_predicate{};
  std::vector<std::size_t> targets        = std::vector<std::size_t>{};
  std::optional<write_capture> capture    = std::nullopt;
  std::optional<assertion_kind> assertion = std::nullopt;
  bool consumes : 1                       = false;
  bool accepts  : 1                       = false;
};

struct deterministic_nfa_graph {
  std::vector<deterministic_nfa_node> nodes = std::vector<deterministic_nfa_node>{};
  std::size_t entry                         = 0;
};

struct deterministic_capture_action {
  std::uint32_t slot = 0;
  bool reset_end : 1 = false;

  bool operator==(deterministic_capture_action const&) const = default;
};

struct deterministic_interval {
  std::uint32_t first    = 0;
  std::uint32_t last     = 0;
  std::uint16_t class_id = 0;
};

struct deterministic_machine {
  std::array<std::uint16_t, 256> byte_classes           = std::array<std::uint16_t, 256>{};
  std::array<std::uint64_t, 4> start_byte_bitmap        = std::array<std::uint64_t, 4>{};
  std::vector<deterministic_interval> unicode_intervals = std::vector<deterministic_interval>{};
  std::vector<std::uint16_t> transitions                = std::vector<std::uint16_t>{};
  std::vector<std::vector<deterministic_capture_action>> transition_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::vector<deterministic_capture_action>> accept_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::uint8_t> boundary_accepts     = std::vector<std::uint8_t>{};
  std::uint16_t initial_state                    = 0;
  std::uint16_t dead_state                       = 0;
  std::uint16_t class_count                      = 0;
  std::uint16_t state_count                      = 0;
  std::uint16_t state_mask                       = 16383;
  std::uint16_t restart_state                    = std::numeric_limits<std::uint16_t>::max();
  std::uint8_t transition_address_space          = 4;
  std::uint8_t boundary_class_count              = 0;
  std::uint8_t assertion_mask                    = 0;
  std::uint8_t start_byte_range_count            = 0;
  std::optional<assertion_kind> accept_assertion = std::nullopt;
  bool scan_input        : 1                     = false;
  bool accept_at_end     : 1                     = false;
  bool capture_one_pass  : 1                     = false;
  bool assertion_aware   : 1                     = false;
  bool start_byte_filter : 1                     = false;
};

struct glushkov_shift {
  std::uint64_t sources = 0;
  std::uint8_t amount   = 0;
};

struct glushkov_machine {
  deterministic_machine alphabet                     = deterministic_machine{};
  std::vector<std::uint64_t> reach_masks             = std::vector<std::uint64_t>{};
  std::vector<glushkov_shift> shifts                 = std::vector<glushkov_shift>{};
  std::array<std::uint64_t, 64> exception_successors = std::array<std::uint64_t, 64>{};
  std::uint64_t first_set                            = 0;
  std::uint64_t accept_mask                          = 0;
  std::uint64_t exception_mask                       = 0;
  std::optional<std::uint8_t> start_byte             = std::nullopt;
  std::uint8_t position_count                        = 0;
  bool scan_input    : 1                             = false;
  bool accept_at_end : 1                             = false;
};

void build_start_byte_filter(deterministic_machine& machine)
{
  if ((machine.initial_state & 0x8000U) != 0 || machine.dead_state > machine.state_mask ||
      machine.class_count == 0U) {
    return;
  }
  auto initial = static_cast<std::size_t>(machine.initial_state & machine.state_mask);
  auto offset  = initial * machine.class_count;
  if (offset + machine.class_count > machine.transitions.size()) return;

  std::size_t candidates        = 0;
  auto previous_ascii_candidate = false;
  for (std::size_t byte = 0; byte < machine.byte_classes.size(); ++byte) {
    auto transition = machine.transitions[offset + machine.byte_classes[byte]];
    auto target     = static_cast<std::uint16_t>(transition & machine.state_mask);
    auto candidate  = target != machine.dead_state;
    if (byte < 128U && candidate && !previous_ascii_candidate) { ++machine.start_byte_range_count; }
    if (byte < 128U) previous_ascii_candidate = candidate;
    if (!candidate) continue;
    machine.start_byte_bitmap[byte / 64U] |= std::uint64_t{1} << (byte % 64U);
    ++candidates;
  }
  // sparse, simple ranges repay the extra candidate-dispatch control flow.
  machine.start_byte_filter =
    candidates != 0U && candidates <= 16U && machine.start_byte_range_count <= 2U;
}

void build_assertion_start_byte_filter(deterministic_machine& machine)
{
  if (!machine.assertion_aware || machine.class_count == 0U || machine.boundary_class_count == 0U ||
      machine.dead_state > machine.state_mask) {
    return;
  }
  std::size_t candidates        = 0;
  auto previous_ascii_candidate = false;
  for (std::size_t byte = 0; byte < machine.byte_classes.size(); ++byte) {
    auto candidate = false;
    for (std::size_t boundary = 0; boundary < machine.boundary_class_count; ++boundary) {
      auto index = (static_cast<std::size_t>(machine.initial_state) * machine.boundary_class_count +
                    boundary) *
                     machine.class_count +
                   machine.byte_classes[byte];
      if (index >= machine.transitions.size()) return;
      auto target = static_cast<std::uint16_t>(machine.transitions[index] & machine.state_mask);
      candidate |= target != machine.dead_state;
    }
    if (byte < 128U && candidate && !previous_ascii_candidate) { ++machine.start_byte_range_count; }
    if (byte < 128U) previous_ascii_candidate = candidate;
    if (!candidate) continue;
    machine.start_byte_bitmap[byte / 64U] |= std::uint64_t{1} << (byte % 64U);
    ++candidates;
  }
  machine.start_byte_filter =
    candidates != 0U && candidates <= 16U && machine.start_byte_range_count <= 2U;
}

void build_restart_acceleration(deterministic_machine& machine)
{
  if ((machine.initial_state & 0x8000U) != 0 || machine.class_count == 0U ||
      machine.dead_state > machine.state_mask) {
    return;
  }
  auto initial    = static_cast<std::uint16_t>(machine.initial_state & machine.state_mask);
  auto transition = [&](std::uint16_t state, std::uint16_t character_class) {
    return machine
      .transitions[static_cast<std::size_t>(state) * machine.class_count + character_class];
  };

  auto prefix_state = std::numeric_limits<std::uint16_t>::max();
  std::vector<bool> prefix_classes(machine.class_count);
  // every skipped suffix must be in the same prefix state at the failure byte.
  for (std::uint16_t character_class = 0; character_class < machine.class_count;
       ++character_class) {
    auto encoded = transition(initial, character_class);
    auto target  = static_cast<std::uint16_t>(encoded & machine.state_mask);
    if (target == machine.dead_state) continue;
    if ((encoded & 0xC000U) != 0U || target == initial) return;
    if (prefix_state == std::numeric_limits<std::uint16_t>::max()) {
      prefix_state = target;
    } else if (target != prefix_state) {
      return;
    }
    prefix_classes[character_class] = true;
  }
  if (prefix_state == std::numeric_limits<std::uint16_t>::max()) return;

  for (std::uint16_t character_class = 0; character_class < machine.class_count;
       ++character_class) {
    if (!prefix_classes[character_class]) continue;
    auto encoded = transition(prefix_state, character_class);
    if ((encoded & 0x4000U) != 0U || (encoded & machine.state_mask) != prefix_state) return;
  }
  for (std::uint16_t state = 0; state < machine.state_count; ++state) {
    if (state == initial || state == prefix_state) continue;
    for (std::uint16_t character_class = 0; character_class < machine.class_count;
         ++character_class) {
      if ((transition(state, character_class) & machine.state_mask) == prefix_state) return;
    }
  }
  machine.restart_state = prefix_state;
}

void set_machine_bit(std::vector<std::uint64_t>& bits, std::size_t index)
{
  bits[index / 64] |= std::uint64_t{1} << (index % 64);
}

bool machine_bit(std::vector<std::uint64_t> const& bits, std::size_t index)
{
  return (bits[index / 64] & (std::uint64_t{1} << (index % 64))) != 0;
}

character_predicate singleton_predicate(char32_t value)
{
  character_predicate result;
  result.ranges.push_back({value, value});
  return result;
}

std::optional<deterministic_nfa_graph> make_deterministic_graph(instruction_ir const& ir)
{
  if (ir.entry >= ir.blocks.size()) return std::nullopt;
  std::vector<std::size_t> block_starts(ir.blocks.size());
  std::vector<std::size_t> block_lengths(ir.blocks.size(), 1);
  std::size_t node_count = 0;
  for (auto& block : ir.blocks) {
    match_character const* match    = nullptr;
    match_literal const* literal    = nullptr;
    can_peek const* peek            = nullptr;
    advance_cursor const* advance   = nullptr;
    test_assertion const* assertion = nullptr;
    bool accepts                    = false;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<can_peek>(&item)) peek = candidate;
      if (auto* candidate = std::get_if<advance_cursor>(&item)) advance = candidate;
      if (auto* candidate = std::get_if<test_assertion>(&item)) {
        if (assertion != nullptr) return std::nullopt;
        assertion = candidate;
      }
      if (std::holds_alternative<emit_accept>(item)) accepts = true;
    }
    if (match != nullptr && literal != nullptr) return std::nullopt;
    if ((match != nullptr || literal != nullptr) && (accepts || assertion != nullptr)) {
      return std::nullopt;
    }
    if (match != nullptr && (peek == nullptr || peek->characters != 1 || advance == nullptr ||
                             advance->characters != 1)) {
      return std::nullopt;
    }
    if (literal != nullptr) {
      if (literal->value.empty() || peek == nullptr || peek->characters != literal->value.size() ||
          advance == nullptr || advance->characters != literal->value.size()) {
        return std::nullopt;
      }
      block_lengths[block.id] = literal->value.size();
    }
    block_starts[block.id] = node_count;
    node_count += block_lengths[block.id];
  }
  if (node_count == 0) return std::nullopt;

  deterministic_nfa_graph graph;
  graph.nodes.resize(node_count);
  graph.entry = block_starts[ir.entry];
  for (auto& block : ir.blocks) {
    auto start                      = block_starts[block.id];
    match_character const* match    = nullptr;
    match_literal const* literal    = nullptr;
    write_capture const* capture    = nullptr;
    test_assertion const* assertion = nullptr;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<write_capture>(&item)) capture = candidate;
      if (auto* candidate = std::get_if<test_assertion>(&item)) assertion = candidate;
      if (std::holds_alternative<emit_accept>(item)) graph.nodes[start].accepts = true;
    }
    if (capture != nullptr) graph.nodes[start].capture = *capture;
    if (assertion != nullptr) graph.nodes[start].assertion = assertion->kind;

    auto append_successors = [&](deterministic_nfa_node& node) {
      auto successors = block.successors;
      std::stable_sort(successors.begin(), successors.end(), [](auto& left, auto& right) {
        return left.priority < right.priority;
      });
      for (auto edge : successors)
        node.targets.push_back(block_starts[edge.target]);
    };

    if (match != nullptr) {
      graph.nodes[start].predicate = match->predicate;
      graph.nodes[start].consumes  = true;
      append_successors(graph.nodes[start]);
    } else if (literal != nullptr) {
      for (std::size_t index = 0; index < literal->value.size(); ++index) {
        auto& node     = graph.nodes[start + index];
        node.predicate = singleton_predicate(literal->value[index]);
        node.consumes  = true;
        if (index + 1 < literal->value.size()) {
          node.targets.push_back(start + index + 1);
        } else {
          append_successors(node);
        }
      }
    } else {
      append_successors(graph.nodes[start]);
    }
  }
  return graph;
}

std::optional<std::vector<std::uint32_t>> make_deterministic_alphabet(
  std::vector<deterministic_nfa_node> const& nodes, deterministic_machine& machine)
{
  constexpr std::uint32_t unicode_limit = 0x110000;
  auto word_count                       = (nodes.size() + 63U) / 64U;
  std::vector<std::uint32_t> boundaries{0, 256, unicode_limit};
  for (auto& node : nodes) {
    if (!node.consumes) continue;
    if (node.predicate.recognized == predicate_class::ANY && !node.predicate.matches_newline) {
      boundaries.insert(boundaries.end(), {10, 11});
      if (node.predicate.extended_newline) {
        boundaries.insert(boundaries.end(), {13, 14, 133, 134, 8232, 8234});
      }
    }
    for (auto range : node.predicate.ranges) {
      auto first = static_cast<std::uint32_t>(range.first);
      auto last  = static_cast<std::uint32_t>(range.last);
      if (first < unicode_limit) boundaries.push_back(first);
      if (last < unicode_limit - 1) boundaries.push_back(last + 1);
    }
  }
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

  std::map<std::vector<std::uint64_t>, std::uint16_t> class_ids;
  std::vector<std::uint32_t> representatives;
  std::vector<deterministic_interval> intervals;
  for (std::size_t index = 0; index + 1 < boundaries.size(); ++index) {
    auto first = boundaries[index];
    auto last  = boundaries[index + 1] - 1;
    if (first > last || first >= unicode_limit) continue;
    std::vector<std::uint64_t> signature(word_count);
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      if (nodes[node_index].consumes &&
          nodes[node_index].predicate.matches(static_cast<char32_t>(first))) {
        set_machine_bit(signature, node_index);
      }
    }
    auto existing          = class_ids.find(signature);
    std::uint16_t class_id = 0;
    if (existing == class_ids.end()) {
      if (class_ids.size() >= 32767) return std::nullopt;
      class_id = static_cast<std::uint16_t>(class_ids.size());
      class_ids.emplace(std::move(signature), class_id);
      representatives.push_back(first);
    } else {
      class_id = existing->second;
    }
    intervals.push_back({first, last, class_id});
  }
  if (class_ids.empty()) return std::nullopt;

  machine.class_count        = static_cast<std::uint16_t>(class_ids.size());
  std::size_t interval_index = 0;
  for (std::size_t value = 0; value < machine.byte_classes.size(); ++value) {
    while (interval_index + 1 < intervals.size() && value > intervals[interval_index].last)
      ++interval_index;
    machine.byte_classes[value] = intervals[interval_index].class_id;
  }
  for (auto interval : intervals) {
    if (interval.last < 256) continue;
    interval.first = std::max(interval.first, 256U);
    if (!machine.unicode_intervals.empty() &&
        machine.unicode_intervals.back().class_id == interval.class_id &&
        machine.unicode_intervals.back().last + 1 == interval.first) {
      machine.unicode_intervals.back().last = interval.last;
    } else {
      machine.unicode_intervals.push_back(interval);
    }
  }
  return representatives;
}

std::optional<glushkov_machine> make_glushkov_machine(instruction_ir const& ir, bool scan_input)
{
  auto graph = make_deterministic_graph(ir);
  if (!graph) return std::nullopt;

  std::vector<std::size_t> positions;
  std::vector<std::size_t> position_ids(graph->nodes.size(),
                                        std::numeric_limits<std::size_t>::max());
  for (std::size_t index = 0; index < graph->nodes.size(); ++index) {
    auto& node = graph->nodes[index];
    if (node.capture.has_value() || node.assertion.has_value()) return std::nullopt;
    if (!node.consumes) continue;
    if (positions.size() == 64U) return std::nullopt;
    position_ids[index] = positions.size();
    positions.push_back(index);
  }
  if (positions.empty()) return std::nullopt;

  struct closure_result {
    std::uint64_t positions = 0;
    bool accepts : 1        = false;
  };
  // epsilon closure stops at consuming nodes because those nodes are the positions.
  auto close = [&](std::vector<std::size_t> seeds) {
    closure_result result;
    std::vector<bool> visited(graph->nodes.size(), false);
    while (!seeds.empty()) {
      auto index = seeds.back();
      seeds.pop_back();
      if (index >= graph->nodes.size() || visited[index]) continue;
      visited[index] = true;

      auto& node = graph->nodes[index];
      if (node.accepts) result.accepts = true;
      if (node.consumes) {
        result.positions |= std::uint64_t{1} << position_ids[index];
        continue;
      }
      seeds.insert(seeds.end(), node.targets.begin(), node.targets.end());
    }
    return result;
  };

  glushkov_machine machine;
  machine.position_count = static_cast<std::uint8_t>(positions.size());
  machine.scan_input     = scan_input;
  machine.accept_at_end  = ir.control.require_end;

  auto initial = close({graph->entry});
  if (initial.accepts || initial.positions == 0U) return std::nullopt;
  machine.first_set = initial.positions;

  auto follow = std::array<std::uint64_t, 64>{};
  for (std::size_t position = 0; position < positions.size(); ++position) {
    auto& node       = graph->nodes[positions[position]];
    auto successors  = close(node.targets);
    follow[position] = successors.positions;
    if (successors.accepts) machine.accept_mask |= std::uint64_t{1} << position;
  }
  if (machine.accept_mask == 0U) return std::nullopt;

  auto representatives = make_deterministic_alphabet(graph->nodes, machine.alphabet);
  if (!representatives) return std::nullopt;
  machine.reach_masks.reserve(representatives->size());
  for (auto representative : *representatives) {
    std::uint64_t reach = 0;
    for (std::size_t position = 0; position < positions.size(); ++position) {
      if (graph->nodes[positions[position]].predicate.matches(
            static_cast<char32_t>(representative))) {
        reach |= std::uint64_t{1} << position;
      }
    }
    machine.reach_masks.push_back(reach);
  }

  std::map<std::int32_t, std::uint64_t> span_sources;
  for (std::size_t position = 0; position < positions.size(); ++position) {
    for (std::size_t successor = 0; successor < positions.size(); ++successor) {
      if ((follow[position] & (std::uint64_t{1} << successor)) == 0U) continue;
      auto span = static_cast<std::int32_t>(successor) - static_cast<std::int32_t>(position);
      if (span > 0) {
        span_sources[span] |= std::uint64_t{1} << position;
      } else {
        machine.exception_mask |= std::uint64_t{1} << position;
        machine.exception_successors[position] |= std::uint64_t{1} << successor;
      }
    }
  }

  std::vector<std::pair<std::int32_t, std::uint64_t>> spans(span_sources.begin(),
                                                            span_sources.end());
  // common forward spans become shifts; uncommon and backward edges stay explicit.
  std::stable_sort(spans.begin(), spans.end(), [](auto& left, auto& right) {
    return std::popcount(left.second) > std::popcount(right.second);
  });
  auto shift_count = std::min<std::size_t>(spans.size(), 8U);
  machine.shifts.reserve(shift_count);
  for (std::size_t index = 0; index < shift_count; ++index) {
    machine.shifts.push_back({spans[index].second, static_cast<std::uint8_t>(spans[index].first)});
  }
  for (std::size_t index = shift_count; index < spans.size(); ++index) {
    auto sources = spans[index].second;
    while (sources != 0U) {
      auto position = static_cast<std::size_t>(std::countr_zero(sources));
      sources &= sources - 1U;
      auto successor = position + static_cast<std::size_t>(spans[index].first);
      machine.exception_mask |= std::uint64_t{1} << position;
      machine.exception_successors[position] |= std::uint64_t{1} << successor;
    }
  }

  auto first_positions = machine.first_set;
  std::optional<std::uint8_t> start_byte;
  while (first_positions != 0U) {
    auto position = static_cast<std::size_t>(std::countr_zero(first_positions));
    first_positions &= first_positions - 1U;
    auto& predicate = graph->nodes[positions[position]].predicate;
    if (!predicate.is_singleton() || predicate.singleton() > 0x7f) {
      start_byte.reset();
      break;
    }
    auto byte = static_cast<std::uint8_t>(predicate.singleton());
    if (start_byte.has_value() && *start_byte != byte) {
      start_byte.reset();
      break;
    }
    start_byte = byte;
  }
  machine.start_byte = start_byte;
  return machine;
}

bool prefer_glushkov(glushkov_machine const& machine,
                     std::optional<deterministic_machine> const& deterministic)
{
  if (!deterministic.has_value()) return true;

  auto exceptions = std::popcount(machine.exception_mask);
  // use the position machine only when it removes a large table or is a long linear graph.
  auto large_linear =
    machine.position_count >= 32U && machine.shifts.size() == 1U && machine.exception_mask == 0U;
  return (deterministic->transition_address_space == 1U && exceptions <= 5) || large_linear;
}

std::uint8_t assertion_bit(assertion_kind assertion)
{
  return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(assertion));
}

std::optional<deterministic_machine> make_assertion_deterministic_machine(
  instruction_ir const& ir, deterministic_nfa_graph const& graph, bool scan_input)
{
  deterministic_machine machine;
  machine.scan_input      = scan_input;
  machine.accept_at_end   = ir.control.require_end;
  machine.state_mask      = 32767U;
  machine.assertion_aware = true;
  for (auto& node : graph.nodes) {
    if (node.capture.has_value()) return std::nullopt;
    if (node.assertion.has_value()) machine.assertion_mask |= assertion_bit(*node.assertion);
  }
  if (machine.assertion_mask == 0) return std::nullopt;

  std::vector<std::uint8_t> context_masks;
  for (std::uint8_t mask = 0; mask < 64U; ++mask) {
    auto relevant = static_cast<std::uint8_t>(mask & machine.assertion_mask);
    if (std::find(context_masks.begin(), context_masks.end(), relevant) == context_masks.end()) {
      context_masks.push_back(relevant);
    }
  }
  machine.boundary_class_count = static_cast<std::uint8_t>(context_masks.size());

  auto representatives = make_deterministic_alphabet(graph.nodes, machine);
  if (!representatives) return std::nullopt;

  struct assertion_closure {
    std::vector<std::size_t> consumers = std::vector<std::size_t>{};
    bool accepts : 1                   = false;
  };
  auto close = [&](std::vector<std::size_t> seeds, std::uint8_t assertion_mask) {
    assertion_closure result;
    std::vector<bool> visited(graph.nodes.size());
    std::vector<std::size_t> work;
    for (auto seed = seeds.rbegin(); seed != seeds.rend(); ++seed)
      work.push_back(*seed);
    while (!work.empty()) {
      auto index = work.back();
      work.pop_back();
      if (visited[index]) continue;
      visited[index] = true;
      auto& node     = graph.nodes[index];
      if (node.assertion.has_value() && (assertion_mask & assertion_bit(*node.assertion)) == 0) {
        continue;
      }
      if (node.accepts) result.accepts = true;
      if (node.consumes) {
        result.consumers.push_back(index);
        continue;
      }
      for (auto target = node.targets.rbegin(); target != node.targets.rend(); ++target)
        work.push_back(*target);
    }
    std::sort(result.consumers.begin(), result.consumers.end());
    result.consumers.erase(std::unique(result.consumers.begin(), result.consumers.end()),
                           result.consumers.end());
    return result;
  };
  auto canonicalize = [](std::vector<std::size_t>& state) {
    std::sort(state.begin(), state.end());
    state.erase(std::unique(state.begin(), state.end()), state.end());
  };

  constexpr std::size_t max_table_items = 4U * 1024U * 1024U;
  std::map<std::vector<std::size_t>, std::uint16_t> state_ids;
  std::vector<std::vector<std::size_t>> states{{graph.entry}};
  state_ids.emplace(states.front(), 0);
  machine.dead_state = std::numeric_limits<std::uint16_t>::max();
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (states.size() * context_masks.size() * representatives->size() > max_table_items) {
      return std::nullopt;
    }
    for (auto context : context_masks) {
      auto seeds = states[state_index];
      if (scan_input) {
        seeds.push_back(graph.entry);
        canonicalize(seeds);
      }
      auto closure = close(std::move(seeds), context);
      machine.boundary_accepts.push_back(closure.accepts ? 1U : 0U);
      for (auto representative : *representatives) {
        std::vector<std::size_t> next;
        for (auto node_index : closure.consumers) {
          auto& node = graph.nodes[node_index];
          if (!node.predicate.matches(static_cast<char32_t>(representative))) continue;
          next.insert(next.end(), node.targets.begin(), node.targets.end());
        }
        canonicalize(next);
        auto existing        = state_ids.find(next);
        std::uint16_t target = 0;
        if (existing == state_ids.end()) {
          if (states.size() >= machine.state_mask) return std::nullopt;
          target = static_cast<std::uint16_t>(states.size());
          state_ids.emplace(next, target);
          states.push_back(std::move(next));
        } else {
          target = existing->second;
        }
        if (states[target].empty()) machine.dead_state = target;
        machine.transitions.push_back(
          static_cast<std::uint16_t>(target | (closure.accepts ? 0x8000U : 0U)));
      }
    }
  }
  machine.initial_state = 0;
  machine.state_count   = static_cast<std::uint16_t>(states.size());
  if (machine.transitions.size() * sizeof(std::uint16_t) > 32U * 1024U) {
    machine.transition_address_space = 1;
  }
  build_assertion_start_byte_filter(machine);
  return machine;
}

std::optional<deterministic_machine> make_deterministic_machine(instruction_ir const& ir,
                                                                bool scan_input,
                                                                bool preserve_priority)
{
  auto block_assertion = [](instruction_block const& block) -> std::optional<assertion_kind> {
    if (block.instructions.size() != 1U ||
        !std::holds_alternative<test_assertion>(block.instructions.front())) {
      return std::nullopt;
    }
    return std::get<test_assertion>(block.instructions.front()).kind;
  };
  std::optional<block_id> begin_anchor;
  std::optional<block_id> end_anchor;
  std::optional<assertion_kind> end_assertion;
  bool has_assertions = false;
  if (ir.entry < ir.blocks.size()) {
    auto assertion = block_assertion(ir.blocks[ir.entry]);
    if (assertion == assertion_kind::BEGIN_INPUT ||
        (assertion == assertion_kind::BEGIN_LINE && !ir.options.multiline)) {
      begin_anchor = ir.entry;
    }
  }
  if (ir.accept < ir.blocks.size()) {
    std::vector<block_id> predecessors;
    for (auto& block : ir.blocks) {
      for (auto edge : block.successors) {
        if (edge.target == ir.accept) predecessors.push_back(block.id);
      }
    }
    if (predecessors.size() == 1U) {
      auto assertion = block_assertion(ir.blocks[predecessors.front()]);
      if (assertion == assertion_kind::END_INPUT || assertion == assertion_kind::END_LINE) {
        end_anchor    = predecessors.front();
        end_assertion = assertion;
      }
    }
  }
  for (auto& block : ir.blocks) {
    for (auto& item : block.instructions) {
      if (std::holds_alternative<test_assertion>(item)) {
        has_assertions = true;
        if (preserve_priority && block.id != begin_anchor && block.id != end_anchor) {
          return std::nullopt;
        }
      }
    }
  }
  if (has_assertions && !preserve_priority) {
    auto graph = make_deterministic_graph(ir);
    if (!graph) return std::nullopt;
    return make_assertion_deterministic_machine(ir, *graph, scan_input);
  }
  if ((begin_anchor.has_value() || end_anchor.has_value()) &&
      ir.control.result != result_shape::BOOLEAN) {
    return std::nullopt;
  }

  std::vector<std::size_t> block_starts(ir.blocks.size());
  std::vector<std::size_t> block_lengths(ir.blocks.size(), 1);
  std::size_t node_count = 0;
  for (auto& block : ir.blocks) {
    match_character const* match  = nullptr;
    match_literal const* literal  = nullptr;
    can_peek const* peek          = nullptr;
    advance_cursor const* advance = nullptr;
    bool accepts                  = false;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<can_peek>(&item)) peek = candidate;
      if (auto* candidate = std::get_if<advance_cursor>(&item)) advance = candidate;
      if (std::holds_alternative<emit_accept>(item)) accepts = true;
    }
    if (match != nullptr && literal != nullptr) return std::nullopt;
    if ((match != nullptr || literal != nullptr) && accepts) return std::nullopt;
    if (match != nullptr && (peek == nullptr || peek->characters != 1 || advance == nullptr ||
                             advance->characters != 1)) {
      return std::nullopt;
    }
    if (literal != nullptr) {
      if (literal->value.empty() || peek == nullptr || peek->characters != literal->value.size() ||
          advance == nullptr || advance->characters != literal->value.size()) {
        return std::nullopt;
      }
      block_lengths[block.id] = literal->value.size();
    }
    block_starts[block.id] = node_count;
    node_count += block_lengths[block.id];
  }
  if (node_count == 0) return std::nullopt;

  std::vector<deterministic_nfa_node> nodes(node_count);
  for (auto& block : ir.blocks) {
    auto start                   = block_starts[block.id];
    match_character const* match = nullptr;
    match_literal const* literal = nullptr;
    write_capture const* capture = nullptr;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<write_capture>(&item)) capture = candidate;
      if (std::holds_alternative<emit_accept>(item)) nodes[start].accepts = true;
    }
    if (capture != nullptr) nodes[start].capture = *capture;

    auto append_successors = [&](deterministic_nfa_node& node) {
      auto successors = block.successors;
      std::stable_sort(successors.begin(), successors.end(), [](auto& left, auto& right) {
        return left.priority < right.priority;
      });
      for (auto edge : successors)
        node.targets.push_back(block_starts[edge.target]);
    };

    if (match != nullptr) {
      nodes[start].predicate = match->predicate;
      nodes[start].consumes  = true;
      append_successors(nodes[start]);
    } else if (literal != nullptr) {
      for (std::size_t index = 0; index < literal->value.size(); ++index) {
        auto& node     = nodes[start + index];
        node.predicate = singleton_predicate(literal->value[index]);
        node.consumes  = true;
        if (index + 1 < literal->value.size()) {
          node.targets.push_back(start + index + 1);
        } else {
          append_successors(node);
        }
      }
    } else {
      append_successors(nodes[start]);
    }
  }

  auto bit_count  = nodes.size() + 1;
  auto word_count = (bit_count + 63) / 64;
  auto accept_bit = nodes.size();
  struct closure_state {
    std::vector<std::uint64_t> bits  = std::vector<std::uint64_t>{};
    std::vector<std::size_t> ordered = std::vector<std::size_t>{};
    std::vector<std::vector<deterministic_capture_action>> captures =
      std::vector<std::vector<deterministic_capture_action>>{};
  };
  struct closure_work_item {
    std::size_t node                                  = 0;
    std::vector<deterministic_capture_action> actions = std::vector<deterministic_capture_action>{};
  };
  auto capture_paths_deterministic = true;
  auto empty_bits                  = [&] { return std::vector<std::uint64_t>(word_count); };
  auto closure                     = [&](std::vector<std::size_t> const& seeds) {
    closure_state result{empty_bits(), {}, {}};
    auto visited = empty_bits();
    std::vector<std::optional<std::vector<deterministic_capture_action>>> visited_actions(
      nodes.size());
    std::vector<closure_work_item> work;
    for (auto seed = seeds.rbegin(); seed != seeds.rend(); ++seed)
      work.push_back({*seed, {}});
    while (!work.empty()) {
      auto current = std::move(work.back());
      work.pop_back();
      auto index = current.node;
      if (machine_bit(visited, index)) {
        if (visited_actions[index] != current.actions) capture_paths_deterministic = false;
        continue;
      }
      set_machine_bit(visited, index);
      visited_actions[index] = current.actions;
      auto& node             = nodes[index];
      if (node.capture.has_value()) {
        auto slot = node.capture->capture_index * 2U +
                    (node.capture->action == capture_action::END ? 1U : 0U);
        current.actions.push_back({slot, node.capture->action == capture_action::BEGIN});
      }
      if (node.accepts) {
        set_machine_bit(result.bits, accept_bit);
        result.ordered.push_back(accept_bit);
        result.captures.push_back(current.actions);
        if (preserve_priority) break;
      }
      if (node.consumes) {
        set_machine_bit(result.bits, index);
        result.ordered.push_back(index);
        result.captures.push_back(current.actions);
        continue;
      }
      for (auto target = node.targets.rbegin(); target != node.targets.rend(); ++target) {
        work.push_back({*target, current.actions});
      }
    }
    if (!preserve_priority) {
      std::sort(result.ordered.begin(), result.ordered.end());
      result.ordered.erase(std::unique(result.ordered.begin(), result.ordered.end()),
                           result.ordered.end());
      result.captures.assign(result.ordered.size(), {});
    }
    return result;
  };

  auto start_state = closure({block_starts[ir.entry]});

  constexpr std::uint32_t unicode_limit = 0x110000;
  std::vector<std::uint32_t> boundaries{0, 256, unicode_limit};
  for (auto& node : nodes) {
    if (!node.consumes) continue;
    if (node.predicate.recognized == predicate_class::ANY && !node.predicate.matches_newline) {
      boundaries.insert(boundaries.end(), {10, 11});
      if (node.predicate.extended_newline) {
        boundaries.insert(boundaries.end(), {13, 14, 133, 134, 8232, 8234});
      }
    }
    for (auto range : node.predicate.ranges) {
      auto first = static_cast<std::uint32_t>(range.first);
      auto last  = static_cast<std::uint32_t>(range.last);
      if (first < unicode_limit) boundaries.push_back(first);
      if (last < unicode_limit - 1) boundaries.push_back(last + 1);
    }
  }
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

  std::map<std::vector<std::uint64_t>, std::uint16_t> class_ids;
  std::vector<std::uint32_t> representatives;
  std::vector<deterministic_interval> intervals;
  for (std::size_t index = 0; index + 1 < boundaries.size(); ++index) {
    auto first = boundaries[index];
    auto last  = boundaries[index + 1] - 1;
    if (first > last || first >= unicode_limit) continue;
    auto signature = empty_bits();
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      if (nodes[node_index].consumes &&
          nodes[node_index].predicate.matches(static_cast<char32_t>(first))) {
        set_machine_bit(signature, node_index);
      }
    }
    auto existing          = class_ids.find(signature);
    std::uint16_t class_id = 0;
    if (existing == class_ids.end()) {
      if (class_ids.size() >= 32767) return std::nullopt;
      class_id = static_cast<std::uint16_t>(class_ids.size());
      class_ids.emplace(std::move(signature), class_id);
      representatives.push_back(first);
    } else {
      class_id = existing->second;
    }
    intervals.push_back({first, last, class_id});
  }
  if (class_ids.empty()) return std::nullopt;

  deterministic_machine machine;
  machine.class_count        = static_cast<std::uint16_t>(class_ids.size());
  machine.scan_input         = scan_input && !begin_anchor.has_value();
  machine.accept_at_end      = ir.control.require_end;
  machine.accept_assertion   = end_assertion;
  machine.state_mask         = preserve_priority ? 16383U : 32767U;
  std::size_t interval_index = 0;
  for (std::size_t value = 0; value < machine.byte_classes.size(); ++value) {
    while (interval_index + 1 < intervals.size() && value > intervals[interval_index].last)
      ++interval_index;
    machine.byte_classes[value] = intervals[interval_index].class_id;
  }
  for (auto interval : intervals) {
    if (interval.last < 256) continue;
    interval.first = std::max(interval.first, 256U);
    if (!machine.unicode_intervals.empty() &&
        machine.unicode_intervals.back().class_id == interval.class_id &&
        machine.unicode_intervals.back().last + 1 == interval.first) {
      machine.unicode_intervals.back().last = interval.last;
    } else {
      machine.unicode_intervals.push_back(interval);
    }
  }

  auto max_dfa_states                   = static_cast<std::size_t>(machine.state_mask);
  constexpr std::size_t max_table_items = 4 * 1024 * 1024;
  std::map<std::vector<std::size_t>, std::uint16_t> state_ids;
  std::vector<std::vector<std::uint64_t>> states;
  std::vector<std::vector<std::size_t>> state_orders;
  std::vector<std::vector<std::vector<deterministic_capture_action>>> state_capture_actions;
  state_ids.emplace(start_state.ordered, 0);
  states.push_back(start_state.bits);
  state_orders.push_back(start_state.ordered);
  state_capture_actions.push_back(start_state.captures);
  machine.dead_state   = std::numeric_limits<std::uint16_t>::max();
  auto strict_one_pass = !machine_bit(start_state.bits, accept_bit);
  auto terminal_accept = true;
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (states.size() * representatives.size() > max_table_items) return std::nullopt;
    for (auto representative : representatives) {
      std::vector<std::size_t> seeds;
      std::vector<deterministic_capture_action> transition_captures;
      std::vector<deterministic_capture_action> deferred_accept_captures;
      auto matching_consumers  = 0U;
      auto has_deferred_accept = false;
      for (std::size_t order_index = 0; order_index < state_orders[state_index].size();
           ++order_index) {
        auto node_index = state_orders[state_index][order_index];
        if (node_index == accept_bit) {
          if (preserve_priority) {
            has_deferred_accept      = true;
            deferred_accept_captures = state_capture_actions[state_index][order_index];
            break;
          }
          continue;
        }
        auto& node = nodes[node_index];
        if (!node.predicate.matches(static_cast<char32_t>(representative))) continue;
        if (matching_consumers++ == 0U) {
          transition_captures = state_capture_actions[state_index][order_index];
        }
        seeds.insert(seeds.end(), node.targets.begin(), node.targets.end());
      }
      if (matching_consumers > 1U) strict_one_pass = false;
      auto next              = closure(seeds);
      auto discovered_accept = machine_bit(next.bits, accept_bit);
      auto stop_before       = preserve_priority && has_deferred_accept && matching_consumers == 0U;
      if (preserve_priority && has_deferred_accept && matching_consumers != 0U &&
          !discovered_accept) {
        // preserve a lower-priority accept while descendants of earlier threads continue.
        set_machine_bit(next.bits, accept_bit);
        next.ordered.push_back(accept_bit);
        next.captures.push_back(std::move(deferred_accept_captures));
      }
      // A deferred acceptance wins before this character. Injecting a new scan start here would
      // make a nullable initial state accepting again and incorrectly consume the character.
      if (machine.scan_input && !stop_before) {
        for (std::size_t word = 0; word < next.bits.size(); ++word)
          next.bits[word] |= start_state.bits[word];
        for (std::size_t start_index = 0; start_index < start_state.ordered.size(); ++start_index) {
          auto node_index = start_state.ordered[start_index];
          if (std::find(next.ordered.begin(), next.ordered.end(), node_index) ==
              next.ordered.end()) {
            next.ordered.push_back(node_index);
            next.captures.push_back(start_state.captures[start_index]);
          }
        }
        if (!preserve_priority) {
          std::sort(next.ordered.begin(), next.ordered.end());
          next.ordered.erase(std::unique(next.ordered.begin(), next.ordered.end()),
                             next.ordered.end());
          next.captures.assign(next.ordered.size(), {});
        }
      }
      auto existing        = state_ids.find(next.ordered);
      std::uint16_t target = 0;
      if (existing == state_ids.end()) {
        if (states.size() >= max_dfa_states) return std::nullopt;
        target = static_cast<std::uint16_t>(states.size());
        state_ids.emplace(next.ordered, target);
        states.push_back(std::move(next.bits));
        state_orders.push_back(std::move(next.ordered));
        state_capture_actions.push_back(std::move(next.captures));
      } else {
        target = existing->second;
        if (state_capture_actions[target] != next.captures) { capture_paths_deterministic = false; }
      }
      if (std::none_of(
            states[target].begin(), states[target].end(), [](auto word) { return word != 0; })) {
        machine.dead_state = target;
      }
      auto accepts  = machine_bit(states[target], accept_bit);
      auto consumes = std::any_of(state_orders[target].begin(),
                                  state_orders[target].end(),
                                  [&](auto node_index) { return node_index != accept_bit; });
      if (accepts && consumes) terminal_accept = false;
      std::vector<deterministic_capture_action> accept_captures;
      if (accepts) {
        auto accept =
          std::find(state_orders[target].begin(), state_orders[target].end(), accept_bit);
        auto index      = static_cast<std::size_t>(accept - state_orders[target].begin());
        accept_captures = state_capture_actions[target][index];
      }
      auto update_accept =
        preserve_priority ? discovered_accept : machine_bit(states[target], accept_bit);
      if (update_accept) target |= 0x8000U;
      if (stop_before) target |= 0x4000U;
      machine.transitions.push_back(target);
      machine.transition_capture_actions.push_back(std::move(transition_captures));
      machine.accept_capture_actions.push_back(std::move(accept_captures));
    }
  }
  machine.state_count      = static_cast<std::uint16_t>(states.size());
  machine.initial_state    = machine_bit(start_state.bits, accept_bit) ? 0x8000U : 0U;
  machine.capture_one_pass = strict_one_pass && terminal_accept && capture_paths_deterministic;
  if (machine.transitions.size() * sizeof(std::uint16_t) > 32U * 1024U) {
    machine.transition_address_space = 1;
  }
  build_start_byte_filter(machine);
  build_restart_acceleration(machine);
  return machine;
}

class nvvm_ir_renderer {
 public:
  nvvm_ir_renderer(instruction_ir const& ir, nvvm_ir_codegen_options const& options)
    : ir_(ir), options_(options), public_execute_function_(options.execute_function)
  {
    exact_ascii_literal_metadata_ = exact_ascii_literal();
    auto result                   = ir_.control.result;
    auto adapt_count              = result == result_shape::MATCH_COUNT;
    auto adapt_find = result == result_shape::MATCH_SPAN &&
                      ir_.selected_operation.kind == operation_kind::FIND &&
                      !ir_.options.find_match_end_observable;
    if (begins_at_input_start() && (adapt_count || adapt_find)) {
      anchored_boolean_result_  = result;
      options_.execute_function = name("anchored_boolean_execute");
      ir_.control.result        = result_shape::BOOLEAN;
      ir_.control.first_only    = true;
    }
  }

  compile_result render()
  {
    require_codegen_ir(ir_);
    require_identifier(options_.symbol_prefix, "symbol_prefix");
    require_identifier(options_.execute_function, "execute_function");
    require_identifier(public_execute_function_, "execute_function");
    whole_match_captures_ = whole_match_captures();
    capture_slots_        = live_capture_slots();
    if (!whole_match_captures_.empty()) {
      for (auto& block : ir_.blocks) {
        std::erase_if(block.instructions, [&](instruction const& item) {
          auto* capture = std::get_if<write_capture>(&item);
          return capture != nullptr &&
                 std::find(whole_match_captures_.begin(),
                           whole_match_captures_.end(),
                           capture->capture_index) != whole_match_captures_.end();
        });
      }
    }
    auto boolean_result = ir_.control.result == result_shape::BOOLEAN;
    string_operations_  = string_operation_plan();
    line_tail_literal_  = string_operations_.has_value() ? std::nullopt : line_tail_literal();
    word_run_minimum_   = string_operations_.has_value() || line_tail_literal_.has_value()
                            ? std::nullopt
                            : word_run_minimum();
    ascii_literal_      = word_run_minimum_.has_value() || line_tail_literal_.has_value()
                            ? std::nullopt
                            : exact_ascii_literal();
    if (string_operations_.has_value()) { ascii_literal_.reset(); }
    utf8_literal_ = string_operations_.has_value() || line_tail_literal_.has_value() ||
                        word_run_minimum_.has_value() || ascii_literal_.has_value()
                      ? std::nullopt
                      : exact_utf8_literal();
    if (boolean_result && !ir_.control.scan_input) { utf8_literal_.reset(); }
    utf8_literal_pivot_ =
      utf8_literal_.has_value() ? utf8_literal_pivot(*utf8_literal_) : std::nullopt;
    prefix_seek_byte_ = required_ascii_prefix();
    if (ir_.has_alternation || string_operations_.has_value() ||
        line_tail_literal_.has_value() || word_run_minimum_.has_value() ||
        ascii_literal_.has_value() || utf8_literal_.has_value()) {
      prefix_seek_byte_.reset();
    }
    // short early-hit scans favor the compact DFA; long literals repay wide candidate scans.
    if (boolean_result && ir_.control.scan_input && ascii_literal_.has_value() &&
        ascii_literal_->size() > 1U && ascii_literal_->size() < 16U) {
      ascii_literal_.reset();
    }
    if (!line_tail_literal_.has_value() && !word_run_minimum_.has_value() &&
        !ascii_literal_.has_value() && !utf8_literal_.has_value()) {
      auto replacement_uses_captures =
        std::any_of(ir_.replacement.begin(), ir_.replacement.end(), [](auto& token) {
          return token.type == replacement_token::kind::CAPTURE;
        });
      auto span_or_count =
        ir_.control.result == result_shape::MATCH_SPAN ||
        ir_.control.result == result_shape::MATCH_COUNT ||
        (ir_.control.result == result_shape::REPLACEMENT && !replacement_uses_captures) ||
        ir_.control.result == result_shape::SPLIT_FIELDS ||
        (ir_.control.result == result_shape::CAPTURES && !uses_capture_buffer());
      // The streaming position automaton wins when every input position is a plausible restart.
      // A known prefix instead favors the deterministic executor's restart acceleration, and
      // capture substitutions need its one-pass capture propagation rather than a second
      // span-recovery pass.
      auto accelerated_restart = required_ascii_prefix().has_value();
      auto streaming_span_result = span_or_count && !uses_capture_buffer() &&
                                   !ir_.has_lazy_quantifier && !accelerated_restart;
      if ((boolean_result && !begins_at_input_start()) || streaming_span_result) {
        glushkov_      = make_glushkov_machine(ir_, ir_.control.scan_input);
        deterministic_ = make_deterministic_machine(ir_, ir_.control.scan_input, false);
      } else if (boolean_result && begins_at_input_start() &&
                 ir_.blocks[ir_.entry].successors.size() == 1) {
        // the selected successor is already constrained to byte zero by the removed assertion.
        auto anchored_ir               = ir_;
        anchored_ir.entry              = ir_.blocks[ir_.entry].successors.front().target;
        anchored_ir.control.scan_input = false;
        anchored_ir.blocks[ir_.entry].instructions.clear();
        anchored_ir.blocks[ir_.entry].successors.clear();
        deterministic_ = make_deterministic_machine(anchored_ir, false, false);
      } else {
        deterministic_ = make_deterministic_machine(
          ir_, boolean_result && ir_.control.scan_input, !boolean_result);
      }
      // Nullable expressions and large graphs may not have a streaming position representation.
      // Rebuild their fallback with priority preservation rather than retaining the speculative
      // unordered DFA used only to compare against a successful streaming plan.
      if (streaming_span_result && !glushkov_.has_value()) {
        deterministic_ = make_deterministic_machine(ir_, false, true);
      }
      // This assertion machine records longest matches, so keep priority-sensitive syntax on the
      // ordered executor until assertion closures carry branch-priority metadata.
      auto assertion_candidate =
        !boolean_result && !deterministic_.has_value() && !uses_capture_buffer() &&
        !ir_.has_alternation && ir_.pattern.find('?') == std::string::npos;
      auto keep_prefix_count =
        ir_.control.result == result_shape::MATCH_COUNT && required_ascii_prefix().has_value();
      if (assertion_candidate && !keep_prefix_count) {
        auto graph = make_deterministic_graph(ir_);
        if (graph.has_value()) {
          deterministic_ = make_assertion_deterministic_machine(ir_, *graph, false);
        }
      }
    }
    if (glushkov_.has_value()) {
      if (!boolean_result || prefer_glushkov(*glushkov_, deterministic_)) {
        deterministic_.reset();
      } else {
        glushkov_.reset();
      }
    }
    if (boolean_result && !glushkov_.has_value() && !deterministic_.has_value() &&
        ir_.control.scan_input) {
      deterministic_ = make_deterministic_machine(ir_, true, true);
    }
    auto tagged_result = ir_.control.result == result_shape::CAPTURES &&
                         deterministic_.has_value() && deterministic_->capture_one_pass;
    if (!boolean_result &&
        (!deterministic_.has_value() || (uses_capture_buffer() && !tagged_result))) {
      deterministic_.reset();
    }
    if (!ascii_literal_.has_value() && !utf8_literal_.has_value() && !glushkov_.has_value() &&
        !deterministic_.has_value()) {
      mandatory_ascii_literal_ = mandatory_ascii_literal();
    }
    auto executor      = executor_kind::RECURSIVE_THOMPSON;
    auto executor_name = std::string_view{"recursive Thompson"};
    if (string_operations_.has_value()) {
      executor      = executor_kind::STRING_OPERATIONS;
      executor_name = "generated string operations";
    } else if (line_tail_literal_.has_value()) {
      executor      = executor_kind::STRING_OPERATIONS;
      executor_name = "generated final-line tail search";
    } else if (word_run_minimum_.has_value()) {
      executor      = executor_kind::WORD_RUN;
      executor_name = ir_.options.ascii_classes ? "ASCII word run" : "Unicode word run";
    } else if (ascii_literal_.has_value()) {
      executor = ascii_literal_->size() == 1U ? executor_kind::SINGLE_BYTE_LITERAL
                                              : executor_kind::PACKED_ASCII_LITERAL;
      executor_name =
        ascii_literal_->size() == 1U ? "single-byte literal scan" : "packed ASCII literal scan";
    } else if (utf8_literal_.has_value()) {
      executor      = utf8_literal_pivot_.has_value() ? executor_kind::PACKED_UTF8_LITERAL
                                                      : executor_kind::UTF8_KMP_LITERAL;
      executor_name = utf8_literal_pivot_.has_value()
                        ? (ir_.control.result == result_shape::MATCH_COUNT
                             ? "guarded-pivot packed UTF-8 literal scan"
                             : "hybrid guarded-pivot/KMP UTF-8 literal scan")
                        : "UTF-8 KMP literal scan";
    } else if (glushkov_.has_value()) {
      executor =
        boolean_result ? executor_kind::GLUSHKOV : executor_kind::STREAMING_PRIORITIZED_GLUSHKOV;
      executor_name =
        boolean_result ? "bit-parallel Glushkov NFA" : "streaming prioritized Glushkov NFA";
    } else if (deterministic_.has_value()) {
      executor      = deterministic_->assertion_aware ? executor_kind::ASSERTION_AWARE_DETERMINISTIC
                      : boolean_result                ? executor_kind::DETERMINISTIC
                      : tagged_result ? executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC
                                      : executor_kind::PRIORITIZED_DETERMINISTIC;
      executor_name = deterministic_->assertion_aware ? "assertion-aware deterministic table"
                      : boolean_result                ? "deterministic table"
                      : tagged_result                 ? "tagged prioritized deterministic table"
                                                      : "prioritized deterministic table";
    }
    output_.emit("{}",
                 std::format(R"NVVM(; NVVM IR generated by Regex IR
; pattern: {0}
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
; executor: {1})NVVM",
                             escaped_comment(ir_.pattern),
                             executor_name));
    if (mandatory_ascii_literal_.has_value()) {
      output_.emit("; mandatory ASCII literal filter: {}",
                   escaped_comment(*mandatory_ascii_literal_));
    }
    if (glushkov_.has_value()) {
      output_.emit("; glushkov positions: {}, alphabet classes: {}, shifts: {}, exceptions: {}",
                   glushkov_->position_count,
                   glushkov_->alphabet.class_count,
                   glushkov_->shifts.size(),
                   std::popcount(glushkov_->exception_mask));
    } else if (deterministic_.has_value()) {
      output_.emit("; dfa states: {}, alphabet classes: {}",
                   deterministic_->state_count,
                   deterministic_->class_count);
    }
    output_.blank();

    emit_load_byte();
    emit_decode_width();
    emit_decode_codepoint();
    if (prefix_seek_byte_.has_value()) { emit_prefix_byte_seek(*prefix_seek_byte_); }
    if (string_operations_.has_value()) {
      emit_string_operation_execute(*string_operations_);
    } else if (line_tail_literal_.has_value()) {
      emit_line_tail_execute(*line_tail_literal_);
    } else if (word_run_minimum_.has_value()) {
      if (!ir_.options.ascii_classes) emit_is_word();
      emit_word_run_execute(*word_run_minimum_);
    } else if (ascii_literal_.has_value() && boolean_result) {
      emit_ascii_literal_execute(*ascii_literal_);
    } else if (utf8_literal_.has_value() && boolean_result) {
      emit_utf8_literal_find_from(*utf8_literal_);
      emit_utf8_kmp_execute();
    } else if (glushkov_.has_value() && boolean_result) {
      emit_glushkov_globals(*glushkov_);
      emit_deterministic_classifier(glushkov_->alphabet);
      emit_glushkov_reach(*glushkov_);
      emit_glushkov_follow(*glushkov_);
      emit_glushkov_execute(*glushkov_);
    } else if (deterministic_.has_value() && boolean_result) {
      if (deterministic_->assertion_aware) {
        auto word_assertions =
          static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                    assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
        if ((deterministic_->assertion_mask & word_assertions) != 0) emit_is_word();
      } else if (deterministic_->accept_assertion.has_value()) {
        emit_advance();
        emit_is_word();
        emit_previous_position();
        emit_assertion();
      }
      emit_deterministic_globals(*deterministic_);
      emit_deterministic_classifier(*deterministic_);
      if (deterministic_->assertion_aware) {
        emit_deterministic_boundary_classifier(*deterministic_);
        emit_assertion_deterministic_execute(*deterministic_);
      } else {
        emit_deterministic_execute(*deterministic_);
      }
    } else {
      emit_optimizer_intrinsics();
      emit_advance();
      emit_replacement_globals();
      if (mandatory_ascii_literal_.has_value()) {
        emit_ascii_literal_at(*mandatory_ascii_literal_);
        emit_mandatory_literal_filter(*mandatory_ascii_literal_);
      }
      if (ascii_literal_.has_value()) {
        if (ascii_literal_->size() == 1U) {
          emit_single_byte_find_from(static_cast<std::uint8_t>(ascii_literal_->front()));
        } else {
          emit_ascii_literal_at(*ascii_literal_);
          emit_ascii_literal_find_from(*ascii_literal_);
        }
      } else if (utf8_literal_.has_value()) {
        emit_utf8_literal_find_from(*utf8_literal_);
      } else if (glushkov_.has_value()) {
        emit_glushkov_globals(*glushkov_);
        emit_deterministic_classifier(glushkov_->alphabet);
        emit_glushkov_reach(*glushkov_);
        emit_glushkov_follow(*glushkov_);
        emit_glushkov_find_from(*glushkov_);
      } else if (deterministic_.has_value()) {
        if (deterministic_->assertion_aware) {
          auto word_assertions =
            static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                      assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
          if ((deterministic_->assertion_mask & word_assertions) != 0) emit_is_word();
          emit_previous_position();
        }
        emit_deterministic_globals(*deterministic_);
        emit_deterministic_classifier(*deterministic_);
        if (deterministic_->assertion_aware) {
          emit_deterministic_boundary_classifier(*deterministic_);
          emit_assertion_deterministic_find_from(*deterministic_);
        } else if (tagged_result) {
          emit_tagged_deterministic_find_from(*deterministic_);
        } else {
          emit_deterministic_find_from(*deterministic_);
        }
      } else {
        emit_can_peek();
        emit_is_word();
        emit_previous_position();
        emit_assertion();
        emit_predicate_helpers();
        emit_literal_helpers();
        emit_blocks();
      }
      if (boolean_result) {
        emit_execute();
      } else {
        if (!deterministic_.has_value() && !glushkov_.has_value() &&
            !ascii_literal_.has_value() && !utf8_literal_.has_value()) {
          emit_find_from();
        }
        switch (ir_.control.result) {
          case result_shape::MATCH_SPAN:
            if (ir_.selected_operation.kind == operation_kind::FIND_ALL) {
              emit_find_all_execute();
            } else {
              emit_find_execute();
            }
            break;
          case result_shape::MATCH_COUNT: emit_count_execute(); break;
          case result_shape::CAPTURES: emit_capture_execute(); break;
          case result_shape::REPLACEMENT: emit_replace_execute(); break;
          case result_shape::SPLIT_FIELDS: emit_split_execute(); break;
          case result_shape::BOOLEAN: break;
        }
      }
    }
    emit_anchored_boolean_adapter();
    output_.emit(
      R"NVVM(!nvvmir.version = !{{!0}}
!0 = !{{i32 2, i32 0}})NVVM");
    auto executor_states  = deterministic_.has_value()
                              ? static_cast<std::uint32_t>(deterministic_->state_count)
                            : glushkov_.has_value() ? glushkov_->position_count
                                                    : std::uint32_t{0};
    auto alphabet_classes = deterministic_.has_value() ? deterministic_->class_count
                            : glushkov_.has_value()    ? glushkov_->alphabet.class_count
                                                       : std::uint32_t{0};
    return {output_.take(),
            ir_.capture_count,
            executor,
            executor_states,
            alphabet_classes,
            std::move(exact_ascii_literal_metadata_)};
  }

 private:
  [[nodiscard]] std::string name(std::string_view suffix) const
  {
    return nvvm_symbol(options_.symbol_prefix, suffix);
  }

  enum class string_operation_kind : std::uint8_t {
    BEGINS_WITH,
    ENDS_WITH,
    ENDS_LINE,
    EQUALS,
    EQUALS_LINE
  };

  struct string_operation {
    string_operation_kind kind;
    std::string literal;
  };

  [[nodiscard]] std::optional<string_operation> string_operation_plan() const
  {
    if (ir_.control.result != result_shape::BOOLEAN || ir_.entry >= ir_.blocks.size() ||
        ir_.accept >= ir_.blocks.size()) {
      return std::nullopt;
    }

    auto begins            = !ir_.control.scan_input;
    auto ends              = ir_.control.require_end;
    auto line_end          = false;
    auto explicit_boundary = false;
    std::vector<bool> visited(ir_.blocks.size(), false);
    std::string literal;
    auto current = ir_.entry;
    while (current < ir_.blocks.size() && !visited[current]) {
      visited[current] = true;
      auto& block      = ir_.blocks[current];
      if (current == ir_.accept) {
        auto accepting = block.instructions.size() == 1U &&
                         std::holds_alternative<emit_accept>(block.instructions.front()) &&
                         block.successors.empty();
        if (!accepting || literal.empty() || !explicit_boundary || (!begins && !ends)) {
          return std::nullopt;
        }
        auto kind = begins && ends && line_end ? string_operation_kind::EQUALS_LINE
                    : begins && ends           ? string_operation_kind::EQUALS
                    : begins                   ? string_operation_kind::BEGINS_WITH
                    : line_end                 ? string_operation_kind::ENDS_LINE
                                               : string_operation_kind::ENDS_WITH;
        return string_operation{kind, std::move(literal)};
      }
      if (block.successors.size() != 1U) return std::nullopt;

      std::u32string consumed;
      std::optional<std::uint32_t> peek_count;
      std::optional<std::uint32_t> advance_count;
      auto reads_character = false;
      for (auto& item : block.instructions) {
        if (auto* assertion = std::get_if<test_assertion>(&item)) {
          if (assertion->kind == assertion_kind::BEGIN_INPUT ||
              (assertion->kind == assertion_kind::BEGIN_LINE && !ir_.options.multiline)) {
            if (!literal.empty()) return std::nullopt;
            begins            = true;
            explicit_boundary = true;
          } else if (assertion->kind == assertion_kind::END_INPUT ||
                     (assertion->kind == assertion_kind::END_LINE && !ir_.options.multiline &&
                      !ir_.options.extended_newline)) {
            if (literal.empty()) return std::nullopt;
            ends              = true;
            line_end          = assertion->kind == assertion_kind::END_LINE;
            explicit_boundary = true;
          } else {
            return std::nullopt;
          }
        } else if (auto* peek = std::get_if<can_peek>(&item)) {
          if (peek_count.has_value()) return std::nullopt;
          peek_count = peek->characters;
        } else if (std::holds_alternative<read_character>(item)) {
          if (reads_character) return std::nullopt;
          reads_character = true;
        } else if (auto* character_match = std::get_if<match_character>(&item)) {
          if (!consumed.empty() || !character_match->predicate.is_singleton()) {
            return std::nullopt;
          }
          consumed.push_back(character_match->predicate.singleton());
        } else if (auto* literal_match = std::get_if<match_literal>(&item)) {
          if (!consumed.empty()) return std::nullopt;
          consumed = literal_match->value;
        } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
          if (advance_count.has_value()) return std::nullopt;
          advance_count = advance->characters;
        } else {
          return std::nullopt;
        }
      }

      if (!consumed.empty()) {
        auto count = static_cast<std::uint32_t>(consumed.size());
        if (!peek_count.has_value() || *peek_count != count || !advance_count.has_value() ||
            *advance_count != count || (reads_character && count != 1U)) {
          return std::nullopt;
        }
        for (auto codepoint : consumed) {
          if (codepoint > 0x7f) return std::nullopt;
          literal.push_back(static_cast<char>(codepoint));
        }
      } else if (!block.instructions.empty() &&
                 !std::holds_alternative<test_assertion>(block.instructions.front())) {
        return std::nullopt;
      }
      current = block.successors.front().target;
    }
    return std::nullopt;
  }

  void emit_prefix_byte_seek(std::uint8_t byte)
  {
    auto repeated = std::uint64_t{0};
    for (std::size_t index = 0; index < 8U; ++index) {
      repeated |= static_cast<std::uint64_t>(byte) << (index * 8U);
    }
    output_.emit(
      R"NVVM(declare i64 @llvm.cttz.i64(i64, i1)

define internal i64 @{0}(i8* %data, i64 %size, i64 %start) alwaysinline nounwind readonly {{
entry:
  br label %chunks
chunks:
  %position = phi i64 [ %start, %entry ], [ %next_chunk, %chunk_continue ]
  %chunk_end = add i64 %position, 8
  %full_chunk = icmp ule i64 %chunk_end, %size
  br i1 %full_chunk, label %load_chunk, label %tail
load_chunk:
  %chunk_byte_ptr = getelementptr i8, i8* %data, i64 %position
  %chunk_ptr = bitcast i8* %chunk_byte_ptr to i64*
  %chunk = load i64, i64* %chunk_ptr, align 1
  %candidate_x = xor i64 %chunk, {1}
  %candidate_minus_ones = sub i64 %candidate_x, 72340172838076673
  %candidate_not = xor i64 %candidate_x, -1
  %candidate_zero_bytes = and i64 %candidate_minus_ones, %candidate_not
  %candidate_mask = and i64 %candidate_zero_bytes, -9187201950435737472
  %has_candidate = icmp ne i64 %candidate_mask, 0
  br i1 %has_candidate, label %candidate, label %chunk_continue
candidate:
  %remaining_candidates = phi i64 [ %candidate_mask, %load_chunk ], [ %next_candidates, %candidate_continue ]
  %candidate_bit = call i64 @llvm.cttz.i64(i64 %remaining_candidates, i1 false)
  %candidate_byte = lshr i64 %candidate_bit, 3
  %candidate_position = add i64 %position, %candidate_byte
  %candidate_ptr = getelementptr i8, i8* %data, i64 %candidate_position
  %candidate_value = call i32 @{2}(i8* %candidate_ptr)
  %candidate_exact = icmp eq i32 %candidate_value, {3}
  br i1 %candidate_exact, label %found_candidate, label %candidate_continue
candidate_continue:
  %candidate_mask_minus_one = sub i64 %remaining_candidates, 1
  %next_candidates = and i64 %remaining_candidates, %candidate_mask_minus_one
  %has_more_candidates = icmp ne i64 %next_candidates, 0
  br i1 %has_more_candidates, label %candidate, label %chunk_continue
chunk_continue:
  %next_chunk = add nuw i64 %position, 8
  br label %chunks
tail:
  %tail_position = phi i64 [ %position, %chunks ], [ %tail_next, %tail_continue ]
  %tail_in_range = icmp ult i64 %tail_position, %size
  br i1 %tail_in_range, label %tail_load, label %not_found
tail_load:
  %tail_ptr = getelementptr i8, i8* %data, i64 %tail_position
  %tail_value = call i32 @{2}(i8* %tail_ptr)
  %tail_match = icmp eq i32 %tail_value, {3}
  br i1 %tail_match, label %found_tail, label %tail_continue
tail_continue:
  %tail_next = add nuw i64 %tail_position, 1
  br label %tail
found_candidate:
  ret i64 %candidate_position
found_tail:
  ret i64 %tail_position
not_found:
  ret i64 %size
}})NVVM",
      name("seek_prefix_byte"),
      repeated,
      name("load_byte"),
      static_cast<std::uint32_t>(byte));
    output_.blank();
  }

  void emit_string_operation_execute(string_operation const& operation)
  {
    emit_ascii_literal_at(operation.literal);
    switch (operation.kind) {
      case string_operation_kind::BEGINS_WITH:
        output_.emit(
          "define zeroext i1 @{}(i8* %data, i64 %size) nounwind readonly {{\n"
          "entry:\n"
          "  %matched = call i1 @{}(i8* %data, i64 %size, i64 0)\n"
          "  ret i1 %matched\n"
          "}}",
          options_.execute_function,
          name("ascii_literal_at"));
        break;
      case string_operation_kind::ENDS_WITH:
        output_.emit(
          "define zeroext i1 @{}(i8* %data, i64 %size) nounwind readonly {{\n"
          "entry:\n"
          "  %enough = icmp uge i64 %size, {}\n"
          "  %position = sub i64 %size, {}\n"
          "  br i1 %enough, label %compare, label %no\n"
          "compare:\n"
          "  %matched = call i1 @{}(i8* %data, i64 %size, i64 %position)\n"
          "  ret i1 %matched\n"
          "no:\n"
          "  ret i1 false\n"
          "}}",
          options_.execute_function,
          operation.literal.size(),
          operation.literal.size(),
          name("ascii_literal_at"));
        break;
      case string_operation_kind::ENDS_LINE:
        output_.emit(
          "define zeroext i1 @{}(i8* %data, i64 %size) nounwind readonly {{\n"
          "entry:\n"
          "  %enough = icmp uge i64 %size, {}\n"
          "  br i1 %enough, label %at_end, label %no\n"
          "at_end:\n"
          "  %end_position = sub i64 %size, {}\n"
          "  %end_match = call i1 @{}(i8* %data, i64 %size, i64 %end_position)\n"
          "  br i1 %end_match, label %yes, label %before_newline_check\n"
          "before_newline_check:\n"
          "  %has_extra = icmp ugt i64 %size, {}\n"
          "  br i1 %has_extra, label %load_last, label %no\n"
          "load_last:\n"
          "  %last_position = sub i64 %size, 1\n"
          "  %last_ptr = getelementptr i8, i8* %data, i64 %last_position\n"
          "  %last = call i32 @{}(i8* %last_ptr)\n"
          "  %is_lf = icmp eq i32 %last, 10\n"
          "  %line_position = sub i64 %last_position, {}\n"
          "  br i1 %is_lf, label %before_newline, label %no\n"
          "before_newline:\n"
          "  %line_match = call i1 @{}(i8* %data, i64 %size, i64 %line_position)\n"
          "  ret i1 %line_match\n"
          "yes:\n"
          "  ret i1 true\n"
          "no:\n"
          "  ret i1 false\n"
          "}}",
          options_.execute_function,
          operation.literal.size(),
          operation.literal.size(),
          name("ascii_literal_at"),
          operation.literal.size(),
          name("load_byte"),
          operation.literal.size(),
          name("ascii_literal_at"));
        break;
      case string_operation_kind::EQUALS:
        output_.emit(
          "define zeroext i1 @{}(i8* %data, i64 %size) nounwind readonly {{\n"
          "entry:\n"
          "  %same_size = icmp eq i64 %size, {}\n"
          "  br i1 %same_size, label %compare, label %no\n"
          "compare:\n"
          "  %matched = call i1 @{}(i8* %data, i64 %size, i64 0)\n"
          "  ret i1 %matched\n"
          "no:\n"
          "  ret i1 false\n"
          "}}",
          options_.execute_function,
          operation.literal.size(),
          name("ascii_literal_at"));
        break;
      case string_operation_kind::EQUALS_LINE:
        output_.emit(
          "define zeroext i1 @{}(i8* %data, i64 %size) nounwind readonly {{\n"
          "entry:\n"
          "  %exact_size = icmp eq i64 %size, {}\n"
          "  %line_size = icmp eq i64 %size, {}\n"
          "  br i1 %exact_size, label %compare, label %line_check\n"
          "line_check:\n"
          "  br i1 %line_size, label %load_last, label %no\n"
          "load_last:\n"
          "  %last_ptr = getelementptr i8, i8* %data, i64 {}\n"
          "  %last = call i32 @{}(i8* %last_ptr)\n"
          "  %is_lf = icmp eq i32 %last, 10\n"
          "  br i1 %is_lf, label %compare, label %no\n"
          "compare:\n"
          "  %matched = call i1 @{}(i8* %data, i64 %size, i64 0)\n"
          "  ret i1 %matched\n"
          "no:\n"
          "  ret i1 false\n"
          "}}",
          options_.execute_function,
          operation.literal.size(),
          operation.literal.size() + 1U,
          operation.literal.size(),
          name("load_byte"),
          name("ascii_literal_at"));
        break;
    }
    output_.blank();
  }

  /**
   * @brief recognizes a case-sensitive ASCII literal followed by `.*$`
   *
   * The direct executor is deliberately restricted to the default, non-multiline LF semantics.
   * In that dialect only the final logical line can match, so scanning every candidate through
   * the recursive Thompson executor is unnecessary. Regex metacharacters and escapes in the
   * literal are rejected until the structural IR analysis can prove the same shape.
   *
   * @return literal prefix when the operation can use the final-line tail executor
   */
  [[nodiscard]] std::optional<std::string> line_tail_literal() const
  {
    auto supported_result =
      ir_.control.result == result_shape::BOOLEAN ||
      ir_.control.result == result_shape::MATCH_COUNT ||
      ir_.control.result == result_shape::MATCH_SPAN ||
      (ir_.control.result == result_shape::CAPTURES && ir_.capture_count == 0U);
    if (!supported_result || !ir_.control.scan_input || ir_.options.case_insensitive ||
        ir_.options.multiline || ir_.options.dot_all || ir_.options.extended_newline) {
      return std::nullopt;
    }

    constexpr auto suffix = std::string_view{R"(.*$)"};
    auto pattern          = std::string_view{ir_.pattern};
    if (!pattern.ends_with(suffix) || pattern.size() == suffix.size()) return std::nullopt;
    auto literal                  = pattern.substr(0, pattern.size() - suffix.size());
    constexpr auto metacharacters = std::string_view{R"(\.^$*+?()[]{}|)"};
    for (auto character : literal) {
      auto byte = static_cast<std::uint8_t>(character);
      if (byte > 0x7fU || metacharacters.find(character) != std::string_view::npos) {
        return std::nullopt;
      }
    }
    return std::string{literal};
  }

  /**
   * @brief emits a direct finder for `literal.*$` with default LF semantics
   *
   * The matcher first locates the start of the final logical line, then searches only that range
   * for the literal. A successful match ends at the input end or immediately before its final LF.
   *
   * @param literal non-empty case-sensitive ASCII prefix
   */
  void emit_line_tail_execute(std::string_view literal)
  {
    emit_ascii_literal_at(literal);
    emit_prefix_byte_seek(static_cast<std::uint8_t>(literal.front()));
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal zeroext i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind readonly {{
entry:
  %empty = icmp eq i64 %size, 0
  br i1 %empty, label %no, label %load_last
load_last:
  %last_position = sub i64 %size, 1
  %last_ptr = getelementptr i8, i8* %data, i64 %last_position
  %last = call i32 @{1}(i8* %last_ptr)
  %trailing_lf = icmp eq i32 %last, 10
  %line_end = select i1 %trailing_lf, i64 %last_position, i64 %size
  br label %reverse
reverse:
  %reverse_position = phi i64 [ %line_end, %load_last ], [ %previous_position, %reverse_continue ]
  %at_begin = icmp eq i64 %reverse_position, 0
  br i1 %at_begin, label %line_start_zero, label %reverse_load
reverse_load:
  %previous_position = sub i64 %reverse_position, 1
  %previous_ptr = getelementptr i8, i8* %data, i64 %previous_position
  %previous = call i32 @{1}(i8* %previous_ptr)
  %previous_lf = icmp eq i32 %previous, 10
  br i1 %previous_lf, label %line_start_after_lf, label %reverse_continue
reverse_continue:
  br label %reverse
line_start_zero:
  br label %search_setup
line_start_after_lf:
  %after_lf = add nuw i64 %previous_position, 1
  br label %search_setup
search_setup:
  %line_start = phi i64 [ 0, %line_start_zero ], [ %after_lf, %line_start_after_lf ]
  %start_before_line = icmp ult i64 %search_start, %line_start
  %effective_start = select i1 %start_before_line, i64 %line_start, i64 %search_start
  br label %search
search:
  %candidate_start = phi i64 [ %effective_start, %search_setup ], [ %next_start, %mismatch ]
  %candidate = call i64 @{2}(i8* %data, i64 %line_end, i64 %candidate_start)
  %candidate_end = add i64 %candidate, {3}
  %in_range = icmp ule i64 %candidate_end, %line_end
  br i1 %in_range, label %verify, label %no
verify:
  %matched = call i1 @{4}(i8* %data, i64 %line_end, i64 %candidate)
  br i1 %matched, label %yes, label %mismatch
mismatch:
  %next_start = add nuw i64 %candidate, 1
  br label %search
yes:
  store i64 %candidate, i64* %match_begin, align 8
  store i64 %line_end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("find_from"),
        name("load_byte"),
        name("seek_prefix_byte"),
        literal.size(),
        name("ascii_literal_at")));
    output_.blank();

    if (ir_.control.result == result_shape::BOOLEAN) {
      output_.emit("{}",
                   std::format(
                     R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 0, i64* %match_begin, i64* %match_end, i64* null)
  ret i1 %matched
}})NVVM",
                     options_.execute_function,
                     name("find_from")));
      output_.blank();
    } else if (ir_.control.result == result_shape::MATCH_COUNT) {
      output_.emit("{}",
                   std::format(
                     R"NVVM(define i64 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 0, i64* %match_begin, i64* %match_end, i64* null)
  %count = zext i1 %matched to i64
  ret i64 %count
}})NVVM",
                     options_.execute_function,
                     name("find_from")));
      output_.blank();
    } else if (ir_.control.result == result_shape::CAPTURES) {
      emit_capture_execute();
    } else if (ir_.selected_operation.kind == operation_kind::FIND_ALL) {
      emit_find_all_execute();
    } else {
      emit_find_execute();
    }
  }

  [[nodiscard]] std::optional<std::uint32_t> word_run_minimum() const
  {
    if (ir_.options.characters == character_mode::BYTES && !ir_.options.ascii_classes) {
      return std::nullopt;
    }
    if (ir_.control.result != result_shape::BOOLEAN &&
        ir_.control.result != result_shape::MATCH_COUNT &&
        ir_.control.result != result_shape::MATCH_SPAN) {
      return std::nullopt;
    }
    constexpr auto prefix = std::string_view{R"(\b\w{)"};
    constexpr auto suffix = std::string_view{R"(,}\b)"};
    auto pattern          = std::string_view{ir_.pattern};
    if (!pattern.starts_with(prefix) || !pattern.ends_with(suffix) ||
        pattern.size() <= prefix.size() + suffix.size()) {
      return std::nullopt;
    }
    auto digits = pattern.substr(prefix.size(), pattern.size() - prefix.size() - suffix.size());
    auto value  = std::uint64_t{0};
    for (auto character : digits) {
      if (character < '0' || character > '9') return std::nullopt;
      value = value * 10U + static_cast<std::uint64_t>(character - '0');
      if (value > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
    }
    return value == 0 ? std::nullopt
                      : std::optional<std::uint32_t>{static_cast<std::uint32_t>(value)};
  }

  void emit_unicode_word_run_matcher(std::uint32_t minimum)
  {
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal zeroext i1 @{0}(i8* %data, i64 %size, i64 %start, i64* %begin_out, i64* %end_out) alwaysinline nounwind readonly {{
entry:
  br label %search
search:
  %position = phi i64 [ %start, %entry ], [ %next_position, %advance ], [ %resume, %short ]
  %in_range = icmp ult i64 %position, %size
  br i1 %in_range, label %load, label %no
load:
  %cp = call i32 @{1}(i8* %data, i64 %size, i64 %position)
  %width = call i64 @{2}(i8* %data, i64 %size, i64 %position)
  %word = call i1 @{3}(i32 %cp)
  br i1 %word, label %run, label %advance
advance:
  %next_position = add nuw i64 %position, %width
  br label %search
run:
  br label %scan
scan:
  %run_position = phi i64 [ %position, %run ], [ %run_next, %run_continue ]
  %run_count = phi i64 [ 0, %run ], [ %run_next_count, %run_continue ]
  %run_at_end = icmp eq i64 %run_position, %size
  br i1 %run_at_end, label %run_done_end, label %run_load
run_load:
  %run_cp = call i32 @{1}(i8* %data, i64 %size, i64 %run_position)
  %run_width = call i64 @{2}(i8* %data, i64 %size, i64 %run_position)
  %run_word = call i1 @{3}(i32 %run_cp)
  br i1 %run_word, label %run_continue, label %run_done_nonword
run_continue:
  %run_next = add nuw i64 %run_position, %run_width
  %run_next_count = add nuw i64 %run_count, 1
  br label %scan
run_done_end:
  br label %evaluate
run_done_nonword:
  br label %evaluate
evaluate:
  %run_end = phi i64 [ %size, %run_done_end ], [ %run_position, %run_done_nonword ]
  %word_count = phi i64 [ %run_count, %run_done_end ], [ %run_count, %run_done_nonword ]
  %delimiter_width = phi i64 [ 0, %run_done_end ], [ %run_width, %run_done_nonword ]
  %qualifies = icmp uge i64 %word_count, {4}
  br i1 %qualifies, label %yes, label %short
short:
  %resume = add nuw i64 %run_end, %delimiter_width
  br label %search
yes:
  store i64 %position, i64* %begin_out, align 8
  store i64 %run_end, i64* %end_out, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("next_word_run"),
        name("decode_codepoint"),
        name("decode_width"),
        name("is_word"),
        minimum));
  }

  void emit_word_run_execute(std::uint32_t minimum)
  {
    if (!ir_.options.ascii_classes) {
      emit_unicode_word_run_matcher(minimum);
    } else {
      output_.emit(
        "{}",
        std::format(
          R"NVVM(define internal zeroext i1 @{0}(i8* %data, i64 %size, i64 %start, i64* %begin_out, i64* %end_out) alwaysinline nounwind readonly {{
entry:
  br label %search
search:
  %position = phi i64 [ %start, %entry ], [ %next_position, %advance ], [ %resume, %short ]
  %in_range = icmp ult i64 %position, %size
  br i1 %in_range, label %load, label %no
load:
  %byte_ptr = getelementptr i8, i8* %data, i64 %position
  %byte = call i32 @{1}(i8* %byte_ptr)
  %digit_low = icmp uge i32 %byte, 48
  %digit_high = icmp ule i32 %byte, 57
  %digit = and i1 %digit_low, %digit_high
  %upper_low = icmp uge i32 %byte, 65
  %upper_high = icmp ule i32 %byte, 90
  %upper = and i1 %upper_low, %upper_high
  %lower_low = icmp uge i32 %byte, 97
  %lower_high = icmp ule i32 %byte, 122
  %lower = and i1 %lower_low, %lower_high
  %underscore = icmp eq i32 %byte, 95
  %alpha = or i1 %upper, %lower
  %alnum = or i1 %alpha, %digit
  %word = or i1 %alnum, %underscore
  br i1 %word, label %run, label %advance
advance:
  %next_position = add nuw i64 %position, 1
  br label %search
run:
  br label %scan
scan:
  %run_position = phi i64 [ %position, %run ], [ %run_next, %run_continue ]
  %run_at_end = icmp eq i64 %run_position, %size
  br i1 %run_at_end, label %run_done_end, label %run_load
run_load:
  %run_byte_ptr = getelementptr i8, i8* %data, i64 %run_position
  %run_byte = call i32 @{1}(i8* %run_byte_ptr)
  %run_digit_low = icmp uge i32 %run_byte, 48
  %run_digit_high = icmp ule i32 %run_byte, 57
  %run_digit = and i1 %run_digit_low, %run_digit_high
  %run_upper_low = icmp uge i32 %run_byte, 65
  %run_upper_high = icmp ule i32 %run_byte, 90
  %run_upper = and i1 %run_upper_low, %run_upper_high
  %run_lower_low = icmp uge i32 %run_byte, 97
  %run_lower_high = icmp ule i32 %run_byte, 122
  %run_lower = and i1 %run_lower_low, %run_lower_high
  %run_underscore = icmp eq i32 %run_byte, 95
  %run_alpha = or i1 %run_upper, %run_lower
  %run_alnum = or i1 %run_alpha, %run_digit
  %run_word = or i1 %run_alnum, %run_underscore
  br i1 %run_word, label %run_continue, label %run_done_nonword
run_continue:
  %run_next = add nuw i64 %run_position, 1
  br label %scan
run_done_end:
  br label %evaluate
run_done_nonword:
  br label %evaluate
evaluate:
  %run_end = phi i64 [ %size, %run_done_end ], [ %run_position, %run_done_nonword ]
  %run_length = sub i64 %run_end, %position
  %qualifies = icmp uge i64 %run_length, {2}
  br i1 %qualifies, label %yes, label %short
short:
  %short_at_end = icmp eq i64 %run_end, %size
  %after_delimiter = add i64 %run_end, 1
  %resume = select i1 %short_at_end, i64 %size, i64 %after_delimiter
  br label %search
yes:
  store i64 %position, i64* %begin_out, align 8
  store i64 %run_end, i64* %end_out, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
          name("next_word_run"),
          name("load_byte"),
          minimum));
    }
    output_.blank();

    switch (ir_.control.result) {
      case result_shape::BOOLEAN:
        output_.emit("{}",
                     std::format(
                       R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %begin = alloca i64, align 8
  %end = alloca i64, align 8
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 0, i64* %begin, i64* %end)
  ret i1 %matched
}})NVVM",
                       options_.execute_function,
                       name("next_word_run")));
        break;
      case result_shape::MATCH_SPAN:
        output_.emit(
          "{}",
          std::format(
            R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size, i64* %span) nounwind readonly {{
entry:
  %begin = getelementptr i64, i64* %span, i64 0
  %end = getelementptr i64, i64* %span, i64 1
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 0, i64* %begin, i64* %end)
  ret i1 %matched
}})NVVM",
            options_.execute_function,
            name("next_word_run")));
        break;
      case result_shape::MATCH_COUNT:
        output_.emit("{}",
                     std::format(
                       R"NVVM(define i64 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %begin = alloca i64, align 8
  %end = alloca i64, align 8
  br label %loop
loop:
  %cursor = phi i64 [ 0, %entry ], [ %matched_end, %found ]
  %count = phi i64 [ 0, %entry ], [ %next_count, %found ]
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 %cursor, i64* %begin, i64* %end)
  br i1 %matched, label %found, label %done
found:
  %matched_end = load i64, i64* %end, align 8
  %next_count = add i64 %count, 1
  br label %loop
done:
  ret i64 %count
}})NVVM",
                       options_.execute_function,
                       name("next_word_run")));
        break;
      default: throw std::invalid_argument("unsupported word-run result shape");
    }
    output_.blank();
  }

  struct utf8_literal {
    std::u32string codepoints;
    std::size_t byte_count;
  };

  [[nodiscard]] static std::string encode_utf8_literal(std::u32string_view codepoints)
  {
    auto bytes = std::string{};
    for (auto codepoint : codepoints) {
      if (codepoint <= 0x7fU) {
        bytes.push_back(static_cast<char>(codepoint));
      } else if (codepoint <= 0x7ffU) {
        bytes.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        bytes.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
      } else if (codepoint <= 0xffffU) {
        bytes.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        bytes.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        bytes.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
      } else {
        bytes.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
        bytes.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
        bytes.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        bytes.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
      }
    }
    return bytes;
  }

  [[nodiscard]] static std::optional<std::size_t> utf8_literal_pivot(utf8_literal const& literal)
  {
    auto bytes = encode_utf8_literal(literal.codepoints);
    if (bytes.size() < sizeof(std::uint64_t)) return std::nullopt;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      auto byte  = static_cast<std::uint8_t>(bytes[index]);
      auto digit = byte >= static_cast<std::uint8_t>('0') && byte <= static_cast<std::uint8_t>('9');
      auto upper = byte >= static_cast<std::uint8_t>('A') && byte <= static_cast<std::uint8_t>('Z');
      auto lower = byte >= static_cast<std::uint8_t>('a') && byte <= static_cast<std::uint8_t>('z');
      if (byte < 0x80U && !digit && !upper && !lower) return index;
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<utf8_literal> exact_utf8_literal() const
  {
    if (ir_.options.characters == character_mode::BYTES || ir_.entry >= ir_.blocks.size() ||
        ir_.accept >= ir_.blocks.size()) {
      return std::nullopt;
    }

    std::vector<bool> visited(ir_.blocks.size(), false);
    std::u32string literal;
    auto byte_count    = std::size_t{0};
    auto has_non_ascii = false;
    auto current       = ir_.entry;
    while (current < ir_.blocks.size() && !visited[current]) {
      visited[current] = true;
      auto& block      = ir_.blocks[current];
      if (current == ir_.accept) {
        auto accepting = block.instructions.size() == 1U &&
                         std::holds_alternative<emit_accept>(block.instructions.front()) &&
                         block.successors.empty();
        if (!accepting || literal.empty() || !has_non_ascii) { return std::nullopt; }
        return utf8_literal{std::move(literal), byte_count};
      }
      if (block.successors.size() != 1U) return std::nullopt;

      std::u32string consumed;
      std::optional<std::uint32_t> peek_count;
      std::optional<std::uint32_t> advance_count;
      auto reads_character = false;
      for (auto& item : block.instructions) {
        if (auto* peek = std::get_if<can_peek>(&item)) {
          if (peek_count.has_value()) return std::nullopt;
          peek_count = peek->characters;
        } else if (std::holds_alternative<read_character>(item)) {
          if (reads_character) return std::nullopt;
          reads_character = true;
        } else if (auto* character_match = std::get_if<match_character>(&item)) {
          if (!consumed.empty() || !character_match->predicate.is_singleton()) {
            return std::nullopt;
          }
          consumed.push_back(character_match->predicate.singleton());
        } else if (auto* literal_match = std::get_if<match_literal>(&item)) {
          if (!consumed.empty()) return std::nullopt;
          consumed = literal_match->value;
        } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
          if (advance_count.has_value()) return std::nullopt;
          advance_count = advance->characters;
        } else {
          return std::nullopt;
        }
      }

      if (!block.instructions.empty()) {
        auto count = static_cast<std::uint32_t>(consumed.size());
        if (count == 0U || !peek_count.has_value() || *peek_count != count ||
            !advance_count.has_value() || *advance_count != count ||
            (reads_character && count != 1U)) {
          return std::nullopt;
        }
        for (auto codepoint : consumed) {
          if (codepoint > 0x10ffffU || (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
            return std::nullopt;
          }
          has_non_ascii = has_non_ascii || codepoint > 0x7fU;
          byte_count += codepoint <= 0x7fU     ? 1U
                        : codepoint <= 0x7ffU  ? 2U
                        : codepoint <= 0xffffU ? 3U
                                               : 4U;
          literal.push_back(codepoint);
        }
      }
      current = block.successors.front().target;
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string> exact_ascii_literal() const
  {
    if (ir_.entry >= ir_.blocks.size() || ir_.accept >= ir_.blocks.size()) { return std::nullopt; }

    std::vector<bool> visited(ir_.blocks.size(), false);
    std::string literal;
    auto current = ir_.entry;
    while (current < ir_.blocks.size() && !visited[current]) {
      visited[current] = true;
      auto& block      = ir_.blocks[current];
      if (current == ir_.accept) {
        auto accepting = block.instructions.size() == 1U &&
                         std::holds_alternative<emit_accept>(block.instructions.front()) &&
                         block.successors.empty();
        return accepting && !literal.empty() ? std::optional<std::string>{std::move(literal)}
                                             : std::nullopt;
      }
      if (block.successors.size() != 1U) return std::nullopt;

      std::u32string consumed;
      std::optional<std::uint32_t> peek_count;
      std::optional<std::uint32_t> advance_count;
      auto reads_character = false;
      for (auto& item : block.instructions) {
        if (auto* peek = std::get_if<can_peek>(&item)) {
          if (peek_count.has_value()) return std::nullopt;
          peek_count = peek->characters;
        } else if (std::holds_alternative<read_character>(item)) {
          if (reads_character) return std::nullopt;
          reads_character = true;
        } else if (auto* character_match = std::get_if<match_character>(&item)) {
          if (!consumed.empty() || !character_match->predicate.is_singleton()) {
            return std::nullopt;
          }
          consumed.push_back(character_match->predicate.singleton());
        } else if (auto* literal_match = std::get_if<match_literal>(&item)) {
          if (!consumed.empty()) return std::nullopt;
          consumed = literal_match->value;
        } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
          if (advance_count.has_value()) return std::nullopt;
          advance_count = advance->characters;
        } else {
          return std::nullopt;
        }
      }

      if (!block.instructions.empty()) {
        auto count = static_cast<std::uint32_t>(consumed.size());
        if (count == 0U || !peek_count.has_value() || *peek_count != count ||
            !advance_count.has_value() || *advance_count != count ||
            (reads_character && count != 1U)) {
          return std::nullopt;
        }
        for (auto codepoint : consumed) {
          if (codepoint > 0x7f) return std::nullopt;
          literal.push_back(static_cast<char>(codepoint));
        }
      }
      current = block.successors.front().target;
    }
    return std::nullopt;
  }

  static std::string escaped_comment(std::string_view value)
  {
    std::string result;
    result.reserve(value.size());
    for (auto character : value) {
      auto byte = static_cast<std::uint8_t>(character);
      if (byte < 0x20 || byte == 0x7f) {
        std::format_to(std::back_inserter(result), "\\x{:02X}", byte);
      } else {
        result.push_back(character);
      }
    }
    return result;
  }

  [[nodiscard]] std::optional<std::uint8_t> required_ascii_prefix() const
  {
    if (!ir_.control.scan_input || ir_.entry >= ir_.blocks.size()) { return std::nullopt; }
    for (auto& instruction : ir_.blocks[ir_.entry].instructions) {
      if (auto* literal = std::get_if<match_literal>(&instruction)) {
        if (!literal->value.empty() && literal->value.front() <= 0x7f) {
          return static_cast<std::uint8_t>(literal->value.front());
        }
        return std::nullopt;
      }
      if (auto* character = std::get_if<match_character>(&instruction)) {
        if (character->predicate.is_singleton() && character->predicate.singleton() <= 0x7f) {
          return static_cast<std::uint8_t>(character->predicate.singleton());
        }
        return std::nullopt;
      }
      if (std::holds_alternative<advance_cursor>(instruction) ||
          std::holds_alternative<emit_accept>(instruction)) {
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string> mandatory_ascii_literal() const
  {
    if (ir_.entry >= ir_.blocks.size() || ir_.accept >= ir_.blocks.size()) { return std::nullopt; }

    std::vector<std::pair<std::size_t, std::string>> candidates;
    for (std::size_t block = 0; block < ir_.blocks.size(); ++block) {
      for (auto& instruction : ir_.blocks[block].instructions) {
        auto* matched = std::get_if<match_literal>(&instruction);
        if (matched == nullptr || matched->value.size() < 2U ||
            std::any_of(matched->value.begin(), matched->value.end(), [](auto codepoint) {
              return codepoint > 0x7f;
            })) {
          continue;
        }
        auto literal = std::string{};
        literal.reserve(matched->value.size());
        for (auto codepoint : matched->value) {
          literal.push_back(static_cast<char>(codepoint));
        }
        candidates.emplace_back(block, std::move(literal));
      }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](auto& lhs, auto& rhs) {
      return lhs.second.size() > rhs.second.size();
    });

    for (auto& [candidate, literal] : candidates) {
      std::vector<bool> reachable(ir_.blocks.size(), false);
      std::vector<std::size_t> pending;
      if (ir_.entry != candidate) {
        reachable[ir_.entry] = true;
        pending.push_back(ir_.entry);
      }
      while (!pending.empty()) {
        auto block = pending.back();
        pending.pop_back();
        for (auto& successor : ir_.blocks[block].successors) {
          if (successor.target >= ir_.blocks.size() || successor.target == candidate ||
              reachable[successor.target]) {
            continue;
          }
          reachable[successor.target] = true;
          pending.push_back(successor.target);
        }
      }
      if (!reachable[ir_.accept]) { return literal; }
    }
    return std::nullopt;
  }

  [[nodiscard]] bool begins_at_input_start() const
  {
    if (ir_.entry >= ir_.blocks.size()) return false;
    auto& entry = ir_.blocks[ir_.entry];
    if (entry.instructions.size() != 1U ||
        !std::holds_alternative<test_assertion>(entry.instructions.front())) {
      return false;
    }
    auto assertion = std::get<test_assertion>(entry.instructions.front()).kind;
    return assertion == assertion_kind::BEGIN_INPUT ||
           (assertion == assertion_kind::BEGIN_LINE && !ir_.options.multiline);
  }

  [[nodiscard]] std::vector<std::size_t> live_capture_slots() const
  {
    std::vector<bool> live(static_cast<std::size_t>(ir_.capture_count + 1U) * 2U, false);
    if (ir_.control.result == result_shape::CAPTURES) {
      for (std::uint32_t capture = 1; capture <= ir_.capture_count; ++capture) {
        if (std::find(whole_match_captures_.begin(), whole_match_captures_.end(), capture) !=
            whole_match_captures_.end()) {
          continue;
        }
        auto slot       = static_cast<std::size_t>(capture) * 2U;
        live[slot]      = true;
        live[slot + 1U] = true;
      }
    } else if (ir_.control.result == result_shape::REPLACEMENT) {
      for (auto& token : ir_.replacement) {
        if (token.type == replacement_token::kind::CAPTURE && token.capture_index != 0 &&
            !is_whole_match_capture(token.capture_index)) {
          auto slot       = static_cast<std::size_t>(token.capture_index) * 2U;
          live[slot]      = true;
          live[slot + 1U] = true;
        }
      }
    }
    std::vector<std::size_t> result;
    for (std::size_t slot = 0; slot < live.size(); ++slot) {
      if (live[slot]) result.push_back(slot);
    }
    return result;
  }

  /**
   * @brief identifies explicit captures whose span is necessarily the whole match
   *
   * These captures can be copied from the whole-match span in the public capture adapter instead
   * of forcing the matcher to carry capture state through every transition.
   *
   * @return explicit capture indices equivalent to the whole match
   */
  [[nodiscard]] std::vector<std::uint32_t> whole_match_captures() const
  {
    if (ir_.control.result != result_shape::CAPTURES) return {};
    auto result = std::vector<std::uint32_t>{};
    for (std::uint32_t capture = 1; capture <= ir_.capture_count; ++capture) {
      if (is_whole_match_capture(capture)) result.push_back(capture);
    }
    return result;
  }

  [[nodiscard]] bool uses_capture_buffer() const { return !capture_slots_.empty(); }

  [[nodiscard]] bool is_whole_match_capture(std::uint32_t capture_index) const
  {
    auto is_capture = [&](instruction_block const& block, capture_action action) {
      return block.instructions.size() == 1U &&
             std::holds_alternative<write_capture>(block.instructions.front()) &&
             std::get<write_capture>(block.instructions.front()).capture_index == capture_index &&
             std::get<write_capture>(block.instructions.front()).action == action;
    };

    auto begin_writes = std::size_t{0};
    auto end_writes   = std::size_t{0};
    for (auto& candidate : ir_.blocks) {
      for (auto& item : candidate.instructions) {
        auto* capture = std::get_if<write_capture>(&item);
        if (capture == nullptr || capture->capture_index != capture_index) continue;
        if (capture->action == capture_action::BEGIN) {
          ++begin_writes;
        } else {
          ++end_writes;
        }
      }
    }
    if (begin_writes != 1U || end_writes != 1U) return false;

    std::vector<bool> visited(ir_.blocks.size(), false);
    auto block           = ir_.entry;
    auto begins_at_start = false;
    while (block < ir_.blocks.size() && !visited[block]) {
      visited[block]  = true;
      auto& candidate = ir_.blocks[block];
      if (is_capture(candidate, capture_action::BEGIN)) begins_at_start = true;
      if (!candidate.instructions.empty() && !is_capture(candidate, capture_action::BEGIN)) {
        break;
      }
      if (candidate.successors.size() != 1U) break;
      block = candidate.successors.front().target;
    }
    if (!begins_at_start) return false;

    std::vector<std::vector<block_id>> predecessors(ir_.blocks.size());
    for (auto& candidate : ir_.blocks) {
      for (auto edge : candidate.successors)
        predecessors[edge.target].push_back(candidate.id);
    }
    std::fill(visited.begin(), visited.end(), false);
    block               = ir_.accept;
    auto ends_at_accept = false;
    while (block < ir_.blocks.size() && !visited[block]) {
      visited[block]  = true;
      auto& candidate = ir_.blocks[block];
      if (is_capture(candidate, capture_action::END)) ends_at_accept = true;
      auto is_accept = candidate.instructions.size() == 1U &&
                       std::holds_alternative<emit_accept>(candidate.instructions.front());
      if (!candidate.instructions.empty() && !is_capture(candidate, capture_action::END) &&
          !is_accept) {
        break;
      }
      if (predecessors[block].size() != 1U) break;
      block = predecessors[block].front();
    }
    return ends_at_accept;
  }

  static std::int32_t llvm_i16(std::uint16_t value) { return static_cast<std::int16_t>(value); }

  static std::int64_t llvm_i64(std::uint64_t value) { return static_cast<std::int64_t>(value); }

  static std::string format_i16_array(std::vector<std::uint16_t> const& values)
  {
    std::string result;
    result.reserve(values.size() * 8);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i16 {}", llvm_i16(values[index]));
    }
    return result;
  }

  static std::string format_i32_array(std::vector<std::uint32_t> const& values)
  {
    std::string result;
    result.reserve(values.size() * 14U);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i32 {}", values[index]);
    }
    return result;
  }

  static std::string format_i64_array(std::vector<std::uint64_t> const& values)
  {
    std::string result;
    result.reserve(values.size() * 24U);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i64 {}", llvm_i64(values[index]));
    }
    return result;
  }

  template <typename Range>
  static std::string format_i8_array(Range const& values)
  {
    std::string result;
    result.reserve(values.size() * 7U);
    std::size_t index = 0;
    for (auto value : values) {
      if (index++ != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i8 {}", value);
    }
    return result;
  }

  static std::string format_byte_array(std::string_view value)
  {
    std::string result;
    result.reserve(value.size() * 3U);
    for (auto byte : value) {
      std::format_to(std::back_inserter(result), R"(\{:02X})", static_cast<unsigned char>(byte));
    }
    return result;
  }

  /**
   * @brief emits constant alphabet and reach-mask tables for a Glushkov executor
   *
   * @param machine bit-parallel machine to render
   */
  void emit_glushkov_globals(glushkov_machine const& machine)
  {
    std::vector<std::uint16_t> byte_classes(machine.alphabet.byte_classes.begin(),
                                            machine.alphabet.byte_classes.end());
    auto globals = std::format("@{} = internal addrspace(4) constant [256 x i16] [{}], align 2",
                               name("dfa_byte_classes"),
                               format_i16_array(byte_classes));
    if (machine.reach_masks.size() > 8U) {
      std::format_to(std::back_inserter(globals),
                     "\n@{} = internal addrspace(4) constant [{} x i64] [{}], align 8",
                     name("glushkov_reach_masks"),
                     machine.reach_masks.size(),
                     format_i64_array(machine.reach_masks));
    }
    if (machine.alphabet.unicode_intervals.size() > 8U) {
      std::vector<std::uint32_t> unicode_ends;
      std::vector<std::uint16_t> unicode_classes;
      unicode_ends.reserve(machine.alphabet.unicode_intervals.size());
      unicode_classes.reserve(machine.alphabet.unicode_intervals.size());
      for (auto interval : machine.alphabet.unicode_intervals) {
        unicode_ends.push_back(interval.last);
        unicode_classes.push_back(interval.class_id);
      }
      std::format_to(std::back_inserter(globals),
                     "\n@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                     name("dfa_unicode_ends"),
                     unicode_ends.size(),
                     format_i32_array(unicode_ends));
      std::format_to(std::back_inserter(globals),
                     "\n@{} = internal addrspace(4) constant [{} x i16] [{}], align 2",
                     name("dfa_unicode_classes"),
                     unicode_classes.size(),
                     format_i16_array(unicode_classes));
    }
    output_.emit("{}", globals);
    output_.blank();
  }

  /**
   * @brief emits the alphabet-class to active-position reach-mask mapper
   *
   * @param machine bit-parallel machine whose reach masks are rendered
   */
  void emit_glushkov_reach(glushkov_machine const& machine)
  {
    auto function = name("glushkov_reach");
    if (machine.reach_masks.size() == 1U) {
      output_.emit(
        R"NVVM(define internal i64 @{}(i32 %character_class) alwaysinline nounwind readnone {{
entry:
  ret i64 {}
}})NVVM",
        function,
        llvm_i64(machine.reach_masks.front()));
      output_.blank();
      return;
    }

    if (machine.reach_masks.size() <= 8U) {
      auto result = std::format("{}", llvm_i64(machine.reach_masks.front()));
      std::string body;
      for (std::size_t index = 1; index < machine.reach_masks.size(); ++index) {
        std::format_to(std::back_inserter(body),
                       R"NVVM(  %reach_is_{0} = icmp eq i32 %character_class, {0}
  %reach_{0} = select i1 %reach_is_{0}, i64 {1}, i64 {2}
)NVVM",
                       index,
                       llvm_i64(machine.reach_masks[index]),
                       result);
        result = std::format("%reach_{}", index);
      }
      output_.emit(
        R"NVVM(define internal i64 @{}(i32 %character_class) alwaysinline nounwind readnone {{
entry:
{}  ret i64 {}
}})NVVM",
        function,
        body,
        result);
      output_.blank();
      return;
    }

    output_.emit(
      R"NVVM(define internal i64 @{0}(i32 %character_class) alwaysinline nounwind readonly {{
entry:
  %reach_index = zext i32 %character_class to i64
  %reach_ptr = getelementptr [{1} x i64], [{1} x i64] addrspace(4)* @{2}, i64 0, i64 %reach_index
  %reach = load i64, i64 addrspace(4)* %reach_ptr, align 8
  ret i64 %reach
}})NVVM",
      function,
      machine.reach_masks.size(),
      name("glushkov_reach_masks"));
    output_.blank();
  }

  /**
   * @brief emits the bit-parallel Glushkov successor computation
   *
   * @param machine bit-parallel machine whose shift and exception transitions are rendered
   */
  void emit_glushkov_follow(glushkov_machine const& machine)
  {
    std::string body;
    auto result                = std::string{"0"};
    std::size_t combined_index = 0;
    auto append                = [&](std::string_view value) {
      if (result == "0") {
        result = value;
        return;
      }
      std::format_to(
        std::back_inserter(body), "  %follow_{} = or i64 {}, {}\n", combined_index, result, value);
      result = std::format("%follow_{}", combined_index++);
    };

    for (std::size_t index = 0; index < machine.shifts.size(); ++index) {
      auto shift = machine.shifts[index];
      std::format_to(std::back_inserter(body),
                     R"NVVM(  %shift_source_{0} = and i64 %state, {1}
  %shift_value_{0} = shl i64 %shift_source_{0}, {2}
)NVVM",
                     index,
                     llvm_i64(shift.sources),
                     shift.amount);
      append(std::format("%shift_value_{}", index));
    }

    for (std::size_t position = 0; position < machine.exception_successors.size(); ++position) {
      auto successors = machine.exception_successors[position];
      if (successors == 0U) continue;
      auto bit = std::uint64_t{1} << position;
      std::format_to(std::back_inserter(body),
                     R"NVVM(  %exception_bit_{0} = and i64 %state, {1}
  %exception_active_{0} = icmp ne i64 %exception_bit_{0}, 0
  %exception_value_{0} = select i1 %exception_active_{0}, i64 {2}, i64 0
)NVVM",
                     position,
                     llvm_i64(bit),
                     llvm_i64(successors));
      append(std::format("%exception_value_{}", position));
    }

    output_.emit(
      R"NVVM(define internal i64 @{}(i64 %state) alwaysinline nounwind readnone {{
entry:
{}  ret i64 {}
}})NVVM",
      name("glushkov_follow"),
      body,
      result);
    output_.blank();
  }

  /**
   * @brief emits constant character-class and DFA-transition tables
   *
   * @param machine deterministic machine to render
   */
  void emit_deterministic_globals(deterministic_machine const& machine)
  {
    std::vector<std::uint16_t> byte_classes(machine.byte_classes.begin(),
                                            machine.byte_classes.end());
    auto common = std::format(
      R"NVVM(@{0} = internal addrspace(4) constant [256 x i16] [{1}], align 2
@{2} = internal addrspace({3}) constant [{4} x i16] [{5}], align 2)NVVM",
      name("dfa_byte_classes"),
      format_i16_array(byte_classes),
      name("dfa_transitions"),
      machine.transition_address_space,
      machine.transitions.size(),
      format_i16_array(machine.transitions));
    if (machine.assertion_aware) {
      output_.emit("{}",
                   std::format(
                     R"NVVM({0}
@{1} = internal addrspace(4) constant [{2} x i8] [{3}], align 1)NVVM",
                     common,
                     name("dfa_boundary_accepts"),
                     machine.boundary_accepts.size(),
                     format_i8_array(machine.boundary_accepts)));
    } else {
      output_.emit("{}", common);
    }
    if (machine.unicode_intervals.size() > 8U) {
      std::vector<std::uint32_t> unicode_ends;
      std::vector<std::uint16_t> unicode_classes;
      unicode_ends.reserve(machine.unicode_intervals.size());
      unicode_classes.reserve(machine.unicode_intervals.size());
      for (auto interval : machine.unicode_intervals) {
        unicode_ends.push_back(interval.last);
        unicode_classes.push_back(interval.class_id);
      }
      output_.emit("@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                   name("dfa_unicode_ends"),
                   unicode_ends.size(),
                   format_i32_array(unicode_ends));
      output_.emit("@{} = internal addrspace(4) constant [{} x i16] [{}], align 2",
                   name("dfa_unicode_classes"),
                   unicode_classes.size(),
                   format_i16_array(unicode_classes));
    }
    output_.blank();
  }

  /**
   * @brief emits the Unicode code-point to deterministic alphabet-class mapper
   *
   * @param machine deterministic machine whose classes are rendered
   */
  void emit_deterministic_classifier(deterministic_machine const& machine)
  {
    auto function = name("dfa_classify");
    auto table    = name("dfa_byte_classes");
    std::vector<std::size_t> frequencies(machine.class_count);
    for (auto character_class : machine.byte_classes)
      ++frequencies[character_class];
    auto default_class = static_cast<std::uint16_t>(
      std::max_element(frequencies.begin(), frequencies.end()) - frequencies.begin());
    std::vector<deterministic_interval> byte_intervals;
    for (std::uint32_t byte = 0; byte < machine.byte_classes.size();) {
      auto character_class = machine.byte_classes[byte];
      if (character_class == default_class) {
        ++byte;
        continue;
      }
      auto first = byte;
      while (byte + 1U < machine.byte_classes.size() &&
             machine.byte_classes[byte + 1U] == character_class) {
        ++byte;
      }
      byte_intervals.push_back({first, byte, character_class});
      ++byte;
    }
    // more fragmented mappings favor one cached table load over comparison chains.
    auto inline_byte_classes = !byte_intervals.empty() && byte_intervals.size() <= 2U;
    std::string byte_classifier;
    if (inline_byte_classes) {
      auto result = std::format("{}", default_class);
      for (std::size_t index = 0; index < byte_intervals.size(); ++index) {
        auto interval = byte_intervals[index];
        if (interval.first == interval.last) {
          std::format_to(std::back_inserter(byte_classifier),
                         "  %byte_in_{} = icmp eq i32 %cp, {}\n",
                         index,
                         interval.first);
        } else {
          std::format_to(std::back_inserter(byte_classifier),
                         R"NVVM(  %byte_offset_{} = sub i32 %cp, {}
  %byte_in_{} = icmp ule i32 %byte_offset_{}, {}
)NVVM",
                         index,
                         interval.first,
                         index,
                         index,
                         interval.last - interval.first);
        }
        std::format_to(std::back_inserter(byte_classifier),
                       "  %byte_class_{} = select i1 %byte_in_{}, i32 {}, i32 {}\n",
                       index,
                       index,
                       interval.class_id,
                       result);
        result = std::format("%byte_class_{}", index);
      }
      std::format_to(std::back_inserter(byte_classifier), "  ret i32 {}", result);
    } else {
      byte_classifier = std::format(
        R"NVVM(  %byte_index = zext i32 %cp to i64
  %byte_class_ptr = getelementptr [256 x i16], [256 x i16] addrspace(4)* @{0}, i64 0, i64 %byte_index
  %byte_class_i16 = load i16, i16 addrspace(4)* %byte_class_ptr, align 2
  %byte_class = zext i16 %byte_class_i16 to i32
  ret i32 %byte_class)NVVM",
        table);
    }
    output_.emit("{}",
                 std::format(
                   R"NVVM(define internal i32 @{0}(i32 %cp) alwaysinline nounwind readonly {{
entry:
  %is_byte = icmp ult i32 %cp, 256
  br i1 %is_byte, label %byte, label %unicode
byte:
{1}
unicode:)NVVM",
                   function,
                   byte_classifier));

    if (machine.unicode_intervals.empty()) {
      output_.emit("  ret i32 0");
    } else if (machine.unicode_intervals.size() == 1) {
      output_.emit("  ret i32 {}", machine.unicode_intervals.front().class_id);
    } else if (machine.unicode_intervals.size() <= 8U) {
      auto result = std::format("{}", machine.unicode_intervals.back().class_id);
      for (std::size_t reverse = machine.unicode_intervals.size() - 1; reverse > 0; --reverse) {
        auto index     = reverse - 1;
        auto& interval = machine.unicode_intervals[index];
        output_.emit("{}",
                     std::format(R"NVVM(  %unicode_le_{0} = icmp ule i32 %cp, {1}
  %unicode_class_{0} = select i1 %unicode_le_{0}, i32 {2}, i32 {3})NVVM",
                                 index,
                                 interval.last,
                                 interval.class_id,
                                 result));
        result = std::format("%unicode_class_{}", index);
      }
      output_.emit("  ret i32 {}", result);
    } else {
      output_.emit(
        R"NVVM(  br label %unicode_search
unicode_search:
  %unicode_low = phi i32 [ 0, %unicode ], [ %unicode_next_low, %unicode_step ]
  %unicode_high = phi i32 [ {0}, %unicode ], [ %unicode_next_high, %unicode_step ]
  %unicode_more = icmp ult i32 %unicode_low, %unicode_high
  br i1 %unicode_more, label %unicode_probe, label %unicode_found
unicode_probe:
  %unicode_span = sub i32 %unicode_high, %unicode_low
  %unicode_half = lshr i32 %unicode_span, 1
  %unicode_mid = add i32 %unicode_low, %unicode_half
  %unicode_mid_i64 = zext i32 %unicode_mid to i64
  %unicode_end_ptr = getelementptr [{0} x i32], [{0} x i32] addrspace(4)* @{1}, i64 0, i64 %unicode_mid_i64
  %unicode_end = load i32, i32 addrspace(4)* %unicode_end_ptr, align 4
  %unicode_before_end = icmp ule i32 %cp, %unicode_end
  %unicode_after_mid = add i32 %unicode_mid, 1
  %unicode_next_low = select i1 %unicode_before_end, i32 %unicode_low, i32 %unicode_after_mid
  %unicode_next_high = select i1 %unicode_before_end, i32 %unicode_mid, i32 %unicode_high
  br label %unicode_step
unicode_step:
  br label %unicode_search
unicode_found:
  %unicode_index_i64 = zext i32 %unicode_low to i64
  %unicode_class_ptr = getelementptr [{0} x i16], [{0} x i16] addrspace(4)* @{2}, i64 0, i64 %unicode_index_i64
  %unicode_class_i16 = load i16, i16 addrspace(4)* %unicode_class_ptr, align 2
  %unicode_class = zext i16 %unicode_class_i16 to i32
  ret i32 %unicode_class)NVVM",
        machine.unicode_intervals.size(),
        name("dfa_unicode_ends"),
        name("dfa_unicode_classes"));
    }
    output_.emit("}}");
    output_.blank();
  }

  /**
   * @brief emits the position-context classifier used by assertion-aware DFA transitions
   *
   * @param machine deterministic machine whose assertion truth values are classified
   */
  void emit_deterministic_boundary_classifier(deterministic_machine const& machine)
  {
    auto function = name("dfa_boundary_classify");
    auto is_word  = name("is_word");
    auto decode   = name("decode_codepoint");
    output_.emit(
      R"NVVM(define internal i32 @{}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 %current_cp, i64 %current_width) alwaysinline nounwind readonly {{
entry:)NVVM",
      function);

    auto mask_value        = std::string{"0"};
    std::size_t mask_index = 0;
    auto append_bit        = [&](std::string_view condition) {
      auto bit = std::uint32_t{1} << mask_index;
      output_.emit("  %boundary_bit_{} = select i1 {}, i32 {}, i32 0", mask_index, condition, bit);
      if (mask_value == "0") {
        mask_value = std::format("%boundary_bit_{}", mask_index);
      } else {
        output_.emit(
          "  %boundary_mask_{} = or i32 {}, %boundary_bit_{}", mask_index, mask_value, mask_index);
        mask_value = std::format("%boundary_mask_{}", mask_index);
      }
      ++mask_index;
    };

    auto begin_input_bit = assertion_bit(assertion_kind::BEGIN_INPUT);
    auto end_input_bit   = assertion_bit(assertion_kind::END_INPUT);
    auto word_bits       = static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                               assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
    auto line_bits       = static_cast<std::uint8_t>(assertion_bit(assertion_kind::BEGIN_LINE) |
                                               assertion_bit(assertion_kind::END_LINE));
    auto needs_begin =
      (machine.assertion_mask &
       static_cast<std::uint8_t>(begin_input_bit | assertion_bit(assertion_kind::BEGIN_LINE))) != 0;
    auto needs_end =
      (machine.assertion_mask &
       static_cast<std::uint8_t>(end_input_bit | assertion_bit(assertion_kind::END_LINE))) != 0;
    if (needs_begin) output_.emit("  %at_begin = icmp eq i64 %position, 0");
    if (needs_end) output_.emit("  %at_end = icmp eq i64 %position, %size");
    if ((machine.assertion_mask & begin_input_bit) != 0) { append_bit("%at_begin"); }
    if ((machine.assertion_mask & end_input_bit) != 0) { append_bit("%at_end"); }

    if ((machine.assertion_mask & word_bits) != 0) {
      output_.emit("{}",
                   std::format(
                     R"NVVM(  %has_previous = icmp ne i64 %position, 0
  %has_current = icmp ult i64 %position, %size
  %previous_word_raw = call i1 @{0}(i32 %previous_cp)
  %current_word_raw = call i1 @{0}(i32 %current_cp)
  %previous_word = and i1 %has_previous, %previous_word_raw
  %current_word = and i1 %has_current, %current_word_raw
  %word_boundary = xor i1 %previous_word, %current_word
  %not_word_boundary = xor i1 %word_boundary, true)NVVM",
                     is_word));
      if ((machine.assertion_mask & assertion_bit(assertion_kind::WORD_BOUNDARY)) != 0) {
        append_bit("%word_boundary");
      }
      if ((machine.assertion_mask & assertion_bit(assertion_kind::NOT_WORD_BOUNDARY)) != 0) {
        append_bit("%not_word_boundary");
      }
    }

    if ((machine.assertion_mask & line_bits) != 0) {
      if (ir_.options.extended_newline) {
        output_.emit(
          R"NVVM(  %current_lf = icmp eq i32 %current_cp, 10
  %current_cr = icmp eq i32 %current_cp, 13
  %current_nel = icmp eq i32 %current_cp, 133
  %current_ls = icmp eq i32 %current_cp, 8232
  %current_ps = icmp eq i32 %current_cp, 8233
  %current_crlf = or i1 %current_lf, %current_cr
  %current_extended_0 = or i1 %current_crlf, %current_nel
  %current_extended_1 = or i1 %current_ls, %current_ps
  %current_newline = or i1 %current_extended_0, %current_extended_1
  %previous_lf = icmp eq i32 %previous_cp, 10
  %previous_cr = icmp eq i32 %previous_cp, 13
  %previous_nel = icmp eq i32 %previous_cp, 133
  %previous_ls = icmp eq i32 %previous_cp, 8232
  %previous_ps = icmp eq i32 %previous_cp, 8233
  %previous_crlf = or i1 %previous_lf, %previous_cr
  %previous_extended_0 = or i1 %previous_crlf, %previous_nel
  %previous_extended_1 = or i1 %previous_ls, %previous_ps
  %previous_newline = or i1 %previous_extended_0, %previous_extended_1
  %mid_crlf = and i1 %previous_cr, %current_lf
  %not_mid_crlf = xor i1 %mid_crlf, true)NVVM");
      } else {
        output_.emit(
          R"NVVM(  %current_lf = icmp eq i32 %current_cp, 10
  %current_newline = icmp eq i32 %current_cp, 10
  %previous_newline = icmp eq i32 %previous_cp, 10
  %not_mid_crlf = icmp eq i32 0, 0)NVVM");
      }

      if ((machine.assertion_mask & assertion_bit(assertion_kind::BEGIN_LINE)) != 0) {
        if (ir_.options.multiline) {
          output_.emit(R"NVVM(  %begin_after_newline = and i1 %previous_newline, %not_mid_crlf
  %begin_line = or i1 %at_begin, %begin_after_newline)NVVM");
        } else {
          output_.emit("  %begin_line = icmp eq i64 %position, 0");
        }
        append_bit("%begin_line");
      }

      if ((machine.assertion_mask & assertion_bit(assertion_kind::END_LINE)) != 0) {
        if (ir_.options.multiline) {
          output_.emit(R"NVVM(  %end_at_newline = and i1 %current_newline, %not_mid_crlf
  %end_line = or i1 %at_end, %end_at_newline)NVVM");
        } else {
          output_.emit(R"NVVM(  %next_position = add i64 %position, %current_width
  %current_is_final = icmp eq i64 %next_position, %size)NVVM");
          auto final_sequence = std::string{"%current_is_final"};
          if (ir_.options.extended_newline) {
            output_.emit(
              "{}",
              std::format(
                R"NVVM(  %next_cp = call i32 @{0}(i8* %data, i64 %size, i64 %next_position)
  %next_width = call i64 @{1}(i8* %data, i64 %size, i64 %next_position)
  %after_next = add i64 %next_position, %next_width
  %next_lf = icmp eq i32 %next_cp, 10
  %after_next_is_end = icmp eq i64 %after_next, %size
  %crlf_tail_0 = and i1 %current_cr, %next_lf
  %crlf_tail = and i1 %crlf_tail_0, %after_next_is_end
  %final_sequence = or i1 %current_is_final, %crlf_tail)NVVM",
                decode,
                name("decode_width")));
            final_sequence = "%final_sequence";
          }
          output_.emit(
            R"NVVM(  %end_final_newline_0 = and i1 %current_newline, %not_mid_crlf
  %end_final_newline = and i1 %end_final_newline_0, {0}
  %end_line = or i1 %at_end, %end_final_newline)NVVM",
            final_sequence);
        }
        append_bit("%end_line");
      }
    }

    output_.emit("  ret i32 {}\n}}", mask_value);
    output_.blank();
  }

  /**
   * @brief emits the bit-parallel Glushkov contains or matches executor
   *
   * @param machine bit-parallel machine to execute
   */
  void emit_glushkov_execute(glushkov_machine const& machine)
  {
    auto classify    = name("dfa_classify");
    auto decode      = name("decode_codepoint");
    auto follow      = name("glushkov_follow");
    auto load_byte   = name("load_byte");
    auto reach       = name("glushkov_reach");
    auto width       = name("decode_width");
    auto ascii_block = std::format(
      R"NVVM(ascii:
  %ascii_class = call i32 @{0}(i32 %first)
  br label %transition)NVVM",
      classify);
    auto ascii_predecessor    = std::string{"%ascii"};
    auto prefix_loop_position = std::string{};
    auto prefix_loop_state    = std::string{};
    if (machine.scan_input && machine.start_byte.has_value()) {
      ascii_predecessor    = "%ascii_classify";
      prefix_loop_position = ", [ %prefix_next_position, %prefix_skip ]";
      prefix_loop_state    = ", [ 0, %prefix_skip ]";
      ascii_block          = std::format(
        R"NVVM(ascii:
  %prefix_state_empty = icmp eq i64 %state, 0
  %prefix_equal = icmp eq i32 %first, {0}
  %prefix_mismatch = xor i1 %prefix_equal, true
  %prefix_skip_candidate = and i1 %prefix_state_empty, %prefix_mismatch
  br i1 %prefix_skip_candidate, label %prefix_skip, label %ascii_classify
ascii_classify:
  %ascii_class = call i32 @{1}(i32 %first)
  br label %transition
prefix_skip:
  %prefix_next_position = add nuw i64 %position, 1
  br label %loop)NVVM",
        *machine.start_byte,
        classify);
    }

    auto inject = machine.scan_input ? std::format("  %candidates = or i64 %follow, {}\n",
                                                   llvm_i64(machine.first_set))
                                     : std::format(R"NVVM(  %at_start = icmp eq i64 %position, 0
  %initial_positions = select i1 %at_start, i64 {0}, i64 0
  %candidates = or i64 %follow, %initial_positions
)NVVM",
                                                   llvm_i64(machine.first_set));

    output_.emit(
      R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %loop
loop:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]{1}
  %state = phi i64 [ 0, %entry ], [ %next_state, %continue ]{2}
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %done, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{3}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
{4}
unicode:
  %codepoint = call i32 @{5}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{6}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{7}(i32 %codepoint)
  br label %transition
transition:
  %character_class = phi i32 [ %ascii_class, {8} ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, {8} ], [ %unicode_width, %unicode ]
  %reach = call i64 @{9}(i32 %character_class)
  %follow = call i64 @{10}(i64 %state)
{11}  %next_state = and i64 %candidates, %reach
  %next_position = add nuw i64 %position, %character_width)NVVM",
      options_.execute_function,
      prefix_loop_position,
      prefix_loop_state,
      load_byte,
      ascii_block,
      decode,
      width,
      classify,
      ascii_predecessor,
      reach,
      follow,
      inject);

    if (machine.accept_at_end) {
      output_.emit(
        R"NVVM(  %dead = icmp eq i64 %next_state, 0
  br i1 %dead, label %reject, label %continue
continue:
  br label %loop
done:
  %accept_bits = and i64 %state, {0}
  %accepted = icmp ne i64 %accept_bits, 0
  ret i1 %accepted
reject:
  ret i1 false
}})NVVM",
        llvm_i64(machine.accept_mask));
    } else {
      output_.emit(
        R"NVVM(  %accept_bits = and i64 %next_state, {0}
  %accepted = icmp ne i64 %accept_bits, 0
  br i1 %accepted, label %yes, label %continue
continue:
  br label %loop
done:
  ret i1 false
yes:
  ret i1 true
}})NVVM",
        llvm_i64(machine.accept_mask));
    }
    output_.blank();
  }

  /**
   * @brief emits a streaming leftmost-prioritized Glushkov span finder
   *
   * The first pass advances all candidate starts together until it finds the greedy match end.
   * The second pass recovers the earliest start that reaches an accepting position. This avoids
   * restarting the complete automaton at every input position while preserving the span ABI used
   * by count, replacement, and split.
   *
   * @param machine bit-parallel machine to execute
   */
  void emit_glushkov_find_from(glushkov_machine const& machine)
  {
    auto accept_at_end = machine.accept_at_end ? "%phase1_at_input_end" : "true";
    auto rescan_accept_at_end = machine.accept_at_end ? "%rescan_at_input_end" : "true";
    if (!prefix_seek_byte_.has_value()) {
      output_.emit("declare i64 @llvm.cttz.i64(i64, i1)");
      output_.blank();
    }
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i64 @{0}(i64 %state) alwaysinline nounwind readnone {{
entry:
  %accept_bits = and i64 %state, {9}
  %accepted = icmp ne i64 %accept_bits, 0
  br i1 %accepted, label %trim, label %done
trim:
  %first_accept = call i64 @llvm.cttz.i64(i64 %accept_bits, i1 false)
  %last_bit = icmp eq i64 %first_accept, 63
  %next_bit = add nuw nsw i64 %first_accept, 1
  %safe_shift = select i1 %last_bit, i64 63, i64 %next_bit
  %one_past = shl i64 1, %safe_shift
  %priority_mask = sub i64 %one_past, 1
  %trimmed = and i64 %state, %priority_mask
  %priority_state = select i1 %last_bit, i64 %state, i64 %trimmed
  br label %done
done:
  %result = phi i64 [ %state, %entry ], [ %priority_state, %trim ]
  ret i64 %result
}}

define internal i1 @{1}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind {{
entry:
  %has_input = icmp ult i64 %search_start, %size
  br i1 %has_input, label %phase1_loop, label %no

phase1_loop:
  %phase1_position = phi i64 [ %search_start, %entry ], [ %phase1_next_position, %phase1_continue ]
  %phase1_state = phi i64 [ 0, %entry ], [ %phase1_next_state, %phase1_continue ]
  %phase1_match_end = phi i64 [ -1, %entry ], [ %phase1_next_match_end, %phase1_continue ]
  %phase1_inject_start = phi i64 [ %search_start, %entry ], [ %phase1_next_inject_start, %phase1_continue ]
  %phase1_ptr = getelementptr i8, i8* %data, i64 %phase1_position
  %phase1_first = call i32 @{2}(i8* %phase1_ptr)
  %phase1_ascii = icmp ult i32 %phase1_first, 128
  br i1 %phase1_ascii, label %phase1_ascii_path, label %phase1_unicode_path

phase1_ascii_path:
  %phase1_ascii_class = call i32 @{5}(i32 %phase1_first)
  br label %phase1_transition

phase1_unicode_path:
  %phase1_codepoint = call i32 @{3}(i8* %data, i64 %size, i64 %phase1_position)
  %phase1_unicode_width = call i64 @{4}(i8* %data, i64 %size, i64 %phase1_position)
  %phase1_unicode_class = call i32 @{5}(i32 %phase1_codepoint)
  br label %phase1_transition

phase1_transition:
  %phase1_class = phi i32 [ %phase1_ascii_class, %phase1_ascii_path ], [ %phase1_unicode_class, %phase1_unicode_path ]
  %phase1_width = phi i64 [ 1, %phase1_ascii_path ], [ %phase1_unicode_width, %phase1_unicode_path ]
  %phase1_reach = call i64 @{6}(i32 %phase1_class)
  %phase1_follow = call i64 @{7}(i64 %phase1_state)
  %phase1_no_match = icmp eq i64 %phase1_match_end, -1
  %phase1_seed = select i1 %phase1_no_match, i64 {8}, i64 0
  %phase1_candidates = or i64 %phase1_follow, %phase1_seed
  %phase1_raw_state = and i64 %phase1_candidates, %phase1_reach
  %phase1_next_position = add nuw i64 %phase1_position, %phase1_width
  %phase1_accept_bits = and i64 %phase1_raw_state, {9}
  %phase1_accept_bits_nonzero = icmp ne i64 %phase1_accept_bits, 0
  %phase1_at_input_end = icmp eq i64 %phase1_next_position, %size
  %phase1_accepted = and i1 %phase1_accept_bits_nonzero, {10}
  %phase1_next_match_end = select i1 %phase1_accepted, i64 %phase1_next_position, i64 %phase1_match_end
  %phase1_state_empty = icmp eq i64 %phase1_state, 0
  %phase1_new_window = and i1 %phase1_state_empty, %phase1_no_match
  %phase1_next_inject_start = select i1 %phase1_new_window, i64 %phase1_position, i64 %phase1_inject_start
  %phase1_next_state = call i64 @{0}(i64 %phase1_raw_state)
  %phase1_dead = icmp eq i64 %phase1_next_state, 0
  %phase1_has_match = icmp ne i64 %phase1_next_match_end, -1
  %phase1_greedy_done = and i1 %phase1_dead, %phase1_has_match
  %phase1_done = or i1 %phase1_greedy_done, %phase1_at_input_end
  br i1 %phase1_done, label %phase1_finish, label %phase1_continue

phase1_continue:
  br label %phase1_loop

phase1_finish:
  br i1 %phase1_has_match, label %rescan_search, label %no

rescan_search:
  %candidate_start = phi i64 [ %phase1_next_inject_start, %phase1_finish ], [ %next_candidate, %rescan_advance ]
  %candidate_in_range = icmp ult i64 %candidate_start, %phase1_next_match_end
  br i1 %candidate_in_range, label %rescan_seed, label %no

rescan_seed:
  %rescan_ptr = getelementptr i8, i8* %data, i64 %candidate_start
  %rescan_first = call i32 @{2}(i8* %rescan_ptr)
  %rescan_ascii = icmp ult i32 %rescan_first, 128
  br i1 %rescan_ascii, label %rescan_ascii_path, label %rescan_unicode_path

rescan_ascii_path:
  %rescan_ascii_class = call i32 @{5}(i32 %rescan_first)
  br label %rescan_seed_transition

rescan_unicode_path:
  %rescan_codepoint = call i32 @{3}(i8* %data, i64 %size, i64 %candidate_start)
  %rescan_unicode_width = call i64 @{4}(i8* %data, i64 %size, i64 %candidate_start)
  %rescan_unicode_class = call i32 @{5}(i32 %rescan_codepoint)
  br label %rescan_seed_transition

rescan_seed_transition:
  %rescan_class = phi i32 [ %rescan_ascii_class, %rescan_ascii_path ], [ %rescan_unicode_class, %rescan_unicode_path ]
  %rescan_width = phi i64 [ 1, %rescan_ascii_path ], [ %rescan_unicode_width, %rescan_unicode_path ]
  %rescan_reach = call i64 @{6}(i32 %rescan_class)
  %rescan_seed_state = and i64 %rescan_reach, {8}
  %rescan_next_position = add nuw i64 %candidate_start, %rescan_width
  %rescan_accept_bits = and i64 %rescan_seed_state, {9}
  %rescan_accept_bits_nonzero = icmp ne i64 %rescan_accept_bits, 0
  %rescan_at_input_end = icmp eq i64 %rescan_next_position, %size
  %rescan_accepted = and i1 %rescan_accept_bits_nonzero, {11}
  %rescan_initial_end = select i1 %rescan_accepted, i64 %rescan_next_position, i64 -1
  %rescan_initial_state = call i64 @{0}(i64 %rescan_seed_state)
  br label %rescan_loop

rescan_loop:
  %rescan_position = phi i64 [ %rescan_next_position, %rescan_seed_transition ], [ %rescan_advanced_position, %rescan_transition ]
  %rescan_state = phi i64 [ %rescan_initial_state, %rescan_seed_transition ], [ %rescan_advanced_state, %rescan_transition ]
  %rescan_match_end = phi i64 [ %rescan_initial_end, %rescan_seed_transition ], [ %rescan_advanced_match_end, %rescan_transition ]
  %rescan_dead = icmp eq i64 %rescan_state, 0
  %rescan_end = icmp eq i64 %rescan_position, %size
  %rescan_stop = or i1 %rescan_dead, %rescan_end
  br i1 %rescan_stop, label %rescan_candidate_done, label %rescan_load

rescan_load:
  %rescan_loop_ptr = getelementptr i8, i8* %data, i64 %rescan_position
  %rescan_loop_first = call i32 @{2}(i8* %rescan_loop_ptr)
  %rescan_loop_ascii = icmp ult i32 %rescan_loop_first, 128
  br i1 %rescan_loop_ascii, label %rescan_loop_ascii_path, label %rescan_loop_unicode_path

rescan_loop_ascii_path:
  %rescan_loop_ascii_class = call i32 @{5}(i32 %rescan_loop_first)
  br label %rescan_transition

rescan_loop_unicode_path:
  %rescan_loop_codepoint = call i32 @{3}(i8* %data, i64 %size, i64 %rescan_position)
  %rescan_loop_unicode_width = call i64 @{4}(i8* %data, i64 %size, i64 %rescan_position)
  %rescan_loop_unicode_class = call i32 @{5}(i32 %rescan_loop_codepoint)
  br label %rescan_transition

rescan_transition:
  %rescan_loop_class = phi i32 [ %rescan_loop_ascii_class, %rescan_loop_ascii_path ], [ %rescan_loop_unicode_class, %rescan_loop_unicode_path ]
  %rescan_loop_width = phi i64 [ 1, %rescan_loop_ascii_path ], [ %rescan_loop_unicode_width, %rescan_loop_unicode_path ]
  %rescan_loop_reach = call i64 @{6}(i32 %rescan_loop_class)
  %rescan_loop_follow = call i64 @{7}(i64 %rescan_state)
  %rescan_raw_state = and i64 %rescan_loop_follow, %rescan_loop_reach
  %rescan_advanced_position = add nuw i64 %rescan_position, %rescan_loop_width
  %rescan_loop_accept_bits = and i64 %rescan_raw_state, {9}
  %rescan_loop_accept_bits_nonzero = icmp ne i64 %rescan_loop_accept_bits, 0
  %rescan_loop_at_input_end = icmp eq i64 %rescan_advanced_position, %size
  %rescan_loop_accepted = and i1 %rescan_loop_accept_bits_nonzero, {12}
  %rescan_advanced_match_end = select i1 %rescan_loop_accepted, i64 %rescan_advanced_position, i64 %rescan_match_end
  %rescan_advanced_state = call i64 @{0}(i64 %rescan_raw_state)
  br label %rescan_loop

rescan_candidate_done:
  %candidate_matched = icmp ne i64 %rescan_match_end, -1
  br i1 %candidate_matched, label %yes, label %rescan_advance

rescan_advance:
  %next_candidate = call i64 @{13}(i8* %data, i64 %size, i64 %candidate_start, i64 1)
  br label %rescan_search

yes:
  store i64 %candidate_start, i64* %match_begin, align 8
  store i64 %rescan_match_end, i64* %match_end, align 8
  ret i1 true

no:
  ret i1 false
}})NVVM",
        name("glushkov_priority_kill"),
        name("find_from"),
        name("load_byte"),
        name("decode_codepoint"),
        name("decode_width"),
        name("dfa_classify"),
        name("glushkov_reach"),
        name("glushkov_follow"),
        llvm_i64(machine.first_set),
        llvm_i64(machine.accept_mask),
        accept_at_end,
        rescan_accept_at_end,
        machine.accept_at_end ? "%rescan_loop_at_input_end" : "true",
        name("advance")));
    output_.blank();
  }

  /**
   * @brief emits the single-pass deterministic contains or matches executor
   *
   * @param machine deterministic machine to execute
   */
  void emit_deterministic_execute(deterministic_machine const& machine)
  {
    if ((machine.initial_state & 0x8000U) != 0 && !machine.accept_at_end &&
        !machine.accept_assertion.has_value()) {
      output_.emit(
        R"NVVM(define zeroext i1 @{}(i8* %data, i64 %size) alwaysinline nounwind readnone {{
entry:
  ret i1 true
}})NVVM",
        options_.execute_function);
      output_.blank();
      return;
    }

    auto load_byte            = name("load_byte");
    auto decode               = name("decode_codepoint");
    auto width                = name("decode_width");
    auto classify             = name("dfa_classify");
    auto transitions          = name("dfa_transitions");
    auto dead_guard           = std::string{};
    auto reject               = std::string{};
    auto prefix_loop_position = std::string{};
    auto prefix_loop_state    = std::string{};
    auto ascii_block          = std::format(
      R"NVVM(ascii:
  %ascii_class = call i32 @{0}(i32 %first)
  br label %transition)NVVM",
      classify);
    auto ascii_predecessor = std::string{"%ascii"};
    auto prefix            = machine.scan_input ? prefix_seek_byte_ : std::nullopt;
    if (prefix.has_value()) {
      prefix_loop_position = ", [ %prefix_next_position, %prefix_skip ]";
      prefix_loop_state    = ", [ " + std::to_string(machine.initial_state) + ", %prefix_skip ]";
      ascii_predecessor    = "%ascii_classify";
      ascii_block          = std::format(
        R"NVVM(ascii:
  %prefix_state_index = and i32 %state, {0}
  %prefix_at_initial = icmp eq i32 %prefix_state_index, {1}
  %prefix_equal = icmp eq i32 %first, {2}
  %prefix_mismatch = xor i1 %prefix_equal, true
  %prefix_skip_candidate = and i1 %prefix_at_initial, %prefix_mismatch
  br i1 %prefix_skip_candidate, label %prefix_skip, label %ascii_classify
ascii_classify:
  %ascii_class = call i32 @{3}(i32 %first)
  br label %transition
prefix_skip:
  %prefix_seek_start = add nuw i64 %position, 1
  %prefix_next_position = call i64 @{4}(i8* %data, i64 %size, i64 %prefix_seek_start)
  br label %loop)NVVM",
        machine.state_mask,
        machine.initial_state & machine.state_mask,
        static_cast<std::uint32_t>(*prefix),
        classify,
        name("seek_prefix_byte"));
    }
    if (!machine.scan_input && machine.dead_state <= machine.state_mask) {
      dead_guard = std::format(
        R"NVVM(  %next_state_index = and i32 %next_state, {0}
  %dead = icmp eq i32 %next_state_index, {1}
  br i1 %dead, label %reject, label %transition_live
transition_live:)NVVM",
        machine.state_mask,
        machine.dead_state);
      reject = R"NVVM(reject:
  ret i1 false
)NVVM";
    }
    output_.emit("{}",
                 std::format(
                   R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %loop
loop:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]{12}
  %state = phi i32 [ {1}, %entry ], [ %next_state, %continue ]{13}
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %done, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{2}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
{14}
unicode:
  %codepoint = call i32 @{4}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{5}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{3}(i32 %codepoint)
  br label %transition
transition:
  %character_class = phi i32 [ %ascii_class, {15} ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, {15} ], [ %unicode_width, %unicode ]
  %state_index = and i32 %state, {7}
  %state_offset = mul nuw i32 %state_index, {6}
  %transition_index = add nuw i32 %state_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{8} x i16], [{8} x i16] addrspace({9})* @{10}, i64 0, i64 %transition_index_i64
  %next_state_i16 = load i16, i16 addrspace({9})* %transition_ptr, align 2
  %next_state = zext i16 %next_state_i16 to i32
  %next_position = add i64 %position, %character_width
{11})NVVM",
                   options_.execute_function,
                   machine.initial_state,
                   load_byte,
                   classify,
                   decode,
                   width,
                   machine.class_count,
                   machine.state_mask,
                   machine.transitions.size(),
                   machine.transition_address_space,
                   transitions,
                   dead_guard,
                   prefix_loop_position,
                   prefix_loop_state,
                   ascii_block,
                   ascii_predecessor));
    if (!machine.accept_at_end && machine.accept_assertion.has_value()) {
      output_.emit("{}",
                   std::format(
                     R"NVVM(  %accept_bits = and i32 %next_state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  br i1 %accepted, label %check_accept, label %continue
check_accept:
  %accept_assertion = call i1 @{0}(i8* %data, i64 %size, i64 %next_position, i32 {1})
  br i1 %accept_assertion, label %yes, label %continue
continue:
  br label %loop
done:
  %done_accept_bits = and i32 %state, 32768
  %done_accepted = icmp ne i32 %done_accept_bits, 0
  %done_assertion = call i1 @{0}(i8* %data, i64 %size, i64 %position, i32 {1})
  %done_result = and i1 %done_accepted, %done_assertion
  ret i1 %done_result
yes:
  ret i1 true
{2}
}})NVVM",
                     name("assertion"),
                     static_cast<std::uint32_t>(*machine.accept_assertion),
                     reject));
    } else if (!machine.accept_at_end) {
      output_.emit("{}",
                   std::format(R"NVVM(  %accept_bits = and i32 %next_state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  br i1 %accepted, label %yes, label %continue
continue:
  br label %loop
done:
  ret i1 false
yes:
  ret i1 true
{0}
}})NVVM",
                               reject));
    } else {
      auto assertion = std::string{};
      auto result    = std::string{"%accepted"};
      if (machine.accept_assertion.has_value()) {
        assertion = std::format(
          R"NVVM(  %done_assertion = call i1 @{0}(i8* %data, i64 %size, i64 %position, i32 {1})
  %done_result = and i1 %accepted, %done_assertion
)NVVM",
          name("assertion"),
          static_cast<std::uint32_t>(*machine.accept_assertion));
        result = "%done_result";
      }
      output_.emit("{}",
                   std::format(
                     R"NVVM(  br label %continue
continue:
  br label %loop
done:
  %accept_bits = and i32 %state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
{0}  ret i1 {1}
{2}
}})NVVM",
                     assertion,
                     result,
                     reject));
    }
    output_.blank();
  }

  /**
   * @brief emits a single-pass DFA whose epsilon closure is selected by boundary context
   *
   * @param machine assertion-aware deterministic machine to execute
   */
  void emit_assertion_deterministic_execute(deterministic_machine const& machine)
  {
    auto load_byte        = name("load_byte");
    auto decode           = name("decode_codepoint");
    auto width            = name("decode_width");
    auto classify         = name("dfa_classify");
    auto boundary         = name("dfa_boundary_classify");
    auto transitions      = name("dfa_transitions");
    auto boundary_accepts = name("dfa_boundary_accepts");
    output_.emit("{}",
                 std::format(
                   R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %loop
loop:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]
  %state = phi i32 [ {1}, %entry ], [ %next_state, %continue ]
  %previous_cp = phi i32 [ 0, %entry ], [ %current_cp, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %finish, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{2}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
ascii:
  %ascii_class = call i32 @{3}(i32 %first)
  br label %boundary
unicode:
  %codepoint = call i32 @{4}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{5}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{3}(i32 %codepoint)
  br label %boundary
boundary:
  %current_cp = phi i32 [ %first, %ascii ], [ %codepoint, %unicode ]
  %character_class = phi i32 [ %ascii_class, %ascii ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, %ascii ], [ %unicode_width, %unicode ]
  %boundary_class = call i32 @{6}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 %current_cp, i64 %character_width)
  br label %transition)NVVM",
                   options_.execute_function,
                   machine.initial_state,
                   load_byte,
                   classify,
                   decode,
                   width,
                   boundary));
    output_.emit("{}",
                 std::format(
                   R"NVVM(transition:
  %transition_state_offset = mul nuw i32 %state, {0}
  %transition_context_index = add nuw i32 %transition_state_offset, %boundary_class
  %transition_context_offset = mul nuw i32 %transition_context_index, {1}
  %transition_index = add nuw i32 %transition_context_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{2} x i16], [{2} x i16] addrspace({3})* @{4}, i64 0, i64 %transition_index_i64
  %next_state_i16 = load i16, i16 addrspace({3})* %transition_ptr, align 2
  %next_state_raw = zext i16 %next_state_i16 to i32
  %accept_bits = and i32 %next_state_raw, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  %next_state = and i32 %next_state_raw, 32767
  %next_position = add i64 %position, %character_width)NVVM",
                   machine.boundary_class_count,
                   machine.class_count,
                   machine.transitions.size(),
                   machine.transition_address_space,
                   transitions));
    if (machine.accept_at_end) {
      output_.emit("  br label %continue");
    } else {
      output_.emit("  br i1 %accepted, label %yes, label %continue");
    }
    output_.emit("{}",
                 std::format(
                   R"NVVM(continue:
  br label %loop
finish:
  %finish_boundary_class = call i32 @{1}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 0, i64 0)
  %finish_state_offset = mul nuw i32 %state, {0}
  %finish_accept_index = add nuw i32 %finish_state_offset, %finish_boundary_class
  %finish_accept_index_i64 = zext i32 %finish_accept_index to i64
  %finish_accept_ptr = getelementptr [{2} x i8], [{2} x i8] addrspace(4)* @{3}, i64 0, i64 %finish_accept_index_i64
  %finish_accept_i8 = load i8, i8 addrspace(4)* %finish_accept_ptr, align 1
  %finish_accepted = icmp ne i8 %finish_accept_i8, 0
  ret i1 %finish_accepted
yes:
  ret i1 true
}})NVVM",
                   machine.boundary_class_count,
                   boundary,
                   machine.boundary_accepts.size(),
                   boundary_accepts));
    output_.blank();
  }

  /**
   * @brief emits NVVM intrinsics used by branch hints and replacement range copies
   */
  void emit_optimizer_intrinsics()
  {
    auto expect      = required_ascii_prefix().has_value();
    auto memory_copy = ir_.control.result == result_shape::REPLACEMENT;
    if (expect && memory_copy) {
      output_.emit(R"NVVM(declare i1 @llvm.expect.i1(i1, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1))NVVM");
    } else if (expect) {
      output_.emit("declare i1 @llvm.expect.i1(i1, i1)");
    } else if (memory_copy) {
      output_.emit("declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)");
    }
    if (expect || memory_copy) output_.blank();
  }

  /**
   * @brief emits the compiler-cached byte load used by every input decoder
   */
  void emit_load_byte()
  {
    auto function = name("load_byte");
    output_.emit(
      R"NVVM(define internal i32 @{}(i8* %ptr) alwaysinline nounwind readonly {{
entry:
  %value8 = load i8, i8* %ptr, align 1
  %value = zext i8 %value8 to i32
  ret i32 %value
}})NVVM",
      function);
    output_.blank();
  }

  /**
   * @brief emits the helper that returns the byte width of the character at a cursor position
   */
  void emit_decode_width()
  {
    auto function  = name("decode_width");
    auto load_byte = name("load_byte");
    output_.emit(
      R"NVVM(define internal i64 @{}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %in_bounds = icmp ult i64 %pos, %size
  br i1 %in_bounds, label %load, label %missing
missing:
  ret i64 0
load:)NVVM",
      function);
    if (ir_.options.characters == character_mode::BYTES) {
      output_.emit(
        R"NVVM(  ret i64 1
}})NVVM");
      output_.blank();
      return;
    }
    output_.emit(
      R"NVVM(  %ptr = getelementptr i8, i8* %data, i64 %pos
  %first = call i32 @{}(i8* %ptr)
  %ascii = icmp ult i32 %first, 128
  br i1 %ascii, label %one, label %classify
one:
  ret i64 1
classify:
  %ge2 = icmp uge i32 %first, 194
  %le2 = icmp ule i32 %first, 223
  %is2 = and i1 %ge2, %le2
  %ge3 = icmp uge i32 %first, 224
  %le3 = icmp ule i32 %first, 239
  %is3 = and i1 %ge3, %le3
  %ge4 = icmp uge i32 %first, 240
  %le4 = icmp ule i32 %first, 244
  %is4 = and i1 %ge4, %le4
  %maybe3 = select i1 %is3, i64 3, i64 1
  %maybe4 = select i1 %is4, i64 4, i64 %maybe3
  %required = select i1 %is2, i64 2, i64 %maybe4
  %is_multibyte = icmp ugt i64 %required, 1
  br i1 %is_multibyte, label %bounds, label %invalid
invalid:
  ret i64 1
bounds:
  %end = add i64 %pos, %required
  %enough = icmp ule i64 %end, %size
  br i1 %enough, label %continuation_loop, label %invalid_short
invalid_short:
  ret i64 1
continuation_loop:
  %index = phi i64 [ 1, %bounds ], [ %next_index, %continuation_ok ]
  %continuation_pos = add i64 %pos, %index
  %continuation_ptr = getelementptr i8, i8* %data, i64 %continuation_pos
  %continuation = call i32 @{}(i8* %continuation_ptr)
  %tag = and i32 %continuation, 192
  %valid = icmp eq i32 %tag, 128
  br i1 %valid, label %continuation_ok, label %invalid_continuation
invalid_continuation:
  ret i64 1
continuation_ok:
  %next_index = add i64 %index, 1
  %done = icmp eq i64 %next_index, %required
  br i1 %done, label %valid_multibyte, label %continuation_loop
valid_multibyte:
  ret i64 %required
}})NVVM",
      load_byte,
      load_byte);
    output_.blank();
  }

  /**
   * @brief emits the helper that decodes the character at a cursor position into a code point
   */
  void emit_decode_codepoint()
  {
    auto function  = name("decode_codepoint");
    auto width     = name("decode_width");
    auto load_byte = name("load_byte");
    output_.emit(
      R"NVVM(define internal i32 @{}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %available = icmp ult i64 %pos, %size
  br i1 %available, label %load, label %missing
missing:
  ret i32 0
load:
  %ptr = getelementptr i8, i8* %data, i64 %pos
  %first = call i32 @{}(i8* %ptr))NVVM",
      function,
      load_byte);
    if (ir_.options.characters == character_mode::BYTES) {
      output_.emit(
        R"NVVM(  ret i32 %first
}})NVVM");
      output_.blank();
      return;
    }
    output_.emit(
      R"NVVM(  %width = call i64 @{}(i8* %data, i64 %size, i64 %pos)
  %single = icmp eq i64 %width, 1
  br i1 %single, label %single_byte, label %initialize
single_byte:
  ret i32 %first
initialize:
  %is2 = icmp eq i64 %width, 2
  %is3 = icmp eq i64 %width, 3
  %mask3 = select i1 %is3, i32 15, i32 7
  %mask = select i1 %is2, i32 31, i32 %mask3
  %initial = and i32 %first, %mask
  br label %loop
loop:
  %index = phi i64 [ 1, %initialize ], [ %next_index, %body ]
  %value = phi i32 [ %initial, %initialize ], [ %combined, %body ]
  %done = icmp uge i64 %index, %width
  br i1 %done, label %exit, label %body
body:
  %byte_pos = add i64 %pos, %index
  %byte_ptr = getelementptr i8, i8* %data, i64 %byte_pos
  %byte = call i32 @{}(i8* %byte_ptr)
  %payload = and i32 %byte, 63
  %shifted = shl i32 %value, 6
  %combined = or i32 %shifted, %payload
  %next_index = add i64 %index, 1
  br label %loop
exit:
  ret i32 %value
}})NVVM",
      width,
      load_byte);
    output_.blank();
  }

  /**
   * @brief emits the helper that advances a byte cursor by a requested number of characters
   */
  void emit_advance()
  {
    auto function = name("advance");
    auto width    = name("decode_width");
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i64 @{0}(i8* %data, i64 %size, i64 %pos, i64 %count) alwaysinline nounwind readonly {{
entry:
  br label %loop
loop:
  %cursor = phi i64 [ %pos, %entry ], [ %next_cursor, %step ]
  %index = phi i64 [ 0, %entry ], [ %next_index, %step ]
  %done = icmp uge i64 %index, %count
  br i1 %done, label %exit, label %check
check:
  %width = call i64 @{1}(i8* %data, i64 %size, i64 %cursor)
  %missing = icmp eq i64 %width, 0
  br i1 %missing, label %exit, label %step
step:
  %next_cursor = add i64 %cursor, %width
  %next_index = add i64 %index, 1
  br label %loop
exit:
  ret i64 %cursor
}})NVVM",
        function,
        width));
    output_.blank();
  }

  /**
   * @brief emits the helper that checks whether a requested number of characters is available
   */
  void emit_can_peek()
  {
    auto function = name("can_peek");
    auto width    = name("decode_width");
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %pos, i64 %count) alwaysinline nounwind readonly {{
entry:
  br label %loop
loop:
  %cursor = phi i64 [ %pos, %entry ], [ %next_cursor, %step ]
  %index = phi i64 [ 0, %entry ], [ %next_index, %step ]
  %done = icmp uge i64 %index, %count
  br i1 %done, label %yes, label %check
check:
  %width = call i64 @{1}(i8* %data, i64 %size, i64 %cursor)
  %missing = icmp eq i64 %width, 0
  br i1 %missing, label %no, label %step
step:
  %next_cursor = add i64 %cursor, %width
  %next_index = add i64 %index, 1
  br label %loop
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
        function,
        width));
    output_.blank();
  }

  /**
   * @brief reports whether the program evaluates a Unicode word boundary
   *
   * @return true when a boundary assertion needs cuDF's Unicode word table
   */
  [[nodiscard]] bool uses_unicode_word_boundaries() const
  {
    if (ir_.options.ascii_classes || ir_.options.characters == character_mode::BYTES) return false;
    for (instruction_block const& block : ir_.blocks) {
      for (instruction const& item : block.instructions) {
        auto* assertion = std::get_if<test_assertion>(&item);
        if (assertion != nullptr && (assertion->kind == assertion_kind::WORD_BOUNDARY ||
                                     assertion->kind == assertion_kind::NOT_WORD_BOUNDARY)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * @brief emits the configured word-character classifier used by boundary assertions
   */
  void emit_is_word()
  {
    auto function = name("is_word");
    if (uses_unicode_word_boundaries()) {
      constexpr auto ascii_range_count = std::size_t{3};
      auto unicode_firsts              = std::vector<std::uint32_t>{};
      auto unicode_ends                = std::vector<std::uint32_t>{};
      unicode_firsts.reserve(std::size(unicode_word_ranges) - ascii_range_count);
      unicode_ends.reserve(std::size(unicode_word_ranges) - ascii_range_count);
      for (auto index = ascii_range_count; index < std::size(unicode_word_ranges); ++index) {
        unicode_firsts.push_back(unicode_word_ranges[index].first);
        unicode_ends.push_back(unicode_word_ranges[index].last);
      }
      output_.emit("@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                   name("word_range_firsts"),
                   unicode_firsts.size(),
                   format_i32_array(unicode_firsts));
      output_.emit("@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                   name("word_range_ends"),
                   unicode_ends.size(),
                   format_i32_array(unicode_ends));
      output_.blank();
      output_.emit(
        R"NVVM(define internal i1 @{0}(i32 %cp) alwaysinline nounwind readonly {{
entry:
  %is_ascii = icmp ult i32 %cp, 128
  br i1 %is_ascii, label %ascii, label %unicode_bounds
ascii:
  %digit_low = icmp uge i32 %cp, 48
  %digit_high = icmp ule i32 %cp, 57
  %digit = and i1 %digit_low, %digit_high
  %upper_low = icmp uge i32 %cp, 65
  %upper_high = icmp ule i32 %cp, 90
  %upper = and i1 %upper_low, %upper_high
  %lower_low = icmp uge i32 %cp, 97
  %lower_high = icmp ule i32 %cp, 122
  %lower = and i1 %lower_low, %lower_high
  %underscore = icmp eq i32 %cp, 95
  %alpha = or i1 %upper, %lower
  %alnum = or i1 %alpha, %digit
  %ascii_result = or i1 %alnum, %underscore
  ret i1 %ascii_result
unicode_bounds:
  %in_table = icmp ule i32 %cp, {1}
  br i1 %in_table, label %unicode_search, label %no
unicode_search:
  %low = phi i32 [ 0, %unicode_bounds ], [ %next_low, %search_step ]
  %high = phi i32 [ {2}, %unicode_bounds ], [ %next_high, %search_step ]
  %more = icmp ult i32 %low, %high
  br i1 %more, label %probe, label %found
probe:
  %span = sub i32 %high, %low
  %half = lshr i32 %span, 1
  %mid = add i32 %low, %half
  %mid_i64 = zext i32 %mid to i64
  %end_ptr = getelementptr [{2} x i32], [{2} x i32] addrspace(4)* @{3}, i64 0, i64 %mid_i64
  %range_end = load i32, i32 addrspace(4)* %end_ptr, align 4
  %before_end = icmp ule i32 %cp, %range_end
  %after_mid = add i32 %mid, 1
  %next_low = select i1 %before_end, i32 %low, i32 %after_mid
  %next_high = select i1 %before_end, i32 %mid, i32 %high
  br label %search_step
search_step:
  br label %unicode_search
found:
  %index_i64 = zext i32 %low to i64
  %first_ptr = getelementptr [{2} x i32], [{2} x i32] addrspace(4)* @{4}, i64 0, i64 %index_i64
  %range_first = load i32, i32 addrspace(4)* %first_ptr, align 4
  %word = icmp uge i32 %cp, %range_first
  ret i1 %word
no:
  ret i1 false
}})NVVM",
        function,
        unicode_ends.back(),
        unicode_ends.size(),
        name("word_range_ends"),
        name("word_range_firsts"));
      output_.blank();
      return;
    }
    output_.emit(
      R"NVVM(define internal i1 @{}(i32 %cp) alwaysinline nounwind readnone {{
entry:
  %ge_lower = icmp uge i32 %cp, 97
  %le_lower = icmp ule i32 %cp, 122
  %lower = and i1 %ge_lower, %le_lower
  %ge_upper = icmp uge i32 %cp, 65
  %le_upper = icmp ule i32 %cp, 90
  %upper = and i1 %ge_upper, %le_upper
  %ge_digit = icmp uge i32 %cp, 48
  %le_digit = icmp ule i32 %cp, 57
  %digit = and i1 %ge_digit, %le_digit
  %alpha = or i1 %lower, %upper
  %alnum = or i1 %alpha, %digit
  %underscore = icmp eq i32 %cp, 95
  %result = or i1 %alnum, %underscore
  ret i1 %result
}})NVVM",
      function);
    output_.blank();
  }

  /**
   * @brief emits the helper that locates the character immediately before a byte position
   */
  void emit_previous_position()
  {
    auto function = name("previous_position");
    if (ir_.options.characters == character_mode::BYTES) {
      output_.emit(
        R"NVVM(define internal i64 @{}(i8* %data, i64 %size, i64 %target) alwaysinline nounwind readonly {{
entry:
  %at_begin = icmp eq i64 %target, 0
  %previous = sub i64 %target, 1
  %result = select i1 %at_begin, i64 0, i64 %previous
  ret i64 %result
}})NVVM",
        function);
      output_.blank();
      return;
    }

    auto load_byte = name("load_byte");
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i64 @{0}(i8* %data, i64 %size, i64 %target) alwaysinline nounwind readonly {{
entry:
  %at_begin = icmp eq i64 %target, 0
  br i1 %at_begin, label %zero, label %start
zero:
  ret i64 0
start:
  %initial = sub i64 %target, 1
  br label %loop
loop:
  %cursor = phi i64 [ %initial, %start ], [ %previous, %continue ]
  %pointer = getelementptr i8, i8* %data, i64 %cursor
  %byte = call i32 @{1}(i8* %pointer)
  %prefix = and i32 %byte, 192
  %continuation = icmp eq i32 %prefix, 128
  %at_zero = icmp eq i64 %cursor, 0
  %leading = xor i1 %continuation, true
  %done = or i1 %leading, %at_zero
  br i1 %done, label %found, label %continue
continue:
  %previous = sub i64 %cursor, 1
  br label %loop
found:
  ret i64 %cursor
}})NVVM",
        function,
        load_byte));
    output_.blank();
  }

  /**
   * @brief emits the dispatcher for begin, end, word-boundary, and non-boundary assertions
   */
  void emit_assertion()
  {
    auto function = name("assertion");
    auto previous = name("previous_position");
    auto decode   = name("decode_codepoint");
    auto is_word  = name("is_word");
    auto advance  = name("advance");
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{3}(i8* %data, i64 %size, i64 %pos, i32 %kind) inlinehint nounwind readonly {{
entry:
  switch i32 %kind, label %not_boundary [
    i32 0, label %begin_input
    i32 1, label %end_input
    i32 2, label %boundary
    i32 4, label %begin_line
    i32 5, label %end_line
  ]
begin_input:
  %is_begin_input = icmp eq i64 %pos, 0
  ret i1 %is_begin_input
end_input:
  %is_end_input = icmp eq i64 %pos, %size
  ret i1 %is_end_input
begin_line:
  %is_begin_line_input = icmp eq i64 %pos, 0
  br i1 %is_begin_line_input, label %true, label %begin_line_nonzero
begin_line_nonzero:
  br i1 {5}, label %begin_line_previous, label %false
begin_line_previous:
  %begin_prev_pos = call i64 @{2}(i8* %data, i64 %size, i64 %pos)
  %begin_prev = call i32 @{0}(i8* %data, i64 %size, i64 %begin_prev_pos)
  %begin_prev_lf = icmp eq i32 %begin_prev, 10
  %begin_prev_cr = icmp eq i32 %begin_prev, 13
  %begin_prev_nel = icmp eq i32 %begin_prev, 133
  %begin_prev_ls = icmp eq i32 %begin_prev, 8232
  %begin_prev_ps = icmp eq i32 %begin_prev, 8233
  %begin_prev_crlf = or i1 %begin_prev_lf, %begin_prev_cr
  %begin_prev_extended_0 = or i1 %begin_prev_crlf, %begin_prev_nel
  %begin_prev_extended_1 = or i1 %begin_prev_ls, %begin_prev_ps
  %begin_prev_extended = or i1 %begin_prev_extended_0, %begin_prev_extended_1
  %begin_prev_newline = select i1 {6}, i1 %begin_prev_extended, i1 %begin_prev_lf
  %begin_current = call i32 @{0}(i8* %data, i64 %size, i64 %pos)
  %begin_current_lf = icmp eq i32 %begin_current, 10
  %begin_mid_crlf_0 = and i1 %begin_prev_cr, %begin_current_lf
  %begin_mid_crlf = and i1 {6}, %begin_mid_crlf_0
  %begin_not_mid_crlf = xor i1 %begin_mid_crlf, true
  %begin_line_result = and i1 %begin_prev_newline, %begin_not_mid_crlf
  ret i1 %begin_line_result
end_line:
  %is_end_line_input = icmp eq i64 %pos, %size
  br i1 %is_end_line_input, label %true, label %end_line_current
end_line_current:
  %end_cp = call i32 @{0}(i8* %data, i64 %size, i64 %pos)
  %end_lf = icmp eq i32 %end_cp, 10
  %end_cr = icmp eq i32 %end_cp, 13
  %end_nel = icmp eq i32 %end_cp, 133
  %end_ls = icmp eq i32 %end_cp, 8232
  %end_ps = icmp eq i32 %end_cp, 8233
  %end_crlf = or i1 %end_lf, %end_cr
  %end_extended_0 = or i1 %end_crlf, %end_nel
  %end_extended_1 = or i1 %end_ls, %end_ps
  %end_extended = or i1 %end_extended_0, %end_extended_1
  %end_newline = select i1 {6}, i1 %end_extended, i1 %end_lf
  br i1 %end_newline, label %end_line_newline, label %false
end_line_newline:
  %end_prev_pos = call i64 @{2}(i8* %data, i64 %size, i64 %pos)
  %end_prev = call i32 @{0}(i8* %data, i64 %size, i64 %end_prev_pos)
  %end_prev_cr = icmp eq i32 %end_prev, 13
  %end_mid_crlf_0 = and i1 %end_prev_cr, %end_lf
  %end_mid_crlf = and i1 {6}, %end_mid_crlf_0
  br i1 %end_mid_crlf, label %false, label %end_line_not_mid
end_line_not_mid:
  br i1 {5}, label %true, label %end_line_final
end_line_final:
  %end_next = call i64 @{4}(i8* %data, i64 %size, i64 %pos, i64 1)
  %end_is_final = icmp eq i64 %end_next, %size
  br i1 %end_is_final, label %true, label %end_line_possible_crlf
end_line_possible_crlf:
  %end_can_be_crlf = and i1 {6}, %end_cr
  br i1 %end_can_be_crlf, label %end_line_after_cr, label %false
end_line_after_cr:
  %end_next_cp = call i32 @{0}(i8* %data, i64 %size, i64 %end_next)
  %end_next_lf = icmp eq i32 %end_next_cp, 10
  %end_after_lf = call i64 @{4}(i8* %data, i64 %size, i64 %end_next, i64 1)
  %end_lf_is_final = icmp eq i64 %end_after_lf, %size
  %end_final_crlf = and i1 %end_next_lf, %end_lf_is_final
  ret i1 %end_final_crlf
boundary:
  %current_in_bounds = icmp ult i64 %pos, %size
  br i1 %current_in_bounds, label %current_decode, label %current_missing
current_decode:
  %current_cp = call i32 @{0}(i8* %data, i64 %size, i64 %pos)
  %current_word_value = call i1 @{1}(i32 %current_cp)
  br label %current_join
current_missing:
  br label %current_join
current_join:
  %current_word = phi i1 [ %current_word_value, %current_decode ], [ false, %current_missing ]
  %has_previous = icmp ugt i64 %pos, 0
  br i1 %has_previous, label %previous_decode, label %previous_missing
previous_decode:
  %previous_pos = call i64 @{2}(i8* %data, i64 %size, i64 %pos)
  %previous_cp = call i32 @{0}(i8* %data, i64 %size, i64 %previous_pos)
  %previous_word_value = call i1 @{1}(i32 %previous_cp)
  br label %previous_join
previous_missing:
  br label %previous_join
previous_join:
  %previous_word = phi i1 [ %previous_word_value, %previous_decode ], [ false, %previous_missing ]
  %is_boundary = xor i1 %previous_word, %current_word
  ret i1 %is_boundary
not_boundary:
  %boundary_value = call i1 @{3}(i8* %data, i64 %size, i64 %pos, i32 2)
  %not_boundary_value = xor i1 %boundary_value, true
  ret i1 %not_boundary_value
true:
  ret i1 true
false:
  ret i1 false
}})NVVM",
        decode,
        is_word,
        previous,
        function,
        advance,
        ir_.options.multiline ? "true" : "false",
        ir_.options.extended_newline ? "true" : "false"));
    output_.blank();
  }

  /**
   * @brief emits a range-test helper for every character-predicate instruction
   */
  void emit_predicate_helpers()
  {
    for (auto& block : ir_.blocks) {
      for (auto& instruction : block.instructions) {
        auto* match = std::get_if<match_character>(&instruction);
        if (match == nullptr) continue;
        auto function = name(std::format("predicate_{}", block.id));
        auto decode   = name("decode_codepoint");
        output_.emit(
          "{}",
          std::format(
            R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %cp = call i32 @{1}(i8* %data, i64 %size, i64 %pos))NVVM",
            function,
            decode));
        if (match->predicate.recognized == predicate_class::ANY) {
          if (match->predicate.matches_newline) {
            output_.emit("  ret i1 true");
          } else if (match->predicate.extended_newline) {
            output_.emit(
              R"NVVM(  %not_lf = icmp ne i32 %cp, 10
  %not_cr = icmp ne i32 %cp, 13
  %not_nel = icmp ne i32 %cp, 133
  %not_ls = icmp ne i32 %cp, 8232
  %not_ps = icmp ne i32 %cp, 8233
  %not_crlf = and i1 %not_lf, %not_cr
  %not_extended_0 = and i1 %not_nel, %not_ls
  %not_extended_1 = and i1 %not_extended_0, %not_ps
  %result = and i1 %not_crlf, %not_extended_1
  ret i1 %result)NVVM");
          } else {
            output_.emit(
              R"NVVM(  %not_lf = icmp ne i32 %cp, 10
  ret i1 %not_lf)NVVM");
          }
        } else {
          std::vector<std::string> comparisons;
          for (std::size_t index = 0; index < match->predicate.ranges.size(); ++index) {
            auto range = match->predicate.ranges[index];
            if (range.first == range.last) {
              auto comparison = std::format("equal_{}", index);
              output_.emit(
                "  %{} = icmp eq i32 %cp, {}", comparison, static_cast<std::uint32_t>(range.first));
              comparisons.push_back(comparison);
            } else {
              auto ge     = std::format("ge_{}", index);
              auto le     = std::format("le_{}", index);
              auto inside = std::format("inside_{}", index);
              output_.emit("{}",
                           std::format(R"NVVM(  %{0} = icmp uge i32 %cp, {3}
  %{1} = icmp ule i32 %cp, {4}
  %{2} = and i1 %{0}, %{1})NVVM",
                                       ge,
                                       le,
                                       inside,
                                       static_cast<std::uint32_t>(range.first),
                                       static_cast<std::uint32_t>(range.last)));
              comparisons.push_back(inside);
            }
          }
          if (comparisons.empty()) {
            output_.emit("  ret i1 {}", match->predicate.negated ? "true" : "false");
          } else {
            auto combined = comparisons.front();
            for (std::size_t index = 1; index < comparisons.size(); ++index) {
              auto next = std::format("combined_{}", index);
              output_.emit("  %{} = or i1 %{}, %{}", next, combined, comparisons[index]);
              combined = next;
            }
            if (match->predicate.negated) {
              output_.emit(
                R"NVVM(  %negated = xor i1 %{}, true
  ret i1 %negated)NVVM",
                combined);
            } else {
              output_.emit("  ret i1 %{}", combined);
            }
          }
        }
        output_.emit("}}");
        output_.blank();
      }
    }
  }

  /**
   * @brief emits a code-point comparison helper for every literal instruction
   */
  void emit_literal_helpers()
  {
    auto can_peek  = name("can_peek");
    auto decode    = name("decode_codepoint");
    auto width     = name("decode_width");
    auto load_byte = name("load_byte");
    for (auto& block : ir_.blocks) {
      for (auto& instruction : block.instructions) {
        auto* literal = std::get_if<match_literal>(&instruction);
        if (literal == nullptr) continue;
        auto function = name(std::format("literal_{}", block.id));
        auto ascii    = !literal->value.empty() &&
                     std::all_of(literal->value.begin(), literal->value.end(), [](char32_t value) {
                       return value <= 0x7f;
                     });
        if (ascii) {
          output_.emit(
            "{}",
            std::format(
              R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %in_bounds = icmp ule i64 %pos, %size
  %remaining = sub i64 %size, %pos
  %enough = icmp uge i64 %remaining, {1}
  %available = and i1 %in_bounds, %enough
  br i1 %available, label %check_0, label %fail)NVVM",
              function,
              literal->value.size()));
          for (std::size_t index = 0; index < literal->value.size(); ++index) {
            auto success =
              index + 1 == literal->value.size() ? "success" : std::format("check_{}", index + 1);
            output_.emit("{}",
                         std::format(R"NVVM(check_{0}:
  %byte_pos_{0} = add i64 %pos, {0}
  %byte_ptr_{0} = getelementptr i8, i8* %data, i64 %byte_pos_{0}
  %byte_{0} = call i32 @{1}(i8* %byte_ptr_{0})
  %matches_{0} = icmp eq i32 %byte_{0}, {2}
  br i1 %matches_{0}, label %{3}, label %fail)NVVM",
                                     index,
                                     load_byte,
                                     static_cast<std::uint32_t>(literal->value[index]),
                                     success));
          }
          output_.emit(
            R"NVVM(success:
  ret i1 true
fail:
  ret i1 false
}})NVVM");
          output_.blank();
          continue;
        }

        output_.emit(
          "{}",
          std::format(
            R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %available = call i1 @{1}(i8* %data, i64 %size, i64 %pos, i64 {2})
  br i1 %available, label %check_0, label %fail)NVVM",
            function,
            can_peek,
            literal->value.size()));
        for (std::size_t index = 0; index < literal->value.size(); ++index) {
          auto position = index == 0 ? "%pos" : std::format("%pos_{}", index);
          output_.emit("{}",
                       std::format(R"NVVM(check_{0}:
  %cp_{0} = call i32 @{1}(i8* %data, i64 %size, i64 {2})
  %matches_{0} = icmp eq i32 %cp_{0}, {3})NVVM",
                                   index,
                                   decode,
                                   position,
                                   static_cast<std::uint32_t>(literal->value[index])));
          if (index + 1 == literal->value.size()) {
            output_.emit("  br i1 %matches_{}, label %success, label %fail", index);
          } else {
            output_.emit("{}",
                         std::format(
                           R"NVVM(  %width_{0} = call i64 @{2}(i8* %data, i64 %size, i64 {3})
  %pos_{1} = add i64 {3}, %width_{0}
  br i1 %matches_{0}, label %check_{1}, label %fail)NVVM",
                           index,
                           index + 1,
                           width,
                           position));
          }
        }
        output_.emit(
          R"NVVM(success:
  ret i1 true
fail:
  ret i1 false
}})NVVM");
        output_.blank();
      }
    }
  }

  /**
   * @brief emits constant byte arrays used by a specialized replacement executor
   */
  void emit_replacement_globals()
  {
    if (ir_.control.result != result_shape::REPLACEMENT) return;
    for (std::size_t index = 0; index < ir_.replacement.size(); ++index) {
      auto& token = ir_.replacement[index];
      if (token.type != replacement_token::kind::LITERAL || token.literal.empty()) continue;
      output_.emit(R"NVVM(@{} = internal addrspace(4) constant [{} x i8] c"{}", align 1)NVVM",
                   name(std::format("replacement_{}", index)),
                   token.literal.size(),
                   format_byte_array(token.literal));
    }
    output_.blank();
  }

  /**
   * @brief emits the recursive dispatcher that executes the instruction block graph
   */
  void emit_blocks()
  {
    // one recursive dispatcher represents cyclic block graphs without forward declarations
    auto function = name("run_block");
    std::string cases;
    for (auto& block : ir_.blocks) {
      std::format_to(
        std::back_inserter(cases), "    i32 {}, label %b{}_op_0\n", block.id, block.id);
    }
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i32 %block, i8* %data, i64 %size, i64* %position, i64* %captures, i64 %steps) nounwind {{
entry:
  %exhausted = icmp eq i64 %steps, 0
  %next_steps = sub i64 %steps, 1
  br i1 %exhausted, label %return_fail, label %dispatch
dispatch:
  switch i32 %block, label %return_fail [
{1}  ])NVVM",
        function,
        cases));
    for (auto& block : ir_.blocks)
      emit_block(block);
    output_.emit(
      R"NVVM(return_success:
  ret i1 true
return_fail:
  ret i1 false
}})NVVM");
    output_.blank();
  }

  /**
   * @brief emits one block's instructions and prioritized successor attempts
   *
   * @param block instruction block to lower into NVVM IR
   */
  void emit_block(instruction_block const& block)
  {
    auto prefix                 = std::format("b{}", block.id);
    std::size_t operation_index = 0;
    bool returned               = false;
    for (auto& item : block.instructions) {
      output_.emit("{}_op_{}:", prefix, operation_index);
      auto next = std::format("{}_op_{}", prefix, operation_index + 1);
      if (auto* peek = std::get_if<can_peek>(&item)) {
        auto* next_literal =
          operation_index + 1 < block.instructions.size()
            ? std::get_if<match_literal>(&block.instructions[operation_index + 1])
            : nullptr;
        auto fused_ascii_literal =
          next_literal != nullptr && next_literal->value.size() == peek->characters &&
          std::all_of(next_literal->value.begin(), next_literal->value.end(), [](char32_t value) {
            return value <= 0x7f;
          });
        if (fused_ascii_literal) {
          // the ASCII literal helper performs one byte-count bounds check for the fused sequence
          output_.emit("  br label %{}", next);
        } else {
          output_.emit("{}",
                       std::format(R"NVVM(  %{0}_pos_{1} = load i64, i64* %position, align 8
  %{0}_peek_{1} = call i1 @{2}(i8* %data, i64 %size, i64 %{0}_pos_{1}, i64 {3})
  br i1 %{0}_peek_{1}, label %{4}, label %return_fail)NVVM",
                                   prefix,
                                   operation_index,
                                   name("can_peek"),
                                   peek->characters,
                                   next));
        }
      } else if (std::holds_alternative<read_character>(item)) {
        output_.emit("  br label %{}", next);
      } else if (auto* capture = std::get_if<write_capture>(&item)) {
        auto slot = static_cast<std::size_t>(capture->capture_index) * 2U +
                    (capture->action == capture_action::END ? 1U : 0U);
        if (std::find(capture_slots_.begin(), capture_slots_.end(), slot) == capture_slots_.end()) {
          output_.emit("  br label %{}", next);
          ++operation_index;
          continue;
        }
        auto reset = std::string{};
        if (capture->action == capture_action::BEGIN) {
          reset = std::format(
            R"NVVM(  %{0}_capture_end_ptr_{1} = getelementptr i64, i64* %captures, i64 {2}
  store i64 -1, i64* %{0}_capture_end_ptr_{1}, align 8
)NVVM",
            prefix,
            operation_index,
            slot + 1U);
        }
        output_.emit("{}",
                     std::format(
                       R"NVVM(  %{0}_capture_pos_{1} = load i64, i64* %position, align 8
  %{0}_capture_ptr_{1} = getelementptr i64, i64* %captures, i64 {2}
  store i64 %{0}_capture_pos_{1}, i64* %{0}_capture_ptr_{1}, align 8
{3}  br label %{4})NVVM",
                       prefix,
                       operation_index,
                       slot,
                       reset,
                       next));
      } else if (std::holds_alternative<match_character>(item)) {
        output_.emit("{}",
                     std::format(R"NVVM(  %{0}_match_pos_{1} = load i64, i64* %position, align 8
  %{0}_match_{1} = call i1 @{2}(i8* %data, i64 %size, i64 %{0}_match_pos_{1})
  br i1 %{0}_match_{1}, label %{3}, label %return_fail)NVVM",
                                 prefix,
                                 operation_index,
                                 name(std::format("predicate_{}", block.id)),
                                 next));
      } else if (std::holds_alternative<match_literal>(item)) {
        output_.emit("{}",
                     std::format(R"NVVM(  %{0}_literal_pos_{1} = load i64, i64* %position, align 8
  %{0}_literal_{1} = call i1 @{2}(i8* %data, i64 %size, i64 %{0}_literal_pos_{1})
  br i1 %{0}_literal_{1}, label %{3}, label %return_fail)NVVM",
                                 prefix,
                                 operation_index,
                                 name(std::format("literal_{}", block.id)),
                                 next));
      } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
        auto* previous_literal =
          operation_index > 0 ? std::get_if<match_literal>(&block.instructions[operation_index - 1])
                              : nullptr;
        auto matched_ascii_literal = previous_literal != nullptr &&
                                     previous_literal->value.size() == advance->characters &&
                                     std::all_of(previous_literal->value.begin(),
                                                 previous_literal->value.end(),
                                                 [](char32_t value) { return value <= 0x7f; });
        if (matched_ascii_literal) {
          output_.emit("{}",
                       std::format(R"NVVM(  %{0}_advance_pos_{1} = load i64, i64* %position, align 8
  %{0}_advanced_{1} = add i64 %{0}_advance_pos_{1}, {2}
  store i64 %{0}_advanced_{1}, i64* %position, align 8
  br label %{3})NVVM",
                                   prefix,
                                   operation_index,
                                   advance->characters,
                                   next));
        } else {
          output_.emit("{}",
                       std::format(
                         R"NVVM(  %{0}_advance_pos_{1} = load i64, i64* %position, align 8
  %{0}_advanced_{1} = call i64 @{2}(i8* %data, i64 %size, i64 %{0}_advance_pos_{1}, i64 {3})
  store i64 %{0}_advanced_{1}, i64* %position, align 8
  br label %{4})NVVM",
                         prefix,
                         operation_index,
                         name("advance"),
                         advance->characters,
                         next));
        }
      } else if (auto* assertion = std::get_if<test_assertion>(&item)) {
        output_.emit("{}",
                     std::format(R"NVVM(  %{0}_assert_pos_{1} = load i64, i64* %position, align 8
  %{0}_assert_{1} = call i1 @{2}(i8* %data, i64 %size, i64 %{0}_assert_pos_{1}, i32 {3})
  br i1 %{0}_assert_{1}, label %{4}, label %return_fail)NVVM",
                                 prefix,
                                 operation_index,
                                 name("assertion"),
                                 static_cast<std::uint32_t>(assertion->kind),
                                 next));
      } else if (std::holds_alternative<emit_accept>(item)) {
        if (ir_.control.require_end) {
          output_.emit(
            R"NVVM(  %{}_accept_pos = load i64, i64* %position, align 8
  %{}_accept = icmp eq i64 %{}_accept_pos, %size
  ret i1 %{}_accept)NVVM",
            prefix,
            prefix,
            prefix,
            prefix);
        } else {
          output_.emit("  ret i1 true");
        }
        returned = true;
      }
      ++operation_index;
    }

    if (returned) return;
    output_.emit("{}_op_{}:", prefix, operation_index);
    auto edges = block.successors;
    std::stable_sort(edges.begin(), edges.end(), [](auto& left, auto& right) {
      return left.priority < right.priority;
    });
    if (edges.empty()) {
      output_.emit("  br label %return_fail");
      return;
    }
    output_.emit("  %{}_saved = load i64, i64* %position, align 8", prefix);
    for (auto slot : capture_slots_) {
      output_.emit("{}",
                   std::format(
                     R"NVVM(  %{0}_saved_capture_{1} = getelementptr i64, i64* %captures, i64 {1}
  %{0}_saved_capture_value_{1} = load i64, i64* %{0}_saved_capture_{1}, align 8)NVVM",
                     prefix,
                     slot));
    }
    output_.emit("  br label %{}_attempt_0", prefix);
    for (std::size_t index = 0; index < edges.size(); ++index) {
      auto failure = index + 1 < edges.size() ? std::format("{}_attempt_{}", prefix, index + 1)
                                              : std::string{"return_fail"};
      output_.emit("{}",
                   std::format(
                     R"NVVM({0}_attempt_{1}:
  %{0}_child_{1} = call i1 @{2}(i32 {3}, i8* %data, i64 %size, i64* %position, i64* %captures, i64 %next_steps)
  br i1 %{0}_child_{1}, label %return_success, label %{0}_restore_{1}
{0}_restore_{1}:
  store i64 %{0}_saved, i64* %position, align 8)NVVM",
                     prefix,
                     index,
                     name("run_block"),
                     edges[index].target,
                     failure));
      for (auto slot : capture_slots_) {
        output_.emit("  store i64 %{}_saved_capture_value_{}, i64* %{}_saved_capture_{}, align 8",
                     prefix,
                     slot,
                     prefix,
                     slot);
      }
      output_.emit("  br label %{}", failure);
    }
  }

  /**
   * @brief emits the externally callable anchored or scanning regex execution function
   */
  void emit_execute()
  {
    auto run_block  = name("run_block");
    auto advance    = name("advance");
    auto multiplier = static_cast<std::uint64_t>(ir_.blocks.size()) * 8U + 32U;
    output_.emit(
      "{}",
      std::format(R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %position = alloca i64, align 8
  %size_plus_one = add i64 %size, 1
  %step_limit = mul i64 %size_plus_one, {1})NVVM",
                  options_.execute_function,
                  multiplier));
    auto mandatory_filter =
      mandatory_ascii_literal_.has_value() && ir_.control.scan_input && !begins_at_input_start();
    if (mandatory_filter) {
      output_.emit("  %mandatory_present = call i1 @{}(i8* %data, i64 %size, i64 0)",
                   name("mandatory_literal_present"));
      output_.emit("  br i1 %mandatory_present, label %mandatory_filter_done, label %no");
      output_.emit("mandatory_filter_done:");
    }
    auto initial_predecessor = mandatory_filter ? "%mandatory_filter_done" : "%entry";
    if (!ir_.control.scan_input || begins_at_input_start()) {
      output_.emit("{}",
                   std::format(R"NVVM(  store i64 0, i64* %position, align 8
  %matched = call i1 @{0}(i32 {1}, i8* %data, i64 %size, i64* %position, i64* null, i64 %step_limit)
  ret i1 %matched
}})NVVM",
                               run_block,
                               ir_.entry));
      output_.blank();
      return;
    }

    auto prefix = prefix_seek_byte_;
    if (prefix.has_value()) {
      auto hint =
        std::string{R"NVVM(  %candidate_likely = call i1 @llvm.expect.i1(i1 %candidate, i1 false)
)NVVM"};
      output_.emit("{}",
                   std::format(
                     R"NVVM(  br label %search
search:
  %start = phi i64 [ 0, {7} ], [ %next_start, %next ]
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %attempt, label %filter
filter:
  %start_ptr = getelementptr i8, i8* %data, i64 %start
  %start_byte = call i32 @{0}(i8* %start_ptr)
  %candidate = icmp eq i32 %start_byte, {1}
{2}  br i1 {3}, label %attempt, label %skip
skip:
  %is_ascii = icmp ult i32 %start_byte, 128
  br i1 %is_ascii, label %continue_ascii, label %continue_utf8
attempt:
  store i64 %start, i64* %position, align 8
  %matched = call i1 @{4}(i32 {5}, i8* %data, i64 %size, i64* %position, i64* null, i64 %step_limit)
  br i1 %matched, label %yes, label %check_end
check_end:
  br i1 %at_end, label %no, label %continue_ascii
continue_ascii:
  %prefix_seek_start = add nuw i64 %start, 1
  %ascii_next = call i64 @{6}(i8* %data, i64 %size, i64 %prefix_seek_start)
  br label %next
continue_utf8:
  br label %continue_ascii
next:
  %next_start = phi i64 [ %ascii_next, %continue_ascii ]
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
                     name("load_byte"),
                     static_cast<std::uint32_t>(*prefix),
                     hint,
                     "%candidate_likely",
                     run_block,
                     ir_.entry,
                     name("seek_prefix_byte"),
                     initial_predecessor));
      output_.blank();
      return;
    }

    output_.emit("{}",
                 std::format(R"NVVM(  br label %search
search:
  %start = phi i64 [ 0, {3} ], [ %next_start, %continue ]
  store i64 %start, i64* %position, align 8
  %matched = call i1 @{0}(i32 {1}, i8* %data, i64 %size, i64* %position, i64* null, i64 %step_limit)
  br i1 %matched, label %yes, label %check_end
check_end:
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %no, label %continue
continue:
  %next_start = call i64 @{2}(i8* %data, i64 %size, i64 %start, i64 1)
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
                             run_block,
                             ir_.entry,
                             advance,
                             initial_predecessor));
    output_.blank();
  }

  /**
   * @brief emits capture stores selected by a deterministic transition index
   *
   * @param prefix unique label and temporary prefix
   * @param programs capture-update program for each deterministic transition
   * @param position SSA value recorded by every capture update
   * @param continuation block reached after applying a program
   */
  void emit_capture_action_dispatch(
    std::string_view prefix,
    std::vector<std::vector<deterministic_capture_action>> const& programs,
    std::string_view position,
    std::string_view continuation)
  {
    std::vector<std::size_t> active;
    for (std::size_t index = 0; index < programs.size(); ++index) {
      if (!programs[index].empty()) active.push_back(index);
    }
    if (active.empty()) {
      output_.emit("  br label %{}", continuation);
      return;
    }

    std::string source;
    std::format_to(
      std::back_inserter(source), "  switch i32 %transition_index, label %{}_done [\n", prefix);
    for (auto index : active) {
      std::format_to(
        std::back_inserter(source), "    i32 {}, label %{}_{}\n", index, prefix, index);
    }
    std::format_to(std::back_inserter(source), "  ]\n");
    for (auto index : active) {
      std::format_to(std::back_inserter(source), "{}_{}:\n", prefix, index);
      for (std::size_t action_index = 0; action_index < programs[index].size(); ++action_index) {
        auto action = programs[index][action_index];
        std::format_to(std::back_inserter(source),
                       R"NVVM(  %{0}_{1}_{2}_ptr = getelementptr i64, i64* %captures, i64 {3}
  store i64 {4}, i64* %{0}_{1}_{2}_ptr, align 8
)NVVM",
                       prefix,
                       index,
                       action_index,
                       action.slot,
                       position);
        if (action.reset_end) {
          std::format_to(std::back_inserter(source),
                         R"NVVM(  %{0}_{1}_{2}_end_ptr = getelementptr i64, i64* %captures, i64 {3}
  store i64 -1, i64* %{0}_{1}_{2}_end_ptr, align 8
)NVVM",
                         prefix,
                         index,
                         action_index,
                         action.slot + 1U);
        }
      }
      std::format_to(std::back_inserter(source), "  br label %{}_done\n", prefix);
    }
    std::format_to(std::back_inserter(source), "{}_done:\n  br label %{}", prefix, continuation);
    output_.emit("{}", source);
  }

  std::string render_start_byte_filter(deterministic_machine const& machine,
                                       std::string_view candidate_target,
                                       bool direct_advance)
  {
    if (!machine.start_byte_filter) return {};

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    for (std::uint32_t byte = 0; byte < 128U;) {
      auto is_candidate =
        (machine.start_byte_bitmap[byte / 64U] & (std::uint64_t{1} << (byte % 64U))) != 0U;
      if (!is_candidate) {
        ++byte;
        continue;
      }
      auto first = byte;
      while (byte + 1U < 128U && (machine.start_byte_bitmap[(byte + 1U) / 64U] &
                                  (std::uint64_t{1} << ((byte + 1U) % 64U))) != 0U) {
        ++byte;
      }
      ranges.emplace_back(first, byte);
      ++byte;
    }

    std::string checks;
    std::string predicate = "false";
    for (std::size_t index = 0; index < ranges.size(); ++index) {
      auto [first, last] = ranges[index];
      auto range_value   = std::format("%start_filter_range_{}", index);
      if (first == last) {
        std::format_to(std::back_inserter(checks),
                       "  {} = icmp eq i32 %start_filter_byte, {}\n",
                       range_value,
                       first);
      } else {
        std::format_to(std::back_inserter(checks),
                       R"NVVM(  %start_filter_offset_{} = sub i32 %start_filter_byte, {}
  {} = icmp ule i32 %start_filter_offset_{}, {}
)NVVM",
                       index,
                       first,
                       range_value,
                       index,
                       last - first);
      }
      if (index == 0U) {
        predicate = range_value;
      } else {
        auto any_value = std::format("%start_filter_any_{}", index);
        std::format_to(
          std::back_inserter(checks), "  {} = or i1 {}, {}\n", any_value, predicate, range_value);
        predicate = std::move(any_value);
      }
    }

    auto miss_target  = direct_advance ? "start_filter_advance" : "advance_start";
    auto direct_block = !direct_advance                 ? std::string{}
                        : prefix_seek_byte_.has_value() ? std::format(R"NVVM(start_filter_advance:
  %start_filter_seek_start = add nuw i64 %start, 1
  %start_filter_next = call i64 @{}(i8* %data, i64 %size, i64 %start_filter_seek_start)
  br label %search
)NVVM",
                                                                      name("seek_prefix_byte"))
                                                        : std::string{R"NVVM(start_filter_advance:
  %start_filter_next = add nuw i64 %start, 1
  br label %search
)NVVM"};
    return std::format(
      R"NVVM(start_filter:
  %filter_at_end = icmp eq i64 %start, %size
  br i1 %filter_at_end, label %{0}, label %start_filter_load
start_filter_load:
  %start_filter_ptr = getelementptr i8, i8* %data, i64 %start
  %start_filter_byte = call i32 @{1}(i8* %start_filter_ptr)
  %start_filter_ascii = icmp ult i32 %start_filter_byte, 128
  br i1 %start_filter_ascii, label %start_filter_ascii_byte, label %{0}
start_filter_ascii_byte:
{2}  br i1 {3}, label %{0}, label %{4}
{5}
)NVVM",
      candidate_target,
      name("load_byte"),
      checks,
      predicate,
      miss_target,
      direct_block);
  }

  /**
   * @brief emits a direct tagged-DFA matcher for capture-safe deterministic automata
   *
   * @param machine deterministic machine and transition capture programs to execute
   */
  void emit_tagged_deterministic_find_from(deterministic_machine const& machine)
  {
    auto search_target = machine.start_byte_filter ? "start_filter" : "initialize";
    auto start_filter  = render_start_byte_filter(machine, "initialize", false);
    auto restart       = machine.restart_state <= machine.state_mask
                           ? std::format(
                         R"NVVM(  %restart_state_match = icmp eq i32 %state, {0}
  %restart_consumed = icmp ugt i64 %position, %start
  %restart_before_end = icmp ult i64 %position, %size
  %restart_prefix_candidate = and i1 %restart_state_match, %restart_consumed
  %restart_prefix = and i1 %restart_prefix_candidate, %restart_before_end
  %restart_base = select i1 %restart_prefix, i64 %position, i64 %start
)NVVM",
                         machine.restart_state)
                           : std::string{};
    auto advance_phi =
      machine.restart_state <= machine.state_mask && machine.start_byte_filter
        ? std::
            string{R"NVVM(  %restart_advance_base = phi i64 [ %restart_base, %candidate_fail ], [ %start, %start_filter_ascii_byte ]
)NVVM"}
        : std::string{};
    auto advance_base      = machine.restart_state <= machine.state_mask
                               ? (machine.start_byte_filter ? "%restart_advance_base" : "%restart_base")
                               : "%start";
    auto advance_candidate = prefix_seek_byte_.has_value()
                               ? std::format(
                                   "  %prefix_seek_start = add nuw i64 {}, 1\n"
                                   "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 "
                                   "%prefix_seek_start)",
                                   advance_base,
                                   name("seek_prefix_byte"))
                               : std::format(
                                   "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 {}, "
                                   "i64 1)",
                                   name("advance"),
                                   advance_base);
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind {{
entry:
  br label %search
search:
  %start = phi i64 [ %search_start, %entry ], [ %next_start, %advance_start ]
  %in_range = icmp ule i64 %start, %size
  br i1 %in_range, label %{1}, label %no
{2}
initialize:)NVVM",
        name("find_from"),
        search_target,
        start_filter));
    for (auto slot : capture_slots_) {
      output_.emit(
        "{}",
        std::format(R"NVVM(  %capture_{0}_ptr = getelementptr i64, i64* %captures, i64 {0}
  store i64 -1, i64* %capture_{0}_ptr, align 8)NVVM",
                    slot));
    }
    output_.emit("{}",
                 std::format(
                   R"NVVM(  store i64 %start, i64* %match_begin, align 8
  br label %loop
loop:
  %position = phi i64 [ %start, %initialize ], [ %next_position, %continue ]
  %state = phi i32 [ {0}, %initialize ], [ %next_state, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %candidate_fail, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{1}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
ascii:
  %ascii_class = call i32 @{2}(i32 %first)
  br label %transition
unicode:
  %codepoint = call i32 @{3}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{4}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{2}(i32 %codepoint)
  br label %transition
transition:
  %character_class = phi i32 [ %ascii_class, %ascii ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, %ascii ], [ %unicode_width, %unicode ]
  %state_index = and i32 %state, 16383
  %state_offset = mul nuw i32 %state_index, {5}
  %transition_index = add nuw i32 %state_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{6} x i16], [{6} x i16] addrspace({7})* @{8}, i64 0, i64 %transition_index_i64
  %encoded_i16 = load i16, i16 addrspace({7})* %transition_ptr, align 2
  %encoded = zext i16 %encoded_i16 to i32
  %next_state = and i32 %encoded, 16383
  %dead = icmp eq i32 %next_state, {9}
  br i1 %dead, label %candidate_fail, label %capture_transition
capture_transition:)NVVM",
                   machine.initial_state,
                   name("load_byte"),
                   name("dfa_classify"),
                   name("decode_codepoint"),
                   name("decode_width"),
                   machine.class_count,
                   machine.transitions.size(),
                   machine.transition_address_space,
                   name("dfa_transitions"),
                   machine.dead_state));
    emit_capture_action_dispatch(
      "transition_capture", machine.transition_capture_actions, "%position", "consume");
    output_.emit(
      R"NVVM(consume:
  %next_position = add i64 %position, %character_width
  %accept_bits = and i32 %encoded, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  br i1 %accepted, label %capture_accept, label %continue
continue:
  br label %loop
capture_accept:)NVVM");
    emit_capture_action_dispatch(
      "accept_capture", machine.accept_capture_actions, "%next_position", "yes");
    output_.emit("{}",
                 std::format(
                   R"NVVM(candidate_fail:
{0}  %at_input_end = icmp eq i64 %start, %size
  br i1 %at_input_end, label %no, label %advance_start
advance_start:
{1}{2}
  br label %search
yes:
  store i64 %next_position, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
                   restart,
                   advance_phi,
                   advance_candidate));
    output_.blank();
  }

  /**
   * @brief emits a packed fixed-width comparison at one byte position
   *
   * @param literal non-empty ASCII literal to compare
   */
  void emit_ascii_literal_at(std::string_view literal)
  {
    std::string comparisons;
    std::string matched;
    std::size_t offset = 0;
    std::size_t index  = 0;
    while (offset < literal.size()) {
      auto remaining      = literal.size() - offset;
      auto width          = remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
      std::uint64_t value = 0;
      for (std::size_t byte = 0; byte < width; ++byte) {
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal[offset + byte]))
                 << (byte * 8U);
      }
      auto bits = width * 8U;
      std::format_to(
        std::back_inserter(comparisons),
        "  %literal_byte_ptr_{} = getelementptr i8, i8* %data, i64 %literal_offset_{}\n",
        index,
        index);
      if (width == 1U) {
        std::format_to(std::back_inserter(comparisons),
                       "  %literal_chunk_{} = load i8, i8* %literal_byte_ptr_{}, align 1\n",
                       index,
                       index);
      } else {
        std::format_to(std::back_inserter(comparisons),
                       R"NVVM(  %literal_chunk_ptr_{} = bitcast i8* %literal_byte_ptr_{} to i{}*
  %literal_chunk_{} = load i{}, i{}* %literal_chunk_ptr_{}, align 1
)NVVM",
                       index,
                       index,
                       bits,
                       index,
                       bits,
                       bits,
                       index);
      }
      std::format_to(std::back_inserter(comparisons),
                     "  %literal_equal_{} = icmp eq i{} %literal_chunk_{}, {}\n",
                     index,
                     bits,
                     index,
                     value);
      if (matched.empty()) {
        matched = std::format("%literal_equal_{}", index);
      } else {
        std::format_to(std::back_inserter(comparisons),
                       "  %literal_equal_through_{} = and i1 {}, %literal_equal_{}\n",
                       index,
                       matched,
                       index);
        matched = std::format("%literal_equal_through_{}", index);
      }
      offset += width;
      ++index;
    }

    std::string offsets;
    offset = 0;
    for (std::size_t chunk = 0; chunk < index; ++chunk) {
      std::format_to(std::back_inserter(offsets),
                     "  %literal_offset_{} = add i64 %position, {}\n",
                     chunk,
                     offset);
      auto remaining = literal.size() - offset;
      offset += remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
    }

    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %position) alwaysinline nounwind readonly {{
entry:
  %position_valid = icmp ule i64 %position, %size
  %remaining = sub i64 %size, %position
  %enough_bytes = icmp uge i64 %remaining, {1}
  %in_bounds = and i1 %position_valid, %enough_bytes
  br i1 %in_bounds, label %compare, label %no
compare:
{2}{3}  ret i1 {4}
no:
  ret i1 false
}})NVVM",
        name("ascii_literal_at"),
        literal.size(),
        offsets,
        comparisons,
        matched));
    output_.blank();
  }

  /**
   * @brief Emits a row-level search for a mandatory ASCII literal.
   *
   * @param literal Mandatory literal already emitted through `emit_ascii_literal_at`
   */
  void emit_mandatory_literal_filter(std::string_view literal)
  {
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start) alwaysinline nounwind readonly {{
entry:
  br label %search
search:
  %position = phi i64 [ %search_start, %entry ], [ %next_position, %continue ]
  %candidate_end = add i64 %position, {1}
  %in_range = icmp ule i64 %candidate_end, %size
  br i1 %in_range, label %load, label %no
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{2}(i8* %input_ptr)
  %candidate = icmp eq i32 %first, {3}
  br i1 %candidate, label %verify, label %continue
verify:
  %matched = call i1 @{4}(i8* %data, i64 %size, i64 %position)
  br i1 %matched, label %yes, label %continue
continue:
  %next_position = add nuw i64 %position, 1
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("mandatory_literal_present"),
        literal.size(),
        name("load_byte"),
        static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
        name("ascii_literal_at")));
    output_.blank();
  }

  /**
   * @brief emits a direct boolean executor for an exact ASCII literal
   *
   * @param literal non-empty ASCII literal to search or match
   */
  void emit_ascii_literal_execute(std::string_view literal)
  {
    emit_ascii_literal_at(literal);
    if (!ir_.control.scan_input) {
      auto required_size =
        ir_.control.require_end
          ? std::format("  %required_size = icmp eq i64 %size, {}\n", literal.size())
          : std::string{"  %required_size = icmp uge i64 %size, 0\n"};
      output_.emit("{}",
                   std::format(
                     R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
{1}  %literal_match = call i1 @{2}(i8* %data, i64 %size, i64 0)
  %matched = and i1 %required_size, %literal_match
  ret i1 %matched
}})NVVM",
                     options_.execute_function,
                     required_size,
                     name("ascii_literal_at")));
      output_.blank();
      return;
    }

    if (literal.size() >= 8U) {
      // one has-zero mask tests eight possible first bytes before packed verification.
      std::uint64_t repeated_first = 0;
      for (std::size_t byte = 0; byte < 8U; ++byte) {
        repeated_first |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal.front()))
                          << (byte * 8U);
      }
      output_.emit("{}",
                   std::format(
                     R"NVVM(declare i64 @llvm.cttz.i64(i64, i1)

define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %search
search:
  %position = phi i64 [ 0, %entry ], [ %next_chunk, %chunk_continue ]
  %candidate_end = add i64 %position, {1}
  %in_range = icmp ule i64 %candidate_end, %size
  br i1 %in_range, label %load_chunk, label %no
load_chunk:
  %chunk_byte_ptr = getelementptr i8, i8* %data, i64 %position
  %chunk_ptr = bitcast i8* %chunk_byte_ptr to i64*
  %chunk = load i64, i64* %chunk_ptr, align 1
  %candidate_x = xor i64 %chunk, {2}
  %candidate_minus_ones = sub i64 %candidate_x, 72340172838076673
  %candidate_not = xor i64 %candidate_x, -1
  %candidate_zero_bytes = and i64 %candidate_minus_ones, %candidate_not
  %candidate_mask = and i64 %candidate_zero_bytes, -9187201950435737472
  %has_candidate = icmp ne i64 %candidate_mask, 0
  br i1 %has_candidate, label %candidate, label %chunk_continue
candidate:
  %remaining_candidates = phi i64 [ %candidate_mask, %load_chunk ], [ %next_candidates, %verify_continue ]
  %candidate_bit = call i64 @llvm.cttz.i64(i64 %remaining_candidates, i1 false)
  %candidate_byte = lshr i64 %candidate_bit, 3
  %candidate_position = add i64 %position, %candidate_byte
  %matched = call i1 @{3}(i8* %data, i64 %size, i64 %candidate_position)
  br i1 %matched, label %yes, label %verify_continue
verify_continue:
  %candidate_mask_minus_one = sub i64 %remaining_candidates, 1
  %next_candidates = and i64 %remaining_candidates, %candidate_mask_minus_one
  %has_more_candidates = icmp ne i64 %next_candidates, 0
  br i1 %has_more_candidates, label %candidate, label %chunk_continue
chunk_continue:
  %next_chunk = add nuw i64 %position, 8
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
                     options_.execute_function,
                     literal.size(),
                     repeated_first,
                     name("ascii_literal_at")));
      output_.blank();
      return;
    }

    auto verify = literal.size() == 1U
                    ? std::string{"  br i1 %candidate, label %yes, label %continue\n"}
                    : std::format(
                        R"NVVM(  br i1 %candidate, label %verify, label %continue
verify:
  %matched = call i1 @{0}(i8* %data, i64 %size, i64 %position)
  br i1 %matched, label %yes, label %continue
)NVVM",
                        name("ascii_literal_at"));
    output_.emit("{}",
                 std::format(
                   R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %search
search:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]
  %candidate_end = add i64 %position, {1}
  %in_range = icmp ule i64 %candidate_end, %size
  br i1 %in_range, label %load, label %no
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{2}(i8* %input_ptr)
  %candidate = icmp eq i32 %first, {3}
{4}continue:
  %next_position = add nuw i64 %position, 1
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
                   options_.execute_function,
                   literal.size(),
                   name("load_byte"),
                   static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
                   verify));
    output_.blank();
  }

  /**
   * @brief emits a selective-pivot seeker with packed verification for an exact UTF-8 literal
   *
   * @param literal Exact UTF-8 literal plan
   * @param pivot Offset of a selective ASCII byte within the encoded literal
   */
  void emit_utf8_pivot_find_from(utf8_literal const& literal,
                                 std::size_t pivot,
                                 std::string_view function)
  {
    auto encoded      = encode_utf8_literal(literal.codepoints);
    auto guard_offset = std::min(pivot, encoded.size() - sizeof(std::uint64_t));
    auto guard_value  = std::uint64_t{0};
    for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
      guard_value |=
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(encoded[guard_offset + byte]))
        << (byte * 8U);
    }
    emit_ascii_literal_at(encoded);
    emit_prefix_byte_seek(static_cast<std::uint8_t>(encoded[pivot]));
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind readonly {{
entry:
  %initial_pivot = add i64 %search_start, {1}
  br label %search
search:
  %pivot_start = phi i64 [ %initial_pivot, %entry ], [ %next_pivot, %mismatch ]
  %pivot_position = call i64 @{2}(i8* %data, i64 %size, i64 %pivot_start)
  %pivot_found = icmp ult i64 %pivot_position, %size
  br i1 %pivot_found, label %candidate, label %no
candidate:
  %begin = sub nuw i64 %pivot_position, {1}
  %end = add i64 %begin, {3}
  %in_range = icmp ule i64 %end, %size
  br i1 %in_range, label %guard, label %no
guard:
  %guard_position = add i64 %begin, {5}
  %guard_byte_ptr = getelementptr i8, i8* %data, i64 %guard_position
  %guard_ptr = bitcast i8* %guard_byte_ptr to i64*
  %guard_value = load i64, i64* %guard_ptr, align 1
  %guard_matches = icmp eq i64 %guard_value, {6}
  br i1 %guard_matches, label %verify, label %mismatch
verify:
  %matched = call i1 @{4}(i8* %data, i64 %size, i64 %begin)
  br i1 %matched, label %yes, label %mismatch
mismatch:
  %next_pivot = add nuw i64 %pivot_position, 1
  br label %search
yes:
  store i64 %begin, i64* %match_begin, align 8
  store i64 %end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        function,
        pivot,
        name("seek_prefix_byte"),
        encoded.size(),
        name("ascii_literal_at"),
        guard_offset,
        llvm_i64(guard_value)));
    output_.blank();
  }

  void emit_utf8_literal_find_from(utf8_literal const& literal)
  {
    if (!utf8_literal_pivot_.has_value()) {
      emit_utf8_kmp_find_from(literal, name("find_from"));
      return;
    }
    if (ir_.control.result == result_shape::MATCH_COUNT) {
      emit_utf8_pivot_find_from(literal, *utf8_literal_pivot_, name("find_from"));
      return;
    }

    auto pivot = name("utf8_pivot_find_from");
    auto kmp   = name("utf8_kmp_find_from");
    emit_utf8_pivot_find_from(literal, *utf8_literal_pivot_, pivot);
    emit_utf8_kmp_find_from(literal, kmp);
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind readonly {{
entry:
  %long_row = icmp uge i64 %size, 256
  br i1 %long_row, label %kmp, label %pivot
pivot:
  %pivot_result = call i1 @{1}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures)
  ret i1 %pivot_result
kmp:
  %kmp_result = call i1 @{2}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures)
  ret i1 %kmp_result
}})NVVM",
        name("find_from"),
        pivot,
        kmp));
    output_.blank();
  }

  /**
   * @brief Emits a byte-domain KMP seeker for an exact UTF-8 literal.
   *
   * Every non-ASCII literal begins with a UTF-8 lead byte, so a byte-domain match cannot begin at a
   * continuation byte. Comparing the complete encoded sequence therefore preserves code-point
   * matching while eliminating decoder work. Input positions and returned spans use byte offsets.
   *
   * @param literal Exact UTF-8 literal plan
   */
  void emit_utf8_kmp_find_from(utf8_literal const& literal, std::string_view function)
  {
    auto encoded = encode_utf8_literal(literal.codepoints);
    auto bytes   = std::vector<std::uint8_t>(encoded.begin(), encoded.end());
    auto failure = std::vector<std::uint32_t>(bytes.size());
    for (std::size_t index = 1, prefix = 0; index < bytes.size();) {
      if (bytes[index] == bytes[prefix]) {
        failure[index++] = static_cast<std::uint32_t>(++prefix);
      } else if (prefix != 0U) {
        prefix = failure[prefix - 1U];
      } else {
        failure[index++] = 0U;
      }
    }
    output_.emit("@{} = internal addrspace(4) constant [{} x i8] [{}], align 1",
                 name("utf8_kmp_literal"),
                 bytes.size(),
                 format_i8_array(bytes));
    output_.emit("@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                 name("utf8_kmp_failure"),
                 failure.size(),
                 format_i32_array(failure));
    output_.blank();
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind {{
entry:
  br label %search
search:
  %position = phi i64 [ %search_start, %entry ], [ %position, %fallback ], [ %next_position, %matched_continue ], [ %mismatch_next, %mismatch_zero ]
  %matched = phi i32 [ 0, %entry ], [ %fallback_matched, %fallback ], [ %next_matched, %matched_continue ], [ 0, %mismatch_zero ]
  %in_range = icmp ult i64 %position, %size
  br i1 %in_range, label %compare, label %no
compare:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %input = call i32 @{1}(i8* %input_ptr)
  %literal_index = zext i32 %matched to i64
  %literal_ptr = getelementptr [{2} x i8], [{2} x i8] addrspace(4)* @{3}, i64 0, i64 %literal_index
  %expected_i8 = load i8, i8 addrspace(4)* %literal_ptr, align 1
  %expected = zext i8 %expected_i8 to i32
  %equal = icmp eq i32 %input, %expected
  br i1 %equal, label %matched_byte, label %mismatch
matched_byte:
  %next_position = add nuw i64 %position, 1
  %next_matched = add nuw i32 %matched, 1
  %complete = icmp eq i32 %next_matched, {2}
  br i1 %complete, label %yes, label %matched_continue
matched_continue:
  br label %search
mismatch:
  %has_prefix = icmp ne i32 %matched, 0
  br i1 %has_prefix, label %fallback, label %mismatch_zero
fallback:
  %fallback_index = sub nuw i32 %matched, 1
  %fallback_index_i64 = zext i32 %fallback_index to i64
  %failure_ptr = getelementptr [{2} x i32], [{2} x i32] addrspace(4)* @{4}, i64 0, i64 %fallback_index_i64
  %fallback_matched = load i32, i32 addrspace(4)* %failure_ptr, align 4
  br label %search
mismatch_zero:
  %mismatch_next = add nuw i64 %position, 1
  br label %search
yes:
  %begin = sub nuw i64 %next_position, {2}
  store i64 %begin, i64* %match_begin, align 8
  store i64 %next_position, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        function,
        name("load_byte"),
        bytes.size(),
        name("utf8_kmp_literal"),
        name("utf8_kmp_failure")));
    output_.blank();
  }

  /**
   * @brief Emits the boolean executor ABI around the UTF-8 KMP seeker.
   */
  void emit_utf8_kmp_execute()
  {
    output_.emit("{}",
                 std::format(
                   R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 0, i64* %match_begin, i64* %match_end, i64* null)
  ret i1 %matched
}})NVVM",
                   options_.execute_function,
                   name("find_from")));
    output_.blank();
  }

  /**
   * @brief emits a packed ASCII-literal finder for span-producing operations
   *
   * @param literal non-empty multi-byte ASCII literal to search
   */
  void emit_ascii_literal_find_from(std::string_view literal)
  {
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind {{
entry:
  br label %search
search:
  %position = phi i64 [ %search_start, %entry ], [ %next_position, %continue ]
  %candidate_end = add i64 %position, {1}
  %in_range = icmp ule i64 %candidate_end, %size
  br i1 %in_range, label %load, label %no
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{2}(i8* %input_ptr)
  %candidate = icmp eq i32 %first, {3}
  br i1 %candidate, label %verify, label %continue
verify:
  %matched = call i1 @{4}(i8* %data, i64 %size, i64 %position)
  br i1 %matched, label %yes, label %continue
continue:
  %next_position = add nuw i64 %position, 1
  br label %search
yes:
  store i64 %position, i64* %match_begin, align 8
  store i64 %candidate_end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("find_from"),
        literal.size(),
        name("load_byte"),
        static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
        name("ascii_literal_at")));
    output_.blank();
  }

  /**
   * @brief emits a direct byte-search primitive for an exact ASCII character
   *
   * @param byte ASCII byte that forms the complete regex
   */
  void emit_single_byte_find_from(std::uint8_t byte)
  {
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind {{
entry:
  br label %search
search:
  %position = phi i64 [ %search_start, %entry ], [ %next_position, %continue ]
  %in_range = icmp ult i64 %position, %size
  br i1 %in_range, label %load, label %no
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %value = call i32 @{1}(i8* %input_ptr)
  %matched = icmp eq i32 %value, {2}
  br i1 %matched, label %yes, label %continue
continue:
  %next_position = add nuw i64 %position, 1
  br label %search
yes:
  %end = add nuw i64 %position, 1
  store i64 %position, i64* %match_begin, align 8
  store i64 %end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("find_from"),
        name("load_byte"),
        static_cast<std::uint32_t>(byte)));
    output_.blank();
  }

  void emit_assertion_deterministic_find_from(deterministic_machine const& machine)
  {
    auto search_target     = machine.start_byte_filter ? "start_filter" : "attempt";
    auto start_filter      = render_start_byte_filter(machine, "attempt", false);
    auto advance_candidate = prefix_seek_byte_.has_value()
                               ? std::format(
                                   "  %prefix_seek_start = add nuw i64 %start, 1\n"
                                   "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 "
                                   "%prefix_seek_start)",
                                   name("seek_prefix_byte"))
                               : std::format(
                                   "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 "
                                   "%start, i64 1)",
                                   name("advance"));
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %start, i64* %match_end) nounwind readonly {{
entry:
  %at_input_begin = icmp eq i64 %start, 0
  br i1 %at_input_begin, label %begin, label %load_previous
load_previous:
  %previous_position = call i64 @{1}(i8* %data, i64 %size, i64 %start)
  %decoded_previous = call i32 @{2}(i8* %data, i64 %size, i64 %previous_position)
  br label %begin
begin:
  %initial_previous = phi i32 [ 0, %entry ], [ %decoded_previous, %load_previous ]
  br label %loop
loop:
  %position = phi i64 [ %start, %begin ], [ %next_position, %continue ]
  %state = phi i32 [ {3}, %begin ], [ %next_state, %continue ]
  %previous_cp = phi i32 [ %initial_previous, %begin ], [ %current_cp, %continue ]
  %last_accept = phi i64 [ -1, %begin ], [ %next_accept, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %finish, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{4}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
ascii:
  %ascii_class = call i32 @{5}(i32 %first)
  br label %boundary
unicode:
  %codepoint = call i32 @{2}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{6}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{5}(i32 %codepoint)
  br label %boundary
boundary:
  %current_cp = phi i32 [ %first, %ascii ], [ %codepoint, %unicode ]
  %character_class = phi i32 [ %ascii_class, %ascii ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, %ascii ], [ %unicode_width, %unicode ]
  %boundary_class = call i32 @{7}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 %current_cp, i64 %character_width)
  %state_offset = mul nuw i32 %state, {8}
  %context_index = add nuw i32 %state_offset, %boundary_class
  %context_offset = mul nuw i32 %context_index, {9}
  %transition_index = add nuw i32 %context_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{10} x i16], [{10} x i16] addrspace({11})* @{12}, i64 0, i64 %transition_index_i64
  %encoded_i16 = load i16, i16 addrspace({11})* %transition_ptr, align 2
  %encoded = zext i16 %encoded_i16 to i32
  %accept_bits = and i32 %encoded, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  %next_accept = select i1 %accepted, i64 %position, i64 %last_accept
  %next_state = and i32 %encoded, 32767
  %next_position = add i64 %position, %character_width
  %dead = icmp eq i32 %next_state, {13}
  br i1 %dead, label %candidate_done, label %continue
continue:
  br label %loop
candidate_done:
  %candidate_matched = icmp ne i64 %next_accept, -1
  br i1 %candidate_matched, label %candidate_yes, label %no
finish:
  %finish_boundary_class = call i32 @{7}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 0, i64 0)
  %finish_state_offset = mul nuw i32 %state, {8}
  %finish_accept_index = add nuw i32 %finish_state_offset, %finish_boundary_class
  %finish_accept_index_i64 = zext i32 %finish_accept_index to i64
  %finish_accept_ptr = getelementptr [{14} x i8], [{14} x i8] addrspace(4)* @{15}, i64 0, i64 %finish_accept_index_i64
  %finish_accept_i8 = load i8, i8 addrspace(4)* %finish_accept_ptr, align 1
  %finish_accepted = icmp ne i8 %finish_accept_i8, 0
  %finish_end = select i1 %finish_accepted, i64 %size, i64 %last_accept
  %finish_matched = icmp ne i64 %finish_end, -1
  br i1 %finish_matched, label %finish_yes, label %no
candidate_yes:
  store i64 %next_accept, i64* %match_end, align 8
  ret i1 true
finish_yes:
  store i64 %finish_end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("assertion_candidate"),
        name("previous_position"),
        name("decode_codepoint"),
        machine.initial_state,
        name("load_byte"),
        name("dfa_classify"),
        name("decode_width"),
        name("dfa_boundary_classify"),
        machine.boundary_class_count,
        machine.class_count,
        machine.transitions.size(),
        machine.transition_address_space,
        name("dfa_transitions"),
        machine.dead_state,
        machine.boundary_accepts.size(),
        name("dfa_boundary_accepts")));
    output_.blank();

    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind readonly {{
entry:
  br label %search
search:
  %start = phi i64 [ %search_start, %entry ], [ %next_start, %advance_start ]
  %in_range = icmp ule i64 %start, %size
  br i1 %in_range, label %{3}, label %no
{4}
attempt:
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 %start, i64* %match_end)
  br i1 %matched, label %yes, label %check_end
check_end:
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %no, label %advance_start
advance_start:
{2}
  br label %search
yes:
  store i64 %start, i64* %match_begin, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("find_from"),
        name("assertion_candidate"),
        advance_candidate,
        search_target,
        start_filter));
    output_.blank();
  }

  /**
   * @brief emits a non-recursive leftmost-match primitive for an ordered deterministic automaton
   *
   * @param machine deterministic machine whose transitions are known to preserve priority
   */
  void emit_deterministic_find_from(deterministic_machine const& machine)
  {
    auto initial_accept = (machine.initial_state & 0x8000U) != 0 ? "%start" : "-1";
    // the direct edge wins for count but disrupts larger materializing control-flow graphs.
    auto direct_filter_advance = ir_.control.result == result_shape::MATCH_COUNT;
    auto search_target         = machine.start_byte_filter ? "start_filter" : "candidate";
    auto start_filter      = render_start_byte_filter(machine, "candidate", direct_filter_advance);
    auto filter_search_phi = machine.start_byte_filter && direct_filter_advance
                               ? ", [ %start_filter_next, %start_filter_advance ]"
                               : "";
    auto restart           = machine.restart_state <= machine.state_mask
                               ? std::format(
                         R"NVVM(  %restart_state_match = icmp eq i32 %state, {0}
  %restart_consumed = icmp ugt i64 %position, %start
  %restart_before_end = icmp ult i64 %position, %size
  %restart_prefix_candidate = and i1 %restart_state_match, %restart_consumed
  %restart_prefix = and i1 %restart_prefix_candidate, %restart_before_end
  %restart_base = select i1 %restart_prefix, i64 %position, i64 %start
)NVVM",
                         machine.restart_state)
                               : std::string{};
    auto advance_phi =
      machine.restart_state <= machine.state_mask && machine.start_byte_filter &&
          !direct_filter_advance
        ? std::
            string{R"NVVM(  %restart_advance_base = phi i64 [ %restart_base, %candidate_fail ], [ %start, %start_filter_ascii_byte ]
)NVVM"}
        : std::string{};
    auto advance_base =
      machine.restart_state <= machine.state_mask
        ? (machine.start_byte_filter && !direct_filter_advance ? "%restart_advance_base"
                                                               : "%restart_base")
        : "%start";
    auto advance_candidate = prefix_seek_byte_.has_value()
                               ? std::format(
                                   "  %prefix_seek_start = add nuw i64 {}, 1\n"
                                   "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 "
                                   "%prefix_seek_start)",
                                   advance_base,
                                   name("seek_prefix_byte"))
                               : std::format(
                                   "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 {}, "
                                   "i64 1)",
                                   name("advance"),
                                   advance_base);
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind {{
entry:
  br label %search
search:
  %start = phi i64 [ %search_start, %entry ], [ %next_start, %advance_start ]{3}
  %in_range = icmp ule i64 %start, %size
  br i1 %in_range, label %{1}, label %no
{2}
candidate:
  br label %loop
loop:
  %position = phi i64 [ %start, %candidate ], [ %next_position, %continue ]
  %state = phi i32 [ {4}, %candidate ], [ %next_state, %continue ]
  %last_accept = phi i64 [ {5}, %candidate ], [ %next_accept, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %candidate_done, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{6}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
ascii:
  %ascii_class = call i32 @{7}(i32 %first)
  br label %transition
unicode:
  %codepoint = call i32 @{8}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{9}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{7}(i32 %codepoint)
  br label %transition
transition:
  %character_class = phi i32 [ %ascii_class, %ascii ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, %ascii ], [ %unicode_width, %unicode ]
  %state_index = and i32 %state, 16383
  %state_offset = mul nuw i32 %state_index, {10}
  %transition_index = add nuw i32 %state_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{11} x i16], [{11} x i16] addrspace({12})* @{13}, i64 0, i64 %transition_index_i64
  %encoded_i16 = load i16, i16 addrspace({12})* %transition_ptr, align 2
  %encoded = zext i16 %encoded_i16 to i32
  %stop_bits = and i32 %encoded, 16384
  %stop_before = icmp ne i32 %stop_bits, 0
  br i1 %stop_before, label %candidate_done, label %check_transition
check_transition:
  %next_state = and i32 %encoded, 16383
  %dead = icmp eq i32 %next_state, {14}
  br i1 %dead, label %candidate_done, label %consume
consume:
  %next_position = add i64 %position, %character_width
  %accept_bits = and i32 %encoded, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  %next_accept = select i1 %accepted, i64 %next_position, i64 %last_accept
  br label %continue
continue:
  br label %loop
candidate_done:
  %accepted_end = phi i64 [ %last_accept, %loop ], [ %last_accept, %transition ], [ %last_accept, %check_transition ]
  %matched = icmp ne i64 %accepted_end, -1
  br i1 %matched, label %yes, label %candidate_fail
candidate_fail:
{15}  %at_input_end = icmp eq i64 %start, %size
  br i1 %at_input_end, label %no, label %advance_start
advance_start:
{16}{17}
  br label %search
yes:
  store i64 %start, i64* %match_begin, align 8
  store i64 %accepted_end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        name("find_from"),
        search_target,
        start_filter,
        filter_search_phi,
        machine.initial_state,
        initial_accept,
        name("load_byte"),
        name("dfa_classify"),
        name("decode_codepoint"),
        name("decode_width"),
        machine.class_count,
        machine.transitions.size(),
        machine.transition_address_space,
        name("dfa_transitions"),
        machine.dead_state,
        restart,
        advance_phi,
        advance_candidate));
    output_.blank();
  }

  /**
   * @brief emits the shared leftmost-match primitive used by materializing operations
   */
  void emit_find_from()
  {
    auto function         = name("find_from");
    auto run_block        = name("run_block");
    auto advance          = name("advance");
    auto multiplier       = static_cast<std::uint64_t>(ir_.blocks.size()) * 8U + 32U;
    auto mandatory_filter = mandatory_ascii_literal_.has_value()
                              ? std::format(
                                  R"NVVM(  %mandatory_at_begin = icmp eq i64 %search_start, 0
  br i1 %mandatory_at_begin, label %mandatory_filter, label %mandatory_filter_done
mandatory_filter:
  %mandatory_present = call i1 @{}(i8* %data, i64 %size, i64 %search_start)
  br i1 %mandatory_present, label %mandatory_filter_done, label %no
mandatory_filter_done:
  br label %search
)NVVM",
                                  name("mandatory_literal_present"))
                              : std::string{"  br label %search\n"};
    auto initial_predecessor =
      mandatory_ascii_literal_.has_value() ? "%mandatory_filter_done" : "%entry";
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind {{
entry:
  %position = alloca i64, align 8
  %size_plus_one = add i64 %size, 1
  %step_limit = mul i64 %size_plus_one, {1}
{2}search:
  %start = phi i64 [ %search_start, {3} ], [ %next_start, %continue ]
  %in_range = icmp ule i64 %start, %size)NVVM",
        function,
        multiplier,
        mandatory_filter,
        initial_predecessor));

    auto prefix = required_ascii_prefix();
    if (prefix.has_value()) {
      auto hint = std::string{
        R"NVVM(  %prefix_likely = call i1 @llvm.expect.i1(i1 %prefix_candidate, i1 false)
)NVVM"};
      output_.emit("{}",
                   std::format(R"NVVM(  br i1 %in_range, label %prefix_end, label %no
prefix_end:
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %initialize, label %prefix_filter
prefix_filter:
  %prefix_ptr = getelementptr i8, i8* %data, i64 %start
  %prefix_byte = call i32 @{0}(i8* %prefix_ptr)
  %prefix_candidate = icmp eq i32 %prefix_byte, {1}
{2}  br i1 {3}, label %initialize, label %continue)NVVM",
                               name("load_byte"),
                               static_cast<std::uint32_t>(*prefix),
                               hint,
                               "%prefix_likely"));
    } else {
      output_.emit("  br i1 %in_range, label %initialize, label %no");
    }

    output_.emit("initialize:");
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit(
        R"NVVM(  %find_capture_ptr_0 = getelementptr i64, i64* %captures, i64 0
  %find_capture_ptr_1 = getelementptr i64, i64* %captures, i64 1)NVVM");
    }
    for (auto slot : capture_slots_) {
      output_.emit("{}",
                   std::format(
                     R"NVVM(  %find_capture_ptr_{0} = getelementptr i64, i64* %captures, i64 {0}
  store i64 -1, i64* %find_capture_ptr_{0}, align 8)NVVM",
                     slot));
    }
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit("  store i64 %start, i64* %find_capture_ptr_0, align 8");
    }
    auto captures = uses_capture_buffer() ? "%captures" : "null";
    output_.emit("{}",
                 std::format(R"NVVM(  store i64 %start, i64* %position, align 8
  %matched = call i1 @{0}(i32 {1}, i8* %data, i64 %size, i64* %position, i64* {2}, i64 %step_limit)
  br i1 %matched, label %yes, label %check_end
check_end:)NVVM",
                             run_block,
                             ir_.entry,
                             captures));
    if (prefix.has_value()) {
      output_.emit("  br i1 %at_end, label %no, label %continue");
    } else {
      output_.emit(R"NVVM(  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %no, label %continue)NVVM");
    }
    auto next_candidate = prefix_seek_byte_.has_value()
                            ? std::format(
                                "  %prefix_seek_start = add nuw i64 %start, 1\n"
                                "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 "
                                "%prefix_seek_start)",
                                name("seek_prefix_byte"))
                            : std::format(
                                "  %next_start = call i64 @{}(i8* %data, i64 %size, i64 %start, "
                                "i64 1)",
                                advance);
    output_.emit("{}",
                 std::format(R"NVVM(continue:
{0}
  br label %search
yes:
  %accepted_end = load i64, i64* %position, align 8
  store i64 %start, i64* %match_begin, align 8
  store i64 %accepted_end, i64* %match_end, align 8)NVVM",
                             next_candidate));
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit("  store i64 %accepted_end, i64* %find_capture_ptr_1, align 8");
    }
    output_.emit(R"NVVM(  ret i1 true
no:
  ret i1 false
}})NVVM");
    output_.blank();
  }

  /**
   * @brief adapts an input-anchored boolean executor to a cardinality or match-start ABI
   */
  void emit_anchored_boolean_adapter()
  {
    if (!anchored_boolean_result_.has_value()) return;
    if (*anchored_boolean_result_ == result_shape::MATCH_COUNT) {
      output_.emit(
        "{}",
        std::format(R"NVVM(define i64 @{0}(i8* %data, i64 %size) alwaysinline nounwind readonly {{
entry:
  %matched = call i1 @{1}(i8* %data, i64 %size)
  %count = zext i1 %matched to i64
  ret i64 %count
}}
)NVVM",
                    public_execute_function_,
                    options_.execute_function));
    } else {
      output_.emit(
        "{}",
        std::format(
          R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size, i64* %span) alwaysinline nounwind {{
entry:
  %matched = call i1 @{1}(i8* %data, i64 %size)
  br i1 %matched, label %store, label %done
store:
  store i64 0, i64* %span, align 8
  br label %done
done:
  ret i1 %matched
}}
)NVVM",
          public_execute_function_,
          options_.execute_function));
    }
    output_.blank();
  }

  /**
   * @brief emits the first-match span ABI for a find operation
   */
  void emit_find_execute()
  {
    output_.emit(
      "{}",
      std::format(R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size, i64* %span) nounwind {{
entry:
  %match_begin = getelementptr i64, i64* %span, i64 0
  %match_end = getelementptr i64, i64* %span, i64 1
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 0, i64* %match_begin, i64* %match_end, i64* null)
  ret i1 %matched
}})NVVM",
                  options_.execute_function,
                  name("find_from")));
    output_.blank();
  }

  /**
   * @brief emits the search-offset whole-match ABI used by findall wrappers
   */
  void emit_find_all_execute()
  {
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %span) nounwind {{
entry:
  %match_begin = getelementptr i64, i64* %span, i64 0
  %match_end = getelementptr i64, i64* %span, i64 1
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* null)
  ret i1 %matched
}})NVVM",
        options_.execute_function,
        name("find_from")));
    output_.blank();
  }

  /**
   * @brief emits the first-match capture ABI used by extract and enumeration wrappers
   */
  void emit_capture_execute()
  {
    auto aliases = std::string{};
    if (!whole_match_captures_.empty()) {
      aliases = R"NVVM(  br i1 %matched, label %alias_whole_match, label %done
alias_whole_match:
  %whole_begin = load i64, i64* %match_begin, align 8
  %whole_end = load i64, i64* %match_end, align 8
)NVVM";
      for (auto capture : whole_match_captures_) {
        auto slot = static_cast<std::size_t>(capture) * 2U;
        std::format_to(
          std::back_inserter(aliases),
          R"NVVM(  %whole_capture_begin_ptr_{0} = getelementptr i64, i64* %captures, i64 {1}
  %whole_capture_end_ptr_{0} = getelementptr i64, i64* %captures, i64 {2}
  store i64 %whole_begin, i64* %whole_capture_begin_ptr_{0}, align 8
  store i64 %whole_end, i64* %whole_capture_end_ptr_{0}, align 8
)NVVM",
          capture,
          slot,
          slot + 1U);
      }
      aliases += R"NVVM(  br label %done
done:
)NVVM";
    }
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define zeroext i1 @{0}(i8* %data, i64 %size, i64 %search_start, i64* %captures) nounwind {{
entry:
  %match_begin = getelementptr i64, i64* %captures, i64 0
  %match_end = getelementptr i64, i64* %captures, i64 1
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures)
{2}
  ret i1 %matched
}})NVVM",
        options_.execute_function,
        name("find_from"),
        aliases));
    output_.blank();
  }

  /**
   * @brief emits a complete non-overlapping match-count executor
   */
  void emit_count_execute()
  {
    output_.emit("{}",
                 std::format(R"NVVM(define i64 @{0}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %end_value, %nonempty ], [ %empty_next, %advance_empty ]
  %count = phi i64 [ 0, %entry ], [ %next_count, %nonempty ], [ %next_count, %advance_empty ]
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* null)
  br i1 %matched, label %found, label %done
found:
  %begin_value = load i64, i64* %match_begin, align 8
  %end_value = load i64, i64* %match_end, align 8
  %next_count = add i64 %count, 1
  %nonempty_match = icmp ne i64 %begin_value, %end_value
  br i1 %nonempty_match, label %nonempty, label %empty
nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %end_value, %size
  br i1 %empty_at_end, label %done_after_match, label %advance_empty
advance_empty:
  %empty_next = call i64 @{2}(i8* %data, i64 %size, i64 %end_value, i64 1)
  br label %loop
done_after_match:
  ret i64 %next_count
done:
  ret i64 %count
}})NVVM",
                             options_.execute_function,
                             name("find_from"),
                             name("advance")));
    output_.blank();
  }

  /**
   * @brief emits the range-copy primitive used by the replacement executor
   */
  void emit_append_range()
  {
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i64 @{0}(i8* %source, i64 %begin, i64 %end, i8* %output, i64 %cursor) alwaysinline nounwind {{
entry:
  %length = sub i64 %end, %begin
  %next_cursor = add i64 %cursor, %length
  %output_missing = icmp eq i8* %output, null
  %empty = icmp eq i64 %length, 0
  %skip = or i1 %output_missing, %empty
  br i1 %skip, label %done, label %copy
copy:
  %source_ptr = getelementptr i8, i8* %source, i64 %begin
  %output_ptr = getelementptr i8, i8* %output, i64 %cursor
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %output_ptr, i8* align 1 %source_ptr, i64 %length, i1 false)
  br label %done
done:
  ret i64 %next_cursor
}})NVVM",
        name("append_range")));
    output_.blank();
  }

  /**
   * @brief emits a sizing-and-materialization executor specialized for a replacement template
   */
  void emit_replace_execute()
  {
    emit_append_range();
    auto capture_argument    = std::string{"null"};
    auto capture_declaration = std::string{};
    if (uses_capture_buffer()) {
      auto slots          = static_cast<std::size_t>(ir_.capture_count + 1U) * 2U;
      capture_declaration = std::format(R"NVVM(  %capture_array = alloca [{0} x i64], align 8
  %captures = getelementptr [{0} x i64], [{0} x i64]* %capture_array, i64 0, i64 0)NVVM",
                                        slots);
      capture_argument    = "%captures";
    }
    output_.emit("{}",
                 std::format(
                   R"NVVM(define i64 @{0}(i8* %data, i64 %size, i8* %output) nounwind {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
{1}
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %advanced_start, %advance_empty ]
  %copied = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %match_end_value, %advance_empty ]
  %cursor = phi i64 [ 0, %entry ], [ %replacement_cursor, %nonempty ], [ %replacement_cursor, %advance_empty ]
  %matched = call i1 @{2}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* {3})
  br i1 %matched, label %found, label %no_match
found:
  %match_begin_value = load i64, i64* %match_begin, align 8
  %match_end_value = load i64, i64* %match_end, align 8
  %unmatched_cursor = call i64 @{4}(i8* %data, i64 %copied, i64 %match_begin_value, i8* %output, i64 %cursor))NVVM",
                   options_.execute_function,
                   capture_declaration,
                   name("find_from"),
                   capture_argument,
                   name("append_range")));
    std::string cursor = "%unmatched_cursor";
    for (std::size_t index = 0; index < ir_.replacement.size(); ++index) {
      auto& token = ir_.replacement[index];
      if (token.type == replacement_token::kind::LITERAL) {
        if (token.literal.empty()) continue;
        output_.emit(
          "{}",
          std::format(
            R"NVVM(  %replacement_constant_{0} = getelementptr [{1} x i8], [{1} x i8] addrspace(4)* @{2}, i64 0, i64 0
  %replacement_literal_{0} = addrspacecast i8 addrspace(4)* %replacement_constant_{0} to i8*
  %replacement_cursor_{0} = call i64 @{3}(i8* %replacement_literal_{0}, i64 0, i64 {1}, i8* %output, i64 {4}))NVVM",
            index,
            token.literal.size(),
            name(std::format("replacement_{}", index)),
            name("append_range"),
            cursor));
      } else if (token.capture_index == 0 || is_whole_match_capture(token.capture_index)) {
        output_.emit(
          "{}",
          std::format(
            R"NVVM(  %replacement_cursor_{0} = call i64 @{1}(i8* %data, i64 %match_begin_value, i64 %match_end_value, i8* %output, i64 {2}))NVVM",
            index,
            name("append_range"),
            cursor));
      } else {
        auto slot = static_cast<std::size_t>(token.capture_index) * 2U;
        output_.emit(
          "{}",
          std::format(
            R"NVVM(  %replacement_capture_begin_ptr_{0} = getelementptr i64, i64* %captures, i64 {1}
  %replacement_capture_end_ptr_{0} = getelementptr i64, i64* %captures, i64 {2}
  %replacement_capture_begin_{0} = load i64, i64* %replacement_capture_begin_ptr_{0}, align 8
  %replacement_capture_end_{0} = load i64, i64* %replacement_capture_end_ptr_{0}, align 8
  %replacement_capture_has_begin_{0} = icmp ne i64 %replacement_capture_begin_{0}, -1
  %replacement_capture_has_end_{0} = icmp ne i64 %replacement_capture_end_{0}, -1
  %replacement_capture_valid_{0} = and i1 %replacement_capture_has_begin_{0}, %replacement_capture_has_end_{0}
  %replacement_capture_safe_begin_{0} = select i1 %replacement_capture_valid_{0}, i64 %replacement_capture_begin_{0}, i64 0
  %replacement_capture_safe_end_{0} = select i1 %replacement_capture_valid_{0}, i64 %replacement_capture_end_{0}, i64 0
  %replacement_cursor_{0} = call i64 @{3}(i8* %data, i64 %replacement_capture_safe_begin_{0}, i64 %replacement_capture_safe_end_{0}, i8* %output, i64 {4}))NVVM",
            index,
            slot,
            slot + 1U,
            name("append_range"),
            cursor));
      }
      cursor = std::format("%replacement_cursor_{}", index);
    }
    output_.emit("{}",
                 std::format(R"NVVM(  %replacement_cursor = add i64 {0}, 0
  %nonempty_match = icmp ne i64 %match_begin_value, %match_end_value
  br i1 %nonempty_match, label %nonempty, label %empty
nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %match_end_value, %size
  br i1 %empty_at_end, label %empty_finish, label %advance_empty
advance_empty:
  %advanced_start = call i64 @{1}(i8* %data, i64 %size, i64 %match_end_value, i64 1)
  br label %loop
no_match:
  br label %finish
empty_finish:
  br label %finish
finish:
  %tail_begin = phi i64 [ %copied, %no_match ], [ %match_end_value, %empty_finish ]
  %tail_cursor = phi i64 [ %cursor, %no_match ], [ %replacement_cursor, %empty_finish ]
  %result_size = call i64 @{2}(i8* %data, i64 %tail_begin, i64 %size, i8* %output, i64 %tail_cursor)
  ret i64 %result_size
}})NVVM",
                             cursor,
                             name("advance"),
                             name("append_range")));
    output_.blank();
  }

  /**
   * @brief emits the optional span-store primitive used by split sizing and materialization
   */
  void emit_write_span()
  {
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal void @{0}(i64* %spans, i64 %index, i64 %begin, i64 %end, i64 %capacity, i8* %overflow) alwaysinline nounwind {{
entry:
  %unbounded = icmp slt i64 %capacity, 0
  %fits = icmp ult i64 %index, %capacity
  %allowed = or i1 %unbounded, %fits
  br i1 %allowed, label %check_missing, label %mark_overflow
mark_overflow:
  store i8 1, i8* %overflow, align 1
  br label %done
check_missing:
  %missing = icmp eq i64* %spans, null
  br i1 %missing, label %done, label %write
write:
  %base = shl i64 %index, 1
  %end_index = add i64 %base, 1
  %begin_ptr = getelementptr i64, i64* %spans, i64 %base
  %end_ptr = getelementptr i64, i64* %spans, i64 %end_index
  store i64 %begin, i64* %begin_ptr, align 8
  store i64 %end, i64* %end_ptr, align 8
  br label %done
done:
  ret void
}})NVVM",
        name("write_span")));
    output_.blank();
  }

  /**
   * @brief emits a sizing-and-span-materialization executor specialized for split
   */
  void emit_split_execute()
  {
    emit_write_span();
    output_.emit(
      "{}",
      std::format(
        R"NVVM(define internal i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* %spans, i64 %maxsplit, i64 %span_capacity, i8* %overflow) nounwind {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %advanced_start, %advance_empty ]
  %copied = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %match_end_value, %advance_empty ]
  %field_count = phi i64 [ 0, %entry ], [ %next_count, %nonempty ], [ %next_count, %advance_empty ]
  %matched = call i1 @{1}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* null)
  br i1 %matched, label %found, label %no_match
found:
  %match_begin_value = load i64, i64* %match_begin, align 8
  %match_end_value = load i64, i64* %match_end, align 8
  call void @{2}(i64* %spans, i64 %field_count, i64 %copied, i64 %match_begin_value, i64 %span_capacity, i8* %overflow)
  %next_count = add i64 %field_count, 1
  %limited = icmp sgt i64 %maxsplit, 0
  %at_limit = icmp eq i64 %next_count, %maxsplit
  %limit_reached = and i1 %limited, %at_limit
  br i1 %limit_reached, label %limited_finish, label %continue
continue:
  %nonempty_match = icmp ne i64 %match_begin_value, %match_end_value
  br i1 %nonempty_match, label %nonempty, label %empty
limited_finish:
  call void @{2}(i64* %spans, i64 %next_count, i64 %match_end_value, i64 %size, i64 %span_capacity, i8* %overflow)
  %limited_result_count = add i64 %next_count, 1
  ret i64 %limited_result_count
nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %match_end_value, %size
  br i1 %empty_at_end, label %empty_finish, label %advance_empty
advance_empty:
  %advanced_start = call i64 @{3}(i8* %data, i64 %size, i64 %match_end_value, i64 1)
  br label %loop
no_match:
  br label %finish
empty_finish:
  br label %finish
finish:
  %tail_begin = phi i64 [ %copied, %no_match ], [ %match_end_value, %empty_finish ]
  %tail_index = phi i64 [ %field_count, %no_match ], [ %next_count, %empty_finish ]
  call void @{2}(i64* %spans, i64 %tail_index, i64 %tail_begin, i64 %size, i64 %span_capacity, i8* %overflow)
  %result_count = add i64 %tail_index, 1
  ret i64 %result_count
}}

define i64 @{0}(i8* %data, i64 %size, i64* %spans) nounwind {{
entry:
  %result = call i64 @libregex_ir_split_execute_limited(i8* %data, i64 %size, i64* %spans, i64 -1, i64 -1, i8* null)
  ret i64 %result
}})NVVM",
        options_.execute_function,
        name("find_from"),
        name("write_span"),
        name("advance")));
    output_.blank();
  }

  instruction_ir ir_;
  nvvm_ir_codegen_options options_;
  std::string public_execute_function_;
  std::optional<result_shape> anchored_boolean_result_     = std::nullopt;
  std::optional<std::string> exact_ascii_literal_metadata_ = std::nullopt;
  std::optional<deterministic_machine> deterministic_      = std::nullopt;
  std::optional<glushkov_machine> glushkov_                = std::nullopt;
  std::optional<string_operation> string_operations_       = std::nullopt;
  std::optional<std::string> line_tail_literal_            = std::nullopt;
  std::optional<std::uint32_t> word_run_minimum_           = std::nullopt;
  std::optional<std::string> ascii_literal_                = std::nullopt;
  std::optional<utf8_literal> utf8_literal_                = std::nullopt;
  std::optional<std::size_t> utf8_literal_pivot_           = std::nullopt;
  std::optional<std::string> mandatory_ascii_literal_      = std::nullopt;
  std::optional<std::uint8_t> prefix_seek_byte_            = std::nullopt;
  std::vector<std::uint32_t> whole_match_captures_         = std::vector<std::uint32_t>{};
  std::vector<std::size_t> capture_slots_                  = std::vector<std::size_t>{};
  source_buffer output_                                    = source_buffer{};
};

std::string_view module_body(std::string_view module)
{
  auto begin = std::string_view::npos;
  for (auto marker :
       {std::string_view{"\n@"}, std::string_view{"\ndefine "}, std::string_view{"\ndeclare "}}) {
    auto candidate = module.find(marker);
    if (candidate != std::string_view::npos) begin = std::min(begin, candidate + 1U);
  }
  if (begin == std::string_view::npos) {
    throw std::invalid_argument("generated alternative module has no declarations");
  }
  auto end = module.find("\n!nvvmir.version", begin);
  return module.substr(begin, end == std::string_view::npos ? end : end - begin);
}

void append_module_body(std::string& output, std::string_view body)
{
  while (!body.empty()) {
    auto newline = body.find(static_cast<char>(10));
    auto length  = newline == std::string_view::npos ? body.size() : newline + 1U;
    auto line    = body.substr(0, length);
    if (!line.starts_with("declare ") || output.find(line) == std::string::npos) {
      output.append(line);
    }
    body.remove_prefix(length);
  }
}

std::string_view module_without_metadata(std::string_view module)
{
  auto end = module.find("\n!nvvmir.version");
  return module.substr(0, end);
}

std::optional<compile_result> render_large_boolean_alternation(
  instruction_ir const& ir, nvvm_ir_codegen_options const& options)
{
  if (ir.control.result != result_shape::BOOLEAN || ir.blocks.size() < 80U ||
      ir.entry >= ir.blocks.size()) {
    return std::nullopt;
  }
  auto& entry = ir.blocks[ir.entry];
  if (!entry.instructions.empty() || entry.successors.size() < 2U) return std::nullopt;

  std::vector<instruction_ir> alternatives;
  alternatives.reserve(entry.successors.size());
  for (auto edge : entry.successors) {
    auto branch    = ir;
    branch.entry   = edge.target;
    auto optimized = optimize(std::move(branch));
    if (!optimized) return std::nullopt;
    alternatives.push_back(std::move(*optimized.value));
  }

  std::string result;
  std::vector<std::string> functions;
  functions.reserve(alternatives.size());
  for (std::size_t index = 0; index < alternatives.size(); ++index) {
    auto branch_options = options;
    branch_options.symbol_prefix += std::format("_alternative_{}", index);
    branch_options.execute_function += std::format("_alternative_{}", index);
    functions.push_back(branch_options.execute_function);
    auto nested    = render_large_boolean_alternation(alternatives[index], branch_options);
    auto generated = nested.has_value()
                       ? std::move(*nested)
                       : nvvm_ir_renderer(alternatives[index], branch_options).render();
    if (index == 0) {
      result = module_without_metadata(generated.nvvm_ir);
    } else {
      result += '\n';
      append_module_body(result, module_body(generated.nvvm_ir));
    }
  }

  std::string branches;
  for (std::size_t index = 0; index < functions.size(); ++index) {
    auto label = index == 0 ? std::string{"entry"} : std::format("alternative_{}", index);
    branches += std::format(
      "{}:\n  %matched_{} = call i1 @{}(i8* %data, i64 %size)\n", label, index, functions[index]);
    if (index + 1U < functions.size()) {
      branches +=
        std::format("  br i1 %matched_{}, label %yes, label %alternative_{}\n", index, index + 1U);
    } else {
      branches += std::format("  ret i1 %matched_{}\n", index);
    }
  }
  result += std::format(
    R"NVVM(
define zeroext i1 @{}(i8* %data, i64 %size) nounwind readonly {{
{}yes:
  ret i1 true
}}
)NVVM",
    options.execute_function,
    branches);
  result += "\n!nvvmir.version = !{!0}\n!0 = !{i32 2, i32 0}\n";
  return compile_result{std::move(result),
                        ir.capture_count,
                        executor_kind::BOOLEAN_ALTERNATION,
                        0,
                        0,
                        std::nullopt};
}

}  // namespace

compile_result generate_nvvm_ir(instruction_ir const& ir, nvvm_ir_codegen_options const& options)
{
  if (auto alternatives = render_large_boolean_alternation(ir, options)) {
    return std::move(*alternatives);
  }
  return nvvm_ir_renderer(ir, options).render();
}

}  // namespace regex_ir

namespace regex_ir {

compile_result compile(std::string_view pattern,
                       operation_kind operation_kind_value,
                       std::optional<std::string> replacement,
                       compile_options const& options)
{
  switch (operation_kind_value) {
    case operation_kind::CONTAINS:
    case operation_kind::MATCHES:
    case operation_kind::COUNT:
    case operation_kind::EXTRACT:
    case operation_kind::FIND:
    case operation_kind::FIND_ALL:
    case operation_kind::SPLIT:
      if (replacement.has_value()) {
        throw std::invalid_argument("replacement is only valid for the REPLACE operation");
      }
      break;
    case operation_kind::REPLACE:
      if (!replacement.has_value()) {
        throw std::invalid_argument("replacement is required for the REPLACE operation");
      }
      break;
    default: throw std::invalid_argument("invalid regex operation");
  }

  auto compiled = compile_instruction_ir(
    pattern, operation{operation_kind_value, replacement.value_or("")}, options);
  if (!compiled) {
    if (compiled.diagnostics.empty()) { throw std::invalid_argument("regex compilation failed"); }
    auto& diagnostic = compiled.diagnostics.front();
    throw std::invalid_argument(std::format(
      "regex compilation failed at byte {}: {}", diagnostic.span.offset, diagnostic.message));
  }
  return generate_nvvm_ir(*compiled.value);
}

}  // namespace regex_ir
