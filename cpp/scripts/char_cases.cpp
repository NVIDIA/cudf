/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Standalone utility to regenerate the special case mapping tables used by libcudf for
 * non-trivial unicode character case conversions.
 *
 * Build:
 *   g++ -std=c++17 -O2 char_cases.cpp -o char_cases
 *
 * Usage:
 *   ./char_cases > special_cases.txt
 *
 * The output contains:
 * - the g_special_case_mappings array which should replace the existing array in
 *   cpp/src/strings/char_types/char_cases.h
 * - the special_case_prime value which should replace the value in get_special_case_hash_index()
 *   in cpp/include/cudf/strings/detail/char_tables.hpp
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

struct special_case_mapping_in {
  uint16_t num_upper_chars;
  uint16_t upper[3];
  uint16_t num_lower_chars;
  uint16_t lower[3];
};

constexpr special_case_mapping_in codepoint_mapping_in[] = {
  {2, {83, 83, 0}, 0, {0, 0, 0}},       {0, {0, 0, 0}, 2, {105, 775, 0}},
  {2, {700, 78, 0}, 0, {0, 0, 0}},      {1, {452, 0, 0}, 1, {454, 0, 0}},
  {1, {455, 0, 0}, 1, {457, 0, 0}},     {1, {458, 0, 0}, 1, {460, 0, 0}},
  {2, {74, 780, 0}, 0, {0, 0, 0}},      {1, {497, 0, 0}, 1, {499, 0, 0}},
  {3, {921, 776, 769}, 0, {0, 0, 0}},   {3, {933, 776, 769}, 0, {0, 0, 0}},
  {2, {1333, 1362, 0}, 0, {0, 0, 0}},   {2, {72, 817, 0}, 0, {0, 0, 0}},
  {2, {84, 776, 0}, 0, {0, 0, 0}},      {2, {87, 778, 0}, 0, {0, 0, 0}},
  {2, {89, 778, 0}, 0, {0, 0, 0}},      {2, {65, 702, 0}, 0, {0, 0, 0}},
  {2, {933, 787, 0}, 0, {0, 0, 0}},     {3, {933, 787, 768}, 0, {0, 0, 0}},
  {3, {933, 787, 769}, 0, {0, 0, 0}},   {3, {933, 787, 834}, 0, {0, 0, 0}},
  {2, {7944, 921, 0}, 0, {0, 0, 0}},    {2, {7945, 921, 0}, 0, {0, 0, 0}},
  {2, {7946, 921, 0}, 0, {0, 0, 0}},    {2, {7947, 921, 0}, 0, {0, 0, 0}},
  {2, {7948, 921, 0}, 0, {0, 0, 0}},    {2, {7949, 921, 0}, 0, {0, 0, 0}},
  {2, {7950, 921, 0}, 0, {0, 0, 0}},    {2, {7951, 921, 0}, 0, {0, 0, 0}},
  {2, {7944, 921, 0}, 1, {8064, 0, 0}}, {2, {7945, 921, 0}, 1, {8065, 0, 0}},
  {2, {7946, 921, 0}, 1, {8066, 0, 0}}, {2, {7947, 921, 0}, 1, {8067, 0, 0}},
  {2, {7948, 921, 0}, 1, {8068, 0, 0}}, {2, {7949, 921, 0}, 1, {8069, 0, 0}},
  {2, {7950, 921, 0}, 1, {8070, 0, 0}}, {2, {7951, 921, 0}, 1, {8071, 0, 0}},
  {2, {7976, 921, 0}, 0, {0, 0, 0}},    {2, {7977, 921, 0}, 0, {0, 0, 0}},
  {2, {7978, 921, 0}, 0, {0, 0, 0}},    {2, {7979, 921, 0}, 0, {0, 0, 0}},
  {2, {7980, 921, 0}, 0, {0, 0, 0}},    {2, {7981, 921, 0}, 0, {0, 0, 0}},
  {2, {7982, 921, 0}, 0, {0, 0, 0}},    {2, {7983, 921, 0}, 0, {0, 0, 0}},
  {2, {7976, 921, 0}, 1, {8080, 0, 0}}, {2, {7977, 921, 0}, 1, {8081, 0, 0}},
  {2, {7978, 921, 0}, 1, {8082, 0, 0}}, {2, {7979, 921, 0}, 1, {8083, 0, 0}},
  {2, {7980, 921, 0}, 1, {8084, 0, 0}}, {2, {7981, 921, 0}, 1, {8085, 0, 0}},
  {2, {7982, 921, 0}, 1, {8086, 0, 0}}, {2, {7983, 921, 0}, 1, {8087, 0, 0}},
  {2, {8040, 921, 0}, 0, {0, 0, 0}},    {2, {8041, 921, 0}, 0, {0, 0, 0}},
  {2, {8042, 921, 0}, 0, {0, 0, 0}},    {2, {8043, 921, 0}, 0, {0, 0, 0}},
  {2, {8044, 921, 0}, 0, {0, 0, 0}},    {2, {8045, 921, 0}, 0, {0, 0, 0}},
  {2, {8046, 921, 0}, 0, {0, 0, 0}},    {2, {8047, 921, 0}, 0, {0, 0, 0}},
  {2, {8040, 921, 0}, 1, {8096, 0, 0}}, {2, {8041, 921, 0}, 1, {8097, 0, 0}},
  {2, {8042, 921, 0}, 1, {8098, 0, 0}}, {2, {8043, 921, 0}, 1, {8099, 0, 0}},
  {2, {8044, 921, 0}, 1, {8100, 0, 0}}, {2, {8045, 921, 0}, 1, {8101, 0, 0}},
  {2, {8046, 921, 0}, 1, {8102, 0, 0}}, {2, {8047, 921, 0}, 1, {8103, 0, 0}},
  {2, {8122, 921, 0}, 0, {0, 0, 0}},    {2, {913, 921, 0}, 0, {0, 0, 0}},
  {2, {902, 921, 0}, 0, {0, 0, 0}},     {2, {913, 834, 0}, 0, {0, 0, 0}},
  {3, {913, 834, 921}, 0, {0, 0, 0}},   {2, {913, 921, 0}, 1, {8115, 0, 0}},
  {2, {8138, 921, 0}, 0, {0, 0, 0}},    {2, {919, 921, 0}, 0, {0, 0, 0}},
  {2, {905, 921, 0}, 0, {0, 0, 0}},     {2, {919, 834, 0}, 0, {0, 0, 0}},
  {3, {919, 834, 921}, 0, {0, 0, 0}},   {2, {919, 921, 0}, 1, {8131, 0, 0}},
  {3, {921, 776, 768}, 0, {0, 0, 0}},   {3, {921, 776, 769}, 0, {0, 0, 0}},
  {2, {921, 834, 0}, 0, {0, 0, 0}},     {3, {921, 776, 834}, 0, {0, 0, 0}},
  {3, {933, 776, 768}, 0, {0, 0, 0}},   {3, {933, 776, 769}, 0, {0, 0, 0}},
  {2, {929, 787, 0}, 0, {0, 0, 0}},     {2, {933, 834, 0}, 0, {0, 0, 0}},
  {3, {933, 776, 834}, 0, {0, 0, 0}},   {2, {8186, 921, 0}, 0, {0, 0, 0}},
  {2, {937, 921, 0}, 0, {0, 0, 0}},     {2, {911, 921, 0}, 0, {0, 0, 0}},
  {2, {937, 834, 0}, 0, {0, 0, 0}},     {3, {937, 834, 921}, 0, {0, 0, 0}},
  {2, {937, 921, 0}, 1, {8179, 0, 0}},  {2, {70, 70, 0}, 0, {0, 0, 0}},
  {2, {70, 73, 0}, 0, {0, 0, 0}},       {2, {70, 76, 0}, 0, {0, 0, 0}},
  {3, {70, 70, 73}, 0, {0, 0, 0}},      {3, {70, 70, 76}, 0, {0, 0, 0}},
  {2, {83, 84, 0}, 0, {0, 0, 0}},       {2, {83, 84, 0}, 0, {0, 0, 0}},
  {2, {1348, 1350, 0}, 0, {0, 0, 0}},   {2, {1348, 1333, 0}, 0, {0, 0, 0}},
  {2, {1348, 1339, 0}, 0, {0, 0, 0}},   {2, {1358, 1350, 0}, 0, {0, 0, 0}},
  {2, {1348, 1341, 0}, 0, {0, 0, 0}},
};
constexpr std::array<uint16_t, 107> codepoints_in = {
  223,   304,   329,   453,   456,   459,   496,   498,   912,   944,  1415, 7830,  7831,  7832,
  7833,  7834,  8016,  8018,  8020,  8022,  8064,  8065,  8066,  8067, 8068, 8069,  8070,  8071,
  8072,  8073,  8074,  8075,  8076,  8077,  8078,  8079,  8080,  8081, 8082, 8083,  8084,  8085,
  8086,  8087,  8088,  8089,  8090,  8091,  8092,  8093,  8094,  8095, 8096, 8097,  8098,  8099,
  8100,  8101,  8102,  8103,  8104,  8105,  8106,  8107,  8108,  8109, 8110, 8111,  8114,  8115,
  8116,  8118,  8119,  8124,  8130,  8131,  8132,  8134,  8135,  8140, 8146, 8147,  8150,  8151,
  8162,  8163,  8164,  8166,  8167,  8178,  8179,  8180,  8182,  8183, 8188, 64256, 64257, 64258,
  64259, 64260, 64261, 64262, 64275, 64276, 64277, 64278, 64279,
};
constexpr std::array<uint16_t, 269> primes = {
  227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,  283,  293,  307,  311,
  313,  317,  331,  337,  347,  349,  353,  359,  367,  373,  379,  383,  389,  397,  401,  409,
  419,  421,  431,  433,  439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
  509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,  599,  601,  607,  613,
  617,  619,  631,  641,  643,  647,  653,  659,  661,  673,  677,  683,  691,  701,  709,  719,
  727,  733,  739,  743,  751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
  829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,  919,  929,  937,  941,
  947,  953,  967,  971,  977,  983,  991,  997,  1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
  1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
  1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283,
  1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423,
  1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
  1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619,
  1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747,
  1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
  1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003,
  2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099,
};

// find a prime number that generates no collisions for all possible input data
uint16_t find_collision_proof_prime()
{
  for (auto const& prime : primes) {
    std::unordered_set<uint16_t> keys;
    std::for_each(std::cbegin(codepoints_in),
                  std::cend(codepoints_in),
                  [&](uint16_t const codepoint) { keys.insert(codepoint % prime); });
    if (keys.size() == codepoints_in.size()) return prime;
  }

  // couldn't find a collision-proof prime
  return 0;
}

/**
 * @brief Regenerates the special case mapping tables used to handle non-trivial unicode
 * character case conversions.
 *
 * 'special' cased characters are those defined as not having trivial single->single character
 * mappings when having upper(), lower() or titlecase() operations applied.  Typically this is
 * for cases where a single character maps to multiple, but there are also cases of
 * non-reversible mappings, where:  codepoint != lower(upper(code_point)).
 *
 * @return true if the tables were generated successfully
 */
bool generate_special_mapping_hash_table()
{
  uint16_t hash_prime = find_collision_proof_prime();
  if (hash_prime == 0) {
    fprintf(stderr, "Could not find a usable prime number for hash table\n");
    return false;
  }

  // generate hash index table
  // size of the table is the prime #, since we're just doing (key % hash_prime)
  std::vector<std::pair<bool, uint16_t>> hash_indices(hash_prime,
                                                      std::pair<bool, uint16_t>(false, 0));
  int index = 0;
  std::for_each(std::begin(codepoints_in), std::end(codepoints_in), [&](uint16_t codepoint) {
    hash_indices[codepoint % hash_prime].first  = true;
    hash_indices[codepoint % hash_prime].second = index++;
  });

  // print out the code

  // the mappings
  printf("constexpr special_case_mapping g_special_case_mappings[] = {\n");
  bool prev_empty = false;
  std::for_each(
    hash_indices.begin(), hash_indices.end(), [&prev_empty](std::pair<bool, uint16_t> entry) {
      if (entry.first) {
        special_case_mapping_in m = codepoint_mapping_in[entry.second];
        printf("%s   { %d, { %d, %d, %d }, %d, {%d, %d, %d} },\n",
               prev_empty ? "\n" : "",
               m.num_upper_chars,
               m.upper[0],
               m.upper[1],
               m.upper[2],
               m.num_lower_chars,
               m.lower[0],
               m.lower[1],
               m.lower[2]);
      } else {
        printf("%s{},", prev_empty ? "" : "   ");
      }
      prev_empty = !entry.first;
    });
  printf("};\n");

  // the prime used by get_special_case_hash_index()
  printf("\nconstexpr uint16_t special_case_prime = %d;\n", hash_prime);
  return true;
}

int main() { return generate_special_mapping_hash_table() ? EXIT_SUCCESS : EXIT_FAILURE; }
