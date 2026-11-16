/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/utilities/host_worker_pool.hpp>

#include <src/io/utilities/base64_utilities.hpp>

#include <chrono>
#include <future>
#include <stdexcept>
#include <vector>

using cudf::io::detail::base64_decode;
using cudf::io::detail::base64_encode;

class IoUtilitiesTest : public cudf::test::BaseFixture {};

TEST(IoUtilitiesTest, Base64EncodeAndDecode)
{
  // a vector of lorem ipsum strings
  std::vector<std::string> strings = {
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut ",
    "labore et dolore magna aliqua. Id ornare arcu odio ut sem. Ultrices neque ornare aenean ",
    "euismod elementum nisi quis. Faucibus pulvinar elementum integer enim. Ut tortor pretium ",
    "viverra suspendisse potenti nullam ac tortor vitae. Elementum pulvinar etiam non quam lacus. ",
    "Fermentum odio eu feugiat pretium nibh. Commodo ullamcorper a lacus vestibulum sed arcu. "
    "Elit ",
    "ut aliquam purus sit amet luctus venenatis lectus magna. Aliquet enim tortor at auctor urna ",
    "nunc id cursus metus. Vivamus at augue eget arcu dictum. Ultricies leo integer malesuada "
    "nunc ",
    "vel risus commodo viverra maecenas.Netus et malesuada fames ac turpis egestas. Erat ",
    "pellentesque adipiscing commodo elit at imperdiet. Commodo nulla facilisi nullam vehicula. ",
    "Morbi tristique senectus et netus et. Cursus vitae congue mauris rhoncus aenean vel elit ",
    "scelerisque mauris. Eros donec ac odio tempor orci dapibus ultrices. Purus in mollis nunc "
    "sed ",
    "id. Justo eget magna fermentum iaculis eu. Diam maecenas ultricies mi eget. Justo laoreet "
    "sit ",
    "amet cursus sit amet. Nibh venenatis cras sed felis eget velit aliquet sagittis id. Dui ut ",
    "ornare lectus sit amet est placerat in egestas. Malesuada nunc vel risus commodo viverra ",
    "maecenas accumsan lacus. Arcu non odio euismod lacinia at. Euismod elementum nisi quis ",
    "eleifend quam adipiscing vitae proin sagittis. Eget sit amet tellus cras adipiscing enim ",
    "eu.Neque ornare aenean euismod elementum nisi quis eleifend quam adipiscing. Posuere ",
    "sollicitudin aliquam ultrices sagittis orci a scelerisque purus. Lobortis elementum nibh ",
    "tellus molestie. Et ligula ullamcorper malesuada proin libero nunc consequat interdum "
    "varius. ",
    "Neque volutpat ac tincidunt vitae semper quis lectus. Nunc mi ipsum faucibus vitae. Congue "
    "eu ",
    "consequat ac felis donec et. Faucibus in ornare quam viverra orci sagittis. Egestas "
    "fringilla ",
    "phasellus faucibus scelerisque eleifend. Sem fringilla ut morbi tincidunt augue. Lobortis ",
    "elementum nibh tellus molestie nunc non. Ultrices neque ornare aenean euismod elementum. ",
    "Cursus turpis massa tincidunt dui ut ornare lectus sit. Eu facilisis sed odio morbi quis "
    "commodo odio. Tortor dignissim convallis aenean et tortor at risus. Sed euismod nisi porta ",
    "lorem. In ornare quam viverra orci sagittis. Sed blandit libero volutpat sed cras. Quis ",
    "viverra nibh cras pulvinar mattis nunc sed blandit libero. Non tellus orci ac auctor augue. ",
    "Mattis molestie a iaculis at erat pellentesque adipiscing. Est lorem ipsum dolor sit amet ",
    "consectetur. Commodo odio aenean sed adipiscing. Nunc lobortis mattis aliquam faucibus "
    "purus. ",
    "Pellentesque massa placerat duis ultricies lacus. Sed viverra tellus in hac habitasse "
    "platea. ",
    "Ut porttitor leo a diam sollicitudin tempor id eu. Rhoncus aenean vel elit scelerisque "
    "mauris ",
    "pellentesque pulvinar pellentesque. Ornare quam viverra orci sagittis. Interdum consectetur ",
    "libero id faucibus nisl tincidunt eget. Eget est lorem ipsum dolor sit amet. Malesuada fames ",
    "ac turpis egestas integer eget aliquet nibh. Scelerisque felis imperdiet proin fermentum "
    "leo. ",
    "Duis convallis convallis tellus id interdum velit. Sit amet massa vitae tortor condimentum ",
    "lacinia quis vel. Eu turpis egestas pretium aenean pharetra. Sed enim ut sem viverra aliquet ",
    "eget sit amet tellus. Feugiat nisl pretium fusce id velit ut tortor. In hendrerit gravida ",
    "rutrum quisque non tellus orci ac auctor. Sit amet nulla facilisi morbi. Nunc congue nisi ",
    "vitae suscipit tellus. Posuere morbi leo urna molestie at elementum eu. Egestas sed tempus ",
    "urna et pharetra pharetra. Sed euismod nisi porta lorem. At elementum eu facilisis sed. Odio ",
    "aenean sed adipiscing diam donec. Congue nisi vitae suscipit tellus mauris a diam. Fringilla ",
    "urna porttitor rhoncus dolor purus non enim praesent. Eget gravida cum sociis natoque. ",
    "Facilisis mauris sit amet massa vitae tortor. Vulputate odio ut enim blandit volutpat ",
    "maecenas volutpat blandit. Ut ornare lectus sit amet est placerat in. Quis vel eros donec ac ",
    "odio tempor orci dapibus ultrices. Venenatis lectus magna fringilla urna porttitor rhoncus ",
    "dolor. Mattis vulputate enim nulla aliquet porttitor lacus. Lectus nulla at volutpat diam ut ",
    "venenatis tellus in. Et ligula ullamcorper malesuada proin libero nunc consequat interdum. "
    "Ut ",
    "enim blandit volutpat maecenas volutpat blandit aliquam etiam erat. Pellentesque pulvinar ",
    "pellentesque habitant morbi tristique senectus et. Auctor eu augue ut lectus arcu bibendum "
    "at ",
    "varius. Posuere ac ut consequat semper viverra nam. Sed euismod nisi porta lorem mollis ",
    "aliquam ut. Porttitor eget dolor morbi non arcu risus quis varius. Adipiscing bibendum est ",
    "ultricies integer quis auctor. Hac habitasse platea dictumst quisque sagittis purus sit amet ",
    "volutpat. Nullam vehicula ipsum a arcu cursus vitae. Velit scelerisque in dictum non ",
    "consectetur a erat nam at. Nulla facilisi cras fermentum odio eu. Tincidunt augue interdum ",
    "velit euismod in pellentesque massa placerat. Suspendisse potenti nullam ac tortor vitae ",
    "purus faucibus ornare. Amet dictum sit amet justo donec enim diam vulputate. Tellus ",
    "pellentesque eu tincidunt tortor aliquam nulla facilisi cras. Mauris in aliquam sem "
    "fringilla ",
    "ut morbi tincidunt. Volutpat diam ut venenatis tellus in metus. Sed pulvinar proin gravida ",
    "hendrerit lectus a. Feugiat nisl pretium fusce id velit ut tortor pretium viverra. Non ",
    "consectetur a erat nam. Fermentum odio eu feugiat pretium nibh ipsum consequat nisl. Donec ",
    "pretium vulputate sapien nec. Purus sit amet luctus venenatis lectus magna fringilla. Mauris ",
    "cursus mattis molestie a iaculis. A iaculis at erat pellentesque adipiscing. Auctor augue ",
    "mauris augue neque gravida in fermentum et sollicitudin. Lectus quam id leo in vitae turpis ",
    "massa sed. Erat nam at lectus urna duis convallis convallis. Dignissim cras tincidunt ",
    "lobortis feugiat vivamus at augue eget arcu. Eleifend mi in nulla posuere sollicitudin ",
    "aliquam ultrices sagittis. Pellentesque nec nam aliquam sem. Feugiat in fermentum posuere ",
    "urna nec tincidunt praesent. Morbi non arcu risus quis varius quam quisque. Morbi tristique ",
    "senectus et netus et malesuada fames ac. Et ligula ullamcorper malesuada proin libero. ",
    "Vivamus at augue eget arcu dictum varius duis at consectetur. Eget mauris pharetra et ",
    "ultrices neque ornare aenean euismod. Sapien faucibus et molestie ac feugiat sed lectus ",
    "vestibulum mattis. Blandit turpis cursus in hac habitasse platea dictumst quisque sagittis. ",
    "Fermentum iaculis eu non diam phasellus vestibulum. Mattis aliquam faucibus purus in massa ",
    "tempor nec feugiat nisl. Lectus sit amet est placerat. Accumsan sit amet nulla facilisi "
    "morbi ",
    "tempus iaculis urna. Magna eget est lorem ipsum dolor sit. Curabitur gravida arcu ac tortor ",
    "dignissim convallis aenean."};

  std::vector<std::string> base64_roundtripped_strings;

  std::transform(strings.begin(),
                 strings.end(),
                 std::back_inserter(base64_roundtripped_strings),
                 [&](auto& str) { return base64_decode(base64_encode(str)); });

  // Create columns for expected and results
  cudf::test::strings_column_wrapper expected(strings.begin(), strings.end());
  cudf::test::strings_column_wrapper results(base64_roundtripped_strings.begin(),
                                             base64_roundtripped_strings.end());

  // Check equal columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results);
}

TEST(IoUtilitiesTest, FuturesWaitGuardWaitsForPendingFutures)
{
  std::promise<int> failed;
  std::promise<int> pending;
  failed.set_exception(std::make_exception_ptr(std::runtime_error("failed")));

  // Like the futures of a host worker pool, these futures do not wait on destruction, so the
  // collector can only stay blocked on `pending` through the guard
  auto collector = std::async(std::launch::async, [&] {
    std::vector<std::future<int>> futures;
    cudf::detail::futures_wait_guard futures_guard{futures};
    futures.push_back(failed.get_future());
    futures.push_back(pending.get_future());
    for (auto& future : futures) {
      future.get();
    }
  });

  EXPECT_EQ(collector.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);

  // The exception from the drained future must not replace the one that started the unwinding
  pending.set_exception(std::make_exception_ptr(std::logic_error("pending")));
  EXPECT_THROW(collector.get(), std::runtime_error);
}
