/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_kafka/kafka_callback.hpp>

#include <librdkafka/rdkafkacpp.h>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

python_oauth_refresh_callback::python_oauth_refresh_callback(
  kafka_oauth_callback_wrapper_type callback_wrapper, python_callable_type python_callable)
  : callback_wrapper_(callback_wrapper), python_callable_(python_callable) {};

void python_oauth_refresh_callback::oauthbearer_token_refresh_cb(
  RdKafka::Handle* handle, std::string const& oauthbearer_config)
{
  std::map<std::string, std::string> resp = callback_wrapper_(python_callable_);

  // Build parameters to pass to librdkafka
  std::string token         = resp["token"];
  int64_t token_lifetime_ms = std::stoll(resp["token_expiration_in_epoch"]);
  std::list<std::string> extensions;  // currently not supported
  std::string errstr;
  CUDF_EXPECTS(
    RdKafka::ErrorCode::ERR_NO_ERROR ==
      handle->oauthbearer_set_token(token, token_lifetime_ms, "kafka", extensions, errstr),
    "Error occurred while setting the oauthbearer token");
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
// CI scenario: libcudf_kafka/src
