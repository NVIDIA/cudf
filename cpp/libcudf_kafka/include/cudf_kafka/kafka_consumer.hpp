/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kafka_callback.hpp"

#include <cudf/io/datasource.hpp>

#include <librdkafka/rdkafkacpp.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

/**
 * @brief libcudf datasource for Apache Kafka
 *
 * @ingroup io_datasources
 */
class kafka_consumer : public cudf::io::datasource {
 public:
  /**
   * @brief Creates an instance of the Kafka consumer object that is in a semi-ready state.
   *
   * A consumer in a semi-ready state does not have all required parameters to make successful
   * consumer interactions with the Kafka broker. However in the semi-ready state Kafka metadata
   * operations are still possible. This is useful for clients who plan to only use those metadata
   * operations. This is useful when the need for delayed partition and topic assignment
   * is not known ahead of time and needs to be delayed to as late as possible.
   * Documentation for librdkafka configurations can be found at
   * https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
   *
   * @param configs key/value pairs of librdkafka configurations that will be
   *                passed to the librdkafka client
   * @param python_callable `python_callable_type` pointer to a Python functools.partial object
   * @param callable_wrapper `kafka_oauth_callback_wrapper_type` Cython wrapper that will
   *                 be used to invoke the `python_callable`. This wrapper serves the purpose
   *                 of preventing us from having to link against the Python development library
   *                 in libcudf_kafka.
   */
  kafka_consumer(std::map<std::string, std::string> configs,
                 python_callable_type python_callable,
                 kafka_oauth_callback_wrapper_type callable_wrapper);

  /**
   * @brief Instantiate a Kafka consumer object. Documentation for librdkafka configurations can be
   * found at https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
   *
   * @param configs key/value pairs of librdkafka configurations that will be
   *                passed to the librdkafka client
   * @param python_callable `python_callable_type` pointer to a Python functools.partial object
   * @param callable_wrapper `kafka_oauth_callback_wrapper_type` Cython wrapper that will
   *                 be used to invoke the `python_callable`. This wrapper serves the purpose
   *                 of preventing us from having to link against the Python development library
   *                 in libcudf_kafka.
   * @param topic_name name of the Kafka topic to consume from
   * @param partition partition index to consume from between `0` and `TOPIC_NUM_PARTITIONS - 1`
   * inclusive
   * @param start_offset seek position for the specified TOPPAR (Topic/Partition combo)
   * @param end_offset position in the specified TOPPAR to read to
   * @param batch_timeout maximum (millisecond) read time allowed. If end_offset is not reached
   * before batch_timeout, a smaller subset will be returned
   * @param delimiter optional delimiter to insert into the output between kafka messages, Ex: "\n"
   */
  kafka_consumer(std::map<std::string, std::string> configs,
                 python_callable_type python_callable,
                 kafka_oauth_callback_wrapper_type callable_wrapper,
                 std::string const& topic_name,
                 int partition,
                 int64_t start_offset,
                 int64_t end_offset,
                 int batch_timeout,
                 std::string const& delimiter);

  /**
   * @brief Returns a buffer with a subset of data from Kafka Topic
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   *
   * @return The data buffer
   */
  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size) override;

  /**
   * @brief Returns the size of the data in Kafka buffer
   *
   * @return size_t The size of the source data in bytes
   */
  size_t size() const override;

  /**
   * @brief Reads a selected range into a preallocated buffer.
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[in] dst Address of the existing host memory
   *
   * @return The number of bytes read (can be smaller than size)
   */
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  /**
   * @brief Commits an offset to a specified Kafka Topic/Partition instance
   *
   * @throws cudf::logic_error on failure to commit the partition offset
   *
   * @param[in] topic Name of the Kafka topic that the offset should be set for
   * @param[in] partition Partition on the specified topic that should be used
   * @param[in] offset Offset that should be set for the topic/partition pair
   *
   */
  void commit_offset(std::string const& topic, int partition, int64_t offset);

  /**
   * @brief Retrieve the watermark offset values for a topic/partition
   *
   * @param[in] topic Name of the Kafka topic that the watermark should be retrieved for
   * @param[in] partition Partition on the specified topic which should be used
   * @param[in] timeout Max milliseconds to wait on a response from the Kafka broker
   * @param[in] cached If True uses the last retrieved value from the Kafka broker, if False
   *            the latest value will be retrieved from the Kafka broker by making a network
   *            request.
   * @return The watermark offset value for the specified topic/partition
   */
  std::map<std::string, int64_t> get_watermark_offset(std::string const& topic,
                                                      int partition,
                                                      int timeout,
                                                      bool cached);

  /**
   * @brief Retrieve the current Kafka client configurations
   *
   * @return Map<string, string> of key/value pairs of the current client configurations
   */
  std::map<std::string, std::string> current_configs();

  /**
   * @brief Get the latest offset that was successfully committed to the Kafka broker
   *
   * @param[in] topic Topic name for the topic/partition pair
   * @param[in] partition Partition number of the topic/partition pair
   *
   * @return Latest offset for the specified topic/partition pair
   */
  int64_t get_committed_offset(std::string const& topic, int partition);

  /**
   * @brief Query the Kafka broker for the list of Topic partitions for a Topic. If no topic is
   * specified then the partitions for all Topics in the broker will be retrieved.
   *
   * @param[in] specific_topic The name of the topic for which to retrieve partitions. If empty then
   * the partitions for all topics will be retrieved.
   *
   * @return Map of Kafka topic names with their corresponding list of topic partition values.
   */
  std::map<std::string, std::vector<int32_t>> list_topics(std::string specific_topic);

  /**
   * @brief Close the underlying socket connection to Kafka and clean up system resources
   *
   * @throws cudf::logic_error on failure to close the connection
   * @param timeout Max milliseconds to wait on a response
   */
  void close(int timeout);

  /**
   * @brief Stop all active consumption and remove consumer subscriptions to topic/partition
   * instances
   *
   * @throws cudf::logic_error on failure to unsubscribe from the active partition assignments.
   */
  void unsubscribe();

  /// @brief Destroy the Kafka consumer object
  virtual ~kafka_consumer() {};

 private:
  std::unique_ptr<RdKafka::Conf> kafka_conf;  // RDKafka configuration object
  std::unique_ptr<RdKafka::KafkaConsumer> consumer;

  std::map<std::string, std::string> configs;
  python_callable_type python_callable_;
  kafka_oauth_callback_wrapper_type callable_wrapper_;

  std::string topic_name;
  int partition;
  int64_t start_offset;
  int64_t end_offset;
  int batch_timeout;
  int default_timeout = 10000;  // milliseconds
  std::string delimiter;

  std::string buffer;

 private:
  RdKafka::ErrorCode update_consumer_topic_partition_assignment(std::string const& topic,
                                                                int partition,
                                                                int64_t offset);

  /**
   * Convenience method for getting "now()" in Kafka's standard format
   */
  int64_t now();

  void consume_to_buffer();
};

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
