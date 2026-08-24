# custreamz - GPU Accelerated Streaming

Built as an extension to [python streamz](https://github.com/python-streamz/streamz), cuStreamz provides GPU accelerated abstractions for streaming data. CuStreamz can be used along side python streamz or as a standalone library for ingesting streaming data to cudf dataframes.

The most common use for cuStreamz is accelerated data ingestion to a cudf dataframe. CuStreamz currently supports ingestion from Apache Kafka in the following message formats; Avro, CSV, JSON, Parquet, and ORC.

For example, the following snippet consumes CSV data from a Kafka topic named `custreamz_tips` and generates a cudf dataframe.

Users can visit [Apache Kafka Quickstart](https://kafka.apache.org/quickstart/) to learn how to install, create `custreamz_tips` topic, and insert the [tips](https://github.com/plotly/datasets/raw/master/tips.csv) data into Kafka.


```python
from custreamz import kafka

# Full list of configurations can be found at: https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
kafka_configs = {
    "metadata.broker.list": "localhost:9092",
    "group.id": "custreamz-client",
}

# Create a reusable Kafka Consumer client; "datasource"
consumer = kafka.Consumer(kafka_configs)

# Read 10,000 messages from `custreamz_tips` topic in CSV format.
tips_df = consumer.read_gdf(topic="custreamz_tips",
                        partition=0,
                        start=0,
                        end=10000,
                        message_format="csv")

print(tips_df.head())
tips_df['tip_percentage'] = tips_df['tip'] / tips_df['total_bill'] * 100

# display average tip by dining party size
print(tips_df.groupby('size').tip_percentage.mean())
```

A "hello world" of using cuStreamz with python streamz can be found [here](https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/hello_worlds/hello_streamz.ipynb)

A more detailed example of [parsing haproxy logs](https://github.com/rapidsai-community/notebooks-contrib/blob/main/the_archive/archived_rapids_demos/custreamz/parsing_haproxy_logs.ipynb) is also available.

## Quick Start

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you're running. This provides a ready to run Docker container with cuStreamz already installed.

## Installation


### CUDA/GPU requirements

* CUDA 12.2+ with a compatible NVIDIA driver
* Volta architecture or better (Compute Capability >=7.0)

### Conda

cuStraamz can be installed with conda (via [miniforge](https://github.com/conda-forge/miniforge)) from the `rapidsai` channel:

Release:
```bash
conda install -c rapidsai cudf_kafka custreamz
```

Nightly:
```bash
conda install -c rapidsai-nightly cudf_kafka custreamz
```

See the [Get RAPIDS version picker](https://docs.rapids.ai/install/) for more OS and version info.
