# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rmm.pylibrmm.stream import Stream

import pylibcudf as plc


@pytest.mark.parametrize(
    "streams, stream",
    [
        ([], Stream()),
        ([Stream()], Stream()),
        ([Stream() for _ in range(10)], Stream()),
    ],
)
def test_join_streams(streams: list[Stream], stream: Stream):
    # Just check that there isn't an error / segfault.
    # We can't easily verify correctness with this type of test.
    plc.experimental.join_streams(streams, stream)


@pytest.mark.uses_custom_stream
def test_join_streams_type_error():
    """Test that join_streams raises appropriate errors for invalid inputs."""
    main_stream = Stream()

    # Test with non-list input
    with pytest.raises(TypeError):
        plc.experimental.join_streams(None, main_stream)

    # Protocol stream should be accepted
    class _CudaStreamProto:
        def __cuda_stream__(self):
            return (0, 0)

    plc.experimental.join_streams([_CudaStreamProto()], main_stream)
    plc.experimental.join_streams([Stream()], _CudaStreamProto())
