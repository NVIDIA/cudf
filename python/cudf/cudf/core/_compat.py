# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
from packaging import version

PANDAS_CURRENT_SUPPORTED_VERSION = version.parse("3.0.2")
PANDAS_VERSION = version.parse(pd.__version__)
PYARROW_GE_26 = version.parse(pa.__version__) >= version.parse("26.0.0")
