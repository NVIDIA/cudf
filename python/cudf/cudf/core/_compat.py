# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from packaging import version

PYARROW_GE_26 = version.parse(pa.__version__) >= version.parse("26.0.0")
