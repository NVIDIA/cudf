# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.io.datasource import Datasource

__all__ = ["AdvisingDatasource"]

class AdvisingDatasource(Datasource):
    def nbytes(self) -> int: ...
