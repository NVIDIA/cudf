/*
 * SPDX-FileCopyrightText: Copyright (C) 1996-2002 Julian R Seward.  All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND bzip2-1.0.6
 */

/*
 *
 * bzip2 license information is available at
 * https://spdx.org/licenses/bzip2-1.0.6.html
 * https://github.com/asimonov-im/bzip2/blob/master/LICENSE
 * original source code available at
 * https://www.sourceware.org/bzip2/
 */

/*--

Copyright (C) 1996-2002 Julian R Seward.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. The origin of this software must not be misrepresented; you must
not claim that you wrote the original software.  If you use this
software in a product, an acknowledgment in the product
documentation would be appreciated but is not required.

3. Altered source versions must be plainly marked as such, and must
not be misrepresented as being the original software.

4. The name of the author may not be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Julian Seward, Cambridge, UK.
jseward@acm.org
bzip2/libbzip2 version 1.0 of 21 March 2000

This program is based on (at least) the work of:
Mike Burrows
David Wheeler
Peter Fenwick
Alistair Moffat
Radford Neal
Ian H. Witten
Robert Sedgewick
Jon L. Bentley

For more information on these sources, see the manual.
--*/

#pragma once

namespace cudf {
namespace io {
// If BZ_OUTBUFF_FULL is returned and block_start is non-NULL, dstlen will be updated to point to
// the end of the last valid block, and block_start will contain the offset in bits of the beginning
// of the block, so it can be passed in to resume decoding later on.
#define BZ_OK               0
#define BZ_RUN_OK           1
#define BZ_FLUSH_OK         2
#define BZ_FINISH_OK        3
#define BZ_STREAM_END       4
#define BZ_SEQUENCE_ERROR   (-1)
#define BZ_PARAM_ERROR      (-2)
#define BZ_MEM_ERROR        (-3)
#define BZ_DATA_ERROR       (-4)
#define BZ_DATA_ERROR_MAGIC (-5)
#define BZ_IO_ERROR         (-6)
#define BZ_UNEXPECTED_EOF   (-7)
#define BZ_OUTBUFF_FULL     (-8)

int32_t cpu_bz2_uncompress(uint8_t const* input,
                           size_t inlen,
                           uint8_t* dst,
                           size_t* dstlen,
                           uint64_t* block_start = nullptr);

}  // namespace io
}  // namespace cudf
