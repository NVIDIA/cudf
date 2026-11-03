/*
 * SPDX-FileCopyrightText: Copyright (C) 1996-2002 Julian R Seward.  All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND bzip2-1.0.6
 */

/*
 * cpu_unbz2.cpp
 *
 * Heavily based on libbzip2's decompress.c (original copyright notice below)
 * Modified version for memory-to-memory decompression.
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

#include "decompression.hpp"
#include "unbz2.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cudf {
namespace io {

// Constants for the fast MTF decoder.
#define MTFA_SIZE 4096
#define MTFL_SIZE 16

// Header bytes.
#define BZ_HDR_B 0x42 /* 'B' */
#define BZ_HDR_Z 0x5a /* 'Z' */
#define BZ_HDR_h 0x68 /* 'h' */
#define BZ_HDR_0 0x30 /* '0' */

// Constants for the back end.

#define BZ_MAX_ALPHA_SIZE 258
#define BZ_MAX_CODE_LEN   23

#define BZ_RUNA 0
#define BZ_RUNB 1

#define BZ_N_GROUPS 6
#define BZ_G_SIZE   50

#define BZ_MAX_SELECTORS (2 + (900000 / BZ_G_SIZE))

using huff_s = struct {
  int32_t minLen;
  int32_t limit[BZ_MAX_CODE_LEN];
  int32_t base[BZ_MAX_CODE_LEN];
  uint16_t perm[BZ_MAX_ALPHA_SIZE];
};

// Decoder state
using unbz_state_s = struct {
  // Input
  uint8_t const* cur;
  uint8_t const* end;
  uint8_t const* base;
  uint64_t bitbuf;
  uint32_t bitpos;

  // Output
  uint8_t* out;
  uint8_t* outend;
  uint8_t* outbase;

  // misc administratium
  uint32_t blockSize100k;
  int32_t currBlockNo;
  int32_t save_nblock;

  // for undoing the Burrows-Wheeler transform
  std::vector<uint32_t> tt;
  uint32_t origPtr;
  int32_t nblock_used;
  int32_t unzftab[256];

  // map of bytes used in block
  uint8_t seqToUnseq[256];

  // for decoding the MTF values
  int32_t mtfbase[256 / MTFL_SIZE];
  uint8_t mtfa[MTFA_SIZE];
  uint8_t selector[BZ_MAX_SELECTORS];
  uint8_t len[BZ_MAX_ALPHA_SIZE];

  huff_s ht[BZ_N_GROUPS];
};

// return next 32 bits
static inline uint32_t next32bits(unbz_state_s const* s)
{
  return (uint32_t)((s->bitbuf << s->bitpos) >> 32);
}

// return next n bits
static inline uint32_t showbits(unbz_state_s const* s, uint32_t n)
{
  return (uint32_t)((s->bitbuf << s->bitpos) >> (64 - n));
}

// update bit position, refill bit buffer if necessary
static void skipbits(unbz_state_s* s, uint32_t n)
{
  uint32_t bitpos = s->bitpos + n;
  if (bitpos >= 32) {
    uint8_t const* cur = s->cur + 4;
    uint32_t next32 =
      (cur + 4 < s->end) ? __builtin_bswap32(*reinterpret_cast<uint32_t const*>(cur + 4)) : 0;
    s->cur    = cur;
    s->bitbuf = (s->bitbuf << 32) | next32;
    bitpos &= 0x1f;
  }
  s->bitpos = bitpos;
}

static inline uint32_t getbits(unbz_state_s* s, uint32_t n)
{
  uint32_t bits = showbits(s, n);
  skipbits(s, n);
  return bits;
}

/*---------------------------------------------------*/
int32_t bz2_decompress_block(unbz_state_s* s)
{
  int nInUse;

  int32_t i;
  int32_t j;
  int32_t t;
  int32_t alphaSize;
  int32_t nGroups;
  int32_t nSelectors;
  int32_t EOB;
  int32_t groupNo;
  int32_t groupPos;
  uint32_t nblock, nblockMAX;
  huff_s const* gSel = nullptr;
  uint32_t inUse16;
  uint32_t sig0, sig1;

  // Start-of-block signature
  sig0 = getbits(s, 24);
  sig1 = getbits(s, 24);
  if (sig0 != 0x31'4159 || sig1 != 0x26'5359) { return BZ_DATA_ERROR; }

  s->currBlockNo++;

  skipbits(s, 32);  // block CRC

  if (getbits(s, 1)) return BZ_DATA_ERROR;  // blockRandomized not supported (old bzip versions)

  s->origPtr = getbits(s, 24);
  if (s->origPtr > 10 + 100000 * s->blockSize100k) return BZ_DATA_ERROR;

  // Receive the mapping table
  inUse16 = getbits(s, 16);
  nInUse  = 0;
  for (i = 0; i < 16; i++, inUse16 <<= 1) {
    if (inUse16 & 0x8000) {
      uint32_t inUse = getbits(s, 16);
      for (j = 0; j < 16; j++, inUse <<= 1) {
        if (inUse & 0x8000) { s->seqToUnseq[nInUse++] = (i << 4) + j; }
      }
    }
  }
  if (nInUse == 0) return BZ_DATA_ERROR;
  alphaSize = nInUse + 2;

  // Now the selectors
  {
    uint32_t pos;  // BZ_N_GROUPS * 4-bit

    nGroups    = getbits(s, 3);
    nSelectors = getbits(s, 15);
    if (nGroups < 2 || nGroups > 6 || nSelectors < 1 || nSelectors > BZ_MAX_SELECTORS)
      return BZ_DATA_ERROR;

    pos = 0x7654'3210;
    for (i = 0; i < nSelectors; i++) {
      uint32_t selectorMtf = 0, mask, tmp;
      for (int32_t v = next32bits(s); v < 0; v <<= 1) {
        if (++selectorMtf >= (uint32_t)nGroups) return BZ_DATA_ERROR;
      }
      skipbits(s, selectorMtf + 1);
      // Undo the MTF values for the selectors.
      tmp            = (pos >> (selectorMtf * 4)) & 0xf;
      s->selector[i] = tmp;
      mask           = (1 << ((selectorMtf * 4) + 4)) - 1;
      pos            = (pos & ~mask) | ((pos << 4) & mask) | tmp;
    }
  }

  // Now the coding tables
  for (t = 0; t < nGroups; t++) {
    int32_t pp, vec;
    uint8_t* length = &s->len[0];
    int32_t curr    = getbits(s, 5);
    int32_t minLen  = BZ_MAX_CODE_LEN - 1;
    int32_t maxLen  = 0;
    huff_s* sel     = &s->ht[t];
    for (i = 0; i < alphaSize; i++) {
      for (;;) {
        uint32_t v = showbits(s, 2);
        if (curr < 1 || curr > 20) return BZ_DATA_ERROR;
        if (v < 2) {
          skipbits(s, 1);
          break;
        } else {
          skipbits(s, 2);
          curr += 1 - (v & 1) * 2;
        }
      }
      length[i] = curr;
      if (curr > maxLen) maxLen = curr;
      if (curr < minLen) minLen = curr;
    }
    // Create the Huffman decoding tables for this group
    pp = 0;
    for (i = minLen; i <= maxLen; i++)
      for (j = 0; j < alphaSize; j++)
        if (length[j] == i) {
          sel->perm[pp] = j;
          pp++;
        };

    for (i = 0; i < BZ_MAX_CODE_LEN; i++) {
      sel->base[i]  = 0;
      sel->limit[i] = 0;
    }
    for (i = 0; i < alphaSize; i++)
      sel->base[length[i] + 1]++;

    for (i = 1; i < BZ_MAX_CODE_LEN; i++)
      sel->base[i] += sel->base[i - 1];

    vec = 0;
    for (i = minLen; i <= maxLen; i++) {
      vec += (sel->base[i + 1] - sel->base[i]);
      sel->limit[i] = vec - 1;
      vec <<= 1;
    }
    for (i = minLen + 1; i <= maxLen; i++)
      sel->base[i] = ((sel->limit[i - 1] + 1) << 1) - sel->base[i];

    sel->minLen = minLen;
  }

  // Now the MTF values

  EOB       = nInUse + 1;
  nblockMAX = 100000 * s->blockSize100k;

  for (i = 0; i <= 255; i++)
    s->unzftab[i] = 0;

  // MTF init
  {
    int32_t kk = MTFA_SIZE - 1;
    for (int32_t ii = 256 / MTFL_SIZE - 1; ii >= 0; ii--) {
      for (int32_t jj = MTFL_SIZE - 1; jj >= 0; jj--) {
        s->mtfa[kk--] = (uint8_t)(ii * MTFL_SIZE + jj);
      }
      s->mtfbase[ii] = kk + 1;
    }
  }
  // end MTF init

  nblock   = 0;
  groupNo  = -1;
  groupPos = 0;

  for (;;) {
    uint32_t es = 0;
    uint32_t N  = 1;
    uint32_t nextSym, nn, uc;
    for (;;) {
      uint32_t next32, zvec;
      int32_t zn;
      if (groupPos == 0) {
        if (++groupNo >= nSelectors) return BZ_DATA_ERROR;
        groupPos = BZ_G_SIZE;
        gSel     = &s->ht[s->selector[groupNo]];
      }
      groupPos--;
      next32 = next32bits(s);
      zn     = gSel->minLen;
      for (;;) {
        zvec = next32 >> (32u - (uint32_t)zn);
        if (zn > 20)  // the longest code
          return BZ_DATA_ERROR;
        if (zvec <= (uint32_t)gSel->limit[zn]) break;
        zn++;
      }
      skipbits(s, zn);
      zvec -= gSel->base[zn];
      if (zvec >= BZ_MAX_ALPHA_SIZE) return BZ_DATA_ERROR;
      nextSym = gSel->perm[zvec];
      if (nextSym > BZ_RUNB) break;
      es += N << nextSym;
      N <<= 1;
    }
    if (es > 0) {
      if (nblock + es > nblockMAX) return BZ_DATA_ERROR;
      uc = s->seqToUnseq[s->mtfa[s->mtfbase[0]]];
      s->unzftab[uc] += es;
      do {
        s->tt[nblock++] = uc;
      } while (--es);
    }

    if (nextSym == static_cast<uint32_t>(EOB)) break;

    if (nblock >= nblockMAX) return BZ_DATA_ERROR;
    nn = nextSym - 1;
    // uc = MTF ( nextSym-1 )
    if (nn < MTFL_SIZE) {
      // avoid general-case expense
      int32_t pp = s->mtfbase[0];
      uc         = s->mtfa[pp + nn];
      while (nn > 3) {
        int32_t z      = pp + nn;
        s->mtfa[(z)]   = s->mtfa[(z)-1];
        s->mtfa[(z)-1] = s->mtfa[(z)-2];
        s->mtfa[(z)-2] = s->mtfa[(z)-3];
        s->mtfa[(z)-3] = s->mtfa[(z)-4];
        nn -= 4;
      }
      while (nn > 0) {
        s->mtfa[(pp + nn)] = s->mtfa[(pp + nn) - 1];
        nn--;
      };
      s->mtfa[pp] = uc;
    } else {
      // general case
      int32_t lno = nn / MTFL_SIZE;
      int32_t off = nn % MTFL_SIZE;
      int32_t pp  = s->mtfbase[lno] + off;
      uc          = s->mtfa[pp];
      while (pp > s->mtfbase[lno]) {
        s->mtfa[pp] = s->mtfa[pp - 1];
        pp--;
      };
      s->mtfbase[lno]++;
      while (lno > 0) {
        s->mtfbase[lno]--;
        s->mtfa[s->mtfbase[lno]] = s->mtfa[s->mtfbase[lno - 1] + MTFL_SIZE - 1];
        lno--;
      }
      s->mtfbase[0]--;
      s->mtfa[s->mtfbase[0]] = uc;
      if (s->mtfbase[0] == 0) {
        int kk = MTFA_SIZE - 1;
        for (int ii = 256 / MTFL_SIZE - 1; ii >= 0; ii--) {
          for (int jj = MTFL_SIZE - 1; jj >= 0; jj--) {
            s->mtfa[kk] = s->mtfa[s->mtfbase[ii] + jj];
            kk--;
          }
          s->mtfbase[ii] = kk + 1;
        }
      }
    }
    uc = s->seqToUnseq[uc];
    s->unzftab[uc]++;
    s->tt[nblock++] = uc;
  }

  // Now we know what nblock is, we can do a better sanity check on s->origPtr.
  if (s->origPtr >= nblock) return BZ_DATA_ERROR;

  // compute the T^(-1) vector
  {
    int32_t prev  = s->unzftab[0];
    s->unzftab[0] = 0;
    for (i = 1; i < 256; i++) {
      int32_t tmp   = s->unzftab[i];
      s->unzftab[i] = prev + s->unzftab[i - 1];
      prev          = tmp;
    }

    for (i = 0; i < (int)nblock; i++) {
      int uc = (s->tt[i] & 0xff);
      s->tt[s->unzftab[uc]] |= (i << 8);
      s->unzftab[uc]++;
    }
  }

  s->save_nblock = nblock;

  // Verify the end-of-block signature: should be followed by another block or an end-of-stream
  // signature
  {
    uint8_t const* save_cur = s->cur;
    uint64_t save_bitbuf    = s->bitbuf;
    uint32_t save_bitpos    = s->bitpos;
    sig0                    = getbits(s, 24);
    sig1                    = getbits(s, 24);
    if (sig0 == 0x31'4159 && sig1 == 0x26'5359) {
      // Start of another block: restore bitstream location
      s->cur    = save_cur;
      s->bitbuf = save_bitbuf;
      s->bitpos = save_bitpos;
      return BZ_OK;
    } else if (sig0 == 0x17'7245 && sig1 == 0x38'5090) {
      // End-of-stream signature
      return BZ_STREAM_END;
    } else {
      return BZ_DATA_ERROR;
    }
  }
}

static void bzUnRLE(unbz_state_s* s)
{
  uint8_t* out    = s->out;
  uint8_t* outend = s->outend;

  int32_t rle_cnt           = s->save_nblock;
  int cprev                 = -1;
  std::vector<uint32_t>& tt = s->tt;
  uint32_t pos              = tt[s->origPtr] >> 8;
  int mask                  = ~0;

  s->nblock_used = rle_cnt + 1;

  while (rle_cnt > 0) {
    int c;

    rle_cnt--;
    pos = tt[pos];
    c   = (pos & 0xff);
    pos >>= 8;
    if (out < outend) { *out = c; }
    out++;
    mask  = (mask * 2 + (c != cprev)) & 7;
    cprev = c;
    if (!mask) {
      int run;
      if (--rle_cnt < 0) {
        printf("run split across blocks! (unsupported)\n");
        break;
      }
      pos = tt[pos];
      run = (pos & 0xff);
      pos >>= 8;
      for (int i = 0; i < run; i++) {
        if (out + i < outend) out[i] = c;
      }
      out += run;
      cprev = -1;
    }
  }
  s->out = out;
}

int32_t cpu_bz2_uncompress(
  uint8_t const* source, size_t sourceLen, uint8_t* dest, size_t* destLen, uint64_t* block_start)
{
  unbz_state_s s{};
  uint32_t v;
  int ret;
  size_t last_valid_block_in, last_valid_block_out;

  if (dest == nullptr || destLen == nullptr || source == nullptr || sourceLen < 12)
    return BZ_PARAM_ERROR;
  s.currBlockNo = 0;

  s.cur  = source;
  s.base = source;
  s.end =
    source + sourceLen - 4;  // We will not read the final combined CRC (last 4 bytes of the file)
  s.bitbuf = __builtin_bswap64(*reinterpret_cast<uint64_t const*>(source));
  s.bitpos = 0;

  s.out     = dest;
  s.outend  = dest + *destLen;
  s.outbase = dest;

  s.save_nblock = 0;

  v = getbits(&s, 24);
  if (v != (('B' << 16) | ('Z' << 8) | 'h')) return BZ_DATA_ERROR_MAGIC;

  v = getbits(&s, 8) - '0';
  if (v < 1 || v > 9) return BZ_DATA_ERROR_MAGIC;
  s.blockSize100k = v;

  last_valid_block_in  = 0;
  last_valid_block_out = 0;

  if (block_start) {
    uint64_t bit_offs = *block_start;
    if (bit_offs > 32)  // 32-bits are used for the file header (0..32 is considered as first block)
    {
      s.cur    = source + (size_t)(bit_offs >> 3);
      s.bitpos = (uint32_t)(bit_offs & 7);
      if (s.cur + 8 > s.end) return BZ_PARAM_ERROR;
      s.bitbuf = __builtin_bswap64(*reinterpret_cast<uint64_t const*>(s.cur));
    }
  }

  s.tt.resize(s.blockSize100k * 100000);

  do {
    last_valid_block_in  = ((s.cur - s.base) << 3) + (s.bitpos);
    last_valid_block_out = s.out - s.outbase;

    ret = bz2_decompress_block(&s);
    if (ret == BZ_OK || ret == BZ_STREAM_END) {
      bzUnRLE(&s);
      if (s.nblock_used != s.save_nblock + 1 || s.out > s.outend) {
        ret = (s.out < s.outend) ? BZ_UNEXPECTED_EOF : BZ_OUTBUFF_FULL;
      }
    }
  } while (ret == BZ_OK);

  if (ret == BZ_STREAM_END) {
    // normal termination
    last_valid_block_in  = ((s.cur - s.base) << 3) + (s.bitpos);
    last_valid_block_out = s.out - s.outbase;
    ret                  = BZ_OK;
  }

  *destLen = last_valid_block_out;
  if (block_start) { *block_start = last_valid_block_in; }

  return ret;
}

}  // namespace io
}  // namespace cudf
