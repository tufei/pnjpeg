// Parallel NanoJPEG -- Copyright (c) 2021 Ngyu-Phee Yen <yyuan@gmx.com>
// NanoJPEG -- KeyJ's Tiny Baseline JPEG Decoder
// version 1.3.5 (2016-11-14)
// Copyright (c) 2009-2016 Martin J. Fiedler <martin.fiedler@gmx.net>
// published under the terms of the MIT license
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// Include statements.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <thread>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>

#include "thread_pool/thread_pool.hpp"

using namespace std;
using namespace thread_pool;
using namespace boost::program_options;

struct block_context
{
    const uint8_t *pos;
    int size;
    int buf;
    int bufbits;
    array<int, 3> dcpred;
    array<int, 64> block;
};

struct vlc_code
{
    uint8_t bits;
    uint8_t code;
};

struct component
{
    int cid;
    int ssx;
    int ssy;
    int width;
    int height;
    int stride;
    int qtsel;
    int actabsel;
    int dctabsel;
    int dcpred;
    uint8_t *pixels;
};

struct context
{
    std::vector<uint8_t> bitstream;
    int width;
    int height;
    int mbwidth;
    int mbheight;
    int mbsizex;
    int mbsizey;
    int ncomp;
    array<component, 3> comp;
    int qtused;
    int qtavail;
    int rstinterval;
    array<array<uint8_t, 64>, 4> qtab;
    array<array<vlc_code, 64 * 1024>, 5> vlctab;
};

static const char njZZ[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18,
11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

inline static uint8_t clip(const int x)
{
    return x < 0 ? 0 : (x > 0xFF ? 0xFF : static_cast<uint8_t>(x));
}

#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565

inline static void row_idct(int *blk)
{
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;

    if (!((x1 = blk[4] << 11) | (x2 = blk[6]) | (x3 = blk[2]) |
          (x4 = blk[1]) | (x5 = blk[7]) | (x6 = blk[5]) | (x7 = blk[3])))
    {
        blk[0] = blk[1] = blk[2] = blk[3] = blk[4] =
        blk[5] = blk[6] = blk[7] = blk[0] << 3;
        return;
    }

    x0 = (blk[0] << 11) + 128;
    x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    blk[0] = (x7 + x1) >> 8;
    blk[1] = (x3 + x2) >> 8;
    blk[2] = (x0 + x4) >> 8;
    blk[3] = (x8 + x6) >> 8;
    blk[4] = (x8 - x6) >> 8;
    blk[5] = (x0 - x4) >> 8;
    blk[6] = (x3 - x2) >> 8;
    blk[7] = (x7 - x1) >> 8;
}

inline static void col_idct(const int *blk,
                            uint8_t *out, int stride)
{
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;

    if (!((x1 = blk[8*4] << 8) | (x2 = blk[8*6]) | (x3 = blk[8*2]) |
          (x4 = blk[8*1]) | (x5 = blk[8*7]) | (x6 = blk[8*5]) | (x7 = blk[8*3])))
    {
        x1 = clip(((blk[0] + 32) >> 6) + 128);
        for (x0 = 8; x0; --x0)
        {
            *out = static_cast<uint8_t>(x1);
            out += stride;
        }
        return;
    }

    x0 = (blk[0] << 8) + 8192;
    x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    *out = clip(((x7 + x1) >> 14) + 128);  out += stride;
    *out = clip(((x3 + x2) >> 14) + 128);  out += stride;
    *out = clip(((x0 + x4) >> 14) + 128);  out += stride;
    *out = clip(((x8 + x6) >> 14) + 128);  out += stride;
    *out = clip(((x8 - x6) >> 14) + 128);  out += stride;
    *out = clip(((x0 - x4) >> 14) + 128);  out += stride;
    *out = clip(((x3 - x2) >> 14) + 128);  out += stride;
    *out = clip(((x7 - x1) >> 14) + 128);
}

inline static std::pair<uint8_t, int> seek_rst(const uint8_t *bitstream,
                                               int size)
{
    uint8_t rst = 0xFF;
    const uint8_t *pos = bitstream;
    while (0 < size)
    {
        if (0xFF == pos[0] && 0xD0 == (pos[1] & 0xF8))
        {
            rst = pos[1] & 0x07;
            break;
        }
        pos++; size--;
    }
    if (0xFF == rst)
    {
        if (0xFF == pos[-2] && 0xD9 == pos[-1])
        {
            return std::make_pair(rst, pos - bitstream - 2);
        }
        else
        {
            throw length_error("data too short to decode RST");
        }
    }

    return std::make_pair(rst, pos - bitstream); 
}

inline static uint16_t decode_16b(const uint8_t *bitstream)
{
    return (bitstream[0] << 8) | bitstream[1];
}

inline static uint16_t decode_len(const uint8_t *&bitstream, const int size)
{
    if (2 > size)
    {
        throw length_error("data too short to decode length");
    }

    uint16_t length = decode_16b(bitstream);

    if (length > size)
    {
        throw length_error("data too short to decode content");
    }

    bitstream += 2;

    return length - 2;
}

inline static int show_bits(block_context *bc, int bits)
{
    uint8_t newbyte;
    if (!bits) return 0;

    while (bc->bufbits < bits)
    {
        if (bc->size <= 0)
        {
            bc->buf = (bc->buf << 8) | 0xFF;
            bc->bufbits += 8;
            continue;
        }
        newbyte = *bc->pos++;
        bc->size--;
        bc->bufbits += 8;
        bc->buf = (bc->buf << 8) | newbyte;
        if (newbyte == 0xFF)
        {
            if (bc->size)
            {
                uint8_t marker = *bc->pos++;
                bc->size--;
                switch (marker)
                {
                    case 0x00:
                    case 0xFF:
                        break;
                    case 0xD9: bc->size = 0; break;
                    default:
                        if ((marker & 0xF8) != 0xD0)
                        {
                            throw runtime_error("invalid semantics in decoding VLC");
                        }
                        else
                        {
                            bc->buf = (bc->buf << 8) | marker;
                            bc->bufbits += 8;
                        }
                }
            }
            else
            {
                throw runtime_error("invalid semantics in decoding VLC");
            }
        }
    }
    return (bc->buf >> (bc->bufbits - bits)) & ((1 << bits) - 1);
}

inline static void skip_bits(block_context *bc, int bits)
{
    if (bc->bufbits < bits) show_bits(bc, bits);
    bc->bufbits -= bits;
}

inline static int get_bits(block_context *bc, int bits)
{
    int res = show_bits(bc, bits);
    skip_bits(bc, bits);

    return res;
}

static int get_vlc(block_context *bc, const vlc_code *vlc, uint8_t *code)
{
    int value = show_bits(bc, 16);
    int bits = vlc[value].bits;
    if (!bits)
    {
        throw runtime_error("invalid semantics in decoding VLC");
        return 0;
    }
    skip_bits(bc, bits);
    value = vlc[value].code;
    if (code) *code = static_cast<uint8_t>(value);
    bits = value & 15;
    if (!bits) return 0;

    value = get_bits(bc, bits);
    if (value < (1 << (bits - 1))) value += ((-1) << bits) + 1;
    return value;
}

inline static void decode_blk(const context &ctx, const int i,
                              block_context *bc, uint8_t *out)
{
    uint8_t code = 0;
    int value, coef = 0;
    const component &c = ctx.comp[i];

    bc->block.fill(0);
    bc->dcpred[i] += get_vlc(bc, &ctx.vlctab[c.dctabsel][0], nullptr);
    bc->block[0] = bc->dcpred[i] * ctx.qtab[c.qtsel][0];

    do
    {
        value = get_vlc(bc, &ctx.vlctab[c.actabsel][0], &code);
        if (!code) break;  // EOB
        if (!(code & 0x0F) && (code != 0xF0))
        {
            throw runtime_error("invalid semantics in decoding BLK");
        }
        coef += (code >> 4) + 1;
        if (coef > 63)
        {
            throw runtime_error("invalid semantics in decoding BLK");
        }
        bc->block[njZZ[coef]] = value * ctx.qtab[c.qtsel][coef];
    } while (coef < 63);

    for (coef = 0; coef < 64; coef += 8) row_idct(&bc->block[coef]);
    for (coef = 0; coef < 8; ++coef)
    {
        col_idct(&bc->block[coef], &out[coef], c.stride);
    }
}

void decode_blks(const uint8_t *bitstream, int size, const context &ctx,
                 const int start_block, const int num_blocks)
{
    int mby = start_block / ctx.mbwidth;
    int mbx = start_block - mby * ctx.mbwidth;

    std::unique_ptr<block_context> bc = std::make_unique<block_context>();

    std::memset(bc.get(), 0, sizeof(block_context));

    bc->pos = bitstream;
    bc->size = size;

    for (int b = 0; b < num_blocks; ++b)
    {
        for (int i = 0; i < ctx.ncomp; ++i)
        {
            const component &c = ctx.comp[i];
            for (int sby = 0; sby < c.ssy; ++sby)
            {
                for (int sbx = 0; sbx < c.ssx; ++sbx)
                {
                    decode_blk(ctx, i, bc.get(),
                               &c.pixels[((mby * c.ssy + sby) * c.stride + mbx * c.ssx + sbx) << 3]);
                }
            }
        }
        if (++mbx >= ctx.mbwidth)
        {
            mbx = 0;
            if (++mby >= ctx.mbheight) break;
        }
    }
}

void decode_dri(const uint8_t *bitstream, int size, context &ctx)
{
    uint32_t length = decode_len(bitstream, size);
    if (2 > length)
    {
        throw length_error("data too short to decode DRI");
    }
    ctx.rstinterval = decode_16b(bitstream);

    //cout << "DRI done\n";
}

void decode_dqt(const uint8_t *bitstream, int size, context &ctx)
{
    uint32_t length = decode_len(bitstream, size);

    ctx.qtavail = 0;

    for (auto &t : ctx.qtab) t.fill(0);

    while (length >= 65)
    {
        int i = bitstream[0];
        if (i & 0xFC)
        {
            throw runtime_error("invalid semantics in decoding DQT");
        }
        ctx.qtavail |= 1 << i;
        std::copy_n(&bitstream[i + 1], 64, ctx.qtab[i].begin());
        bitstream += 65; length -= 65;
    }
    if (length)
    {
        throw length_error("extra data after decoding DQT");
    }

    //cout << "DQT done\n";
}

void decode_sos(const uint8_t *bitstream, int size, context &ctx)
{
    uint32_t length = decode_len(bitstream, size);

    if (4 + ctx.ncomp * 2 > length)
    {
        throw length_error("data too short to decode SOS");
    }
    if (bitstream[0] != ctx.ncomp)
    {
        throw runtime_error("invalid semantics in decoding SOS");
    }
    bitstream += 1; length -= 1;
    for (int i = 0; i < ctx.ncomp; ++i)
    {
        component &c = ctx.comp[i];
        if (c.cid != bitstream[0] || (bitstream[1] & 0xEE))
        {
            throw runtime_error("invalid semantics in decoding SOS");
        }
        c.dctabsel = bitstream[1] >> 4;
        c.actabsel = (bitstream[1] & 1) | 2;
        bitstream += 2; length -= 2;
    }
    if (bitstream[0] || bitstream[1] != 63 || bitstream[2])
    {
        throw runtime_error("invalid semantics in decoding SOS");
    }

    //cout << "SOS done\n";
}

void decode_dht(const uint8_t *bitstream, int size, context &ctx)
{
    uint32_t length = decode_len(bitstream, size);

    std::array<uint8_t, 16> counts;

    for (auto &t : ctx.vlctab) t.fill({0, 0});

    while (length >= 17)
    {
        int i = bitstream[0];
        if (i & 0xEC)
        {
            throw runtime_error("invalid semantics in decoding DHT");
        }
        if (i & 0x02)
        {
            throw runtime_error("unsupported semantics in decoding DHT");
        }
        i = (i | (i >> 3)) & 3;  // combined DC/AC + tableid value

        std::copy_n(&bitstream[1], 16, counts.begin());

        bitstream += 17; length -= 17;

        vlc_code *vlc = ctx.vlctab[i].data();
        int spread = 65536;
        int remain = spread;
        for (int codelen = 1; codelen <= 16; ++codelen)
        {
            spread >>= 1;
            int currcnt = counts[codelen - 1];
            if (!currcnt) continue;

            if (length < currcnt)
            {
                throw length_error("data too short to decode DHT");
            }
            remain -= currcnt << (16 - codelen);
            if (remain < 0)
            {
                throw length_error("data too short to decode DHT");
            }
            for (i = 0; i < currcnt; ++i)
            {
                uint8_t code = bitstream[i];
                for (int j = spread; j; --j, ++vlc)
                {
                    *vlc = {static_cast<uint8_t>(codelen), code};
                }
            }
            bitstream += currcnt; length -= currcnt;
        }
        while (remain--)
        {
            *vlc = {0, 0};
            ++vlc;
        }
    }
    if (length)
    {
        throw length_error("extra data after decoding DHT");
    }
    //cout << "DHT done\n";
}

void decode_sof(const uint8_t *bitstream, int size, context &ctx)
{
    uint32_t length = decode_len(bitstream, size);

    if (length < 9)
    {
        throw length_error("data too short to decode SOF");
    }
    if (bitstream[0] != 8)
    {
        throw runtime_error("unsupported semantics in decoding SOF");
    }

    ctx.height = decode_16b(&bitstream[1]);
    ctx.width = decode_16b(&bitstream[3]);
    if (!ctx.width || !ctx.height)
    {
        throw runtime_error("invalid semantics in decoding SOF");
    }
    ctx.ncomp = bitstream[5];
    bitstream += 6; length -= 6;

    switch (ctx.ncomp)
    {
        case 1:
        case 3:
            break;
        default:
            throw runtime_error("unsupported semantics in decoding SOF");
            break;
    }

    if (length < ctx.ncomp * 3)
    {
        throw length_error("data too short to decode SOF");
    }

    int qtused = 0;
    int ssxmax = 0;
    int ssymax = 0;
    component *c = nullptr;
    for (int i = 0; i < ctx.ncomp; ++i)
    {
        c = &ctx.comp[i];

        c->cid = bitstream[0];
        if (!(c->ssx = bitstream[1] >> 4))
        {
            throw runtime_error("invalid semantics in decoding SOF");
        }
        if (c->ssx & (c->ssx - 1))
        {
            throw runtime_error("unsupported semantics in decoding SOF");
        }
        if (!(c->ssy = bitstream[1] & 15))
        {
            throw runtime_error("invalid semantics in decoding SOF");
        }
        if (c->ssy & (c->ssy - 1))
        {
            throw runtime_error("unsupported semantics in decoding SOF");
        }
        if ((c->qtsel = bitstream[2]) & 0xFC)
        {
            throw runtime_error("invalid semantics in decoding SOF");
        }
        bitstream += 3; length -= 3;
        ctx.qtused |= 1 << c->qtsel;
        if (c->ssx > ssxmax) ssxmax = c->ssx;
        if (c->ssy > ssymax) ssymax = c->ssy;
    }

    if (ctx.ncomp == 1)
    {
        c = &ctx.comp[0];
        c->ssx = c->ssy = ssxmax = ssymax = 1;
    }
    ctx.mbsizex = ssxmax << 3;
    ctx.mbsizey = ssymax << 3;
    ctx.mbwidth = (ctx.width + ctx.mbsizex - 1) / ctx.mbsizex;
    ctx.mbheight = (ctx.height + ctx.mbsizey - 1) / ctx.mbsizey;
    for (int i = 0; i < ctx.ncomp; ++i)
    {
        c = &ctx.comp[i];
        c->width = (ctx.width * c->ssx + ssxmax - 1) / ssxmax;
        c->height = (ctx.height * c->ssy + ssymax - 1) / ssymax;
        c->stride = ctx.mbwidth * c->ssx << 3;
        if (((c->width < 3) && (c->ssx != ssxmax)) ||
            ((c->height < 3) && (c->ssy != ssymax)))
        {
            throw runtime_error("invalid semantics in decoding SOF");
        }
    }

    //cout << "SOF done\n";
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    options_description desc
    (
        "\nThread-pool based NanoJPEG.\n"
        "\nAllowed arguments"
    );

    // Define command line arguments using either formats:
    //
    //     (“long-name,short-name”, “Description of argument”)
    //     for flag values or
    //
    //     (“long-name,short-name”, <data-type>, 
    //     “Description of argument”) arguments with values
    //
    // Remember that arguments with values may be multi-values
    // and must be vectors
    desc.add_options()
    ("help", "Print help message.")
    ("threads,t", value<int>(), "Number of threads in pool, default: 0")
    ("dump-width,w", value<int>(),
     "dumped width default: min(width, 1920)")
    ("dump-height,h", value<int>(),
     "dumped height default: min(height, 1080)")
    ("ow", value<int>(),
     "dumped width offset default: 0")
    ("oh", value<int>(),
     "dumped height offset default: 0")
    ("output-file,o", value<string>(),
     "Path to the output image, default: out.yuv")
    ("input-file,i", value<string>(), "Path to the input image.");

    // Map positional parameters to their tag valued types 
    // (e.g. –input-file parameters)
    positional_options_description p;
    p.add("input-file", -1);

    // Parse the command line catching and displaying any 
    // parser errors
    variables_map vm;
    try
    {
        store(command_line_parser(argc, argv).options(desc)
                                             .positional(p).run(), vm);
        notify(vm);
    }
    catch (exception &e)
    {
        cout << '\n' << e.what() << '\n';
        cout << desc << '\n';
    }

    // Display help text when requested
    if (vm.count("help"))
    {
        cout << "–help specified\n";
        cout << desc << '\n';
        return 0;
    }

    int threads = 0;
    if (vm.count("threads"))
    {
        threads = vm["threads"].as<int>();
        cout << "–threads specified with value = " << threads << '\n';
    }

    string out_filename("out.yuv");
    if (vm.count("output-file"))
    {
        out_filename = vm["output-file"].as<string>();
        cout << "–output-file specified with value = " << out_filename << '\n';
    }

    string in_filename;
    if (vm.count("input-file"))
    {
        in_filename = vm["input-file"].as<string>();
        cout << "–input-file specified with value = " << in_filename << '\n';
    }

    ThreadPool thread_pool{threads ? threads : thread::hardware_concurrency()};

    if (in_filename.empty())
    {
        throw invalid_argument("input file path not given");
    }

    context jpg_ctx;
    memset(&jpg_ctx, 0, sizeof(context));

    std::ifstream ifs(in_filename, std::ios::binary | std::ios::ate);
    if (false == ifs.is_open())
    {
        throw system_error(error_code(), "cannot open input file");
    }
    const auto fsize = ifs.tellg() & 0x7FFFFFF;

    jpg_ctx.bitstream.resize(fsize);
    ifs.seekg(std::ios::beg);
    ifs.read(reinterpret_cast<char *>(jpg_ctx.bitstream.data()), fsize);
    const uint8_t *bitstream = jpg_ctx.bitstream.data();

    uint32_t length = fsize;

    if (2 > length)
    {
        throw length_error("data too short to decode JPEG");
    }
    if ((bitstream[0] ^ 0xFF) | (bitstream[1] ^ 0xD8))
    {
        throw runtime_error("invalid semantics in decoding JPEG");
    }
    bitstream += 2; length -= 2;

    uint32_t bitmask = 0;
    vector<future<void>> futures;

    bool no_rst = false;
    while (length && 15 != bitmask)
    {
        if (2 > length || (bitstream[0] != 0xFF))
        {
            throw length_error("data too short to decode SOF");
        }
        bitstream += 2; length -= 2;
        int marker_len = decode_16b(bitstream);
        switch (bitstream[-1])
        {
            case 0xC0:
                cout << "SOF buffer = "
                     << static_cast<const void *>(bitstream)
                     << ", length = " << marker_len << '\n';
                futures.emplace_back(
                    thread_pool.Submit(decode_sof, bitstream, marker_len,
                                       std::ref(jpg_ctx)));
                bitmask |= 1;
                break;
            case 0xC4:
                cout << "DHT buffer = "
                     << static_cast<const void *>(bitstream)
                     << ", length = " << marker_len << '\n';
                futures.emplace_back(
                    thread_pool.Submit(decode_dht, bitstream, marker_len,
                                       std::ref(jpg_ctx)));
                bitmask |= 2;
                break;
            case 0xDA:
                cout << "No DRI/RST in file\n";
                no_rst = true;
                bitstream -= 2; length += 2;
                break;
            case 0xDB:
                cout << "DQT buffer = "
                     << static_cast<const void *>(bitstream)
                     << ", length = " << marker_len << '\n';
                futures.emplace_back(
                    thread_pool.Submit(decode_dqt, bitstream, marker_len,
                                       std::ref(jpg_ctx)));
                bitmask |= 4;
                break;
            case 0xDD:
                cout << "DRI buffer = "
                     << static_cast<const void *>(bitstream)
                     << ", length = " << marker_len << '\n';
                futures.emplace_back(
                    thread_pool.Submit(decode_dri, bitstream, marker_len,
                                       std::ref(jpg_ctx)));
                bitmask |= 8;
                break;
            case 0xFE:
                break;
            default:
                if ((bitstream[-1] & 0xF0) != 0xE0)
                {
                    throw runtime_error("unsupported semantics in decoding JPEG");
                }
                break;
        }

        if (no_rst) break;

        bitstream += marker_len; length -= marker_len;
    }

    for (const auto &f : futures)
    {
        f.wait();
    }

    array<unique_ptr<uint8_t []>, 3> out_buffer;
    for (int i = 0; i < jpg_ctx.ncomp; ++i)
    {
        component &c = jpg_ctx.comp[i];
        out_buffer[i] = make_unique<uint8_t[]>(c.stride * jpg_ctx.mbheight *
                                               c.ssy << 3);
        c.pixels = out_buffer[i].get();
    }

#if 1
    cout << "RST interval = " << jpg_ctx.rstinterval << '\n';
    cout << "width = " << jpg_ctx.width << ", "
         << "height = " << jpg_ctx.height << '\n';
    cout << "mbwidth = " << jpg_ctx.mbwidth << ", "
         << "mbheight = " << jpg_ctx.mbheight << '\n';

    for (int i = 0; i < jpg_ctx.ncomp; ++i)
    {
        const component &c = jpg_ctx.comp[i];
        cout << "Component " << i << ":\n";
        cout << "\tssx = " << c.ssx << ", ssy = " << c.ssy << '\n';
        cout << "\tw = " << c.width << ", h = " << c.height << ", "
             << "stride = " << c.stride << '\n';
        cout << "\tqtsel = " << c.qtsel << ", "
             << "acsel = " << c.actabsel << ", "
             << "dcsel = " << c.dctabsel << '\n';
        cout << "\tbuffer = " << static_cast<void *>(c.pixels) << '\n';
    }
#endif

    futures.resize(0);
    bitmask = 0;
    while (length && 1 != bitmask)
    {
        if (2 > length || (bitstream[0] != 0xFF))
        {
            throw length_error("data too short to decode SOS");
        }
        bitstream += 2; length -= 2;
        const int marker_len = decode_16b(bitstream);
        switch (bitstream[-1])
        {
            case 0xDA:
                cout << "SOS buffer = "
                     << static_cast<const void *>(bitstream)
                     << ", length = " << marker_len << '\n';
                futures.emplace_back(
                    thread_pool.Submit(decode_sos, bitstream, marker_len,
                                       std::ref(jpg_ctx)));
                bitmask = 1;
                break;
            case 0xC0:
            case 0xC4:
            case 0xDB:
            case 0xDD:
                throw runtime_error("unsupported semantics in decoding JPEG");
                break;
            case 0xFE:
                break;
            default:
                if ((bitstream[-1] & 0xF0) != 0xE0)
                {
                    throw runtime_error("unsupported semantics in decoding JPEG");
                }
                break;
        }
        bitstream += marker_len; length -= marker_len;
    }

    for (const auto &f : futures)
    {
        f.wait();
    }

#if 0
    for (auto &t : jpg_ctx.vlctab)
    {
        for (auto &v : t)
        {
            cout << "bits = " << static_cast<uint32_t>(v.bits)
                 << ", code = " << static_cast<uint32_t>(v.code) << '\n';
        }
    }

    for (auto &qt : jpg_ctx.qtab)
    {
        cout << '\n';
        for (auto &q : qt) cout << static_cast<uint32_t>(q) << ", ";
        cout << '\n';
    }
#endif

    if (no_rst)
    {
        int num_blocks = jpg_ctx.mbheight * jpg_ctx.mbwidth;
        decode_blks(bitstream, length, jpg_ctx, 0, num_blocks);
    }
    else
    {
        int mbxy = 0;
        int curr_rst = 7;

        futures.resize(0);
        while (length)
        {
            auto blocks = seek_rst(bitstream, length);
            int num_blocks = ((8 + blocks.first - curr_rst) & 7) *
                             jpg_ctx.rstinterval;

            if (0xFF == blocks.first)
            {
                num_blocks = jpg_ctx.mbwidth * jpg_ctx.mbheight - mbxy;
            }

#if 0
            format_to(cout, "BLOCKS buffer = {1}, length = {2}\n",
                      bitstream, blocks.second) << flush;
            format_to(cout, "BLOCKS start_mb = {1}, length_mb = {2}\n",
                      mbxy, num_blocks) << flush;
#endif

            futures.emplace_back(
                thread_pool.Submit(decode_blks, bitstream, blocks.second,
                                   std::cref(jpg_ctx), mbxy, num_blocks));
            curr_rst = blocks.first;
            mbxy += num_blocks;

            bitstream += blocks.second + 2;
            length -= blocks.second + 2;
            if (0xFF == blocks.first)
            {
                cout << "EOI reached\n";
            }
        }

        for (const auto &f : futures)
        {
            f.wait();
        }
    }

    std::ofstream ofs(out_filename, std::ios::binary);
    if (false == ofs.is_open())
    {
        throw system_error(error_code(), "cannot open output file");
    }

    int width = min(jpg_ctx.width, 1920);
    int height = min(jpg_ctx.height, 1080);
    if (vm.count("dump-width"))
    {
        width = vm["dump-width"].as<int>();
    }
    if (vm.count("dump-height"))
    {
        height = vm["dump-height"].as<int>();
    }

    int offset_w = 0;
    int offset_h = 0;
    if (vm.count("ow"))
    {
        offset_w = vm["ow"].as<int>() & ~1;
    }
    if (vm.count("oh"))
    {
        offset_h = vm["oh"].as<int>() & ~1;
    }

    if (offset_w + width > jpg_ctx.width)
    {
        throw invalid_argument("dumped region exceeds width");
    }
    if (offset_h + height > jpg_ctx.height)
    {
        throw invalid_argument("dumped region exceeds height");
    }

    for (int i = 0; i < jpg_ctx.ncomp; ++i)
    {
        component &c = jpg_ctx.comp[i];
        const int scale_w = jpg_ctx.comp[0].ssx / c.ssx;
        const int scale_h = jpg_ctx.comp[0].ssy / c.ssy;
        for (int j = 0; j < (height + scale_h - 1) / scale_h; ++j)
        {
            const int start_offset = (offset_h / scale_h + j) * c.stride +
                                     offset_w / scale_w;
            ofs.write(reinterpret_cast<const char *>(&c.pixels[start_offset]),
                      (width + scale_w - 1) / scale_w);
        }
    }
    ofs.close();
}

