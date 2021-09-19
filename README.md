# Parallel NanoJPEG Decoder Using Thread Pool 

This is a program that demonstrates how to decode a JPEG image in parallel using
the error resilience features in the standard.

When a JPEG image contains DRI and RST markers, we could take advantage of them
to split the bitstream into multiple individually-decodable segments, and decode
segments in parallel.

To simplify thread management, we could use any decent thread-pool implementation,
so that each segment is decoded in a dedicated task.

The source code is based on
[NanoJPEG](https://keyj.emphy.de/nanojpeg/), which is in C and not thread-safe.

There are many thread-pool implementation in source form, I picked
[rvases/thread_pool](https://github.com/rvaser/thread_pool) and added the project
as a submodule.

## Limitations

* It will not run faster if a JPEG file does not use the RST feature
* It will throw exceptions if there are tables updated after SOS marker, it
should be straight forward to create a new copy of context and pass it to new
tasks, but I chose not to implement for a demonstration program.
* Output is in the original coding color space in planar format, which is most
likely in YCbCr4:2:0 or YCbCr4:4:4. An raw YUV buffer viewer can be used to
view the content.  Most of the times, I am interested in feeding the dump to a
video encoder.

## Prerequisites

* CMake is used to simplify building process
* Boost is used for parsing command line options, it can easily be removed
* Compiler that supports C++17 features

## Build

```bash
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
ninja
```

## Run
```bash
./pnjpeg -i input.jpg
```
