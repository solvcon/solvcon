#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * CUDA fast Fourier transform entry points, implemented by in-house
 * kernels. The declarations are always visible; a CUDA build defines them
 * in fft.cpp (backed by the nvcc unit fft_kernel.cu) and any other build
 * defines them in fft_stub.cpp, so callers never need a build-time macro.
 *
 * @ingroup group_numerics
 */

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/math/Complex.hpp>

namespace solvcon
{

namespace device
{

namespace cuda
{

/**
 * Report whether a CUDA device is usable at runtime. Always false in a
 * build without CUDA support.
 */
bool available();

/**
 * Forward FFT of a one-dimensional complex signal on the CUDA device.
 * Throws std::runtime_error when CUDA is unusable or a CUDA call fails.
 */
void fft(SimpleArray<Complex<float>> const & in, SimpleArray<Complex<float>> & out);
void fft(SimpleArray<Complex<double>> const & in, SimpleArray<Complex<double>> & out);

} /* end namespace cuda */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
