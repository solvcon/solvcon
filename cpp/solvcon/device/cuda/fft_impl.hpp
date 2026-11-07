#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Raw-pointer interface between the host-compiled CUDA FFT adapter
 * (fft.cpp) and the nvcc-compiled kernel unit (fft_kernel.cu). The
 * interface names no solvcon or CUDA type so the kernel unit stays free
 * of the C++23 core headers, which nvcc may not parse.
 *
 * @ingroup group_numerics
 */

namespace solvcon
{

namespace device
{

namespace cuda
{

namespace detail
{

bool device_available();

/**
 * Forward complex-to-complex FFT of n interleaved (real, imag) pairs of
 * float or double. Throws std::runtime_error on any CUDA failure.
 */
void fft_float(void const * in, void * out, int n);
void fft_double(void const * in, void * out, int n);

} /* end namespace detail */

} /* end namespace cuda */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
