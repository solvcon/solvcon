/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/device/cuda/fft.hpp>

#include <stdexcept>

namespace solvcon
{

namespace device
{

namespace cuda
{

[[noreturn]] static void unavailable()
{
    throw std::runtime_error("CUDA FFT is not available; rebuild with BUILD_CUDA=ON");
}

bool available()
{
    return false;
}

void fft(SimpleArray<Complex<float>> const &, SimpleArray<Complex<float>> &)
{
    unavailable();
}

void fft(SimpleArray<Complex<double>> const &, SimpleArray<Complex<double>> &)
{
    unavailable();
}

} /* end namespace cuda */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
