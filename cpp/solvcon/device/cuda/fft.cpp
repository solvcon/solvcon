/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/device/cuda/fft.hpp>
#include <solvcon/device/cuda/fft_impl.hpp>

#include <format>
#include <limits>
#include <stdexcept>

namespace solvcon
{

namespace device
{

namespace cuda
{

namespace
{

template <typename T>
void fft_checked(SimpleArray<Complex<T>> const & in, SimpleArray<Complex<T>> & out, void (*impl)(void const *, void *, int))
{
    static_assert(is_std_complex_layout_compatible_v<T> && sizeof(Complex<T>) == 2 * sizeof(T));

    if (!available())
    {
        throw std::runtime_error("CUDA FFT is not available: no usable CUDA device");
    }
    if (in.size() != out.size())
    {
        throw std::invalid_argument(std::format(
            "CUDA FFT input size {} does not match output size {}", in.size(), out.size()));
    }
    if (in.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument(std::format("CUDA FFT size {} exceeds the supported limit", in.size()));
    }

    impl(in.data(), out.data(), static_cast<int>(in.size()));
}

} /* end namespace */

bool available()
{
    static bool const ok = detail::device_available();
    return ok;
}

void fft(SimpleArray<Complex<float>> const & in, SimpleArray<Complex<float>> & out)
{
    fft_checked(in, out, detail::fft_float);
}

void fft(SimpleArray<Complex<double>> const & in, SimpleArray<Complex<double>> & out)
{
    fft_checked(in, out, detail::fft_double);
}

} /* end namespace cuda */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
