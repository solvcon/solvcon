/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/device/cuda/fft.hpp>
#include <solvcon/device/cuda/fft_impl.hpp>

#include <cstddef>
#include <format>
#include <stdexcept>

namespace solvcon
{

namespace device
{

namespace cuda
{

namespace
{

// The Bluestein path pads to the next power of two at or above 2n-1 and
// carries that length in an int, so cap n where the padded length still
// fits. The cap is far above any length a device can hold.
constexpr size_t MAX_SIZE = size_t(1) << 29;

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
    if (in.size() > MAX_SIZE)
    {
        throw std::invalid_argument(std::format("CUDA FFT size {} exceeds the supported limit {}", in.size(), MAX_SIZE));
    }

    impl(in.data(), out.data(), static_cast<int>(in.size()));
}

} /* end namespace */

bool available()
{
    // Probed once and latched: the device set does not change under a
    // running process, and the probe initializes the CUDA runtime.
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
