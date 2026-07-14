#pragma once

/**
 * @file
 * Fourier transform algorithms (FFT, inverse FFT, and DFT) for complex
 * SimpleArray data.
 *
 * @ingroup group_numerics
 */

#include <solvcon/math/math.hpp>
#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace detail
{

size_t bit_reverse(size_t n, size_t bits);
size_t next_power_of_two(size_t n);
template <template <typename> class T1, typename T2>
void fft_bluestein(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out);

template <template <typename> class T1, typename T2>
void fft_radix_2(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
{
    size_t const N = in.size();
    const auto bits = static_cast<unsigned int>(std::log2(N));

    // bit reversed reordering
    for (size_t i = 0; i < N; ++i)
    {
        out[detail::bit_reverse(i, bits)] = in[i];
    }

    // Cooly-Tukey FFT algorithm, radix-2
    for (size_t size = 2; size <= N; size *= 2)
    {
        size_t const half_size = size / 2;
        T2 angle_inc = -2.0 * pi<T2> / static_cast<T2>(size);

        for (size_t i = 0; i < N; i += size)
        {
            for (size_t k = 0; k < half_size; ++k)
            {
                // Twiddle factor = exp(-2 * pi * i * k / N)
                T2 angle = angle_inc * k;
                T1<T2> const twiddle_factor{std::cos(angle), std::sin(angle)};

                T1<T2> const even(out[i + k]);
                T1<T2> const odd(out[i + k + half_size] * twiddle_factor);

                out[i + k] = even + odd;
                out[i + k + half_size] = even - odd;
            }
        }
    }
}

} /* end namespace detail */

/**
 * Discrete Fourier transform of complex-valued arrays.
 *
 * The static methods operate on a SimpleArray of complex elements
 * (T1<T2>, for example a complex type over double) holding one signal.
 * fft() computes the forward transform, dispatching to a radix-2
 * Cooley-Tukey step when the length N is a power of two and to the
 * Bluestein algorithm otherwise. ifft() computes the inverse by
 * conjugating the input, reusing fft(), and scaling by 1/N. dft()
 * evaluates the direct O(N^2) sum. The forward transform uses the
 * twiddle factor exp(-2 * pi * i * k / N).
 *
 * @ingroup group_numerics
 */
class FourierTransform
{
public:
    FourierTransform() = default;
    ~FourierTransform() = default;
    FourierTransform(const FourierTransform & other) = delete;
    FourierTransform(FourierTransform && other) = delete;
    FourierTransform & operator=(const FourierTransform & other) = delete;
    FourierTransform & operator=(FourierTransform && other) = delete;

    template <template <typename> class T1, typename T2>
    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    static void fft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
    {
        const size_t N = in.size();

        if ((N & (N - 1)) == 0)
        {
            detail::fft_radix_2<T1, T2>(in, out);
        }
        else
        {
            detail::fft_bluestein<T1, T2>(in, out);
        }
    }

    template <template <typename> class T1, typename T2>
    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    static void ifft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
    {
        size_t const N = in.size();
        SimpleArray<T1<T2>> in_conj{solvcon::small_vector<ssize_t>{static_cast<ssize_t>(N)}, T1<T2>{0.0, 0.0}};

        for (size_t i = 0; i < N; ++i)
        {
            in_conj[i] = in[i].conj();
        }

        fft<T1, T2>(in_conj, out);

        for (size_t i = 0; i < N; ++i)
        {
            out[i] = out[i].conj() / static_cast<T2>(N);
        }
    }

    // TODO: The template of template is too complicate, we should find a way to make it easier.
    template <template <typename> typename T1, typename T2>
    static void dft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
    {
        size_t const N = in.size();
        for (size_t i = 0; i < N; ++i)
        {
            out[i] = 0;
            for (size_t j = 0; j < N; ++j)
            {
                T2 tmp = -2.0 * pi<T2> * i * j / static_cast<T2>(N);
                T1<T2> const twiddle_factor{std::cos(tmp), std::sin(tmp)};
                out[i] += in[j] * twiddle_factor;
            }
        }
    }
};

namespace detail
{

template <template <typename> class T1, typename T2>
// FIXME: NOLINTNEXTLINE(misc-no-recursion)
void fft_bluestein(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
{
    const size_t N = in.size();
    // Calculate a length with power of 2 and at least 2N-1
    const size_t K = detail::next_power_of_two(2 * N - 1);

    SimpleArray<T1<T2>> a{solvcon::small_vector<ssize_t>{static_cast<ssize_t>(K)}, T1<T2>{0.0, 0.0}};
    SimpleArray<T1<T2>> A{solvcon::small_vector<ssize_t>{static_cast<ssize_t>(K)}, T1<T2>{0.0, 0.0}};
    SimpleArray<T1<T2>> b{solvcon::small_vector<ssize_t>{static_cast<ssize_t>(K)}, T1<T2>{0.0, 0.0}};
    SimpleArray<T1<T2>> B{solvcon::small_vector<ssize_t>{static_cast<ssize_t>(K)}, T1<T2>{0.0, 0.0}};

    // Calculate a[0], b[0] first, becuase it can avoid a branch in the following
    // for loop!
    a[0] = in[0];
    b[0] = T1<T2>{1.0, 0.0};

    for (size_t i = 1; i < N; ++i)
    {
        T2 const tmp = -pi<T2> * i * i / static_cast<T2>(N);
        T1<T2> const twiddle_factor{std::cos(tmp), std::sin(tmp)};

        a[i] = in[i] * twiddle_factor;
        b[i] = twiddle_factor.conj();
        // Convert circular convolution to linear convolution
        b[K - i] = b[i];
    }

    fft_radix_2<T1, T2>(a, A);
    fft_radix_2<T1, T2>(b, B);

    for (size_t i = 0; i < K; ++i)
    {
        A[i] *= B[i];
    }

    FourierTransform::ifft<T1, T2>(A, a);

    for (size_t i = 0; i < N; ++i)
    {
        T2 const tmp = -pi<T2> * i * i / static_cast<T2>(N);
        T1<T2> const twiddle_factor{std::cos(tmp), std::sin(tmp)};
        out[i] = a[i] * twiddle_factor;
    }
}

} /* end of namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
