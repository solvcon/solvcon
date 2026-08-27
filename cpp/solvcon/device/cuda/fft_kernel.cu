/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

// In-house CUDA FFT: radix-2 Cooley-Tukey for power-of-two lengths and the
// Bluestein algorithm otherwise, mirroring the CPU implementation in
// transform/fourier.hpp so both backends share the same numerics.
//
// This unit is compiled by nvcc and deliberately includes no solvcon core
// header: nvcc may not parse the C++23 the core requires. fft.cpp adapts
// SimpleArray to the raw-pointer interface in fft_impl.hpp.
#include <solvcon/device/cuda/fft_impl.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace solvcon
{

namespace device
{

namespace cuda
{

namespace detail
{

namespace
{

void check(cudaError_t err, char const * what)
{
    if (cudaSuccess != err)
    {
        throw std::runtime_error(std::string(what) + " failed with CUDA error " +
                                 std::to_string(static_cast<int>(err)) + ": " + cudaGetErrorString(err));
    }
}

struct DeviceBuffer
{
    explicit DeviceBuffer(size_t nbytes) { check(cudaMalloc(&m_ptr, nbytes), "cudaMalloc"); }
    ~DeviceBuffer() { cudaFree(m_ptr); }
    DeviceBuffer(DeviceBuffer const &) = delete;
    DeviceBuffer(DeviceBuffer &&) = delete;
    DeviceBuffer & operator=(DeviceBuffer const &) = delete;
    DeviceBuffer & operator=(DeviceBuffer &&) = delete;

    template <typename T>
    T * as() const { return static_cast<T *>(m_ptr); }

    void * m_ptr = nullptr;
}; /* end struct DeviceBuffer */

template <typename T>
struct DevComplex
{
    T r;
    T i;
}; /* end struct DevComplex */

template <typename T>
__host__ __device__ constexpr T pi()
{
    return static_cast<T>(3.14159265358979323846);
}

__device__ inline void sincos_t(float a, float * s, float * c) { sincosf(a, s, c); }
__device__ inline void sincos_t(double a, double * s, double * c) { sincos(a, s, c); }

template <typename T>
__device__ inline DevComplex<T> cmul(DevComplex<T> a, DevComplex<T> b)
{
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

template <typename T>
__global__ void bit_reverse_kernel(DevComplex<T> const * in, DevComplex<T> * out, int n, int bits)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    int rev = 0;
    for (int b = 0; b < bits; ++b)
    {
        if (idx & (1 << b))
        {
            rev |= 1 << (bits - 1 - b);
        }
    }
    out[rev] = in[idx];
}

// One radix-2 stage: threads pair up the elements `half` apart inside each
// block of `size` and apply the twiddle exp(-2 pi i k / size).
template <typename T>
__global__ void butterfly_kernel(DevComplex<T> * data, int n, int size)
{
    int const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n / 2)
    {
        return;
    }
    int const half = size / 2;
    int const base = (t / half) * size;
    int const k = t % half;

    T s, c;
    T const angle = T(-2.0) * pi<T>() / static_cast<T>(size) * static_cast<T>(k);
    sincos_t(angle, &s, &c);
    DevComplex<T> const even = data[base + k];
    DevComplex<T> const odd = cmul(data[base + k + half], DevComplex<T>{c, s});

    data[base + k] = {even.r + odd.r, even.i + odd.i};
    data[base + k + half] = {even.r - odd.r, even.i - odd.i};
}

template <typename T>
__global__ void pointwise_mul_kernel(DevComplex<T> * a, DevComplex<T> const * b, int n)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        a[idx] = cmul(a[idx], b[idx]);
    }
}

template <typename T>
__global__ void conj_kernel(DevComplex<T> * a, int n)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        a[idx].i = -a[idx].i;
    }
}

template <typename T>
__global__ void conj_scale_kernel(DevComplex<T> * a, int n, T scale)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        a[idx] = {a[idx].r * scale, -a[idx].i * scale};
    }
}

constexpr int BLOCK = 256;

int grid(int n)
{
    return (n + BLOCK - 1) / BLOCK;
}

int ilog2(int n)
{
    int bits = 0;
    while ((1 << bits) < n)
    {
        ++bits;
    }
    return bits;
}

bool is_pow2(int n)
{
    return 0 == (n & (n - 1));
}

int next_pow2(int n)
{
    int power = 1;
    while (power < n)
    {
        power <<= 1;
    }
    return power;
}

// Forward FFT of `in` into `out`, both device pointers, n a power of two.
// TODO: One global-memory kernel launch per stage; fuse stages through
// shared memory to cut launch and bandwidth overhead.
template <typename T>
void fft_pow2_device(DevComplex<T> const * in, DevComplex<T> * out, int n)
{
    bit_reverse_kernel<<<grid(n), BLOCK>>>(in, out, n, ilog2(n));
    for (int size = 2; size <= n; size *= 2)
    {
        butterfly_kernel<<<grid(n / 2), BLOCK>>>(out, n, size);
    }
    check(cudaGetLastError(), "fft kernel launch");
}

// Inverse FFT in place through the conjugation identity, matching
// FourierTransform::ifft() on the CPU: conj, forward FFT, conj and 1/n.
template <typename T>
void ifft_pow2_device(DevComplex<T> * data, DevComplex<T> * scratch, int n)
{
    conj_kernel<<<grid(n), BLOCK>>>(data, n);
    fft_pow2_device<T>(data, scratch, n);
    conj_scale_kernel<<<grid(n), BLOCK>>>(scratch, n, T(1.0) / static_cast<T>(n));
    check(cudaMemcpy(data, scratch, sizeof(DevComplex<T>) * n, cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

template <typename T>
void fft_bluestein(DevComplex<T> const * in, DevComplex<T> * out, int n)
{
    int const K = next_pow2(2 * n - 1);

    // Build the Bluestein sequences on the host exactly as the CPU
    // fft_bluestein() does: a[i] = in[i] * w(i) and the circular kernel b,
    // with w(i) = exp(-pi i^2 / n).
    std::vector<DevComplex<T>> a(K, DevComplex<T>{T(0.0), T(0.0)});
    std::vector<DevComplex<T>> b(K, DevComplex<T>{T(0.0), T(0.0)});
    std::vector<DevComplex<T>> w(n);
    w[0] = {T(1.0), T(0.0)};
    a[0] = in[0];
    b[0] = {T(1.0), T(0.0)};
    for (int i = 1; i < n; ++i)
    {
        T const tmp = -pi<T>() * static_cast<T>(i) * static_cast<T>(i) / static_cast<T>(n);
        DevComplex<T> const tw{std::cos(tmp), std::sin(tmp)};
        w[i] = tw;
        a[i] = {in[i].r * tw.r - in[i].i * tw.i, in[i].r * tw.i + in[i].i * tw.r};
        b[i] = {tw.r, -tw.i};
        b[K - i] = b[i];
    }

    size_t const kbytes = sizeof(DevComplex<T>) * K;
    DeviceBuffer dev_a(kbytes), dev_fa(kbytes), dev_b(kbytes), dev_fb(kbytes);
    check(cudaMemcpy(dev_a.m_ptr, a.data(), kbytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    check(cudaMemcpy(dev_b.m_ptr, b.data(), kbytes, cudaMemcpyHostToDevice), "cudaMemcpy");

    // Linear convolution as pointwise product in the frequency domain.
    fft_pow2_device<T>(dev_a.as<DevComplex<T>>(), dev_fa.as<DevComplex<T>>(), K);
    fft_pow2_device<T>(dev_b.as<DevComplex<T>>(), dev_fb.as<DevComplex<T>>(), K);
    pointwise_mul_kernel<<<grid(K), BLOCK>>>(dev_fa.as<DevComplex<T>>(), dev_fb.as<DevComplex<T>>(), K);
    ifft_pow2_device<T>(dev_fa.as<DevComplex<T>>(), dev_a.as<DevComplex<T>>(), K);

    std::vector<DevComplex<T>> conv(n);
    check(cudaMemcpy(conv.data(), dev_fa.m_ptr, sizeof(DevComplex<T>) * n, cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (int i = 0; i < n; ++i)
    {
        out[i] = {conv[i].r * w[i].r - conv[i].i * w[i].i, conv[i].r * w[i].i + conv[i].i * w[i].r};
    }
}

template <typename T>
void fft_impl(void const * in, void * out, int n)
{
    auto const * host_in = static_cast<DevComplex<T> const *>(in);
    auto * host_out = static_cast<DevComplex<T> *>(out);

    if (is_pow2(n))
    {
        size_t const nbytes = sizeof(DevComplex<T>) * n;
        DeviceBuffer dev_in(nbytes), dev_out(nbytes);
        check(cudaMemcpy(dev_in.m_ptr, host_in, nbytes, cudaMemcpyHostToDevice), "cudaMemcpy");
        fft_pow2_device<T>(dev_in.as<DevComplex<T>>(), dev_out.as<DevComplex<T>>(), n);
        check(cudaMemcpy(host_out, dev_out.m_ptr, nbytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
    }
    else
    {
        fft_bluestein<T>(host_in, host_out, n);
    }
}

} /* end namespace */

bool device_available()
{
    int count = 0;
    return cudaSuccess == cudaGetDeviceCount(&count) && count > 0;
}

void fft_float(void const * in, void * out, int n)
{
    fft_impl<float>(in, out, n);
}

void fft_double(void const * in, void * out, int n)
{
    fft_impl<double>(in, out, n);
}

} /* end namespace detail */

} /* end namespace cuda */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
