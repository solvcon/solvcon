/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/simd/simd_support.hpp>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#if defined(__aarch64__) || defined(__arm__)
#include <asm/hwcap.h>
#endif
#elifdef _WIN32
#include <windows.h>
#include <intrin.h>
#elifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace solvcon
{

namespace simd
{

namespace detail
{

SimdFeature detect_simd()
{
    static SimdFeature CurrentFeature = SIMD_UNKNOWN;

    if (CurrentFeature != SIMD_UNKNOWN)
    {
        return CurrentFeature;
    }

#if defined(__aarch64__) || defined(__arm__)
// ARM architecture
#if defined(__linux__) || defined(__ANDROID__)
    unsigned long hwcaps = getauxval(AT_HWCAP);
#ifdef HWCAP_NEON
    if (hwcaps & HWCAP_NEON)
    {
        CurrentFeature = SIMD_NEON;
    }
#endif /* HWCAP_NEON */
#elifdef __APPLE__
    int neon_supported = 0;
    size_t size = sizeof(neon_supported);
    if (sysctlbyname("hw.optional.neon", &neon_supported, &size, nullptr, 0) == 0 && neon_supported)
    {
        CurrentFeature = SIMD_NEON;
    }
#elifdef _WIN32
    if (IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE))
    {
        CurrentFeature = SIMD_NEON;
    }
#endif
#endif /* defined(__aarch64__) || defined(__arm__) */

    if (CurrentFeature == SIMD_UNKNOWN)
    {
        CurrentFeature = SIMD_NONE;
    }
    return CurrentFeature;
}

} /* end namespace detail */

} /* end namespace simd */

} /* end namespace solvcon */
