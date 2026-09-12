#include <solvcon/math/Complex.hpp>

#include <gtest/gtest.h>

#include <complex>
#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif
#ifdef _MSC_VER
#pragma fenv_access(on)
#endif

namespace
{

template <typename T>
concept HasStdComplexConversion = requires(T const & value) { value.to_std_complex(); };

template <typename T>
concept HasStdComplexPointerBridge = requires(T * value) { solvcon::as_std_complex_pointer(value); };

namespace sc = solvcon;

static_assert(!HasStdComplexConversion<sc::Complex32>);
static_assert(!HasStdComplexPointerBridge<sc::Complex32>);
static_assert(!HasStdComplexPointerBridge<sc::Complex32 const>);

static_assert(HasStdComplexConversion<sc::Complex<float>>);
static_assert(HasStdComplexPointerBridge<sc::Complex<float>>);
static_assert(HasStdComplexPointerBridge<sc::Complex<float> const>);

static_assert(HasStdComplexConversion<sc::Complex<double>>);
static_assert(HasStdComplexPointerBridge<sc::Complex<double>>);
static_assert(HasStdComplexPointerBridge<sc::Complex<double> const>);

} /* end namespace */

TEST(Complex32, number_conversion)
{
    namespace sc = solvcon;

    // widening conversion
    sc::Complex32 const narrow{sc::Float16(1.5F), sc::Float16(-2.0F)};
    sc::Complex<float> const wide{narrow};
    EXPECT_FLOAT_EQ(1.5F, wide.real());
    EXPECT_FLOAT_EQ(-2.0F, wide.imag());

    // narrowing conversion
    sc::Complex<sc::Float16> const converted{wide};
    EXPECT_EQ(sc::Float16(1.5F).bits(), converted.real().bits());
    EXPECT_EQ(sc::Float16(-2.0F).bits(), converted.imag().bits());

    // from real number
    sc::Float16 const real = sc::Float16::from_bits(0x3555U);
    sc::Complex<sc::Float16> const value = real;

    EXPECT_EQ(real.bits(), value.real().bits());
    EXPECT_EQ(0x0000U, value.imag().bits());

    // from std::complex object
    std::complex<float> const standard{3.0F, -4.0F};
    sc::Complex<float> const existing_conversion{standard};
    EXPECT_EQ(standard, existing_conversion.to_std_complex());
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
