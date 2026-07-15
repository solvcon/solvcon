#include <solvcon/transform/fourier.hpp>

namespace solvcon
{

namespace detail
{

size_t bit_reverse(size_t n, const size_t bits)
{
    size_t reversed = 0;
    for (size_t i = 0; i < bits; i++)
    {
        if (n & (1 << i))
        {
            reversed |= 1 << (bits - 1 - i);
        }
    }
    return reversed;
}

size_t next_power_of_two(size_t n)
{
    size_t power = 1;
    while (power < n)
    {
        power = power << 1;
    }
    return power;
}

} /* end namespace detail */

} /* end namespace solvcon */
