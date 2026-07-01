#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Typed multi-dimensional array over a shared memory buffer.
 *
 * @ingroup group_core
 */

#include <solvcon/buffer/ConcreteBuffer.hpp>
#include <solvcon/buffer/matmul.hpp>
#include <solvcon/math/math.hpp>
#include <solvcon/simd/simd.hpp>

// TODO: Solve circular include between <solvcon/toggle/toggle.hpp> and SimpleArray class.
// Since it will happen circulate include when using <solvcon/toggle/toggle.hpp>,
// I use <solvcon/toggle/RadixTree.hpp> instead.
#include <solvcon/toggle/RadixTree.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <format>
#include <functional>
#include <limits>
#include <mdspan>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace solvcon
{

template <typename T>
class SimpleArray; // forward declaration

template <typename T>
concept IntegralType = std::is_integral_v<T>;

template <typename T>
concept ArithmeticType = std::is_arithmetic_v<T>;

template <typename T>
concept SimpleArrayType = requires(T t) {
    { t.data() } -> std::convertible_to<typename T::value_type *>;
    { t.size() } -> std::convertible_to<size_t>;
};

template <typename T, typename U>
concept IsSameRemoveConstType = std::is_same_v<std::remove_const_t<T>, U>;

template <typename It>
concept InputIterator = requires(It it) {
    { *it };
    { ++it } -> std::same_as<It &>;
    { it++ } -> std::same_as<It>;
};

namespace detail
{

template <size_t D, typename S>
ssize_t buffer_offset_impl(S const &)
{
    return 0;
}

template <size_t D, typename S, typename Arg, typename... Args>
ssize_t buffer_offset_impl(S const & strides, Arg arg, Args... args)
{
    return static_cast<ssize_t>(arg) * strides[D] + buffer_offset_impl<D + 1>(strides, args...);
}

} /* end namespace detail */

template <typename S, typename... Args>
ssize_t buffer_offset(S const & strides, Args... args)
{
    return detail::buffer_offset_impl<0>(strides, args...);
}

inline ssize_t buffer_offset(small_vector<ssize_t> const & stride, small_vector<ssize_t> const & idx)
{
    if (stride.size() != idx.size())
    {
        throw std::out_of_range(std::format("stride size {} != index size {}", stride.size(), idx.size()));
    }
    ssize_t offset = 0;
    for (size_t it = 0; it < stride.size(); ++it)
    {
        offset += stride[it] * idx[it];
    }
    return offset;
}

namespace detail
{

using shape_type = small_vector<size_t>;
using sshape_type = small_vector<ssize_t>;

/**
 * @brief Enumerate SimpleArray indices with nghost included.
 *
 * The first dimension starts at -nghost, while the other dimensions start
 * at 0. For example, an array with shape [4, 3, 2] and nghost = 1 is
 * enumerated from [-1, 0, 0] to [2, 2, 1].
 */
class IndexRange
{
public:

    IndexRange(IndexRange const &) = default;
    IndexRange(IndexRange &&) noexcept = default;
    IndexRange & operator=(IndexRange const &) = default;
    IndexRange & operator=(IndexRange &&) noexcept = default;
    ~IndexRange() = default;

    template <typename Array>
    explicit IndexRange(Array const & array)
        : m_lower(array.ndim(), 0)
        , m_upper(array.ndim(), 0)
    {
        for (size_t dim = 0; dim < array.ndim(); ++dim)
        {
            m_lower[dim] = lower_bound(array, dim);
            m_upper[dim] = upper_bound(array, dim);
        }
    }

    template <typename Array>
    IndexRange(Array const & array, small_vector<size_t> const & axes)
        : m_lower(axes.size(), 0)
        , m_upper(axes.size(), 0)
    {
        for (size_t k = 0; k < axes.size(); ++k)
        {
            size_t const dim = axes[k];
            m_lower[k] = lower_bound(array, dim);
            m_upper[k] = upper_bound(array, dim);
        }
    }

    sshape_type first() const { return m_lower; }

    bool next(sshape_type & idx) const
    {
        for (size_t i = idx.size(); i > 0; --i)
        {
            if (++idx[i - 1] < m_upper[i - 1])
            {
                return true;
            }
            idx[i - 1] = m_lower[i - 1];
        }
        return false;
    }

    template <typename Array>
    static ssize_t index_from_offset(Array const & array, size_t dim, ssize_t offset)
    {
        return lower_bound(array, dim) + offset;
    }

private:

    sshape_type m_lower;
    sshape_type m_upper;

    template <typename Array>
    static ssize_t lower_bound(Array const & array, size_t dim)
    {
        if (dim == 0)
        {
            return -static_cast<ssize_t>(array.nghost());
        }
        return 0;
    }

    template <typename Array>
    static ssize_t upper_bound(Array const & array, size_t dim)
    {
        return lower_bound(array, dim) + static_cast<ssize_t>(array.shape(dim));
    }
};

std::string format_shape(shape_type const & shape);
std::string format_flat_index(shape_type const & shape, size_t offset);

template <typename T>
struct SimpleArrayInternalTypes
{
    using value_type = T;
    using shape_type = detail::shape_type;
    using sshape_type = detail::sshape_type;
    using buffer_type = ConcreteBuffer;
}; /* end class SimpleArrayInternalType */

template <typename A, typename T>
class SimpleArrayMixinModifiers
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;

    A & fill(value_type const & value)
    {
        auto athis = static_cast<A *>(this);
        std::fill(athis->begin(), athis->end(), value);
        return *athis;
    }

}; /* end class SimpleArrayMixinModifiers */

template <typename U>
struct select_real_t
{
    using type = U;
};

template <typename U>
struct select_real_t<Complex<U>>
{
    using type = U;
};

template <typename A, typename T>
class SimpleArrayMixinSum
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;
    using shape_type = typename internal_types::shape_type;
    using sshape_type = typename internal_types::sshape_type;

    value_type sum() const
    {
        auto athis = static_cast<A const *>(this);
        const size_t n = athis->size();
        if (n == 0)
        {
            return zero();
        }
        // Either C- or F-contiguous arrays occupy a single dense block in
        // memory, so a linear buffer sweep visits every element exactly
        // once, in C order or F order respectively.
        if (athis->is_c_contiguous() || athis->is_f_contiguous())
        {
            return sum_contiguous(athis->data(), n);
        }
        return sum_strided(athis->data(), athis->shape(), athis->stride());
    }

private:

    static constexpr value_type zero()
    {
        if constexpr (is_complex_v<value_type>)
        {
            return value_type{};
        }
        else
        {
            return value_type{0};
        }
    }

    static void accumulate(value_type & acc, value_type v)
    {
        if constexpr (std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            acc |= v;
        }
        else
        {
            acc += v;
        }
    }

    static value_type sum_contiguous(value_type const * data, size_t n)
    {
        value_type acc = zero();
        for (size_t i = 0; i < n; ++i)
        {
            accumulate(acc, data[i]);
        }
        return acc;
    }

    // Walk a strided array by its innermost dimension: compute the row base
    // offset once per outer iteration, then accumulate along the last axis.
    // This avoids the per-element multi-dimensional index arithmetic that
    // at(sidx) performs.
    static value_type sum_strided(value_type const * data,
                                  shape_type const & shape,
                                  sshape_type const & stride)
    {
        const size_t ndim = shape.size();
        auto const last_dim = static_cast<ssize_t>(shape[ndim - 1]);
        const ssize_t last_stride = stride[ndim - 1];

        value_type acc = zero();
        sshape_type prefix(ndim - 1, 0);
        do
        {
            ssize_t offset = 0;
            for (size_t i = 0; i + 1 < ndim; ++i)
            {
                offset += prefix[i] * stride[i];
            }
            value_type const * row = data + offset;
            for (ssize_t j = 0; j < last_dim; ++j)
            {
                accumulate(acc, row[j * last_stride]);
            }
        } while (next_prefix(prefix, shape));
        return acc;
    }

    static bool next_prefix(sshape_type & idx,
                            shape_type const & shape)
    {
        for (size_t i = idx.size(); i > 0; --i)
        {
            if (++idx[i - 1] < static_cast<ssize_t>(shape[i - 1]))
            {
                return true;
            }
            idx[i - 1] = 0;
        }
        return false;
    }

}; /* end class SimpleArrayMixinSum */

template <typename A, typename T>
class SimpleArrayMixinCalculators
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;
    using real_type = typename detail::select_real_t<value_type>::type;

    template <typename RedFn, typename... RedArgs>
    auto reduce(const shape_type & axis, RedFn red_fn, RedArgs &&... red_args) const
    {
        using element_type = std::invoke_result_t<
            RedFn,
            const A *,
            small_vector<value_type> &,
            RedArgs...>;

        using ret_type = typename A::template rebind<element_type>;

        auto athis = static_cast<const A *>(this);
        const size_t ndim = athis->ndim();

        small_vector<bool> reduce_mask(ndim, false);
        for (size_t const ax : axis)
        {
            if (ax >= ndim || ax < 0)
            {
                throw std::out_of_range("reduce: axis out of range");
            }
            reduce_mask[ax] = true;
        }

        size_t const red_count = reduce_mask.count(true);
        if (red_count == 0 || red_count == ndim)
        {
            throw std::runtime_error("reduce: no axis to reduce or all axes are reduced");
        }

        small_vector<size_t> out_shape(ndim - red_count);
        for (size_t i = 0, l = 0; i < ndim; ++i)
        {
            if (!reduce_mask[i])
            {
                out_shape[l++] = athis->shape(i);
            }
        }
        ret_type result(out_shape);

        small_vector<size_t> red_axes(red_count);
        for (size_t i = 0, l = 0; i < ndim; ++i)
        {
            if (reduce_mask[i])
            {
                red_axes[l++] = i;
            }
        }
        auto const red_range = IndexRange(*athis, red_axes);

        sshape_type full_idx(ndim, 0);
        auto const out_range = IndexRange(result);
        sshape_type out_idx = out_range.first();

        do
        {
            for (size_t i = 0, out_dim = 0; i < ndim; ++i)
            {
                if (!reduce_mask[i])
                {
                    full_idx[i] = IndexRange::index_from_offset(*athis, i, out_idx[out_dim++]);
                }
            }

            small_vector<value_type> slice;
            sshape_type red_idx = red_range.first();

            do
            {
                for (size_t k = 0; k < red_axes.size(); ++k)
                {
                    full_idx[red_axes[k]] = red_idx[k];
                }
                slice.push_back(athis->at(full_idx));

            } while (red_range.next(red_idx));

            // FIXME: NOLINTNEXTLINE(bugprone-use-after-move)
            element_type const mv = std::invoke(red_fn, this, slice, std::forward<RedArgs>(red_args)...);
            result.at(out_idx) = mv;
        } while (out_range.next(out_idx));

        return result;
    }

    value_type median_op(small_vector<value_type> & sv) const;

    value_type median_freq(small_vector<value_type> & sv) const;

    A median(const small_vector<size_t> & axis) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::median_op);
    }

    value_type median() const
    {
        SOLVCON_PROFILE_SCOPE("SimpleArray::median()");
        auto athis = static_cast<A const *>(this);
        const size_t n = athis->size();
        small_vector<T> acopy(n);
        auto const range = IndexRange(*athis);
        sshape_type sidx = range.first();
        size_t i = 0;
        do
        {
            acopy[i] = athis->at(sidx);
            ++i;
        } while (range.next(sidx));
        return median_op(acopy);
    }

    value_type average_op(small_vector<value_type> & sv, small_vector<value_type> & weight) const
    {
        const size_t n = sv.size();
        if (n != weight.size())
        {
            throw std::runtime_error("SimpleArray::average_op(): weight size does not match array size");
        }
        value_type sum = 0;
        value_type total_weight = 0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += sv[i] * weight[i];
            total_weight += weight[i];
        }
        if (total_weight == static_cast<value_type>(0))
        {
            throw std::runtime_error("SimpleArray::average_op(): total weight is zero");
        }
        return sum / total_weight;
    }

    A average(const small_vector<size_t> & axis, A const & weight) const
    {
        small_vector<value_type> weight_sv(weight.size());
        auto const range = IndexRange(weight);
        sshape_type sidx = range.first();
        size_t i = 0;
        do
        {
            weight_sv[i] = weight.at(sidx);
            ++i;
        } while (range.next(sidx));
        return reduce(axis, &SimpleArrayMixinCalculators::average_op, weight_sv);
    }

    value_type average(A const & weight) const
    {
        auto athis = static_cast<A const *>(this);
        const shape_type & weight_shape = weight.shape();
        const shape_type & athis_shape = athis->shape();
        for (size_t i = 0; i < athis_shape.size(); ++i)
        {
            if (weight_shape[i] != athis_shape[i])
            {
                throw std::runtime_error("SimpleArray::average(): weight shape does not match array shape");
            }
        }
        value_type sum = 0;
        value_type total_weight = 0;
        auto const range = IndexRange(*athis);
        sshape_type sidx = range.first();
        do
        {
            sum += athis->at(sidx) * weight.at(sidx);
            total_weight += weight.at(sidx);
        } while (range.next(sidx));
        if (total_weight == static_cast<value_type>(0))
        {
            throw std::runtime_error("SimpleArray::average(): total weight is zero");
        }
        return sum / total_weight;
    }

    value_type mean_op(small_vector<value_type> & sv) const
    {
        const size_t n = sv.size();
        value_type sum = 0;
        for (const auto & v : sv)
        {
            sum += v;
        }
        return sum / static_cast<value_type>(n);
    }

    A mean(const small_vector<size_t> & axis) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::mean_op);
    }

    value_type mean() const
    {
        auto athis = static_cast<A const *>(this);
        const size_t n = athis->size();
        if (n == 0)
        {
            throw std::runtime_error("SimpleArray::mean(): empty array");
        }
        return athis->sum() / static_cast<value_type>(n);
    }

    real_type var_op(small_vector<value_type> & sv, size_t ddof) const
    {
        const size_t n = sv.size();
        if (n <= ddof)
        {
            throw std::runtime_error("SimpleArray::var_op(): ddof must be less than the number of elements");
        }
        value_type const mu = mean_op(sv);
        real_type acc = 0;
        // The complex and real branches differ; clang-tidy cannot tell the
        // type-dependent expressions apart in the uninstantiated template.
        // NOLINTBEGIN(bugprone-branch-clone)
        if constexpr (is_complex_v<value_type>)
        {
            for (const auto & v : sv)
            {
                acc += (v - mu).norm();
            }
        }
        else
        {
            for (const auto & v : sv)
            {
                acc += (v - mu) * (v - mu);
            }
        }
        // NOLINTEND(bugprone-branch-clone)
        return acc / static_cast<real_type>(n - ddof);
    }

    auto var(const small_vector<size_t> & axis, size_t ddof) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::var_op, ddof);
    }

    real_type var(size_t ddof) const
    {
        auto athis = static_cast<A const *>(this);
        const size_t n = athis->size();
        if (n <= ddof)
        {
            throw std::runtime_error("SimpleArray::var(): ddof must be less than the number of elements");
        }

        auto const range = IndexRange(*athis);
        sshape_type sidx = range.first();
        value_type const mu = athis->mean();
        real_type acc = 0;
        if constexpr (is_complex_v<value_type>)
        {
            do
            {
                acc += athis->at(sidx).norm();
            } while (range.next(sidx));
        }
        else
        {
            do
            {
                value_type const value = athis->at(sidx);
                acc += value * value;
            } while (range.next(sidx));
        }
        if constexpr (is_complex_v<value_type>)
        {
            acc -= n * mu.norm();
        }
        else
        {
            acc -= n * mu * mu;
        }
        return acc / static_cast<real_type>(n - ddof);
    }

    real_type std_op(small_vector<value_type> & sv, size_t ddof) const
    {
        return std::sqrt(var_op(sv, ddof));
    }

    auto std(const small_vector<size_t> & axis, size_t ddof) const
    {
        return reduce(axis, &SimpleArrayMixinCalculators::std_op, ddof);
    }

    real_type std(size_t ddof) const
    {
        auto athis = static_cast<A const *>(this);
        return std::sqrt(athis->var(ddof));
    }

    value_type min() const
    {
        value_type initial = std::numeric_limits<value_type>::max();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) < initial)
            {
                initial = athis->data(i);
            }
        }
        return initial;
    }

    value_type max() const
    {
        value_type initial = std::numeric_limits<value_type>::lowest();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) > initial)
            {
                initial = athis->data(i);
            }
        }
        return initial;
    }

    A abs() const
    {
        auto athis = static_cast<A const *>(this);
        A ret(*athis);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>> && std::is_signed_v<value_type>)
        {
            for (size_t i = 0; i < athis->size(); ++i)
            {
                ret.data(i) = std::abs(athis->data(i));
            }
        }
        return ret;
    }

    A add(A const & other) const
    {
        return A(*static_cast<A const *>(this)).iadd(other);
    }

    A add(value_type scalar) const
    {
        return A(*static_cast<A const *>(this)).iadd(scalar);
    }

    A sub(A const & other) const
    {
        return A(*static_cast<A const *>(this)).isub(other);
    }

    A sub(value_type scalar) const
    {
        return A(*static_cast<A const *>(this)).isub(scalar);
    }

    A mul(A const & other) const
    {
        return A(*static_cast<A const *>(this)).imul(other);
    }

    A mul(value_type scalar) const
    {
        return A(*static_cast<A const *>(this)).imul(scalar);
    }

    A div(A const & other) const
    {
        return A(*static_cast<A const *>(this)).idiv(other);
    }

    A div(value_type scalar) const
    {
        return A(*static_cast<A const *>(this)).idiv(scalar);
    }

    SimpleArray<bool> eq(A const & other) const;
    SimpleArray<bool> eq(value_type scalar) const;

    A & iadd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr += *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            size_t const size = athis->size();
            for (size_t i = 0; i < size; ++i)
            {
                athis->data(i) = athis->data(i) || other.data(i);
            }
        }

        return *athis;
    }

    A & iadd(value_type scalar)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();

            while (ptr < end)
            {
                *ptr += scalar;
                ++ptr;
            }
        }
        else
        {
            if (scalar)
            {
                std::fill(athis->begin(), athis->end(), true);
            }
        }
        return *athis;
    }

    A & isub(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr -= *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            throw std::runtime_error(
                "SimpleArray<bool>::isub(): boolean value doesn't support this operation");
        }
        return *athis;
    }

    A & isub(value_type scalar)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();

            while (ptr < end)
            {
                *ptr -= scalar;
                ++ptr;
            }
        }
        else
        {
            throw std::runtime_error(
                "SimpleArray<bool>::isub(): boolean value doesn't support this operation");
        }
        return *athis;
    }

    A & imul(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr *= *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            size_t const size = athis->size();
            for (size_t i = 0; i < size; ++i)
            {
                athis->data(i) = athis->data(i) && other.data(i);
            }
        }
        return *athis;
    }

    A & imul(value_type scalar)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();

            while (ptr < end)
            {
                *ptr *= scalar;
                ++ptr;
            }
        }
        else
        {
            if (!scalar)
            {
                size_t const size = athis->size();
                for (size_t i = 0; i < size; ++i)
                {
                    athis->data(i) = false;
                }
            }
        }
        return *athis;
    }

    A & idiv(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();
            const value_type * other_ptr = other.begin();

            while (ptr < end)
            {
                *ptr /= *other_ptr;
                ++ptr;
                ++other_ptr;
            }
        }
        else
        {
            throw std::runtime_error(
                "SimpleArray<bool>::idiv(): boolean value doesn't support this operation");
        }
        return *athis;
    }

    A & idiv(value_type scalar)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            const value_type * const end = athis->end();
            value_type * ptr = athis->begin();

            while (ptr < end)
            {
                *ptr /= scalar;
                ++ptr;
            }
        }
        else
        {
            throw std::runtime_error(
                "SimpleArray<bool>::idiv(): boolean value doesn't support this operation");
        }
        return *athis;
    }

    A add_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::add<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A sub_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::sub<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A mul_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::mul<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A div_simd(A const & other) const
    {
        A const * athis = static_cast<A const *>(this);
        A ret(athis->shape());

        simd::div<T>(ret.begin(), ret.end(), athis->begin(), other.begin());

        return ret;
    }

    A & iadd_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::add<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->iadd(other);
        }
    }

    A & isub_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::sub<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->isub(other);
        }
    }

    A & imul_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::mul<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->imul(other);
        }
    }

    A & idiv_simd(A const & other)
    {
        auto athis = static_cast<A *>(this);
        if constexpr (!std::is_same_v<bool, std::remove_const_t<value_type>>)
        {
            simd::div<T>(athis->begin(), athis->end(), athis->begin(), other.begin());
            return *athis;
        }
        else
        {
            return athis->idiv(other);
        }
    }

    A matmul(A const & other) const;
    A & imatmul(A const & other);
    A matmul_blas(A const & other) const;
    A & imatmul_blas(A const & other);
    A matmul_fast(A const & other,
                  size_t tile_x,
                  size_t tile_y,
                  size_t tile_z) const;
    A & imatmul_fast(A const & other,
                     size_t tile_x,
                     size_t tile_y,
                     size_t tile_z);

private:
    static void find_two_bins(const uint32_t * freq, size_t n, int & bin1, int & bin2);
}; /* end class SimpleArrayMixinCalculators */

template <typename A, typename T>
typename detail::SimpleArrayMixinCalculators<A, T>::value_type
detail::SimpleArrayMixinCalculators<A, T>::median_op(small_vector<value_type> & sv) const
{
    SOLVCON_PROFILE_SCOPE("SimpleArray::median_op()");
    const size_t n = sv.size();

    if constexpr (std::is_same_v<value_type, int8_t> ||
                  std::is_same_v<value_type, uint8_t> ||
                  std::is_same_v<value_type, bool>)
    {
        return median_freq(sv);
    }
    else
    {
        value_type const v1 = sv.select_kth(n / 2);
        if (n % 2 != 0)
        {
            return v1;
        }
        value_type const v2 = simd::max(sv.begin(), sv.begin() + n / 2);
        value_type const result = static_cast<value_type>(v1 + v2) / static_cast<value_type>(2.0);
        return result;
    }
}

/**
 * Calculate median using frequency counting for small data types.
 * This algorithm is optimized for uint8_t, int8_t, and bool types where
 * the range of possible values is small (<=256). Instead of sorting,
 * it counts the frequency of each value and finds the median position.
 *
 * Algorithm:
 * 1. Count frequency of each possible value (0-255 for uint8, -128-127 for int8, 0-1 for bool)
 * 2. Find the two bins that contain the median elements using cumulative frequency
 * 3. Calculate median based on the positions found
 *
 * Time complexity: O(n)
 * Space complexity: O(k) for the frequency array (k = 256 for 8-bit types)
 */
template <typename A, typename T>
typename detail::SimpleArrayMixinCalculators<A, T>::value_type
detail::SimpleArrayMixinCalculators<A, T>::median_freq(small_vector<value_type> & sv) const
{
    SOLVCON_PROFILE_SCOPE("SimpleArray::median_freq()");

    const size_t n = sv.size();
    if (n == 0)
    {
        throw std::runtime_error("median_freq(): empty container");
    }

    uint32_t freq[256] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

    // Each branch indexes freq differently per element type; clang-tidy
    // cannot tell the type-dependent bodies apart in the template.
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (std::is_same_v<value_type, uint8_t>)
    {
        for (uint8_t const v : sv) { ++freq[v]; }
    }
    else if constexpr (std::is_same_v<value_type, int8_t>)
    {
        for (int8_t const v : sv) { ++freq[static_cast<unsigned>(static_cast<int>(v) + 128)]; }
    }
    else
    {
        for (bool const v : sv) { ++freq[v ? 1 : 0]; }
    }
    // NOLINTEND(bugprone-branch-clone)

    int b1, b2;
    find_two_bins(freq, n, b1, b2);

    if constexpr (std::is_same_v<value_type, uint8_t>)
    {
        const int m = (b1 + b2) / 2;
        return static_cast<value_type>(m);
    }
    else if constexpr (std::is_same_v<value_type, int8_t>)
    {
        const int v1 = b1 - 128;
        const int v2 = b2 - 128;
        return static_cast<value_type>((v1 + v2) / 2);
    }
    else
    {
        const uint32_t ones = freq[1];
        return static_cast<value_type>(ones * 2 >= n);
    }
}

/**
 * Perform matrix multiplication for SimpleArrays.
 * This implementation supports 1D x 1D, 1D x 2D, 2D x 1D, and 2D x 2D matrix multiplication.
 */
template <typename A, typename T>
A SimpleArrayMixinCalculators<A, T>::matmul(A const & other) const
{
    auto const * athis = static_cast<A const *>(this);
    SimpleArrayMatmulHelper<A, T> helper(*athis, other);
    return helper.matmul();
}

/**
 * Perform in-place matrix multiplication for SimpleArrays.
 * This implementation supports 1D x 1D, 1D x 2D, 2D x 1D, and 2D x 2D matrix multiplication.
 * The result replaces the content of the current array.
 */
template <typename A, typename T>
A & SimpleArrayMixinCalculators<A, T>::imatmul(A const & other)
{
    auto athis = static_cast<A *>(this);
    A result = athis->matmul(other);
    *athis = std::move(result);

    return *athis;
}

/**
 * Perform matrix multiplication using vendor BLAS when available.
 */
template <typename A, typename T>
A SimpleArrayMixinCalculators<A, T>::matmul_blas(A const & other) const
{
    auto const * athis = static_cast<A const *>(this);
    SimpleArrayMatmulHelper<A, T> helper(*athis, other);
    return helper.matmul_blas();
}

/**
 * Perform in-place matrix multiplication using vendor BLAS when available.
 * The result replaces the content of the current array.
 */
template <typename A, typename T>
A & SimpleArrayMixinCalculators<A, T>::imatmul_blas(A const & other)
{
    auto athis = static_cast<A *>(this);
    A result = athis->matmul_blas(other);
    *athis = std::move(result);

    return *athis;
}

/**
 * Perform fast matrix multiplication for SimpleArrays.
 * This implementation supports 1D x 1D, 1D x 2D, 2D x 1D, and 2D x 2D matrix multiplication.
 */
template <typename A, typename T>
A SimpleArrayMixinCalculators<A, T>::matmul_fast(A const & other,
                                                 size_t tile_x,
                                                 size_t tile_y,
                                                 size_t tile_z) const
{
    auto const * athis = static_cast<A const *>(this);
    SimpleArrayMatmulHelper<A, T> helper(*athis, other, tile_x, tile_y, tile_z);
    return helper.matmul_fast();
}

/**
 * Perform in-place fast matrix multiplication for SimpleArrays.
 * This implementation supports 1D x 1D, 1D x 2D, 2D x 1D, and 2D x 2D matrix multiplication.
 * The result replaces the content of the current array.
 */
template <typename A, typename T>
A & SimpleArrayMixinCalculators<A, T>::imatmul_fast(A const & other,
                                                    size_t tile_x,
                                                    size_t tile_y,
                                                    size_t tile_z)
{
    auto athis = static_cast<A *>(this);
    A result = athis->matmul_fast(other, tile_x, tile_y, tile_z);
    *athis = std::move(result);

    return *athis;
}

/**
 * Find the two bins that correspond to the median values for frequency-based median calculation.
 * This function is used for small data types (uint8_t, int8_t, bool) where frequency counting
 * is more efficient than sorting.
 */
template <typename A, typename T>
void SimpleArrayMixinCalculators<A, T>::find_two_bins(const uint32_t * freq, size_t n, int & bin1, int & bin2)
{
    const size_t k1 = (n - 1) / 2;
    const size_t k2 = n / 2;

    uint32_t cumulative = 0;
    bin1 = -1;
    bin2 = -1;

    for (int i = 0; i < 256; ++i)
    {
        cumulative += freq[i];
        if (bin1 < 0 && cumulative > k1)
        {
            bin1 = i;
        }
        if (bin2 < 0 && cumulative > k2)
        {
            bin2 = i;
        }
        if (bin1 >= 0 && bin2 >= 0)
        {
            break;
        }
    }
}

template <typename A, typename T>
class SimpleArrayMixinSort
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;

    void sort();
    SimpleArray<uint64_t> argsort();
    template <IntegralType I>
    A take_along_axis(SimpleArray<I> const & indices);
    template <IntegralType I>
    A take_along_axis_simd(SimpleArray<I> const & indices);

}; /* end class SimpleArrayMixinSort */

template <typename A, typename T>
void SimpleArrayMixinSort<A, T>::sort()
{
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        throw std::runtime_error(
            std::format(
                "SimpleArray::sort(): currently only support 1D array but the array is {} dimension", athis->ndim()));
    }

    std::sort(athis->begin(), athis->end());
}

template <typename T, IntegralType I>
void indexed_copy(T * dest, T const * data, I const * begin, I const * const end); // NOLINT(readability-avoid-const-params-in-decls)

template <IntegralType T>
T const * check_index_range(SimpleArray<T> const & indices, size_t max_idx);

template <typename A, typename T>
class SimpleArrayMixinSearch
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;

    size_t argmin() const
    {
        size_t min_index = 0;
        value_type min_value = std::numeric_limits<value_type>::max();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) < min_value)
            {
                min_value = athis->data(i);
                min_index = i;
            }
        }
        return min_index;
    }

    size_t argmax() const
    {
        size_t max_index = 0;
        value_type max_value = std::numeric_limits<value_type>::lowest();
        auto athis = static_cast<A const *>(this);
        for (size_t i = 0; i < athis->size(); ++i)
        {
            if (athis->data(i) > max_value)
            {
                max_value = athis->data(i);
                max_index = i;
            }
        }
        return max_index;
    }

    SimpleArray<uint64_t> argwhere() const;

}; /* end class SimpleArrayMixinSearch */

template <typename A, typename T>
class SimpleArrayMixinMatrix
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:

    using value_type = typename internal_types::value_type;
    using shape_type = typename internal_types::shape_type;
    using sshape_type = typename internal_types::sshape_type;

    static A eye(ssize_t n)
    {
        validate_positive("eye", n);
        shape_type const shape{static_cast<size_t>(n), static_cast<size_t>(n)};
        A result(shape, static_cast<value_type>(0));

        for (ssize_t i = 0; i < n; ++i)
        {
            result(i, i) = static_cast<value_type>(1);
        }

        return result;
    }

    static A scaled_eye(ssize_t n, value_type scale)
    {
        validate_positive("scaled_eye", n);
        shape_type const shape{static_cast<size_t>(n), static_cast<size_t>(n)};
        A result(shape, static_cast<value_type>(0));

        for (ssize_t i = 0; i < n; ++i)
        {
            result(i, i) = scale;
        }

        return result;
    }

    A hermitian() const
    {
        auto athis = static_cast<A const *>(this);
        validate_2d("hermitian");
        A result = *athis;
        result.transpose();
        if constexpr (is_complex_v<value_type>)
        {
            for (auto ptr = result.begin(); ptr != result.end(); ++ptr)
            {
                *ptr = ptr->conj();
            }
        }
        return result;
    }

    A symmetrize() const
    {
        auto athis = static_cast<A const *>(this);
        validate_square("symmetrize");
        A result = *athis;
        if constexpr (is_complex_v<value_type>)
        {
            for (ssize_t i = 0; i < athis->shape(0); ++i)
            {
                for (ssize_t j = 0; j < athis->shape(1); ++j)
                {
                    result(i, j) = (result(i, j) + (*athis)(j, i).conj()) / static_cast<value_type>(2.0);
                }
            }
        }
        else
        {
            for (ssize_t i = 0; i < athis->shape(0); ++i)
            {
                for (ssize_t j = 0; j < athis->shape(1); ++j)
                {
                    result(i, j) = (result(i, j) + (*athis)(j, i)) / static_cast<value_type>(2.0);
                }
            }
        }
        return result;
    }

    value_type trace() const
    {
        auto athis = static_cast<A const *>(this);
        validate_square("trace");
        auto result = static_cast<value_type>(0);
        for (ssize_t i = 0; i < athis->shape(0); ++i)
        {
            result += (*athis)(i, i);
        }
        return result;
    }

    A pow(ssize_t n) const;

private:

    static void validate_positive(const char * operation_name, ssize_t n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument(
                std::format("SimpleArray::{}(): size must be greater than 0, but got {}",
                            operation_name,
                            n));
        }
    }

    void validate_2d(const char * operation_name) const
    {
        auto athis = static_cast<A const *>(this);
        if (athis->ndim() != 2)
        {
            throw std::runtime_error(
                std::format("SimpleArray::{}(): operation requires 2D SimpleArray, "
                            "but got {}D SimpleArray",
                            operation_name,
                            athis->ndim()));
        }
    }

    void validate_square(const char * operation_name) const
    {
        auto athis = static_cast<A const *>(this);
        validate_2d(operation_name);
        if (athis->shape(0) != athis->shape(1))
        {
            throw std::runtime_error(
                std::format("SimpleArray::{}(): operation requires square SimpleArray, "
                            "but got {}x{} shape",
                            operation_name,
                            athis->shape(0),
                            athis->shape(1)));
        }
    }

}; /* end class SimpleArrayMixinMatrix */

/**
 * Compute the matrix power A^n for a square SimpleArray and a non-negative
 * integer exponent. The exponentiation-by-squaring algorithm keeps the
 * number of matrix multiplications at O(log n). A^0 is the identity matrix.
 * Negative exponents are not supported because they require matrix
 * inversion.
 */
template <typename A, typename T>
A SimpleArrayMixinMatrix<A, T>::pow(ssize_t n) const
{
    auto const * athis = static_cast<A const *>(this);
    validate_square("pow");
    if (n < 0)
    {
        throw std::invalid_argument(
            std::format("SimpleArray::pow(): exponent must be non-negative, "
                        "but got {}",
                        n));
    }

    auto const dim = static_cast<ssize_t>(athis->shape(0));
    A result = eye(dim);
    if (n == 0)
    {
        return result;
    }

    // Exponentiation by squaring.  n is non-negative here, so scanning its
    // bits is well-defined.
    A base = *athis;
    while (n > 0)
    {
        if ((n & 1) != 0)
        {
            result.imatmul(base);
        }
        n >>= 1;
        if (n > 0)
        {
            base.imatmul(base);
        }
    }
    return result;
}

} /* end namespace detail */

// Tag type for explicit alignment constructor
struct with_alignment_t
{
};

/**
 * Simple array type for contiguous memory storage. Size does not change. The
 * copy semantics performs data copy. The move semantics invalidates the
 * existing memory buffer.
 *
 * @ingroup group_core
 */
template <typename T>
class SimpleArray
    : public detail::SimpleArrayMixinModifiers<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinSum<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinCalculators<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinSort<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinSearch<SimpleArray<T>, T>
    , public detail::SimpleArrayMixinMatrix<SimpleArray<T>, T>
{

private:

    using internal_types = detail::SimpleArrayInternalTypes<T>;

public:
    template <typename U>
    using rebind = SimpleArray<U>;
    using value_type = typename internal_types::value_type;
    using shape_type = typename internal_types::shape_type;
    using sshape_type = typename internal_types::sshape_type;
    using buffer_type = typename internal_types::buffer_type;

    static constexpr size_t ITEMSIZE = sizeof(value_type);

    static constexpr size_t itemsize() { return ITEMSIZE; }

    /// Constructor with length and optional alignment
    /// @param length the length of the array in items
    /// @param alignment the memory alignment in bytes (default: 0, no special alignment, and valid values are 16, 32, and 64)
    explicit SimpleArray(size_t length, size_t alignment = 0)
        : m_buffer(buffer_type::construct(length * ITEMSIZE, alignment))
        , m_shape{length}
        , m_stride{1}
        , m_body(m_buffer->template data<T>())
    {
    }

    template <InputIterator InputIt>
    SimpleArray(InputIt first, InputIt last, size_t alignment = 0)
        : SimpleArray(last - first, alignment)
    {
        std::copy(first, last, data());
    }

    // NOLINTNEXTLINE(modernize-pass-by-value)
    explicit SimpleArray(shape_type const & shape)
        : m_shape(shape)
        , m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty())
        {
            m_buffer = buffer_type::construct(m_shape[0] * static_cast<size_t>(m_stride[0]) * ITEMSIZE, 0);
            m_body = m_buffer->template data<T>();
        }
    }

    // NOLINTNEXTLINE(modernize-pass-by-value)
    SimpleArray(shape_type const & shape, size_t alignment, with_alignment_t const & /* unnamed argument for tagging */)
        : m_shape(shape)
        , m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty())
        {
            m_buffer = buffer_type::construct(m_shape[0] * static_cast<size_t>(m_stride[0]) * ITEMSIZE, alignment);
            m_body = m_buffer->template data<T>();
        }
    }

    SimpleArray(shape_type const & shape, value_type const & value, size_t alignment)
        : SimpleArray(shape, alignment, with_alignment_t{})
    {
        std::fill(begin(), end(), value);
    }

    SimpleArray(shape_type const & shape, value_type const & value)
        : SimpleArray(shape)
    {
        std::fill(begin(), end(), value);
    }

    explicit SimpleArray(std::vector<size_t> const & shape)
        : m_shape(shape)
        , m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty())
        {
            m_buffer = buffer_type::construct(m_shape[0] * static_cast<size_t>(m_stride[0]) * ITEMSIZE, 0);
            m_body = m_buffer->template data<T>();
        }
    }

    SimpleArray(std::vector<size_t> const & shape, size_t alignment, with_alignment_t)
        : m_shape(shape)
        , m_stride(calc_stride(m_shape))
    {
        if (!m_shape.empty())
        {
            m_buffer = buffer_type::construct(m_shape[0] * static_cast<size_t>(m_stride[0]) * ITEMSIZE, alignment);
            m_body = m_buffer->template data<T>();
        }
    }

    SimpleArray(std::vector<size_t> const & shape, value_type const & value, size_t alignment)
        : SimpleArray(shape, alignment, with_alignment_t{})
    {
        std::fill(begin(), end(), value);
    }

    SimpleArray(std::vector<size_t> const & shape, value_type const & value)
        : SimpleArray(shape)
    {
        std::fill(begin(), end(), value);
    }

    explicit SimpleArray(std::shared_ptr<buffer_type> const & buffer)
    {
        if (buffer)
        {
            const size_t nitem = buffer->nbytes() / ITEMSIZE;
            if (buffer->nbytes() != nitem * ITEMSIZE)
            {
                throw std::runtime_error("SimpleArray: input buffer size must be divisible");
            }
            m_shape = shape_type{nitem};
            m_stride = sshape_type{1};
            m_buffer = buffer;
            m_body = m_buffer->template data<T>();
        }
        else
        {
            throw std::runtime_error("SimpleArray: buffer cannot be null");
        }
    }

    explicit SimpleArray(shape_type const & shape, std::shared_ptr<buffer_type> const & buffer)
        : SimpleArray(buffer)
    {
        if (buffer)
        {
            m_shape = shape;
            m_stride = calc_stride(m_shape);
            const size_t nbytes = m_shape[0] * static_cast<size_t>(m_stride[0]) * ITEMSIZE;
            if (nbytes != buffer->nbytes())
            {
                throw std::runtime_error(
                    std::format("SimpleArray: shape byte count {} differs from buffer {}",
                                nbytes,
                                buffer->nbytes()));
            }
        }
    }

    explicit SimpleArray(shape_type const & shape,
                         sshape_type const & stride,
                         std::shared_ptr<buffer_type> const & buffer,
                         bool c_contiguous,
                         bool f_contiguous)
        : SimpleArray(shape, stride, buffer)
    {
        if (shape.size() != stride.size())
        {
            throw std::runtime_error("SimpleArray: shape and stride size mismatch");
        }
        if (c_contiguous)
        {
            check_c_contiguous(shape, stride);
        }
        if (f_contiguous)
        {
            check_f_contiguous(shape, stride);
        }
    }

    explicit SimpleArray(shape_type const & shape,
                         sshape_type const & stride,
                         std::shared_ptr<buffer_type> const & buffer)
        : SimpleArray(buffer)
    {
        if (buffer)
        {
            m_shape = shape;
            m_stride = stride;
        }
    }

    SimpleArray(std::initializer_list<T> init)
        : SimpleArray(init.size())
    {
        std::copy_n(init.begin(), init.size(), data());
    }

    SimpleArray()
        : SimpleArray(0)
    {
    }

    SimpleArray(SimpleArray const & other)
        : m_buffer(other.m_buffer->clone())
        , m_shape(other.m_shape)
        , m_stride(other.m_stride)
        , m_nghost(other.m_nghost)
        , m_body(calc_body(m_buffer->template data<T>(), m_stride, other.m_nghost))
    {
    }

    SimpleArray(SimpleArray && other) noexcept
        : m_buffer(std::move(other.m_buffer))
        , m_shape(std::move(other.m_shape))
        , m_stride(std::move(other.m_stride))
        , m_nghost(other.m_nghost)
        , m_body(other.m_body)
    {
    }

    SimpleArray & operator=(SimpleArray const & other)
    {
        if (this != &other)
        {
            *m_buffer = *(other.m_buffer); // Size is checked inside.
            m_shape = other.m_shape;
            m_stride = other.m_stride;
            m_nghost = other.m_nghost;
            m_body = calc_body(m_buffer->template data<T>(), m_stride, other.m_nghost);
        }
        return *this;
    }

    SimpleArray & operator=(SimpleArray && other) noexcept
    {
        if (this != &other)
        {
            m_buffer = std::move(other.m_buffer);
            m_shape = std::move(other.m_shape);
            m_stride = std::move(other.m_stride);
            m_nghost = other.m_nghost;
            m_body = other.m_body;
        }
        return *this;
    }

    ~SimpleArray() = default;

    template <typename... Args>
    SimpleArray & remake(Args &&... args)
    {
        SimpleArray(std::forward<Args>(args)...).swap(*this);
        return *this;
    }

    static sshape_type calc_stride(shape_type const & shape)
    {
        sshape_type stride(shape.size());
        if (!shape.empty())
        {
            stride[shape.size() - 1] = 1;
            for (size_t it = shape.size() - 1; it > 0; --it)
            {
                stride[it - 1] = stride[it] * static_cast<ssize_t>(shape[it]);
            }
        }
        return stride;
    }

    static T * calc_body(T * data, sshape_type const & stride, size_t nghost)
    {
        if (nullptr == data || stride.empty() || 0 == nghost)
        {
            // Do nothing.
        }
        else
        {
            sshape_type idx(stride.size(), 0);
            idx[0] = static_cast<ssize_t>(nghost);
            data += buffer_offset(stride, idx);
        }
        return data;
    }

    explicit operator bool() const noexcept { return bool(m_buffer) && bool(*m_buffer); }

    size_t nbytes() const noexcept { return size() * ITEMSIZE; }
    /**
     * In the following numpy documentaions,
     * `size()` are calculated based on the number of the elements.
     * https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.size.html
     * SimpleArray matchs the behavior of numpy.ndarray.
     */
    size_t size() const noexcept
    {
        size_t size = 1;
        for (size_t it = 0; it < m_shape.size(); ++it)
        {
            size *= m_shape[it];
        }
        return size;
    }

    /// Return the underlying buffer alignment in bytes. If no buffer or no alignment, return 0.
    size_t alignment() const noexcept { return m_buffer ? m_buffer->alignment() : 0; }

    using iterator = T *;
    using const_iterator = T const *;

    iterator begin() noexcept { return data(); }
    iterator end() noexcept { return data() + size(); }
    const_iterator begin() const noexcept { return data(); }
    const_iterator end() const noexcept { return data() + size(); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }

    value_type const & operator[](size_t it) const noexcept { return data(it); }
    value_type & operator[](size_t it) noexcept { return data(it); }

    value_type const & at(ssize_t it) const
    {
        sshape_type const idx{normalize_index(it)};
        ssize_t const offset = buffer_offset(m_stride, idx);
        return *(data() + offset);
    }
    value_type & at(ssize_t it)
    {
        sshape_type const idx{normalize_index(it)};
        ssize_t const offset = buffer_offset(m_stride, idx);
        return *(data() + offset);
    }

    value_type const & at(std::vector<ssize_t> const & idx) const { return at(sshape_type(idx)); }
    value_type & at(std::vector<ssize_t> const & idx) { return at(sshape_type(idx)); }

    value_type const & at(sshape_type const & sidx) const
    {
        sshape_type const idx = normalize_index(sidx);
        ssize_t const offset = buffer_offset(m_stride, idx);
        return *(data() + offset);
    }
    value_type & at(sshape_type const & sidx)
    {
        sshape_type const idx = normalize_index(sidx);
        ssize_t const offset = buffer_offset(m_stride, idx);
        return *(data() + offset);
    }

    size_t ndim() const noexcept { return m_shape.size(); }
    shape_type const & shape() const { return m_shape; }
    size_t shape(size_t it) const noexcept { return m_shape[it]; }
    size_t & shape(size_t it) noexcept { return m_shape[it]; }
    sshape_type const & stride() const { return m_stride; }
    ssize_t stride(size_t it) const noexcept { return m_stride[it]; }
    ssize_t & stride(size_t it) noexcept { return m_stride[it]; }

    size_t nghost() const { return m_nghost; }
    size_t nbody() const { return m_shape.empty() ? 0 : m_shape[0] - m_nghost; }
    bool has_ghost() const { return m_nghost != 0; }
    void set_nghost(size_t nghost)
    {
        if (0 != nghost)
        {
            if (0 == ndim())
            {
                throw std::out_of_range(
                    std::format("SimpleArray: cannot set nghost {} > 0 to an empty array", nghost));
            }
            if (nghost > shape(0))
            {
                throw std::out_of_range(
                    std::format("SimpleArray: cannot set nghost {} > shape(0) {}",
                                nghost,
                                shape(0)));
            }
        }
        m_nghost = nghost;
        if (bool(*this))
        {
            m_body = calc_body(m_buffer->template data<T>(), m_stride, m_nghost);
        }
    }

    template <typename U>
    SimpleArray<U> reshape(shape_type const & shape) const
    {
        return SimpleArray<U>(shape, m_buffer);
    }

    SimpleArray reshape(shape_type const & shape) const
    {
        return SimpleArray(shape, m_buffer);
    }

    SimpleArray reshape() const
    {
        return SimpleArray(m_shape, m_buffer);
    }

    void swap(SimpleArray & other) noexcept
    {
        if (this != &other)
        {
            std::swap(m_buffer, other.m_buffer);
            std::swap(m_shape, other.m_shape);
            std::swap(m_stride, other.m_stride);
            std::swap(m_nghost, other.m_nghost);
            std::swap(m_body, other.m_body);
        }
    }

    void transpose(bool copy = false);

    void transpose(shape_type const & axis, bool copy = false);

    SimpleArray transpose_copy() const;

    SimpleArray to_row_major() const;

    SimpleArray to_column_major() const;

    template <typename... Args>
    value_type const & operator()(Args... args) const { return *vptr(args...); }
    template <typename... Args>
    value_type & operator()(Args... args) { return *vptr(args...); }

    template <typename... Args>
    value_type const * vptr(Args... args) const { return m_body + buffer_offset(m_stride, args...); }
    template <typename... Args>
    value_type * vptr(Args... args) { return m_body + buffer_offset(m_stride, args...); }

    std::span<value_type> as_span()
    {
        return std::span<value_type>(data(), buffer().nbytes() / ITEMSIZE);
    }
    std::span<value_type const> as_span() const
    {
        return std::span<value_type const>(data(), buffer().nbytes() / ITEMSIZE);
    }

    template <size_t N>
    std::mdspan<value_type, std::dextents<size_t, N>, std::layout_stride> as_mdspan()
    {
        if (ndim() != N)
        {
            throw std::out_of_range(
                std::format("SimpleArray::as_mdspan: rank {} does not match ndim() {}", N, ndim()));
        }
        return std::mdspan<value_type, std::dextents<size_t, N>, std::layout_stride>(
            data(), make_mdspan_mapping<N>());
    }

    template <size_t N>
    std::mdspan<value_type const, std::dextents<size_t, N>, std::layout_stride> as_mdspan() const
    {
        if (ndim() != N)
        {
            throw std::out_of_range(
                std::format("SimpleArray::as_mdspan: rank {} does not match ndim() {}", N, ndim()));
        }
        return std::mdspan<value_type const, std::dextents<size_t, N>, std::layout_stride>(
            data(), make_mdspan_mapping<N>());
    }

    /* Backdoor */
    value_type const & data(size_t it) const { return data()[it]; }
    value_type & data(size_t it) { return data()[it]; }
    value_type const * data() const { return buffer().template data<value_type>(); }
    value_type * data() { return buffer().template data<value_type>(); }

    buffer_type const & buffer() const { return *m_buffer; }
    buffer_type & buffer() { return *m_buffer; }

    value_type const * body() const { return m_body; }
    value_type * body() { return m_body; }

    bool is_c_contiguous() const { return is_c_contiguous(m_shape, m_stride); }
    bool is_f_contiguous() const { return is_f_contiguous(m_shape, m_stride); }

private:
    void copy_logical_into(SimpleArray & out) const;

    template <size_t N, size_t... I>
    std::layout_stride::mapping<std::dextents<size_t, N>> make_mdspan_mapping_impl(std::index_sequence<I...>) const
    {
        std::array<size_t, N> strides;
        for (size_t i = 0; i < N; ++i)
        {
            if (stride(i) < 0)
            {
                throw std::runtime_error("SimpleArray::as_mdspan: negative stride is not supported");
            }
            strides[i] = static_cast<size_t>(stride(i));
        }
        return std::layout_stride::mapping<std::dextents<size_t, N>>(
            std::dextents<size_t, N>(shape(I)...), strides);
    }

    template <size_t N>
    std::layout_stride::mapping<std::dextents<size_t, N>> make_mdspan_mapping() const
    {
        return make_mdspan_mapping_impl<N>(std::make_index_sequence<N>{});
    }

    static bool is_c_contiguous(shape_type const & shape,
                                sshape_type const & stride)
    {
        if (stride[stride.size() - 1] != 1)
        {
            return false;
        }
        for (size_t it = 0; it < shape.size() - 1; ++it)
        {
            if (stride[it] != static_cast<ssize_t>(shape[it + 1]) * stride[it + 1])
            {
                return false;
            }
        }
        return true;
    }

    static bool is_f_contiguous(shape_type const & shape,
                                sshape_type const & stride)
    {
        if (stride[0] != 1)
        {
            return false;
        }
        for (size_t it = 0; it < shape.size() - 1; ++it)
        {
            if (stride[it + 1] != static_cast<ssize_t>(shape[it]) * stride[it])
            {
                return false;
            }
        }
        return true;
    }

    static void check_c_contiguous(shape_type const & shape,
                                   sshape_type const & stride)
    {
        if (!is_c_contiguous(shape, stride))
        {
            throw std::runtime_error("SimpleArray: C contiguous stride must match shape and end with 1");
        }
    }

    void check_f_contiguous(shape_type const & shape,
                            sshape_type const & stride) const
    {
        if (!is_f_contiguous(shape, stride))
        {
            throw std::runtime_error("SimpleArray: F contiguous stride must match shape and start with 1");
        }
    }

    ssize_t normalize_index(ssize_t it) const
    {
        if (ndim() != 1)
        {
            throw std::out_of_range(
                std::format("SimpleArray::normalize_index(): "
                            "cannot use scalar index for {}-dimensional array",
                            ndim()));
        }
        auto const dim_length = static_cast<ssize_t>(shape(0));
        auto const ghost_offset = static_cast<ssize_t>(m_nghost);
        ssize_t const shifted_index = it + ghost_offset;

        // This overload handles scalar indexing, not slicing. Python sequence
        // indexing substitutes "len(s) + i" for a negative i and raises
        // "IndexError" when i is outside the sequence range. Therefore the
        // normalized index must satisfy 0 <= len(s) + i < len(s), so the valid
        // scalar interval is [-len(s), len(s)).
        // https://docs.python.org/3/library/stdtypes.html#common-sequence-operations
        if (shifted_index < -dim_length)
        {
            throw std::out_of_range(
                std::format("SimpleArray: index {} < -nghost - shape[0]: {}",
                            it,
                            -dim_length - ghost_offset));
        }
        if (shifted_index >= dim_length)
        {
            throw std::out_of_range(
                std::format("SimpleArray: index {} >= {} (shape[0]: {} - nghost: {})",
                            it,
                            dim_length - ghost_offset,
                            shape(0),
                            m_nghost));
        }
        return to_nonnegative_index(shifted_index, dim_length);
    }

    sshape_type normalize_index(small_vector<ssize_t> const & idx) const
    {
        auto index2string = [&idx]() -> std::string
        {
            if (idx.empty())
            {
                return "[]";
            }
            std::string result = "[";
            for (size_t it = 0; it < idx.size(); ++it)
            {
                if (it > 0)
                {
                    result += ", ";
                }
                result += std::to_string(idx[it]);
            }
            result += "]";
            return result;
        };

        // Test for the "index shape".
        if (idx.empty())
        {
            throw std::out_of_range("SimpleArray::normalize_index(): empty index");
        }
        if (idx.size() != m_shape.size())
        {
            throw std::out_of_range(
                std::format("SimpleArray: dimension of input indices {} != array dimension {}",
                            index2string(),
                            m_shape.size()));
        }

        sshape_type normalized(idx.size());
        for (size_t dim = 0; dim < m_shape.size(); ++dim)
        {
            auto const dim_length = static_cast<ssize_t>(m_shape[dim]);
            ssize_t ghost_offset = 0;
            if (dim == 0)
            {
                ghost_offset = static_cast<ssize_t>(m_nghost);
            }
            ssize_t const shifted_index = idx[dim] + ghost_offset;
            if (shifted_index < -dim_length)
            {
                if (dim == 0)
                {
                    throw std::out_of_range(
                        std::format("SimpleArray: dim 0 in {} < -nghost - shape[0]: {}",
                                    index2string(),
                                    -dim_length - ghost_offset));
                }
                throw std::out_of_range(std::format("SimpleArray: dim {} in {} < 0",
                                                    dim,
                                                    index2string()));
            }
            if (shifted_index >= dim_length)
            {
                if (dim == 0)
                {
                    throw std::out_of_range(
                        std::format("SimpleArray: dim 0 in {} >= nbody: {} (shape[0]: {} - nghost: {})",
                                    index2string(),
                                    dim_length - ghost_offset,
                                    m_shape[0],
                                    nghost()));
                }
                throw std::out_of_range(
                    std::format("SimpleArray: dim {} in {} >= shape[{}]: {}",
                                dim,
                                index2string(),
                                dim,
                                m_shape[dim]));
            }
            normalized[dim] = to_nonnegative_index(shifted_index, dim_length);
        }
        return normalized;
    }

    static ssize_t to_nonnegative_index(ssize_t index, ssize_t length)
    {
        if (index < 0)
        {
            index += length;
        }
        return index;
    }

    /// Contiguous data buffer for the array.
    std::shared_ptr<buffer_type> m_buffer;
    /// Each element in this vector is the number of element in the
    /// corresponding dimension.
    shape_type m_shape;
    /// Each element in this vector is the number of elements (not number of
    /// bytes) to skip for advancing an index in the corresponding dimension.
    sshape_type m_stride;

    size_t m_nghost = 0;
    value_type * m_body = nullptr;
}; /* end class SimpleArray */

template <typename A, typename T>
SimpleArray<bool> detail::SimpleArrayMixinCalculators<A, T>::eq(A const & other) const
{
    auto const * athis = static_cast<A const *>(this);
    if (athis->shape() != other.shape())
    {
        throw std::invalid_argument(
            std::format("SimpleArray::eq(): shape mismatch: this={} other={}",
                        format_shape(athis->shape()),
                        format_shape(other.shape())));
    }
    SimpleArray<bool> ret(athis->shape());
    const value_type * ptr = athis->begin();
    const value_type * const end = athis->end();
    const value_type * other_ptr = other.begin();
    bool * ret_ptr = ret.begin();
    while (ptr < end)
    {
        *ret_ptr = (*ptr == *other_ptr);
        ++ptr;
        ++other_ptr;
        ++ret_ptr;
    }
    return ret;
}

template <typename A, typename T>
SimpleArray<bool> detail::SimpleArrayMixinCalculators<A, T>::eq(value_type scalar) const
{
    auto const * athis = static_cast<A const *>(this);
    SimpleArray<bool> ret(athis->shape());
    const value_type * ptr = athis->begin();
    const value_type * const end = athis->end();
    bool * ret_ptr = ret.begin();
    while (ptr < end)
    {
        *ret_ptr = (*ptr == scalar);
        ++ptr;
        ++ret_ptr;
    }
    return ret;
}

/**
 * @brief Type-erased element-copy kernels over a pair of ConcreteBuffers.
 *
 * @details
 *      Non-template helper bundling the family of physical copy kernels that
 *      SimpleArray::copy_logical_into routes to.  Construction binds source
 *      and destination ConcreteBuffers together with the layout metadata
 *      (shape, element-unit strides, itemsize); each member function selects a
 *      specific kernel.  Picking the kernel that suits the stride and
 *      contiguity is the caller's job.
 */
class SimpleArrayCopier
{

public:

    using shape_type = small_vector<size_t>;
    using sshape_type = detail::sshape_type;
    using buffer_type = ConcreteBuffer;

    SimpleArrayCopier(
        buffer_type const & src_buffer,
        ssize_t src_body_offset,
        sshape_type const & src_stride,
        buffer_type & dst_buffer,
        ssize_t dst_body_offset,
        sshape_type const & dst_stride,
        shape_type const & shape,
        size_t itemsize);

    SimpleArrayCopier() = delete;
    SimpleArrayCopier(SimpleArrayCopier const &) = delete;
    SimpleArrayCopier(SimpleArrayCopier &&) = delete;
    SimpleArrayCopier & operator=(SimpleArrayCopier const &) = delete;
    SimpleArrayCopier & operator=(SimpleArrayCopier &&) = delete;
    ~SimpleArrayCopier() = default;

    void memcpy() const;
    void tiled_2d() const;
    void tiled_nd() const;
    void naive() const;

private:

    int8_t const * m_src;
    int8_t * m_dst;
    shape_type const & m_shape;
    sshape_type const & m_src_stride;
    sshape_type const & m_dst_stride;
    size_t m_itemsize;

}; /* end class SimpleArrayCopier */

/**
 * @brief Transpose by reversing all axes in place.
 *
 * @param copy
 *      When false (default), perform a view-only flip.  When true, replace the
 *      buffer with a freshly allocated C-contiguous one holding the physically
 *      transposed contents.
 */
template <typename T>
void SimpleArray<T>::transpose(bool copy)
{
    if (!copy)
    {
        std::reverse(m_shape.begin(), m_shape.end());
        std::reverse(m_stride.begin(), m_stride.end());
        return;
    }
    SimpleArray tc = transpose_copy();
    swap(tc);
}

/**
 * @brief Transpose with axes permuted by the given index vector in place.
 *
 * @param axis
 *      Permutation: the i-th new axis is sourced from the `axis[i]`-th old
 *      axis.  Must be a valid permutation of `[0, ndim)`.
 * @param copy
 *      When false (default), perform a view-only permutation of the metadata.
 *      When true, physically permute contents into a freshly allocated
 *      C-contiguous buffer.
 */
template <typename T>
void SimpleArray<T>::transpose(shape_type const & axis, bool copy)
{
    // Build the permuted shape and stride by gathering each axis from the
    // source.  It is needed by both view-only and with-copy transpose.
    if (axis.size() != m_shape.size())
    {
        throw std::runtime_error("SimpleArray::transpose: axis size mismatch");
    }
    shape_type new_shape(m_shape.size(), -1);
    sshape_type new_stride(m_stride.size());
    for (size_t it = 0; it < m_shape.size(); ++it)
    {
        if (axis[it] >= m_shape.size() || axis[it] < 0)
        {
            throw std::runtime_error("SimpleArray::transpose: axis out of range");
        }
        if (new_shape[it] != -1)
        {
            throw std::runtime_error("SimpleArray::transpose: axis already set");
        }
        new_shape[it] = m_shape[axis[it]];
        new_stride[it] = m_stride[axis[it]];
    }
    if (!copy)
    {
        // View-only flip: install the permuted metadata in place.
        m_shape = new_shape;
        m_stride = new_stride;
        return;
    }
    // Deep-copy path: 1. stage a strided view with the permuted shape/stride.
    SimpleArray const view(new_shape, new_stride, m_buffer);
    SimpleArray out(new_shape);
    // 2. Use the same helper to keep the iteration logic the same as the other
    // deep-copy variants.
    view.copy_logical_into(out);
    swap(out);
}

/**
 * @brief Transpose with a fresh C-contiguous array returned.
 *
 * @details
 *      The column-major byte layout against the original shape is structurally
 *      identical to the C-contiguous byte layout of a transposed array.
 *
 * @return Freshly allocated C-contiguous SimpleArray with reversed axes.
 */
template <typename T>
SimpleArray<T> SimpleArray<T>::transpose_copy() const
{
    // 0-D and 1-D arrays have no axes to reverse; a plain clone suffices.
    if (m_shape.size() <= 1)
    {
        return SimpleArray(*this);
    }
    // Use to_column_major() to copy the array and then flip the shape and
    // stride back.
    SimpleArray out = to_column_major();
    std::reverse(out.m_shape.begin(), out.m_shape.end());
    std::reverse(out.m_stride.begin(), out.m_stride.end());
    return out;
}

/**
 * @brief
 *      Return a fresh C-contiguous (row-major) array with the same logical
 *      shape and values as `*this`.
 *
 * @details
 *      Clones the buffer when the source is already C-contiguous; otherwise
 *      allocates a fresh buffer and copies element-wise.
 *
 * @return Freshly allocated C-contiguous SimpleArray.
 */
template <typename T>
SimpleArray<T> SimpleArray<T>::to_row_major() const
{
    if (is_c_contiguous())
    {
        return SimpleArray(*this);
    }
    SimpleArray out(m_shape);
    copy_logical_into(out);
    return out;
}

/**
 * @brief
 *      Return a fresh F-contiguous (column-major) array with the same logical
 *      shape and values as `*this`.
 *
 * @details
 *      Clones the buffer when the source is already F-contiguous; otherwise
 *      allocates a fresh buffer with F-contiguous strides and copies
 *      element-wise.
 *
 * @return Freshly allocated F-contiguous SimpleArray.
 */
template <typename T>
SimpleArray<T> SimpleArray<T>::to_column_major() const
{
    // Empty shape or already-F-contiguous source: a buffer clone is enough.
    if (m_shape.empty())
    {
        return SimpleArray(*this);
    }
    if (is_f_contiguous())
    {
        return SimpleArray(*this);
    }
    // Compute column-major strides: the fastest-varying axis is the leading
    // one (stride[0] == 1).
    sshape_type fstride(m_shape.size());
    fstride[0] = 1;
    for (size_t i = 1; i < m_shape.size(); ++i)
    {
        fstride[i] = fstride[i - 1] * m_shape[i - 1];
    }
    // Calculate buffer size.
    size_t nelem = 1;
    for (size_t const s : m_shape)
    {
        nelem *= s;
    }
    // Create a fresh array, copy, and return.
    auto buf = buffer_type::construct(nelem * ITEMSIZE, 0);
    SimpleArray out(m_shape, fstride, buf);
    copy_logical_into(out);
    return out;
}

/**
 * @brief Copy `*this` into `out` element-wise.
 *
 * @details
 *      Heavy-lifting helper to copy data. `out` must share the same shape but
 *      may carry a different stride.  Dispatches the work to one of the
 *      SimpleArrayCopier kernels based on the source/destination layout.
 *
 * @param out Destination array; its shape must match the receiver's.
 */
template <typename T>
void SimpleArray<T>::copy_logical_into(SimpleArray & out) const
{
    if (m_shape.empty())
    {
        return;
    }
    size_t total = 1;
    for (size_t const s : m_shape)
    {
        total *= s;
    }
    if (total == 0)
    {
        return;
    }
    // Subtract on T* so no reinterpret_cast is needed; ITEMSIZE then maps
    // the element offset to the byte offset the helper expects.
    auto const src_body_offset = static_cast<ssize_t>(m_body - m_buffer->template data<value_type>()) * static_cast<ssize_t>(ITEMSIZE);
    auto const dst_body_offset = static_cast<ssize_t>(out.m_body - out.m_buffer->template data<value_type>()) * static_cast<ssize_t>(ITEMSIZE);
    SimpleArrayCopier const copier(
        *m_buffer,
        src_body_offset,
        m_stride,
        *out.m_buffer,
        dst_body_offset,
        out.m_stride,
        m_shape,
        ITEMSIZE);
    // Matching-stride fast-path: when source and destination share the same
    // stride vector and the layout is contiguous, both buffers hold the same
    // byte pattern and a single memcpy moves every element.
    if (m_stride == out.m_stride && (is_c_contiguous() || is_f_contiguous()))
    {
        copier.memcpy();
        return;
    }
    if (m_shape.size() == 2)
    {
        copier.tiled_2d();
        return;
    }
    copier.tiled_nd();
}

template <typename A, typename T>
SimpleArray<uint64_t> detail::SimpleArrayMixinSort<A, T>::argsort()
{
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        throw std::runtime_error(
            std::format("SimpleArray::argsort(): "
                        "currently only support 1D array but the array is {} dimension",
                        athis->ndim()));
    }

    SimpleArray<uint64_t> ret(athis->shape());

    { // Return array initialization
        uint64_t cnt = 0;
        std::for_each(ret.begin(), ret.end(), [&cnt](uint64_t & v)
                      { v = cnt++; });
    }

    value_type const * buf = athis->body();
    auto cmp = [buf](uint64_t a, uint64_t b)
    {
        return buf[a] < buf[b];
    };
    std::sort(ret.begin(), ret.end(), cmp);
    return ret;
}

template <typename A, typename T>
template <IntegralType I>
A detail::SimpleArrayMixinSort<A, T>::take_along_axis(SimpleArray<I> const & indices)
{
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        throw std::runtime_error(
            std::format("SimpleArray::take_along_axis(): "
                        "currently only support 1D array but the array is {} dimension",
                        athis->ndim()));
    }

    size_t max_idx = athis->shape()[0];
    I const * src = indices.begin();
    I const * const end = indices.end();
    while (src < end)
    {
        if (*src < 0 || *src > max_idx)
        {
            size_t const offset = src - indices.begin();
            std::string const indices_str = format_flat_index(indices.shape(), offset);

            throw std::out_of_range(
                std::format("SimpleArray::take_along_axis(): "
                            "indices{} is {}, which is out of range of the array size {}",
                            indices_str,
                            *src,
                            max_idx));
        }
        src++;
    }

    src = indices.begin();
    A ret(indices.shape());
    T const * data = athis->begin();
    T * dst = ret.begin();
    while (src < end)
    {
        T const * valp = data + static_cast<size_t>(*src);
        *dst = *valp;
        ++dst;
        ++src;
    }
    return ret;
}

template <typename A, typename T>
SimpleArray<uint64_t> detail::SimpleArrayMixinSearch<A, T>::argwhere() const
{
    auto athis = static_cast<A const *>(this);

    size_t count = 0;
    for (size_t i = 0; i < athis->size(); ++i)
    {
        if (athis->data(i) != value_type(0))
        {
            ++count;
        }
    }

    size_t const ndim = athis->ndim();
    shape_type res_shape{count, ndim};
    SimpleArray<uint64_t> result(res_shape);

    size_t idx = 0;
    for (size_t i = 0; i < athis->size(); ++i)
    {
        if (athis->data(i) == value_type(0)) { continue; }

        size_t offset = i;
        for (size_t dim = 0; dim < ndim; ++dim)
        {
            size_t stride = athis->stride(dim);
            result(idx, dim) = static_cast<uint64_t>(offset / stride);
            offset %= stride;
        }
        ++idx;
    }

    return result;
}

// The declaration lives in namespace detail and is excluded from the API
// reference (EXCLUDE_SYMBOLS=detail); @cond stops Doxygen from trying to match
// this out-of-line definition against that excluded declaration.
/// @cond
template <IntegralType T>
T const * detail::check_index_range(SimpleArray<T> const & indices, size_t max_idx)
{
    constexpr T DataTypeMax = std::numeric_limits<T>::max();
    constexpr T DataTypeMin = std::numeric_limits<T>::min();
    if (max_idx >= DataTypeMax && DataTypeMin == 0)
    {
        return nullptr;
    }

    return simd::check_between<T>(indices.begin(), indices.end(), 0, max_idx);
}
/// @endcond

template <typename T, IntegralType I>
void detail::indexed_copy(T * dest, T const * data, I const * index0, I const * const index1)
{
    T * dst = dest;
    I const * src = index0;
    while (src < index1)
    {
        T const * valp = data + static_cast<size_t>(*src);
        *dst = *valp;
        ++dst;
        ++src;
    }
}

template <typename A, typename T>
template <IntegralType I>
A detail::SimpleArrayMixinSort<A, T>::take_along_axis_simd(SimpleArray<I> const & indices)
{
    auto athis = static_cast<A *>(this);
    if (athis->ndim() != 1)
    {
        const auto err = std::format("SimpleArray::take_along_axis(): "
                                     "currently only support 1D array but the array is {} dimension",
                                     athis->ndim());
        throw std::runtime_error(err);
    }

    if (indices.size() == 0)
    {
        return A(indices.shape());
    }

    size_t max_idx = athis->shape()[0];

    I const * oor_ptr = check_index_range(indices, max_idx);
    if (oor_ptr != nullptr)
    {
        size_t const offset = oor_ptr - indices.begin();
        std::string const indices_str = format_flat_index(indices.shape(), offset);

        const auto err = std::format("SimpleArray::take_along_axis_simd(): "
                                     "indices{} is {}, which is out of range of the array size {}",
                                     indices_str,
                                     *oor_ptr,
                                     max_idx);
        throw std::out_of_range(err);
    }

    I const * src = indices.begin();
    I const * const end = indices.end();
    A ret(indices.shape());
    T const * data = athis->begin();
    T * dest = ret.begin();
    detail::indexed_copy(dest, data, src, end);
    return ret;
}

template <typename S>
using is_simple_array = std::is_same<
    std::remove_reference_t<S>,
    SimpleArray<typename std::remove_reference_t<S>::value_type>>;

template <typename S>
inline constexpr bool is_simple_array_v = is_simple_array<S>::value; // NOLINT(modernize-type-traits)

using SimpleArrayBool = SimpleArray<bool>;
using SimpleArrayInt8 = SimpleArray<int8_t>;
using SimpleArrayInt16 = SimpleArray<int16_t>;
using SimpleArrayInt32 = SimpleArray<int32_t>;
using SimpleArrayInt64 = SimpleArray<int64_t>;
using SimpleArrayUint8 = SimpleArray<uint8_t>;
using SimpleArrayUint16 = SimpleArray<uint16_t>;
using SimpleArrayUint32 = SimpleArray<uint32_t>;
using SimpleArrayUint64 = SimpleArray<uint64_t>;
using SimpleArrayFloat32 = SimpleArray<float>;
using SimpleArrayFloat64 = SimpleArray<double>;
using SimpleArrayComplex64 = SimpleArray<Complex<float>>;
using SimpleArrayComplex128 = SimpleArray<Complex<double>>;

/**
 * Runtime element-type tag mirroring the scalar types a SimpleArray supports.
 *
 * Covers bool, the signed and unsigned 8- to 64-bit integers, the 32- and
 * 64-bit floats, and the 64- and 128-bit complex types. Converts to and from
 * its enum and a type string, and DataType::from<T>() maps a C++ type to its
 * tag.
 *
 * @ingroup group_core
 */
class DataType
{
public:
    enum enum_type : std::uint8_t
    {
        Undefined,
        Bool,
        Int8,
        Int16,
        Int32,
        Int64,
        Uint8,
        Uint16,
        Uint32,
        Uint64,
        Float32,
        Float64,
        Complex64,
        Complex128
    }; /* end enum enum_type */

    DataType() = default;

    constexpr DataType(const enum_type datatype) // NOLINT(google-explicit-constructor)
        : m_data_type(datatype)
    {
    }

    DataType(const std::string & data_type_string); // NOLINT(google-explicit-constructor)

    enum_type type() const { return m_data_type; }

    constexpr operator enum_type() const { return m_data_type; } // Allow implicit switch and comparisons. // NOLINT(google-explicit-constructor)
    explicit operator bool() const = delete; // Prevent usage: if(datatype)

    template <typename T>
    static DataType from();

private:
    enum_type m_data_type;

}; /* end class DataType */

/**
 * Type-erased SimpleArray pairing a buffer and shape with a runtime DataType.
 *
 * Holds an array of any supported element type behind one handle, dispatching
 * on the DataType, so code generic over the element type can store and pass
 * arrays uniformly.
 *
 * @ingroup group_core
 */
class SimpleArrayPlex
{
public:
    using shape_type = detail::shape_type;

    SimpleArrayPlex() = default;

    explicit SimpleArrayPlex(const shape_type & shape, const std::string & data_type_string)
        : SimpleArrayPlex(shape, DataType(data_type_string))
    {
    }

    explicit SimpleArrayPlex(const shape_type & shape, const std::shared_ptr<ConcreteBuffer> & buffer, const std::string & data_type_string)
        : SimpleArrayPlex(shape, buffer, DataType(data_type_string))
    {
    }

    explicit SimpleArrayPlex(const shape_type & shape, DataType data_type);
    explicit SimpleArrayPlex(const shape_type & shape, const std::shared_ptr<ConcreteBuffer> & buffer, DataType data_type);
    explicit SimpleArrayPlex(const shape_type & shape, DataType data_type, size_t alignment);

    template <typename T>
    // FIXME: NOLINTNEXTLINE(google-explicit-constructor)
    SimpleArrayPlex(const SimpleArray<T> & array)
        : m_has_instance_ownership(true)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        , m_instance_ptr(reinterpret_cast<void *>(new SimpleArray<T>(array)))
        , m_data_type(DataType::from<T>())
    {
    }

    SimpleArrayPlex(SimpleArrayPlex const & other);
    SimpleArrayPlex(SimpleArrayPlex && other) noexcept;
    SimpleArrayPlex & operator=(SimpleArrayPlex const & other);
    SimpleArrayPlex & operator=(SimpleArrayPlex && other) noexcept;

    ~SimpleArrayPlex();

    DataType data_type() const
    {
        return m_data_type;
    }

    size_t alignment() const;

    /// Get the pointer to the const instance of SimpleArray<T>.
    const void * instance_ptr() const
    {
        return m_instance_ptr;
    }

    /// Get the pointer to the mutable instance of SimpleArray<T>.
    void * mutable_instance_ptr() const
    {
        return m_instance_ptr;
    }

    /// TODO: add all SimpleArray public methods

private:
    bool m_has_instance_ownership = false; /// ownership of the instance
    void * m_instance_ptr = nullptr; /// the pointer of the SimpleArray<T> instance
    DataType m_data_type = DataType::Undefined; /// the data type for array casting
}; /* end class SimpleArrayPlex */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
