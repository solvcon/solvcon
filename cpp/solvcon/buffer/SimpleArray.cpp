/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/math/math.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>

namespace solvcon
{

/**
 * Typed element-copy helper used by the per-itemsize specializations of the
 * SimpleArrayCopier kernels.
 */
template <size_t N>
static inline void copy_one(int8_t * dst, int8_t const * src)
{
    // Reduces every element move in the inner loops to a std::memcpy with a
    // compile-time-constant size, which the compiler inlines into a single
    // load/store on common dtypes.
    std::memcpy(dst, src, N);
}

template <size_t N>
static void tiled_2d_impl(
    int8_t * const dst_body, int8_t const * const src_body, ssize_t const n0, ssize_t const n1, ssize_t const ss0, ssize_t const ss1, ssize_t const os0, ssize_t const os1)
{
    constexpr ssize_t BLOCK = 32;
    for (ssize_t i0 = 0; i0 < n0; i0 += BLOCK)
    {
        ssize_t const i_end = std::min(i0 + BLOCK, n0);
        for (ssize_t j0 = 0; j0 < n1; j0 += BLOCK)
        {
            ssize_t const j_end = std::min(j0 + BLOCK, n1);
            for (ssize_t i = i0; i < i_end; ++i)
            {
                int8_t const * src_row = src_body + i * ss0;
                int8_t * dst_row = dst_body + i * os0;
                for (ssize_t j = j0; j < j_end; ++j)
                {
                    copy_one<N>(dst_row + j * os1, src_row + j * ss1);
                }
            }
        }
    }
}

/**
 * Generic per-itemsize 2-D kernel that falls back to memcpy.  Used only for
 * itemsizes that are not in the specialized {1, 2, 4, 8, 16} set.
 */
static inline void tiled_2d_generic(
    int8_t * const dst_body, int8_t const * const src_body, ssize_t const n0, ssize_t const n1, ssize_t const ss0, ssize_t const ss1, ssize_t const os0, ssize_t const os1, size_t const itemsize)
{
    constexpr ssize_t BLOCK = 32;
    for (ssize_t i0 = 0; i0 < n0; i0 += BLOCK)
    {
        ssize_t const i_end = std::min(i0 + BLOCK, n0);
        for (ssize_t j0 = 0; j0 < n1; j0 += BLOCK)
        {
            ssize_t const j_end = std::min(j0 + BLOCK, n1);
            for (ssize_t i = i0; i < i_end; ++i)
            {
                int8_t const * src_row = src_body + i * ss0;
                int8_t * dst_row = dst_body + i * os0;
                for (ssize_t j = j0; j < j_end; ++j)
                {
                    std::memcpy(dst_row + j * os1, src_row + j * ss1, itemsize);
                }
            }
        }
    }
}

template <size_t N>
static void tiled_nd_inner(
    int8_t * const dst_body, int8_t const * const src_body, ssize_t const n_a, ssize_t const n_b, ssize_t const ss_a, ssize_t const ss_b, ssize_t const os_a, ssize_t const os_b)
{
    constexpr ssize_t BLOCK = 32;
    for (ssize_t a0 = 0; a0 < n_a; a0 += BLOCK)
    {
        ssize_t const a_end = std::min(a0 + BLOCK, n_a);
        for (ssize_t b0 = 0; b0 < n_b; b0 += BLOCK)
        {
            ssize_t const b_end = std::min(b0 + BLOCK, n_b);
            for (ssize_t i = a0; i < a_end; ++i)
            {
                int8_t const * src_row = src_body + i * ss_a;
                int8_t * dst_row = dst_body + i * os_a;
                for (ssize_t j = b0; j < b_end; ++j)
                {
                    copy_one<N>(dst_row + j * os_b, src_row + j * ss_b);
                }
            }
        }
    }
}

static inline void tiled_nd_inner_generic(
    int8_t * const dst_body, int8_t const * const src_body, ssize_t const n_a, ssize_t const n_b, ssize_t const ss_a, ssize_t const ss_b, ssize_t const os_a, ssize_t const os_b, size_t const itemsize)
{
    constexpr ssize_t BLOCK = 32;
    for (ssize_t a0 = 0; a0 < n_a; a0 += BLOCK)
    {
        ssize_t const a_end = std::min(a0 + BLOCK, n_a);
        for (ssize_t b0 = 0; b0 < n_b; b0 += BLOCK)
        {
            ssize_t const b_end = std::min(b0 + BLOCK, n_b);
            for (ssize_t i = a0; i < a_end; ++i)
            {
                int8_t const * src_row = src_body + i * ss_a;
                int8_t * dst_row = dst_body + i * os_a;
                for (ssize_t j = b0; j < b_end; ++j)
                {
                    std::memcpy(dst_row + j * os_b, src_row + j * ss_b, itemsize);
                }
            }
        }
    }
}

/**
 * Dispatch the inner tile by itemsize.  Specialized kernels run for the common
 * dtypes; everything else falls through to the memcpy version.
 */
static inline void dispatch_tile_inner(
    int8_t * const dst_body, int8_t const * const src_body, ssize_t const n_a, ssize_t const n_b, ssize_t const ss_a, ssize_t const ss_b, ssize_t const os_a, ssize_t const os_b, size_t const itemsize)
{
    switch (itemsize)
    {
    case 1: tiled_nd_inner<1>(dst_body, src_body, n_a, n_b, ss_a, ss_b, os_a, os_b); break;
    case 2: tiled_nd_inner<2>(dst_body, src_body, n_a, n_b, ss_a, ss_b, os_a, os_b); break;
    case 4: tiled_nd_inner<4>(dst_body, src_body, n_a, n_b, ss_a, ss_b, os_a, os_b); break;
    case 8: tiled_nd_inner<8>(dst_body, src_body, n_a, n_b, ss_a, ss_b, os_a, os_b); break;
    case 16: tiled_nd_inner<16>(dst_body, src_body, n_a, n_b, ss_a, ss_b, os_a, os_b); break;
    default: tiled_nd_inner_generic(dst_body, src_body, n_a, n_b, ss_a, ss_b, os_a, os_b, itemsize); break;
    }
}

/**
 * @param src_buffer Source buffer.
 * @param src_body_offset
 *      Byte offset from the source buffer start to the first logical
 *      element ("body").
 * @param src_stride Source strides in element units.
 * @param dst_buffer Destination buffer.
 * @param dst_body_offset
 *      Byte offset from the destination buffer start to its body.
 * @param dst_stride Destination strides in element units.
 * @param shape Logical shape shared by source and destination.
 * @param itemsize Element size in bytes.
 */
SimpleArrayCopier::SimpleArrayCopier(
    buffer_type const & src_buffer,
    ssize_t const src_body_offset,
    sshape_type const & src_stride,
    buffer_type & dst_buffer,
    ssize_t const dst_body_offset,
    sshape_type const & dst_stride,
    shape_type const & shape,
    size_t const itemsize)
    : m_src(src_buffer.data<int8_t>() + src_body_offset)
    , m_dst(dst_buffer.data<int8_t>() + dst_body_offset)
    , m_shape(shape)
    , m_src_stride(src_stride)
    , m_dst_stride(dst_stride)
    , m_itemsize(itemsize)
{
}

/**
 * Single-buffer memcpy fast-path.  Valid only when source and destination
 * share strides and the layout is contiguous.
 */
void SimpleArrayCopier::memcpy() const
{
    size_t total = 1;
    for (size_t const s : m_shape)
    {
        total *= s;
    }
    std::memcpy(m_dst, m_src, total * m_itemsize);
}

/**
 * 2-D 32x32 tile kernel.  Valid only for ndim == 2.
 */
void SimpleArrayCopier::tiled_2d() const
{
    ssize_t const n0 = m_shape[0];
    ssize_t const n1 = m_shape[1];
    // Element strides scaled to byte strides once; the inner loop uses byte
    // arithmetic throughout.
    auto const itemsize = static_cast<ssize_t>(m_itemsize);
    ssize_t const ss0 = m_src_stride[0] * itemsize;
    ssize_t const ss1 = m_src_stride[1] * itemsize;
    ssize_t const os0 = m_dst_stride[0] * itemsize;
    ssize_t const os1 = m_dst_stride[1] * itemsize;
    switch (m_itemsize)
    {
    case 1: tiled_2d_impl<1>(m_dst, m_src, n0, n1, ss0, ss1, os0, os1); break;
    case 2: tiled_2d_impl<2>(m_dst, m_src, n0, n1, ss0, ss1, os0, os1); break;
    case 4: tiled_2d_impl<4>(m_dst, m_src, n0, n1, ss0, ss1, os0, os1); break;
    case 8: tiled_2d_impl<8>(m_dst, m_src, n0, n1, ss0, ss1, os0, os1); break;
    case 16: tiled_2d_impl<16>(m_dst, m_src, n0, n1, ss0, ss1, os0, os1); break;
    default: tiled_2d_generic(m_dst, m_src, n0, n1, ss0, ss1, os0, os1, m_itemsize); break;
    }
}

/**
 * N-D kernel: 32x32 tile on the two innermost axes, carry-walk on the outer
 * axes.  Handles ndim >= 1.
 */
void SimpleArrayCopier::tiled_nd() const
{
    size_t const ndim = m_shape.size();
    auto const itemsize = static_cast<ssize_t>(m_itemsize);
    if (ndim == 1)
    {
        ssize_t const n = m_shape[0];
        ssize_t const ss = m_src_stride[0] * itemsize;
        ssize_t const os = m_dst_stride[0] * itemsize;
        for (ssize_t i = 0; i < n; ++i)
        {
            std::memcpy(m_dst + i * os, m_src + i * ss, m_itemsize);
        }
        return;
    }
    // ndim >= 2: tile the two innermost axes, carry-walk the outer axes.
    // See tiled_2d for the rationale behind the block size.
    size_t const ia = ndim - 2;
    size_t const ib = ndim - 1;
    ssize_t const n_a = m_shape[ia];
    ssize_t const n_b = m_shape[ib];
    ssize_t const ss_a = m_src_stride[ia] * itemsize;
    ssize_t const ss_b = m_src_stride[ib] * itemsize;
    ssize_t const os_a = m_dst_stride[ia] * itemsize;
    ssize_t const os_b = m_dst_stride[ib] * itemsize;

    ssize_t outer_total = 1;
    for (size_t k = 0; k < ia; ++k)
    {
        outer_total *= m_shape[k];
    }

    detail::sshape_type outer_idx(ia, 0);
    for (ssize_t step = 0; step < outer_total; ++step)
    {
        // Resolve outer-axis base offsets (in bytes) for this slab.
        ssize_t src_base = 0;
        ssize_t dst_base = 0;
        for (size_t k = 0; k < ia; ++k)
        {
            src_base += m_src_stride[k] * outer_idx[k] * itemsize;
            dst_base += m_dst_stride[k] * outer_idx[k] * itemsize;
        }
        dispatch_tile_inner(
            m_dst + dst_base, m_src + src_base, n_a, n_b, ss_a, ss_b, os_a, os_b, m_itemsize);
        // Carry-propagating increment of the outer index.
        for (size_t i = ia; i-- > 0;)
        {
            if (++outer_idx[i] < m_shape[i])
            {
                break;
            }
            // After the last slab the carry rolls every outer axis back to 0,
            // but the outer loop terminates before outer_idx is used again.
            outer_idx[i] = 0;
        }
    }
}

/**
 * Naive single-element walker, kept as a reference implementation.  Not called
 * by the dispatcher.
 */
void SimpleArrayCopier::naive() const
{
    if (m_shape.empty())
    {
        return;
    }
    ssize_t total = 1;
    for (ssize_t const s : m_shape)
    {
        total *= s;
    }
    if (total == 0)
    {
        return;
    }
    size_t const ndim = m_shape.size();
    size_t const itemsize = m_itemsize;
    auto const signed_itemsize = static_cast<ssize_t>(itemsize);
    detail::sshape_type idx(ndim, 0);
    for (ssize_t step = 0; step < total; ++step)
    {
        ssize_t src_off = 0;
        ssize_t dst_off = 0;
        for (size_t k = 0; k < ndim; ++k)
        {
            src_off += m_src_stride[k] * idx[k];
            dst_off += m_dst_stride[k] * idx[k];
        }
        std::memcpy(
            m_dst + dst_off * signed_itemsize,
            m_src + src_off * signed_itemsize,
            itemsize);
        // Carry-propagating increment: bump the trailing axis; on overflow,
        // wrap to 0 and carry into the next-most-significant axis.
        for (size_t i = ndim; i-- > 0;)
        {
            if (++idx[i] < m_shape[i])
            {
                break;
            }
            // After the last element the carry rolls every axis back to 0, but
            // the outer loop terminates before idx is used again.
            idx[i] = 0;
        }
    }
}

namespace detail
{
struct DataTypeHasher
{
    // The std::hash<std::string> is not deterministic, so we implement our own.
    // We can use the first and last character to hash the string.
    // The first character indicates the type of the data, and the last character indicates the size of the data (except for bool, which is a special case).
    // The combination of the first and last character can uniquely identify the data type.
    // e.g. "int32"   -> 'i' << 8 | '2' = 26930
    //      "uint32"  -> 'u' << 8 | '2' = 30002
    //      "float64" -> 'f' << 8 | '4' = 26164
    std::size_t operator()(const std::string & data_type_string) const
    {
        assert(data_type_string.size() >= 4); // the length of "bool" and "int8" is 4, and all other data types are longer than 4
        const char ch1 = data_type_string.front();
        const char ch2 = data_type_string.back();
        return static_cast<std::size_t>(ch1) << 8 | static_cast<std::size_t>(ch2);
    }
}; /* end struct DataTypeHasher */

// NOLINTNEXTLINE(misc-use-anonymous-namespace, cert-err58-cpp, bugprone-throwing-static-initialization, cppcoreguidelines-avoid-non-const-global-variables, fuchsia-statically-constructed-objects)
static std::unordered_map<std::string, DataType, DataTypeHasher> string_data_type_map = {
    {"bool", DataType::Bool},
    {"int8", DataType::Int8},
    {"int16", DataType::Int16},
    {"int32", DataType::Int32},
    {"int64", DataType::Int64},
    {"uint8", DataType::Uint8},
    {"uint16", DataType::Uint16},
    {"uint32", DataType::Uint32},
    {"uint64", DataType::Uint64},
    {"float32", DataType::Float32},
    {"float64", DataType::Float64},
    {"complex64", DataType::Complex64},
    {"complex128", DataType::Complex128}};

} /* end namespace detail */

DataType::DataType(const std::string & data_type_string)
{
    auto it = detail::string_data_type_map.find(data_type_string);
    if (it == detail::string_data_type_map.end())
    {
        throw std::invalid_argument("Unsupported datatype");
    }
    m_data_type = it->second;
}

template <>
DataType DataType::from<bool>()
{
    return DataType::Bool;
}

template <>
DataType DataType::from<int8_t>()
{
    return DataType::Int8;
}

template <>
DataType DataType::from<int16_t>()
{
    return DataType::Int16;
}

template <>
DataType DataType::from<int32_t>()
{
    return DataType::Int32;
}

template <>
DataType DataType::from<int64_t>()
{
    return DataType::Int64;
}

template <>
DataType DataType::from<uint8_t>()
{
    return DataType::Uint8;
}

template <>
DataType DataType::from<uint16_t>()
{
    return DataType::Uint16;
}

template <>
DataType DataType::from<uint32_t>()
{
    return DataType::Uint32;
}

template <>
DataType DataType::from<uint64_t>()
{
    return DataType::Uint64;
}

template <>
DataType DataType::from<float>()
{
    return DataType::Float32;
}

template <>
DataType DataType::from<double>()
{
    return DataType::Float64;
}

template <>
DataType DataType::from<Complex<float>>()
{
    return DataType::Complex64;
}

template <>
DataType DataType::from<Complex<double>>()
{
    return DataType::Complex128;
}

// According to the `DataType`, create the corresponding `SimpleArray<T>` instance
// and assign it to `m_instance_ptr`. The `m_instance_ptr` is a void pointer, so
// we need to use `reinterpret_cast` to convert the pointer of the array instance.
#define DECL_MM_CREATE_SIMPLE_ARRAY(DataType, ArrayType, ...)                  \
    case DataType:                                                             \
        /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */      \
        m_instance_ptr = reinterpret_cast<void *>(new ArrayType(__VA_ARGS__)); \
        break;

SimpleArrayPlex::SimpleArrayPlex(const shape_type & shape, const DataType data_type)
    : m_data_type(data_type)
    , m_has_instance_ownership(true)
{
    switch (data_type)
    {
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Bool, SimpleArrayBool, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int8, SimpleArrayInt8, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int16, SimpleArrayInt16, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int32, SimpleArrayInt32, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int64, SimpleArrayInt64, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint8, SimpleArrayUint8, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint16, SimpleArrayUint16, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint32, SimpleArrayUint32, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint64, SimpleArrayUint64, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float32, SimpleArrayFloat32, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float64, SimpleArrayFloat64, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex64, SimpleArrayComplex64, shape)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex128, SimpleArrayComplex128, shape)
    default:
        throw std::invalid_argument("Unsupported datatype");
    }
}

SimpleArrayPlex::SimpleArrayPlex(const shape_type & shape, const std::shared_ptr<ConcreteBuffer> & buffer, const DataType data_type)
    : m_data_type(data_type)
    , m_has_instance_ownership(true)
{
    switch (data_type)
    {
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Bool, SimpleArrayBool, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int8, SimpleArrayInt8, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int16, SimpleArrayInt16, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int32, SimpleArrayInt32, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Int64, SimpleArrayInt64, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint8, SimpleArrayUint8, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint16, SimpleArrayUint16, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint32, SimpleArrayUint32, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Uint64, SimpleArrayUint64, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float32, SimpleArrayFloat32, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Float64, SimpleArrayFloat64, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex64, SimpleArrayComplex64, shape, buffer)
        DECL_MM_CREATE_SIMPLE_ARRAY(DataType::Complex128, SimpleArrayComplex128, shape, buffer)
    default:
        throw std::invalid_argument("Unsupported datatype");
    }
}

#undef DECL_MM_CREATE_SIMPLE_ARRAY

SimpleArrayPlex::SimpleArrayPlex(const shape_type & shape, const DataType data_type, size_t alignment)
    : m_data_type(data_type)
    , m_has_instance_ownership(true)
{
#define DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType, ArrayType)                            \
    case DataType:                                                                                 \
        m_instance_ptr = static_cast<void *>(new ArrayType(shape, alignment, with_alignment_t{})); \
        break;

    switch (data_type)
    {
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Bool, SimpleArrayBool)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Int8, SimpleArrayInt8)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Int16, SimpleArrayInt16)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Int32, SimpleArrayInt32)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Int64, SimpleArrayInt64)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Uint8, SimpleArrayUint8)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Uint16, SimpleArrayUint16)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Uint32, SimpleArrayUint32)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Uint64, SimpleArrayUint64)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Float32, SimpleArrayFloat32)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Float64, SimpleArrayFloat64)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Complex64, SimpleArrayComplex64)
        DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT(DataType::Complex128, SimpleArrayComplex128)
    default:
        throw std::invalid_argument("Unsupported datatype");
    }

#undef DECL_MM_CREATE_SIMPLE_ARRAY_WITH_ALIGNMENT
}

SimpleArrayPlex::SimpleArrayPlex(SimpleArrayPlex const & other)
    : m_data_type(other.m_data_type)
{
    if (!other.m_instance_ptr)
    {
        return; // other does not have instance
    }

    m_has_instance_ownership = true;

    switch (other.m_data_type)
    {
    case DataType::Bool:
    {
        const auto * array = static_cast<SimpleArrayBool *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayBool(*array));
        break;
    }
    case DataType::Int8:
    {
        const auto * array = static_cast<SimpleArrayInt8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt8(*array));
        break;
    }
    case DataType::Int16:
    {
        const auto * array = static_cast<SimpleArrayInt16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt16(*array));
        break;
    }
    case DataType::Int32:
    {
        const auto * array = static_cast<SimpleArrayInt32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt32(*array));
        break;
    }
    case DataType::Int64:
    {
        const auto * array = static_cast<SimpleArrayInt64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt64(*array));
        break;
    }
    case DataType::Uint8:
    {
        const auto * array = static_cast<SimpleArrayUint8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint8(*array));
        break;
    }
    case DataType::Uint16:
    {
        const auto * array = static_cast<SimpleArrayUint16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint16(*array));
        break;
    }
    case DataType::Uint32:
    {
        const auto * array = static_cast<SimpleArrayUint32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint32(*array));
        break;
    }
    case DataType::Uint64:
    {
        const auto * array = static_cast<SimpleArrayUint64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint64(*array));
        break;
    }
    case DataType::Float32:
    {
        const auto * array = static_cast<SimpleArrayFloat32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat32(*array));
        break;
    }
    case DataType::Float64:
    {
        const auto * array = static_cast<SimpleArrayFloat64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat64(*array));
        break;
    }
    case DataType::Complex64:
    {
        const auto * array = static_cast<SimpleArrayComplex64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex64(*array));
        break;
    }
    case DataType::Complex128:
    {
        const auto * array = static_cast<SimpleArrayComplex128 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex128(*array));
        break;
    }
    default:
    {
        throw std::invalid_argument("Unsupported datatype");
    }
    }
}

SimpleArrayPlex::SimpleArrayPlex(SimpleArrayPlex && other) noexcept
    : m_data_type(other.m_data_type)
{
    if (!other.m_instance_ptr)
    {
        return; // other does not have instance
    }

    // take onwership
    m_has_instance_ownership = true;
    other.m_has_instance_ownership = false;

    m_instance_ptr = other.m_instance_ptr;
}

SimpleArrayPlex & SimpleArrayPlex::operator=(SimpleArrayPlex const & other)
{
    if (this == &other)
    {
        return *this;
    }

    m_data_type = other.m_data_type;

    if (!other.m_instance_ptr)
    {
        return *this; // other does not have instance
    }

    m_has_instance_ownership = true;

    switch (other.m_data_type)
    {
    case DataType::Bool:
    {
        const auto * array = static_cast<SimpleArrayBool *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayBool(*array));
        break;
    }
    case DataType::Int8:
    {
        const auto * array = static_cast<SimpleArrayInt8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt8(*array));
        break;
    }
    case DataType::Int16:
    {
        const auto * array = static_cast<SimpleArrayInt16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt16(*array));
        break;
    }
    case DataType::Int32:
    {
        const auto * array = static_cast<SimpleArrayInt32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt32(*array));
        break;
    }
    case DataType::Int64:
    {
        const auto * array = static_cast<SimpleArrayInt64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayInt64(*array));
        break;
    }
    case DataType::Uint8:
    {
        const auto * array = static_cast<SimpleArrayUint8 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint8(*array));
        break;
    }
    case DataType::Uint16:
    {
        const auto * array = static_cast<SimpleArrayUint16 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint16(*array));
        break;
    }
    case DataType::Uint32:
    {
        const auto * array = static_cast<SimpleArrayUint32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint32(*array));
        break;
    }
    case DataType::Uint64:
    {
        const auto * array = static_cast<SimpleArrayUint64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayUint64(*array));
        break;
    }
    case DataType::Float32:
    {
        const auto * array = static_cast<SimpleArrayFloat32 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat32(*array));
        break;
    }
    case DataType::Float64:
    {
        const auto * array = static_cast<SimpleArrayFloat64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayFloat64(*array));
        break;
    }
    case DataType::Complex64:
    {
        const auto * array = static_cast<SimpleArrayComplex64 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex64(*array));
        break;
    }
    case DataType::Complex128:
    {
        const auto * array = static_cast<SimpleArrayComplex128 *>(other.m_instance_ptr);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        m_instance_ptr = reinterpret_cast<void *>(new SimpleArrayComplex128(*array));
        break;
    }
    default:
    {
        throw std::invalid_argument("Unsupported datatype");
    }
    }
    return *this;
}

SimpleArrayPlex & SimpleArrayPlex::operator=(SimpleArrayPlex && other) noexcept
{
    m_data_type = other.m_data_type;

    if (!other.m_instance_ptr)
    {
        return *this; // other does not have instance
    }

    // take onwership
    m_has_instance_ownership = true;
    other.m_has_instance_ownership = false;

    m_instance_ptr = other.m_instance_ptr;
    return *this;
}

size_t SimpleArrayPlex::alignment() const
{
    if (m_instance_ptr == nullptr)
    {
        return 0;
    }

    switch (m_data_type)
    {
    case DataType::Bool:
        return static_cast<const SimpleArrayBool *>(m_instance_ptr)->alignment();
    case DataType::Int8:
        return static_cast<const SimpleArrayInt8 *>(m_instance_ptr)->alignment();
    case DataType::Int16:
        return static_cast<const SimpleArrayInt16 *>(m_instance_ptr)->alignment();
    case DataType::Int32:
        return static_cast<const SimpleArrayInt32 *>(m_instance_ptr)->alignment();
    case DataType::Int64:
        return static_cast<const SimpleArrayInt64 *>(m_instance_ptr)->alignment();
    case DataType::Uint8:
        return static_cast<const SimpleArrayUint8 *>(m_instance_ptr)->alignment();
    case DataType::Uint16:
        return static_cast<const SimpleArrayUint16 *>(m_instance_ptr)->alignment();
    case DataType::Uint32:
        return static_cast<const SimpleArrayUint32 *>(m_instance_ptr)->alignment();
    case DataType::Uint64:
        return static_cast<const SimpleArrayUint64 *>(m_instance_ptr)->alignment();
    case DataType::Float32:
        return static_cast<const SimpleArrayFloat32 *>(m_instance_ptr)->alignment();
    case DataType::Float64:
        return static_cast<const SimpleArrayFloat64 *>(m_instance_ptr)->alignment();
    case DataType::Complex64:
        return static_cast<const SimpleArrayComplex64 *>(m_instance_ptr)->alignment();
    case DataType::Complex128:
        return static_cast<const SimpleArrayComplex128 *>(m_instance_ptr)->alignment();
    default:
        return 0;
    }
}

SimpleArrayPlex::~SimpleArrayPlex()
{
    if (m_instance_ptr == nullptr || !m_has_instance_ownership)
    {
        return;
    }

    switch (m_data_type)
    {
    case DataType::Bool:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayBool *>(m_instance_ptr);
        break;
    }
    case DataType::Int8:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt8 *>(m_instance_ptr);
        break;
    }
    case DataType::Int16:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt16 *>(m_instance_ptr);
        break;
    }
    case DataType::Int32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt32 *>(m_instance_ptr);
        break;
    }
    case DataType::Int64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayInt64 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint8:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint8 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint16:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint16 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint32 *>(m_instance_ptr);
        break;
    }
    case DataType::Uint64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayUint64 *>(m_instance_ptr);
        break;
    }
    case DataType::Float32:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayFloat32 *>(m_instance_ptr);
        break;
    }
    case DataType::Float64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayFloat64 *>(m_instance_ptr);
        break;
    }
    case DataType::Complex64:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayComplex64 *>(m_instance_ptr);
        break;
    }
    case DataType::Complex128:
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        delete reinterpret_cast<SimpleArrayComplex128 *>(m_instance_ptr);
        break;
    }
    default:
        break;
    }

    m_instance_ptr = nullptr;
    m_has_instance_ownership = false;
}

namespace detail
{

// Format a shape such as (3, 4) for diagnostic messages.
std::string format_shape(shape_type const & shape)
{
    std::string ret = "(";
    for (size_t it = 0; it < shape.size(); ++it)
    {
        if (it != 0)
        {
            ret += ", ";
        }
        ret += std::to_string(shape[it]);
    }
    ret += ")";
    return ret;
}

std::string format_flat_index(shape_type const & shape, ssize_t offset)
{
    if (shape.empty())
    {
        return "[]";
    }

    shape_type row_major_stride(shape.size(), 1);
    for (size_t it = shape.size() - 1; it > 0; --it)
    {
        row_major_stride[it - 1] = row_major_stride[it] * shape[it];
    }

    std::string ret = "[";
    for (size_t it = 0; it < shape.size(); ++it)
    {
        if (it != 0)
        {
            ret += ", ";
        }
        ret += std::to_string(offset / row_major_stride[it]);
        offset %= row_major_stride[it];
    }
    ret += "]";
    return ret;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
