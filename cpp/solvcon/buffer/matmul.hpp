#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/loop.hpp>
#include <solvcon/buffer/small_vector.hpp>
#include <solvcon/math/math.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace solvcon
{

namespace detail
{

template <typename T>
inline constexpr bool can_matmul_blas_v = std::is_same_v<T, float> ||
                                          std::is_same_v<T, double> ||
                                          std::is_same_v<T, Complex<float>> ||
                                          std::is_same_v<T, Complex<double>>;

// Whether a matmul over T reaches BLAS at all: the type has to be one BLAS
// takes and the build has to have a backend behind the wrappers.
template <typename T>
inline constexpr bool use_matmul_blas_v = has_blas_backend && can_matmul_blas_v<T>;

/**
 * @brief Identify the contraction kernel fixed before batch traversal.
 */
enum class MatmulKernel : std::uint8_t
{
    Generic,
    BlasDot,
    BlasGevm,
    BlasGemv,
    BlasGemm,
}; /* end enum class MatmulKernel */

/**
 * @brief Identify operands that must be materialized into row-major storage.
 */
struct PackingState
{
    bool lhs = false;
    bool rhs = false;

    explicit operator bool() const noexcept { return lhs || rhs; }
}; /* end struct PackingState */

/**
 * @brief Group measured thresholds by matmul dispatch decision.
 *
 * A tuning table supplies the workload boundaries used to compare generic,
 * direct BLAS, and packing routes. Keeping the values separate from dispatch
 * code allows a backend or value type to provide another table.
 *
 * @note MatmulExecutor currently uses one compile-time table for every
 * supported BLAS backend and value type.
 */
struct MatmulTuning
{
    /**
     * @brief Select BLAS when the supplied operands have compatible layouts.
     */
    struct DirectBlas
    {
        ssize_t dot_min_length;
        ssize_t compact_gevm_min_elements;
        ssize_t gemv_min_dimension;
        ssize_t gemm_min_dimension;
    }; /* end struct DirectBlas */

    /**
     * @brief Pack reused matrices that cannot be described directly to BLAS.
     */
    struct MatrixPacking
    {
        ssize_t gemm_min_dimension;
    }; /* end struct MatrixPacking */

    /**
     * @brief Select direct or packed BLAS for batched vector-matrix products.
     */
    struct BatchedVector
    {
        ssize_t direct_min_matrix_elements;
        size_t always_pack_min_matrix_elements;
        size_t pack_min_matrix_elements;
        size_t pack_min_batch_size;
        size_t reuse_min_matrix_elements;
        size_t reuse_min_total_output_elements;
    }; /* end struct BatchedVector */

    DirectBlas direct_blas;
    MatrixPacking matrix_packing;
    BatchedVector batched_vector;
}; /* end struct MatmulTuning */

/**
 * @brief Record the fixed kernel and packing for one matmul call.
 */
struct MatmulSelection
{
    MatmulKernel kernel = MatmulKernel::Generic;
    PackingState packing;
}; /* end struct MatmulSelection */

/**
 * @brief Describe matmul operands as an execution-independent contraction.
 *
 * MatmulPlan interprets trailing axes as vector or matrix roles and leading
 * axes as the batch domain. It validates the contracted dimension, derives
 * the result shape, and records the signed-stride mappings used to locate
 * operands in the result batch domain. Broadcast batch axes are represented
 * by zero strides, so an executor can advance operand offsets without
 * reinterpreting operand dimensions or broadcasting rules.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the plan records batch shape `(2,5)`, M=3,
 * N=6, K=4, output shape `(2,5,3,6)`, and zero-stride batch mappings for
 * the broadcast axes.
 *
 * For `(K,) @ (B,K,N)`, the plan keeps a synthetic M=1 for execution but
 * omits that axis from the `(B,N)` output. Matrix-vector products similarly
 * omit synthetic N=1, while vector-vector products retain SimpleArray's
 * existing `(1,)` dot-result convention.
 */
class MatmulPlan
{
public:
    using shape_type = small_vector<ssize_t>;

    template <typename Array>
    static MatmulPlan make(Array const & lhs, Array const & rhs);

    shape_type const & output_shape() const noexcept { return m_output_shape; }
    ssize_t rows() const noexcept { return m_contraction.m_rows; }
    ssize_t columns() const noexcept { return m_contraction.m_columns; }
    ssize_t inner_size() const noexcept { return m_contraction.m_inner_size; }
    ssize_t lhs_row_stride() const noexcept { return m_strides.m_lhs_row_stride; }
    ssize_t lhs_inner_stride() const noexcept { return m_strides.m_lhs_inner_stride; }
    ssize_t rhs_inner_stride() const noexcept { return m_strides.m_rhs_inner_stride; }
    ssize_t rhs_column_stride() const noexcept { return m_strides.m_rhs_column_stride; }

    bool lhs_is_vector() const noexcept { return m_contraction.m_lhs_vector; }
    bool rhs_is_vector() const noexcept { return m_contraction.m_rhs_vector; }
    bool lhs_is_broadcast() const noexcept { return m_batch.m_lhs_is_broadcast; }
    bool rhs_is_broadcast() const noexcept { return m_batch.m_rhs_is_broadcast; }
    bool lhs_has_zero_batch_stride() const noexcept { return m_batch.m_lhs_has_zero_stride; }
    bool rhs_has_zero_batch_stride() const noexcept { return m_batch.m_rhs_has_zero_stride; }
    bool has_batch_axes() const noexcept { return m_batch.m_domain.rank() != 0; }
    size_t batch_size() const noexcept { return m_batch.m_domain.size(); }
    MappedOffsetCursor batch_cursor() const & { return MappedOffsetCursor(m_batch.m_domain, m_batch.m_mappings); }
    MappedOffsetCursor batch_cursor() const && = delete;

private:
    using batch_stride_type = OperandMapping::stride_type;
    using mapping_type = MappedOffsetCursor::mapping_type;

    struct Contraction
    {
        bool m_lhs_vector;
        bool m_rhs_vector;
        ssize_t m_rows;
        ssize_t m_columns;
        ssize_t m_inner_size;
    }; /* end struct Contraction */

    struct ContractionStrides
    {
        ssize_t m_lhs_row_stride;
        ssize_t m_lhs_inner_stride;
        ssize_t m_rhs_inner_stride;
        ssize_t m_rhs_column_stride;
    }; /* end struct ContractionStrides */

    struct BatchOperand
    {
        OperandMapping m_mapping;
        bool m_is_broadcast = false;
        bool m_has_zero_stride = false;
    }; /* end struct BatchOperand */

    struct BatchMappings
    {
        LoopDomain m_domain;
        mapping_type m_mappings;
        bool m_lhs_is_broadcast;
        bool m_rhs_is_broadcast;
        bool m_lhs_has_zero_stride;
        bool m_rhs_has_zero_stride;
    }; /* end struct BatchMappings */

    MatmulPlan(shape_type output_shape, Contraction contraction, ContractionStrides strides, BatchMappings batch);

    template <typename Array>
    static Contraction make_contraction(Array const & lhs, Array const & rhs);

    template <typename Array>
    static ContractionStrides make_contraction_strides(
        Array const & lhs,
        Array const & rhs,
        Contraction const & contraction);

    template <typename Array>
    static BatchMappings make_batch_mappings(Array const & lhs, Array const & rhs, ssize_t output_block_size);

    template <typename Array>
    static shape_type make_batch_shape(Array const & lhs, Array const & rhs);

    template <typename Array>
    static BatchOperand make_batch_operand(Array const & operand, LoopDomain const & domain);

    static shape_type make_output_shape(BatchMappings const & batch, Contraction const & contraction);

    template <typename Array>
    static std::string shape_string(Array const & array);

    shape_type m_output_shape;
    Contraction m_contraction;
    ContractionStrides m_strides;
    BatchMappings m_batch;
}; /* end class MatmulPlan */

/**
 * @brief Select and execute one contraction kernel for a MatmulPlan.
 *
 * MatmulExecutor combines plan metadata with MatmulTuning to select a
 * MatmulSelection. It applies the selected operand preparation, rebuilds the
 * plan when a physical layout changes, and traverses every batch offset with
 * the same kernel. Generic kernels read signed strides directly, while BLAS
 * kernels consume compatible vector and matrix descriptors.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the executor visits ten batch offsets and
 * evaluates one `(3,4) @ (4,6)` contraction at each offset. The results are
 * written into the allocated `(2,5,3,6)` output.
 *
 * @note The current implementation selects generic or direct BLAS kernels for
 * DOT, GEVM, GEMV, and GEMM. Packing materializes a reused broadcast matrix or
 * vector once in row-major storage before traversal. Unsupported equal-batch
 * matrix layouts remain generic.
 */
template <typename Array>
class MatmulExecutor
{
public:
    MatmulExecutor(MatmulPlan plan, Array & output, Array const & lhs, Array const & rhs);
    ~MatmulExecutor() = default;

    MatmulExecutor(MatmulExecutor const &) = delete;
    MatmulExecutor(MatmulExecutor &&) = delete;
    MatmulExecutor & operator=(MatmulExecutor const &) = delete;
    MatmulExecutor & operator=(MatmulExecutor &&) = delete;

    void execute();

private:
    using value_type = typename Array::value_type;
    using matrix_view_type = BlasMatrixView<value_type>;
    using vector_view_type = BlasVectorView<value_type>;

    enum class MappingSlot : std::uint8_t
    {
        Output,
        Lhs,
        Rhs,
    };

    static constexpr MatmulTuning TUNING = {
        .direct_blas = {
            .dot_min_length = 128,
            .compact_gevm_min_elements = 729,
            .gemv_min_dimension = 32,
            .gemm_min_dimension = 8,
        },
        .matrix_packing = {
            .gemm_min_dimension = 16,
        },
        .batched_vector = {
            .direct_min_matrix_elements = 512,
            .always_pack_min_matrix_elements = 4096,
            .pack_min_matrix_elements = 1024,
            .pack_min_batch_size = 4,
            .reuse_min_matrix_elements = 576,
            .reuse_min_total_output_elements = 128,
        },
    };

    MatmulSelection select_execution() const;
    MatmulSelection select_dot() const;
    MatmulSelection select_gevm() const;
    MatmulSelection select_gemv() const;
    MatmulSelection select_gemm() const;
    PackingState select_matrix_packing(PackingState required) const;
    PackingState select_vector_packing(PackingState required) const;
    bool should_pack_vector() const;
    void pack(PackingState const & packing);

    template <MatmulKernel Kernel>
    void execute_contractions();

    template <MatmulKernel Kernel>
    void execute_at(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base);

    void execute_dot_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);
    void execute_gevm_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);
    void execute_gemv_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);
    void execute_gemm_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);
    std::optional<matrix_view_type> lhs_matrix_view(value_type const * data) const;
    std::optional<matrix_view_type> rhs_matrix_view(value_type const * data) const;
    static matrix_view_type require_matrix_view(std::optional<matrix_view_type> view);
    static std::optional<matrix_view_type> make_matrix_view(
        value_type const * data,
        ssize_t row_stride,
        ssize_t column_stride,
        ssize_t rows,
        ssize_t columns);
    void execute_generic(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base);

    MatmulPlan m_plan;
    Array const & m_lhs;
    Array const & m_rhs;
    std::optional<Array> m_packed_lhs;
    std::optional<Array> m_packed_rhs;
    value_type * m_output_data;
    value_type const * m_lhs_data;
    value_type const * m_rhs_data;
}; /* end class MatmulExecutor */

inline MatmulPlan::MatmulPlan(
    shape_type output_shape,
    Contraction contraction,
    ContractionStrides strides,
    BatchMappings batch)
    : m_output_shape(std::move(output_shape))
    , m_contraction(contraction)
    , m_strides(strides)
    , m_batch(std::move(batch))
{
}

template <typename Array>
MatmulPlan MatmulPlan::make(Array const & lhs, Array const & rhs)
{
    Contraction const contraction = make_contraction(lhs, rhs);
    ContractionStrides const strides = make_contraction_strides(lhs, rhs, contraction);
    BatchMappings batch = make_batch_mappings(lhs, rhs, contraction.m_rows * contraction.m_columns);
    shape_type output_shape = make_output_shape(batch, contraction);
    return MatmulPlan{
        std::move(output_shape),
        contraction,
        strides,
        std::move(batch),
    };
}

template <typename Array>
MatmulPlan::Contraction MatmulPlan::make_contraction(Array const & lhs, Array const & rhs)
{
    if (lhs.ndim() == 0 || rhs.ndim() == 0)
    {
        throw std::invalid_argument(
            "planned matmul requires non-scalar operands");
    }

    bool const lhs_vector = lhs.ndim() == 1;
    bool const rhs_vector = rhs.ndim() == 1;
    size_t const lhs_inner_axis = lhs.ndim() - 1;
    size_t rhs_inner_axis = rhs.ndim() - 1;
    if (!rhs_vector)
    {
        --rhs_inner_axis;
    }
    if (lhs.shape(lhs_inner_axis) != rhs.shape(rhs_inner_axis))
    {
        throw std::invalid_argument(
            std::format("SimpleArray::matmul_planned(): shape mismatch: "
                        "this={} other={}",
                        shape_string(lhs),
                        shape_string(rhs)));
    }

    ssize_t rows = 1;
    if (!lhs_vector)
    {
        rows = lhs.shape(lhs.ndim() - 2);
    }
    ssize_t columns = 1;
    if (!rhs_vector)
    {
        columns = rhs.shape(rhs.ndim() - 1);
    }
    return Contraction{
        .m_lhs_vector = lhs_vector,
        .m_rhs_vector = rhs_vector,
        .m_rows = rows,
        .m_columns = columns,
        .m_inner_size = lhs.shape(lhs_inner_axis),
    };
}

template <typename Array>
MatmulPlan::ContractionStrides MatmulPlan::make_contraction_strides(
    Array const & lhs,
    Array const & rhs,
    Contraction const & contraction)
{
    ssize_t lhs_row_stride = 0;
    if (!contraction.m_lhs_vector)
    {
        lhs_row_stride = lhs.stride(lhs.ndim() - 2);
    }

    size_t rhs_inner_axis = rhs.ndim() - 1;
    ssize_t rhs_column_stride = 0;
    if (!contraction.m_rhs_vector)
    {
        --rhs_inner_axis;
        rhs_column_stride = rhs.stride(rhs.ndim() - 1);
    }
    return ContractionStrides{
        .m_lhs_row_stride = lhs_row_stride,
        .m_lhs_inner_stride = lhs.stride(lhs.ndim() - 1),
        .m_rhs_inner_stride = rhs.stride(rhs_inner_axis),
        .m_rhs_column_stride = rhs_column_stride,
    };
}

template <typename Array>
MatmulPlan::BatchMappings MatmulPlan::make_batch_mappings(
    Array const & lhs,
    Array const & rhs,
    ssize_t output_block_size)
{
    LoopDomain domain{make_batch_shape(lhs, rhs)};
    BatchOperand lhs_operand = make_batch_operand(lhs, domain);
    BatchOperand rhs_operand = make_batch_operand(rhs, domain);
    mapping_type mappings{
        OperandMapping::contiguous_blocks(domain, output_block_size),
        std::move(lhs_operand.m_mapping),
        std::move(rhs_operand.m_mapping),
    };
    return BatchMappings{
        .m_domain = std::move(domain),
        .m_mappings = std::move(mappings),
        .m_lhs_is_broadcast = lhs_operand.m_is_broadcast,
        .m_rhs_is_broadcast = rhs_operand.m_is_broadcast,
        .m_lhs_has_zero_stride = lhs_operand.m_has_zero_stride,
        .m_rhs_has_zero_stride = rhs_operand.m_has_zero_stride,
    };
}

template <typename Array>
MatmulPlan::shape_type MatmulPlan::make_batch_shape(Array const & lhs, Array const & rhs)
{
    size_t lhs_batch_axis_count = 0;
    if (lhs.ndim() > 1)
    {
        lhs_batch_axis_count = lhs.ndim() - 2;
    }
    size_t rhs_batch_axis_count = 0;
    if (rhs.ndim() > 1)
    {
        rhs_batch_axis_count = rhs.ndim() - 2;
    }
    size_t const batch_axis_count = std::max(lhs_batch_axis_count, rhs_batch_axis_count);
    shape_type batch_shape(batch_axis_count, 1);
    for (size_t offset = 0; offset < batch_axis_count; ++offset)
    {
        ssize_t const lhs_extent =
            offset < lhs_batch_axis_count ? lhs.shape(lhs_batch_axis_count - offset - 1) : 1;
        ssize_t const rhs_extent =
            offset < rhs_batch_axis_count ? rhs.shape(rhs_batch_axis_count - offset - 1) : 1;
        if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1)
        {
            throw std::invalid_argument(
                std::format("SimpleArray::matmul_planned(): batch shape "
                            "mismatch: this={} other={}",
                            shape_string(lhs),
                            shape_string(rhs)));
        }
        batch_shape[batch_axis_count - offset - 1] = lhs_extent == 1 ? rhs_extent : lhs_extent;
    }
    return batch_shape;
}

template <typename Array>
MatmulPlan::BatchOperand MatmulPlan::make_batch_operand(Array const & operand, LoopDomain const & domain)
{
    size_t batch_axis_count = 0;
    if (operand.ndim() > 1)
    {
        batch_axis_count = operand.ndim() - 2;
    }
    size_t const batch_axis_offset = domain.rank() - batch_axis_count;
    batch_stride_type strides(domain.rank(), 0);
    bool is_broadcast = false;
    bool has_zero_stride = false;
    for (size_t domain_axis = 0; domain_axis < domain.rank(); ++domain_axis)
    {
        if (domain_axis < batch_axis_offset)
        {
            is_broadcast = is_broadcast || domain.extent(domain_axis) > 1;
            continue;
        }

        size_t const operand_axis = domain_axis - batch_axis_offset;
        ssize_t const operand_extent = operand.shape(operand_axis);
        ssize_t const operand_stride = operand.stride(operand_axis);
        if (operand_extent == domain.extent(domain_axis))
        {
            strides[domain_axis] = operand_stride;
        }
        else if (domain.extent(domain_axis) > 1)
        {
            is_broadcast = true;
        }
        has_zero_stride = has_zero_stride || (operand_extent > 1 && operand_stride == 0);
    }
    return BatchOperand{
        .m_mapping = OperandMapping(std::move(strides)),
        .m_is_broadcast = is_broadcast,
        .m_has_zero_stride = has_zero_stride,
    };
}

inline MatmulPlan::shape_type MatmulPlan::make_output_shape(
    BatchMappings const & batch,
    Contraction const & contraction)
{
    size_t const batch_axis_count = batch.m_domain.rank();
    if (contraction.m_lhs_vector && contraction.m_rhs_vector)
    {
        return shape_type{1};
    }

    size_t output_axis_count = batch_axis_count;
    if (!contraction.m_lhs_vector)
    {
        ++output_axis_count;
    }
    if (!contraction.m_rhs_vector)
    {
        ++output_axis_count;
    }
    shape_type output_shape(output_axis_count);
    std::ranges::copy(batch.m_domain.shape(), output_shape.begin());
    size_t output_axis = batch_axis_count;
    if (!contraction.m_lhs_vector)
    {
        output_shape[output_axis++] = contraction.m_rows;
    }
    if (!contraction.m_rhs_vector)
    {
        output_shape[output_axis] = contraction.m_columns;
    }
    return output_shape;
}

template <typename Array>
std::string MatmulPlan::shape_string(Array const & array)
{
    std::string result = "(";
    for (ssize_t axis = 0; axis < array.ndim(); ++axis)
    {
        if (axis > 0)
        {
            result += ",";
        }
        result += std::to_string(array.shape(axis));
    }
    result += ")";
    return result;
}

template <typename Array>
MatmulExecutor<Array>::MatmulExecutor(MatmulPlan plan, Array & output, Array const & lhs, Array const & rhs)
    : m_plan(std::move(plan))
    , m_lhs(lhs)
    , m_rhs(rhs)
    , m_output_data(output.logical_data())
    , m_lhs_data(lhs.logical_data())
    , m_rhs_data(rhs.logical_data())
{
}

template <typename Array>
void MatmulExecutor<Array>::execute()
{
    MatmulSelection const selection = select_execution();

    if (selection.packing)
    {
        pack(selection.packing);
    }

    // `select_execution()` names a BLAS kernel only where one exists, so the
    // dispatch stays uninstantiated for a value type that has none, and the
    // generic kernel is the one path every instantiation keeps.
    if constexpr (use_matmul_blas_v<value_type>)
    {
        switch (selection.kernel)
        {
        case MatmulKernel::Generic:
            break;
        case MatmulKernel::BlasDot:
            execute_contractions<MatmulKernel::BlasDot>();
            return;
        case MatmulKernel::BlasGevm:
            execute_contractions<MatmulKernel::BlasGevm>();
            return;
        case MatmulKernel::BlasGemv:
            execute_contractions<MatmulKernel::BlasGemv>();
            return;
        case MatmulKernel::BlasGemm:
            execute_contractions<MatmulKernel::BlasGemm>();
            return;
        }
    }
    execute_contractions<MatmulKernel::Generic>();
}

template <typename Array>
MatmulSelection MatmulExecutor<Array>::select_execution() const
{
    if constexpr (use_matmul_blas_v<value_type>)
    {
        if (m_plan.lhs_is_vector() && m_plan.rhs_is_vector())
        {
            return select_dot();
        }
        if (m_plan.lhs_is_vector())
        {
            return select_gevm();
        }
        if (m_plan.rhs_is_vector())
        {
            return select_gemv();
        }
        return select_gemm();
    }
    else
    {
        return MatmulSelection{};
    }
}

template <typename Array>
MatmulSelection MatmulExecutor<Array>::select_dot() const
{
    ssize_t const lhs_stride = m_plan.lhs_inner_stride();
    ssize_t const rhs_stride = m_plan.rhs_inner_stride();
    bool const strides_supported = (lhs_stride == 1 && rhs_stride == 1) ||
                                   (lhs_stride == -1 && rhs_stride == -1);
    bool const use_blas = m_plan.inner_size() >= TUNING.direct_blas.dot_min_length && strides_supported;
    return MatmulSelection{
        .kernel = use_blas ? MatmulKernel::BlasDot : MatmulKernel::Generic,
        .packing = {},
    };
}

template <typename Array>
MatmulSelection MatmulExecutor<Array>::select_gevm() const
{
    auto const matrix = rhs_matrix_view(m_rhs_data);
    if (!matrix)
    {
        return MatmulSelection{};
    }
    if (m_plan.has_batch_axes())
    {
        PackingState const required{.lhs = m_plan.lhs_inner_stride() <= 0};
        PackingState const packing = select_vector_packing(required);
        bool const use_blas = packing ||
                              (m_plan.lhs_inner_stride() > 0 &&
                               m_plan.inner_size() * m_plan.columns() >=
                                   TUNING.batched_vector.direct_min_matrix_elements);
        return MatmulSelection{
            .kernel = use_blas ? MatmulKernel::BlasGevm : MatmulKernel::Generic,
            .packing = packing,
        };
    }
    if (m_plan.lhs_inner_stride() <= 0)
    {
        return MatmulSelection{};
    }

    bool const is_compact = matrix->m_transpose == BlasTranspose::None &&
                            matrix->m_leading_dimension == m_plan.columns();
    bool const use_blas = is_compact
                              ? m_plan.inner_size() * m_plan.columns() >=
                                    TUNING.direct_blas.compact_gevm_min_elements
                              : std::min(m_plan.inner_size(), m_plan.columns()) >=
                                    TUNING.direct_blas.gemv_min_dimension;
    return MatmulSelection{
        .kernel = use_blas ? MatmulKernel::BlasGevm : MatmulKernel::Generic,
        .packing = {},
    };
}

template <typename Array>
MatmulSelection MatmulExecutor<Array>::select_gemv() const
{
    auto const matrix = lhs_matrix_view(m_lhs_data);
    if (!matrix)
    {
        return MatmulSelection{};
    }
    if (m_plan.has_batch_axes())
    {
        PackingState const required{.rhs = m_plan.rhs_inner_stride() <= 0};
        PackingState const packing = select_vector_packing(required);
        bool const use_blas = packing ||
                              (m_plan.rhs_inner_stride() > 0 &&
                               m_plan.rows() * m_plan.inner_size() >=
                                   TUNING.batched_vector.direct_min_matrix_elements);
        return MatmulSelection{
            .kernel = use_blas ? MatmulKernel::BlasGemv : MatmulKernel::Generic,
            .packing = packing,
        };
    }

    bool const use_blas = m_plan.rhs_inner_stride() > 0 &&
                          std::min(m_plan.rows(), m_plan.inner_size()) >= TUNING.direct_blas.gemv_min_dimension;
    return MatmulSelection{
        .kernel = use_blas ? MatmulKernel::BlasGemv : MatmulKernel::Generic,
        .packing = {},
    };
}

template <typename Array>
MatmulSelection MatmulExecutor<Array>::select_gemm() const
{
    ssize_t const minimum_dimension =
        std::min({m_plan.rows(), m_plan.columns(), m_plan.inner_size()});
    bool const lhs_blas_compatible = bool(lhs_matrix_view(m_lhs_data));
    bool const rhs_blas_compatible = bool(rhs_matrix_view(m_rhs_data));
    if (minimum_dimension >= TUNING.direct_blas.gemm_min_dimension && lhs_blas_compatible && rhs_blas_compatible)
    {
        return MatmulSelection{.kernel = MatmulKernel::BlasGemm, .packing = {}};
    }
    if (minimum_dimension >= TUNING.matrix_packing.gemm_min_dimension)
    {
        PackingState const required{
            .lhs = !lhs_blas_compatible,
            .rhs = !rhs_blas_compatible,
        };
        PackingState const packing = select_matrix_packing(required);
        if (packing)
        {
            return MatmulSelection{.kernel = MatmulKernel::BlasGemm, .packing = packing};
        }
    }
    return MatmulSelection{};
}

template <typename Array>
PackingState MatmulExecutor<Array>::select_matrix_packing(PackingState required) const
{
    bool const lhs_supported =
        !required.lhs || (m_plan.lhs_is_broadcast() && !m_plan.lhs_has_zero_batch_stride());
    bool const rhs_supported =
        !required.rhs || (m_plan.rhs_is_broadcast() && !m_plan.rhs_has_zero_batch_stride());
    if (!required || !lhs_supported || !rhs_supported)
    {
        return PackingState{};
    }
    return required;
}

template <typename Array>
PackingState MatmulExecutor<Array>::select_vector_packing(PackingState required) const
{
    if (!required)
    {
        return PackingState{};
    }
    bool const vector_reused = required.lhs ? m_plan.lhs_is_broadcast() : m_plan.rhs_is_broadcast();
    return vector_reused && should_pack_vector() ? required : PackingState{};
}

template <typename Array>
bool MatmulExecutor<Array>::should_pack_vector() const
{
    auto const output_size = static_cast<size_t>(
        m_plan.lhs_is_vector() ? m_plan.columns() : m_plan.rows());
    size_t const matrix_elements = static_cast<size_t>(m_plan.inner_size()) * output_size;
    if (matrix_elements >= TUNING.batched_vector.always_pack_min_matrix_elements)
    {
        return true;
    }
    if (matrix_elements >= TUNING.batched_vector.pack_min_matrix_elements &&
        m_plan.batch_size() >= TUNING.batched_vector.pack_min_batch_size)
    {
        return true;
    }
    if (matrix_elements < TUNING.batched_vector.reuse_min_matrix_elements || output_size == 0)
    {
        return false;
    }
    size_t const minimum_batch_size =
        (TUNING.batched_vector.reuse_min_total_output_elements + output_size - 1) / output_size;
    return m_plan.batch_size() >= minimum_batch_size;
}

template <typename Array>
void MatmulExecutor<Array>::pack(PackingState const & packing)
{
    if (packing.lhs)
    {
        m_packed_lhs.emplace(m_lhs.to_row_major());
    }
    if (packing.rhs)
    {
        m_packed_rhs.emplace(m_rhs.to_row_major());
    }

    Array const & lhs = m_packed_lhs ? *m_packed_lhs : m_lhs;
    Array const & rhs = m_packed_rhs ? *m_packed_rhs : m_rhs;
    m_plan = MatmulPlan::make(lhs, rhs);
    m_lhs_data = lhs.logical_data();
    m_rhs_data = rhs.logical_data();
}

template <typename Array>
template <MatmulKernel Kernel>
void MatmulExecutor<Array>::execute_contractions()
{
    if (!m_plan.has_batch_axes())
    {
        execute_at<Kernel>(0, 0, 0);
        return;
    }

    for (MappedOffsetCursor cursor = m_plan.batch_cursor(); cursor; cursor.advance())
    {
        execute_at<Kernel>(
            cursor.offset(MappingSlot::Output),
            cursor.offset(MappingSlot::Lhs),
            cursor.offset(MappingSlot::Rhs));
    }
}

template <typename Array>
template <MatmulKernel Kernel>
void MatmulExecutor<Array>::execute_at(
    ssize_t output_base,
    ssize_t lhs_base,
    ssize_t rhs_base)
{
    if constexpr (Kernel == MatmulKernel::Generic)
    {
        execute_generic(output_base, lhs_base, rhs_base);
    }
    else
    {
        static_assert(use_matmul_blas_v<value_type>,
                      "execute() dispatches a BLAS kernel only where one exists");
        value_type * output = m_output_data + output_base;
        value_type const * lhs_data = m_lhs_data + lhs_base;
        value_type const * rhs_data = m_rhs_data + rhs_base;
        if constexpr (Kernel == MatmulKernel::BlasDot)
        {
            execute_dot_blas(output, lhs_data, rhs_data);
        }
        else if constexpr (Kernel == MatmulKernel::BlasGevm)
        {
            execute_gevm_blas(output, lhs_data, rhs_data);
        }
        else if constexpr (Kernel == MatmulKernel::BlasGemv)
        {
            execute_gemv_blas(output, lhs_data, rhs_data);
        }
        else if constexpr (Kernel == MatmulKernel::BlasGemm)
        {
            execute_gemm_blas(output, lhs_data, rhs_data);
        }
    }
}

template <typename Array>
void MatmulExecutor<Array>::execute_dot_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data)
{
    ssize_t lhs_stride = m_plan.lhs_inner_stride();
    ssize_t rhs_stride = m_plan.rhs_inner_stride();
    if (lhs_stride == -1 && rhs_stride == -1)
    {
        lhs_data += (m_plan.inner_size() - 1) * lhs_stride;
        rhs_data += (m_plan.inner_size() - 1) * rhs_stride;
        lhs_stride = -lhs_stride;
        rhs_stride = -rhs_stride;
    }

    vector_view_type const lhs{lhs_data, lhs_stride};
    vector_view_type const rhs{rhs_data, rhs_stride};
    output[0] = dot_blas(m_plan.inner_size(), lhs, rhs);
}

template <typename Array>
void MatmulExecutor<Array>::execute_gevm_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data)
{
    matrix_view_type const matrix = require_matrix_view(rhs_matrix_view(rhs_data));
    vector_view_type const vector{lhs_data, m_plan.lhs_inner_stride()};
    gemv_blas(m_plan.inner_size(), m_plan.columns(), matrix, vector, output, BlasTranspose::Transpose);
}

template <typename Array>
void MatmulExecutor<Array>::execute_gemv_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data)
{
    matrix_view_type const matrix = require_matrix_view(lhs_matrix_view(lhs_data));
    vector_view_type const vector{rhs_data, m_plan.rhs_inner_stride()};
    gemv_blas(m_plan.rows(), m_plan.inner_size(), matrix, vector, output, BlasTranspose::None);
}

template <typename Array>
void MatmulExecutor<Array>::execute_gemm_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data)
{
    matrix_view_type const lhs = require_matrix_view(lhs_matrix_view(lhs_data));
    matrix_view_type const rhs = require_matrix_view(rhs_matrix_view(rhs_data));
    gemm_blas(m_plan.rows(), m_plan.columns(), m_plan.inner_size(), lhs, rhs, output);
}

template <typename Array>
std::optional<typename MatmulExecutor<Array>::matrix_view_type> MatmulExecutor<Array>::lhs_matrix_view(
    value_type const * data) const
{
    return make_matrix_view(
        data, m_plan.lhs_row_stride(), m_plan.lhs_inner_stride(), m_plan.rows(), m_plan.inner_size());
}

template <typename Array>
std::optional<typename MatmulExecutor<Array>::matrix_view_type> MatmulExecutor<Array>::rhs_matrix_view(
    value_type const * data) const
{
    return make_matrix_view(
        data, m_plan.rhs_inner_stride(), m_plan.rhs_column_stride(), m_plan.inner_size(), m_plan.columns());
}

template <typename Array>
typename MatmulExecutor<Array>::matrix_view_type MatmulExecutor<Array>::require_matrix_view(
    std::optional<matrix_view_type> view)
{
    if (!view)
    {
        throw std::logic_error("MatmulExecutor::require_matrix_view(): missing BLAS matrix view");
    }
    return *view;
}

template <typename Array>
std::optional<typename MatmulExecutor<Array>::matrix_view_type> MatmulExecutor<Array>::make_matrix_view(
    value_type const * data,
    ssize_t row_stride,
    ssize_t column_stride,
    ssize_t rows,
    ssize_t columns)
{
    if (column_stride == 1 && row_stride >= columns)
    {
        return BlasMatrixView<value_type>{data, row_stride, BlasTranspose::None};
    }
    if (row_stride == 1 && column_stride >= rows)
    {
        return BlasMatrixView<value_type>{data, column_stride, BlasTranspose::Transpose};
    }
    return std::nullopt;
}

template <typename Array>
void MatmulExecutor<Array>::execute_generic(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base)
{
    for (ssize_t row = 0; row < m_plan.rows(); ++row)
    {
        ssize_t const lhs_row_base = lhs_base + row * m_plan.lhs_row_stride();
        ssize_t const output_row_base = output_base + row * m_plan.columns();
        for (ssize_t column = 0; column < m_plan.columns(); ++column)
        {
            value_type total{};
            ssize_t lhs_offset = lhs_row_base;
            ssize_t rhs_offset = rhs_base + column * m_plan.rhs_column_stride();
            for (ssize_t inner = 0; inner < m_plan.inner_size(); ++inner)
            {
                total += m_lhs_data[lhs_offset] * m_rhs_data[rhs_offset];
                lhs_offset += m_plan.lhs_inner_stride();
                rhs_offset += m_plan.rhs_inner_stride();
            }
            m_output_data[output_row_base + column] = total;
        }
    }
}

template <typename A, typename T>
class SimpleArrayMatmulHelper
{

public:

    using value_type = T;
    using shape_type = typename A::shape_type;

    SimpleArrayMatmulHelper() = delete;
    SimpleArrayMatmulHelper(A const & lhs, A const & rhs);
    SimpleArrayMatmulHelper(A const & lhs,
                            A const & rhs,
                            ssize_t tile_x,
                            ssize_t tile_y,
                            ssize_t tile_z);
    ~SimpleArrayMatmulHelper() = default;

    SimpleArrayMatmulHelper(SimpleArrayMatmulHelper const &) = delete;
    SimpleArrayMatmulHelper(SimpleArrayMatmulHelper &&) = delete;
    SimpleArrayMatmulHelper & operator=(SimpleArrayMatmulHelper const &) = delete;
    SimpleArrayMatmulHelper & operator=(SimpleArrayMatmulHelper &&) = delete;

    A matmul();
    A matmul_fast();
    A matmul_blas();

private:

    static std::string shape_str(A const & arr);
    void check_dims() const;
    void check_inner(size_t lhs_idx, size_t rhs_idx) const;
    void check_tiles() const;
    A matmul_vec_vec();
    A matmul_vec_vec_blas();
    A matmul_vec_mat();
    A matmul_vec_mat_blas();
    A matmul_mat_vec();
    A matmul_mat_vec_blas();
    A matmul_mat_mat();
    A matmul_mat_mat_blas();
    A pack_rhs(ssize_t n, ssize_t k);
    void accumulate_tile(A const & packed_rhs,
                         ssize_t row_begin,
                         ssize_t row_end,
                         ssize_t col_begin,
                         ssize_t col_end,
                         ssize_t inner_begin,
                         ssize_t inner_end);
    A matmul_mat_mat_tiled();

    A const & m_lhs;
    A const & m_rhs;
    A m_result;
    ssize_t m_tile_x;
    ssize_t m_tile_y;
    ssize_t m_tile_z;

}; /* end class SimpleArrayMatmulHelper */

template <typename A, typename T>
SimpleArrayMatmulHelper<A, T>::SimpleArrayMatmulHelper(A const & lhs, A const & rhs)
    : SimpleArrayMatmulHelper(lhs, rhs, 0, 0, 0)
{
}

template <typename A, typename T>
SimpleArrayMatmulHelper<A, T>::SimpleArrayMatmulHelper(A const & lhs,
                                                       A const & rhs,
                                                       ssize_t tile_x,
                                                       ssize_t tile_y,
                                                       ssize_t tile_z)
    : m_lhs(lhs)
    , m_rhs(rhs)
    , m_tile_x(tile_x)
    , m_tile_y(tile_y)
    , m_tile_z(tile_z)
{
    check_dims();

    size_t const lhs_ndim = m_lhs.ndim();
    size_t const rhs_ndim = m_rhs.ndim();

    if (lhs_ndim == 1 && rhs_ndim == 1)
    {
        check_inner(0, 0);
        m_result = A(1);
        return;
    }

    if (lhs_ndim == 1)
    {
        check_inner(0, 0);
        m_result = A(m_rhs.shape(1));
        return;
    }

    if (rhs_ndim == 1)
    {
        check_inner(1, 0);
        m_result = A(m_lhs.shape(0));
        return;
    }

    check_inner(1, 0);
    shape_type const result_shape{m_lhs.shape(0), m_rhs.shape(1)};
    m_result = A(result_shape);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul()
{
    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec();
    }

    return matmul_mat_mat();
}

/**
 * Perform fast matrix multiplication for SimpleArrays.
 * This implementation currently uses tiling for 2D x 2D matrix multiplication.
 * Future optimizations may add other techniques such as SIMD kernels.
 */
template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_fast()
{
    check_tiles();

    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec();
    }

    return matmul_mat_mat_tiled();
}

/**
 * Perform matrix multiplication using vendor BLAS when available.
 */
template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_blas()
{
    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec_blas();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat_blas();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec_blas();
    }

    return matmul_mat_mat_blas();
}

/**
 * Format shape for matrix multiplication diagnostics.
 */
template <typename A, typename T>
std::string SimpleArrayMatmulHelper<A, T>::shape_str(A const & arr)
{
    if (arr.ndim() == 0)
    {
        return "()";
    }

    std::string result = "(";
    for (ssize_t i = 0; i < arr.ndim(); ++i)
    {
        if (i > 0)
        {
            result += ",";
        }
        result += std::to_string(arr.shape(i));
    }
    result += ")";
    return result;
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_dims() const
{
    bool const lhs_is_supported = m_lhs.ndim() == 1 || m_lhs.ndim() == 2;
    bool const rhs_is_supported = m_rhs.ndim() == 1 || m_rhs.ndim() == 2;
    if (lhs_is_supported && rhs_is_supported)
    {
        return;
    }

    std::string const err = std::format("SimpleArray::matmul(): unsupported dimensions: "
                                        "this={} other={}. SimpleArray must be 1D or 2D.",
                                        shape_str(m_lhs),
                                        shape_str(m_rhs));
    throw std::out_of_range(err);
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_inner(size_t lhs_idx, size_t rhs_idx) const
{
    if (m_lhs.shape(lhs_idx) == m_rhs.shape(rhs_idx))
    {
        return;
    }

    throw std::out_of_range(
        std::format("SimpleArray::matmul(): shape mismatch: this={} other={}",
                    shape_str(m_lhs),
                    shape_str(m_rhs)));
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_tiles() const
{
    if (m_tile_x > 0 && m_tile_y > 0 && m_tile_z > 0)
    {
        return;
    }

    throw std::out_of_range(
        std::format("SimpleArray::matmul_fast(): tile sizes must be positive: "
                    "tile_x={} tile_y={} tile_z={}",
                    m_tile_x,
                    m_tile_y,
                    m_tile_z));
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_vec()
{
    ssize_t const k = m_lhs.shape(0);
    value_type v = 0;
    for (ssize_t i = 0; i < k; ++i)
    {
        v += m_lhs(i) * m_rhs(i);
    }
    m_result.data(0) = v;
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_vec_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_vec_vec();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const k = m_lhs.shape(0);
        m_result.data(0) = dot_blas(k, m_lhs.data(), m_rhs.data());
        return std::move(m_result);
    }
    else
    {
        return matmul_vec_vec();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_mat()
{
    ssize_t const n = m_result.shape(0);
    ssize_t const k = m_lhs.shape(0);
    for (ssize_t j = 0; j < n; ++j)
    {
        value_type v = 0;
        for (ssize_t l = 0; l < k; ++l)
        {
            v += m_lhs(l) * m_rhs(l, j);
        }
        m_result.data(j) = v;
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_mat_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_vec_mat();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const k = m_rhs.shape(0);
        ssize_t const n = m_rhs.shape(1);
        bool const transpose_matrix = true;
        gemv_blas(k,
                  n,
                  m_rhs.data(),
                  m_lhs.data(),
                  m_result.data(),
                  transpose_matrix);
        return std::move(m_result);
    }
    else
    {
        return matmul_vec_mat();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_vec()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const k = m_lhs.shape(1);
    for (ssize_t i = 0; i < m; ++i)
    {
        value_type v = 0;
        for (ssize_t l = 0; l < k; ++l)
        {
            v += m_lhs(i, l) * m_rhs(l);
        }
        m_result.data(i) = v;
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_vec_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_mat_vec();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const m = m_lhs.shape(0);
        ssize_t const k = m_lhs.shape(1);
        bool const transpose_matrix = false;
        gemv_blas(m,
                  k,
                  m_lhs.data(),
                  m_rhs.data(),
                  m_result.data(),
                  transpose_matrix);
        return std::move(m_result);
    }
    else
    {
        return matmul_mat_vec();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const n = m_result.shape(1);
    ssize_t const k = m_lhs.shape(1);
    for (ssize_t i = 0; i < m; ++i)
    {
        for (ssize_t j = 0; j < n; ++j)
        {
            value_type v = 0;
            for (ssize_t l = 0; l < k; ++l)
            {
                v += m_lhs(i, l) * m_rhs(l, j);
            }
            m_result(i, j) = v;
        }
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_mat_mat();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const m = m_result.shape(0);
        ssize_t const n = m_result.shape(1);
        ssize_t const k = m_lhs.shape(1);
        gemm_blas(m, n, k, m_lhs.data(), m_rhs.data(), m_result.data());
        return std::move(m_result);
    }
    else
    {
        return matmul_mat_mat();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::pack_rhs(ssize_t n, ssize_t k)
{
    shape_type const packing_shape{n, k};
    A packing(packing_shape);
    for (ssize_t i = 0; i < n; ++i)
    {
        for (ssize_t j = 0; j < k; ++j)
        {
            packing(i, j) = m_rhs(j, i);
        }
    }
    return packing;
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::accumulate_tile(A const & packed_rhs,
                                                    ssize_t row_begin,
                                                    ssize_t row_end,
                                                    ssize_t col_begin,
                                                    ssize_t col_end,
                                                    ssize_t inner_begin,
                                                    ssize_t inner_end)
{
    for (ssize_t i = row_begin; i < row_end; ++i)
    {
        for (ssize_t j = col_begin; j < col_end; ++j)
        {
            value_type v = m_result(i, j);
            for (ssize_t l = inner_begin; l < inner_end; ++l)
            {
                v += m_lhs(i, l) * packed_rhs(j, l);
            }
            m_result(i, j) = v;
        }
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat_tiled()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const n = m_result.shape(1);
    ssize_t const k = m_lhs.shape(1);
    A packed_rhs = pack_rhs(n, k);
    for (size_t i = 0; i < m_result.size(); ++i)
    {
        m_result.data(i) = value_type{0};
    }
    for (ssize_t row = 0; row < m; row += m_tile_x)
    {
        ssize_t const row_end = std::min(row + m_tile_x, m);
        for (ssize_t col = 0; col < n; col += m_tile_y)
        {
            ssize_t const col_end = std::min(col + m_tile_y, n);
            for (ssize_t inner = 0; inner < k; inner += m_tile_z)
            {
                ssize_t const inner_end = std::min(inner + m_tile_z, k);
                accumulate_tile(packed_rhs, row, row_end, col, col_end, inner, inner_end);
            }
        }
    }
    return std::move(m_result);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
