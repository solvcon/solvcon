#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Select and execute SimpleArray matrix multiplication kernels.
 *
 * @ingroup group_core
 */

#include <solvcon/buffer/matmul_plan.hpp>
#include <solvcon/math/Winograd.hpp>
#include <solvcon/math/math.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace solvcon
{

namespace detail
{

template <typename T>
inline constexpr bool is_blas_element_v = std::is_same_v<T, float> ||
                                          std::is_same_v<T, double> ||
                                          std::is_same_v<T, Complex<float>> ||
                                          std::is_same_v<T, Complex<double>>;

// Whether a matmul over T reaches BLAS at all: the type has to be one BLAS
// takes and the build has to have a backend behind the wrappers.
template <typename T>
inline constexpr bool use_matmul_blas_v = has_blas_backend && is_blas_element_v<T>;

/**
 * @brief Identify the contraction kernel fixed before batch traversal.
 */
enum class MatmulKernel : std::uint8_t
{
    Naive,
    BlasDot,
    BlasGevm,
    BlasGemv,
    BlasGemm,
    Winograd,
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
 * A tuning table supplies the workload boundaries used to compare naive,
 * direct BLAS, and packing routes. Keeping the values separate from dispatch
 * code allows a backend or value type to provide another table.
 *
 * @note MatmulExecutor currently shares one compile-time table across
 * supported CBLAS backends.
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
     * @brief Set the GEMM crossover for packing BLAS-incompatible matrices.
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

    /**
     * @brief Configure one-level Winograd dispatch for square GEMMs.
     */
    struct Winograd
    {
        ssize_t minimum_side;
    }; /* end struct Winograd */

    DirectBlas direct_blas;
    MatrixPacking matrix_packing;
    BatchedVector batched_vector;
    Winograd winograd;
}; /* end struct MatmulTuning */

constexpr bool meets_winograd_threshold(
    MatmulTuning::Winograd const & tuning,
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size) noexcept
{
    return rows == columns &&
           columns == inner_size &&
           rows >= tuning.minimum_side &&
           rows % 2 == 0;
}

inline constexpr MatmulTuning::Winograd WINOGRAD_TUNING{
    .minimum_side = 16384,
};

/**
 * @brief Record the fixed kernel and packing for one matmul call.
 */
struct MatmulSelection
{
    MatmulKernel kernel = MatmulKernel::Naive;
    PackingState packing;
}; /* end struct MatmulSelection */

std::string_view matmul_kernel_name(MatmulKernel kernel) noexcept;
std::optional<MatmulKernel> matmul_kernel_from_name(std::string_view name) noexcept;

/**
 * @brief Select and execute one contraction kernel for a MatmulPlan.
 *
 * MatmulExecutor combines plan metadata with MatmulTuning to select a
 * MatmulSelection. It applies the selected operand preparation, rebuilds the
 * plan when a physical layout changes, and traverses every batch offset with
 * the same kernel. Naive kernels read signed strides directly, while BLAS
 * kernels consume compatible vector and matrix descriptors. Eligible compact
 * square GEMMs may use one-level Winograd multiplication.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the executor visits ten batch offsets and
 * evaluates one `(3,4) @ (4,6)` contraction at each offset. The results are
 * written into the allocated `(2,5,3,6)` output.
 *
 * @note The current implementation selects naive, BLAS, or Winograd kernels
 * using measured thresholds. Packing materializes reused broadcast operands
 * once and streams the remaining operands one matrix at a time through
 * bounded row-major scratch.
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
    void execute(MatmulKernel kernel);

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
            .gemm_min_dimension = is_complex_v<value_type> ? 8 : 16,
        },
        .batched_vector = {
            .direct_min_matrix_elements = 256,
            .always_pack_min_matrix_elements = 4096,
            .pack_min_matrix_elements = 1024,
            .pack_min_batch_size = 4,
            .reuse_min_matrix_elements = 576,
            .reuse_min_total_output_elements = 128,
        },
        .winograd = WINOGRAD_TUNING,
    };
    static constexpr size_t MAX_SCRATCH_ELEMENTS = std::min(
        std::numeric_limits<size_t>::max() / sizeof(value_type),
        static_cast<size_t>(std::numeric_limits<ssize_t>::max()));

    MatmulSelection select() const;
    std::optional<MatmulSelection> select(MatmulKernel kernel) const;
    std::optional<MatmulSelection> select_blas(MatmulKernel kernel) const;
    MatmulSelection select_dot() const;
    MatmulSelection select_gevm() const;
    MatmulSelection select_gemv() const;
    MatmulSelection select_gemm() const;
    PackingState select_matrix_packing(PackingState required) const;
    PackingState select_vector_packing(PackingState required) const;
    bool should_pack_vector() const;
    void pack(PackingState const & packing);
    void run(MatmulSelection const & selection);

    template <MatmulKernel Kernel>
    void execute_contractions();

    template <MatmulKernel Kernel>
    void execute_at(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base);

    void execute_dot_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);
    void execute_gevm_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);
    void execute_gemv_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);
    void prepare_gemm_scratch();
    void execute_gemm_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data);

    static size_t checked_scratch_elements(ssize_t rows, ssize_t columns);
    static matrix_view_type pack_matrix_to_scratch(
        value_type const * source,
        value_type * scratch,
        value_type const *& cached_source,
        ssize_t rows,
        ssize_t columns,
        ssize_t row_stride,
        ssize_t column_stride);

    void execute_winograd();

    std::optional<matrix_view_type> lhs_matrix_view(value_type const * data) const;
    std::optional<matrix_view_type> rhs_matrix_view(value_type const * data) const;
    static matrix_view_type require_matrix_view(std::optional<matrix_view_type> view);
    static std::optional<matrix_view_type> make_matrix_view(
        value_type const * data,
        ssize_t row_stride,
        ssize_t column_stride,
        ssize_t rows,
        ssize_t columns);
    template <size_t ColumnBlock>
    void multiply_naive_column_block(value_type * output, value_type const * lhs, value_type const * rhs);
    void execute_naive(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base);

    MatmulPlan m_plan;
    Array const & m_lhs;
    Array const & m_rhs;
    std::optional<Array> m_packed_lhs;
    std::optional<Array> m_packed_rhs;
    std::optional<Array> m_gemm_scratch;
    size_t m_lhs_scratch_elements = 0;
    value_type const * m_cached_lhs_source = nullptr;
    value_type const * m_cached_rhs_source = nullptr;
    value_type * m_output_data;
    value_type const * m_lhs_data;
    value_type const * m_rhs_data;
}; /* end class MatmulExecutor */

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
    run(select());
}

template <typename Array>
void MatmulExecutor<Array>::execute(MatmulKernel kernel)
{
    std::optional<MatmulSelection> const selection = select(kernel);
    if (!selection)
    {
        std::string_view const reason = use_matmul_blas_v<value_type>
                                            ? "is not eligible for these operands"
                                            : "requires a BLAS backend";
        throw std::invalid_argument(
            std::format("matmul(): kernel '{}' {}", matmul_kernel_name(kernel), reason));
    }
    run(selection.value());
}

template <typename Array>
void MatmulExecutor<Array>::run(MatmulSelection const & selection)
{
    if (m_plan.batch_size() == 0)
    {
        return;
    }

    if (selection.packing)
    {
        pack(selection.packing);
    }

    // Non-naive selections exist only for value types with a BLAS backend,
    // so other instantiations keep only the naive dispatch.
    if constexpr (use_matmul_blas_v<value_type>)
    {
        switch (selection.kernel)
        {
        case MatmulKernel::Naive:
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
            prepare_gemm_scratch();
            execute_contractions<MatmulKernel::BlasGemm>();
            return;
        case MatmulKernel::Winograd:
            execute_winograd();
            return;
        }
    }
    execute_contractions<MatmulKernel::Naive>();
}

template <typename Array>
MatmulSelection MatmulExecutor<Array>::select() const
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
        .kernel = use_blas ? MatmulKernel::BlasDot : MatmulKernel::Naive,
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
            .kernel = use_blas ? MatmulKernel::BlasGevm : MatmulKernel::Naive,
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
        .kernel = use_blas ? MatmulKernel::BlasGevm : MatmulKernel::Naive,
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
            .kernel = use_blas ? MatmulKernel::BlasGemv : MatmulKernel::Naive,
            .packing = packing,
        };
    }

    bool const use_blas = m_plan.rhs_inner_stride() > 0 &&
                          std::min(m_plan.rows(), m_plan.inner_size()) >= TUNING.direct_blas.gemv_min_dimension;
    return MatmulSelection{
        .kernel = use_blas ? MatmulKernel::BlasGemv : MatmulKernel::Naive,
        .packing = {},
    };
}

template <typename Array>
MatmulSelection MatmulExecutor<Array>::select_gemm() const
{
    bool const compact_row_major = !m_plan.has_batch_axes() &&
                                   m_plan.lhs_row_stride() == m_plan.inner_size() &&
                                   m_plan.lhs_inner_stride() == 1 &&
                                   m_plan.rhs_inner_stride() == m_plan.columns() &&
                                   m_plan.rhs_column_stride() == 1;
    if (compact_row_major &&
        meets_winograd_threshold(
            TUNING.winograd, m_plan.rows(), m_plan.columns(), m_plan.inner_size()))
    {
        return MatmulSelection{.kernel = MatmulKernel::Winograd, .packing = {}};
    }

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
        return MatmulSelection{
            .kernel = MatmulKernel::BlasGemm,
            .packing = select_matrix_packing(required),
        };
    }
    return MatmulSelection{};
}

template <typename Array>
std::optional<MatmulSelection> MatmulExecutor<Array>::select(MatmulKernel kernel) const
{
    if (kernel == MatmulKernel::Naive)
    {
        return MatmulSelection{};
    }
    if constexpr (use_matmul_blas_v<value_type>)
    {
        return select_blas(kernel);
    }
    else
    {
        return std::nullopt;
    }
}

template <typename Array>
std::optional<MatmulSelection> MatmulExecutor<Array>::select_blas(MatmulKernel kernel) const
{
    if (m_plan.rows() <= 0 || m_plan.columns() <= 0 || m_plan.inner_size() <= 0)
    {
        return std::nullopt;
    }
    MatmulKernel operand_kernel;
    if (m_plan.lhs_is_vector())
    {
        operand_kernel = m_plan.rhs_is_vector() ? MatmulKernel::BlasDot : MatmulKernel::BlasGevm;
    }
    else
    {
        operand_kernel = m_plan.rhs_is_vector() ? MatmulKernel::BlasGemv : MatmulKernel::BlasGemm;
    }
    if (kernel != operand_kernel && kernel != MatmulKernel::Winograd)
    {
        return std::nullopt;
    }

    switch (kernel)
    {
    case MatmulKernel::Naive:
        return MatmulSelection{};
    case MatmulKernel::BlasDot:
    {
        ssize_t const lhs_stride = m_plan.lhs_inner_stride();
        ssize_t const rhs_stride = m_plan.rhs_inner_stride();
        bool const direct_negative = lhs_stride == -1 && rhs_stride == -1;
        return MatmulSelection{
            .kernel = MatmulKernel::BlasDot,
            .packing = {
                .lhs = lhs_stride <= 0 && !direct_negative,
                .rhs = rhs_stride <= 0 && !direct_negative,
            },
        };
    }
    case MatmulKernel::BlasGevm:
        return MatmulSelection{
            .kernel = MatmulKernel::BlasGevm,
            .packing = {
                .lhs = m_plan.lhs_inner_stride() <= 0,
                .rhs = !rhs_matrix_view(m_rhs_data),
            },
        };
    case MatmulKernel::BlasGemv:
        return MatmulSelection{
            .kernel = MatmulKernel::BlasGemv,
            .packing = {
                .lhs = !lhs_matrix_view(m_lhs_data),
                .rhs = m_plan.rhs_inner_stride() <= 0,
            },
        };
    case MatmulKernel::BlasGemm:
        return MatmulSelection{
            .kernel = MatmulKernel::BlasGemm,
            .packing = select_matrix_packing(PackingState{
                .lhs = !lhs_matrix_view(m_lhs_data),
                .rhs = !rhs_matrix_view(m_rhs_data),
            }),
        };
    case MatmulKernel::Winograd:
    {
        bool const even_dimensions = m_plan.rows() % 2 == 0 &&
                                     m_plan.columns() % 2 == 0 &&
                                     m_plan.inner_size() % 2 == 0;
        if (operand_kernel != MatmulKernel::BlasGemm ||
            m_plan.has_batch_axes() ||
            !even_dimensions)
        {
            return std::nullopt;
        }
        auto const lhs_view = lhs_matrix_view(m_lhs_data);
        auto const rhs_view = rhs_matrix_view(m_rhs_data);
        return MatmulSelection{
            .kernel = MatmulKernel::Winograd,
            .packing = {
                .lhs = !lhs_view || lhs_view->m_transpose != BlasTranspose::None,
                .rhs = !rhs_view || rhs_view->m_transpose != BlasTranspose::None,
            },
        };
    }
    }
    return std::nullopt;
}

template <typename Array>
PackingState MatmulExecutor<Array>::select_matrix_packing(PackingState required) const
{
    return PackingState{
        .lhs = required.lhs && m_plan.lhs_is_broadcast() && !m_plan.lhs_has_zero_batch_stride(),
        .rhs = required.rhs && m_plan.rhs_is_broadcast() && !m_plan.rhs_has_zero_batch_stride(),
    };
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
    if constexpr (Kernel == MatmulKernel::Naive)
    {
        execute_naive(output_base, lhs_base, rhs_base);
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
void MatmulExecutor<Array>::prepare_gemm_scratch()
{
    if (m_gemm_scratch)
    {
        m_cached_lhs_source = nullptr;
        m_cached_rhs_source = nullptr;
        return;
    }

    std::optional<matrix_view_type> const lhs_view = lhs_matrix_view(m_lhs_data);
    std::optional<matrix_view_type> const rhs_view = rhs_matrix_view(m_rhs_data);
    if (lhs_view && rhs_view)
    {
        return;
    }

    m_lhs_scratch_elements = lhs_view ? 0 : checked_scratch_elements(m_plan.rows(), m_plan.inner_size());
    size_t const rhs_elements = rhs_view ? 0 : checked_scratch_elements(m_plan.inner_size(), m_plan.columns());
    if (m_lhs_scratch_elements > MAX_SCRATCH_ELEMENTS - rhs_elements)
    {
        throw std::length_error("MatmulExecutor::prepare_gemm_scratch(): scratch size overflows");
    }
    typename Array::shape_type const scratch_shape{static_cast<ssize_t>(m_lhs_scratch_elements + rhs_elements)};
    m_gemm_scratch.emplace(scratch_shape);
}

template <typename Array>
void MatmulExecutor<Array>::execute_gemm_blas(value_type * output, value_type const * lhs_data, value_type const * rhs_data)
{
    std::optional<matrix_view_type> lhs_view = lhs_matrix_view(lhs_data);
    std::optional<matrix_view_type> rhs_view = rhs_matrix_view(rhs_data);
    value_type * scratch = nullptr;
    if (!lhs_view || !rhs_view)
    {
        if (!m_gemm_scratch)
        {
            throw std::logic_error("MatmulExecutor::execute_gemm_blas(): missing GEMM scratch");
        }
        scratch = m_gemm_scratch->data();
    }
    if (!lhs_view)
    {
        lhs_view = pack_matrix_to_scratch(
            lhs_data,
            scratch,
            m_cached_lhs_source,
            m_plan.rows(),
            m_plan.inner_size(),
            m_plan.lhs_row_stride(),
            m_plan.lhs_inner_stride());
    }
    if (!rhs_view)
    {
        rhs_view = pack_matrix_to_scratch(
            rhs_data,
            scratch + m_lhs_scratch_elements,
            m_cached_rhs_source,
            m_plan.inner_size(),
            m_plan.columns(),
            m_plan.rhs_inner_stride(),
            m_plan.rhs_column_stride());
    }

    BlasGemmOperation<value_type> const operation{
        .rows = m_plan.rows(),
        .columns = m_plan.columns(),
        .inner_size = m_plan.inner_size(),
        .lhs = require_matrix_view(lhs_view),
        .rhs = require_matrix_view(rhs_view),
        .output = {
            .m_data = output,
            .m_leading_dimension = m_plan.columns(),
        },
        .alpha = value_type{1},
        .beta = value_type{0},
    };
    gemm_blas(operation);
}

template <typename Array>
size_t MatmulExecutor<Array>::checked_scratch_elements(ssize_t rows, ssize_t columns)
{
    if (rows < 0 || columns < 0)
    {
        throw std::length_error("MatmulExecutor::prepare_gemm_scratch(): scratch size overflows");
    }
    auto const unsigned_rows = static_cast<size_t>(rows);
    auto const unsigned_columns = static_cast<size_t>(columns);
    if (unsigned_columns != 0 && unsigned_rows > MAX_SCRATCH_ELEMENTS / unsigned_columns)
    {
        throw std::length_error("MatmulExecutor::prepare_gemm_scratch(): scratch size overflows");
    }
    return unsigned_rows * unsigned_columns;
}

template <typename Array>
typename MatmulExecutor<Array>::matrix_view_type MatmulExecutor<Array>::pack_matrix_to_scratch(
    value_type const * source,
    value_type * scratch,
    value_type const *& cached_source,
    ssize_t rows,
    ssize_t columns,
    ssize_t row_stride,
    ssize_t column_stride)
{
    matrix_view_type const view{
        .m_data = scratch,
        .m_leading_dimension = columns,
        .m_transpose = BlasTranspose::None,
    };
    if (source == cached_source)
    {
        return view;
    }

    for (ssize_t row = 0; row < rows; ++row)
    {
        value_type const * source_row = source + row * row_stride;
        value_type * scratch_row = scratch + row * columns;
        if (column_stride == 1)
        {
            std::copy_n(source_row, static_cast<size_t>(columns), scratch_row);
        }
        else if (column_stride == -1)
        {
            std::reverse_copy(source_row - columns + 1, source_row + 1, scratch_row);
        }
        else
        {
            for (ssize_t column = 0; column < columns; ++column)
            {
                scratch_row[column] = source_row[column * column_stride];
            }
        }
    }
    cached_source = source;
    return view;
}

template <typename Array>
void MatmulExecutor<Array>::execute_winograd()
{
    if constexpr (use_matmul_blas_v<value_type>)
    {
        BlasOutputView<value_type> const output{
            .m_data = m_output_data,
            .m_leading_dimension = m_plan.columns(),
        };
        gemm_winograd(
            m_plan.rows(),
            m_plan.columns(),
            m_plan.inner_size(),
            require_matrix_view(lhs_matrix_view(m_lhs_data)),
            require_matrix_view(rhs_matrix_view(m_rhs_data)),
            output);
    }
    else
    {
        throw std::logic_error("MatmulExecutor::execute_winograd(): unavailable Winograd kernel");
    }
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
template <size_t ColumnBlock>
void MatmulExecutor<Array>::multiply_naive_column_block(
    value_type * output,
    value_type const * lhs,
    value_type const * rhs)
{
    std::array<value_type, ColumnBlock> totals{};
    ssize_t lhs_offset = 0;
    ssize_t rhs_inner_offset = 0;
    for (ssize_t inner = 0; inner < m_plan.inner_size(); ++inner)
    {
        value_type const lhs_value = lhs[lhs_offset];
        ssize_t rhs_column_offset = rhs_inner_offset;
        for (value_type & total : totals)
        {
            total += lhs_value * rhs[rhs_column_offset];
            rhs_column_offset += m_plan.rhs_column_stride();
        }
        lhs_offset += m_plan.lhs_inner_stride();
        rhs_inner_offset += m_plan.rhs_inner_stride();
    }
    std::ranges::copy(totals, output);
}

template <typename Array>
void MatmulExecutor<Array>::execute_naive(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base)
{
    constexpr ssize_t LARGE_COLUMN_BLOCK = 8;
    constexpr ssize_t SMALL_COLUMN_BLOCK = 4;
    bool const block_columns = m_plan.columns() >= SMALL_COLUMN_BLOCK &&
                               m_plan.inner_size() != 0 &&
                               (m_plan.rhs_column_stride() == 1 || m_plan.rhs_column_stride() == -1);

    for (ssize_t row = 0; row < m_plan.rows(); ++row)
    {
        ssize_t const lhs_row_base = lhs_base + row * m_plan.lhs_row_stride();
        ssize_t const output_row_base = output_base + row * m_plan.columns();
        ssize_t column = 0;
        if (block_columns)
        {
            value_type const * lhs = m_lhs_data + lhs_row_base;
            for (; column + LARGE_COLUMN_BLOCK <= m_plan.columns(); column += LARGE_COLUMN_BLOCK)
            {
                multiply_naive_column_block<LARGE_COLUMN_BLOCK>(
                    m_output_data + output_row_base + column,
                    lhs,
                    m_rhs_data + rhs_base + column * m_plan.rhs_column_stride());
            }
            if (column + SMALL_COLUMN_BLOCK <= m_plan.columns())
            {
                multiply_naive_column_block<SMALL_COLUMN_BLOCK>(
                    m_output_data + output_row_base + column,
                    lhs,
                    m_rhs_data + rhs_base + column * m_plan.rhs_column_stride());
                column += SMALL_COLUMN_BLOCK;
            }
        }
        for (; column < m_plan.columns(); ++column)
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

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
