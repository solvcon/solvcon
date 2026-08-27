/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mcap/mcap_decoder.hpp>

#include <algorithm>
#include <stdexcept>

namespace solvcon
{

namespace mcap
{

/// Number of instructions a container body runs per element; they follow the step.
static size_t body_length(PlanStep const & step)
{
    return PlanOp::SkipArrayBody == step.op ? step.extra : step.operand;
}

static bool is_power_of_two(uint32_t value)
{
    return 0 != value && 0 == (value & (value - 1));
}

/**
 * Whether a CDR primitive decodes into the type.  IDL declares no
 * half-precision and no complex type, so a column is never of Float16,
 * Complex64, or Complex128.
 */
static bool is_cdr_primitive(DataType type)
{
    switch (type)
    {
    case DataType::Bool:
    case DataType::Int8:
    case DataType::Int16:
    case DataType::Int32:
    case DataType::Int64:
    case DataType::Uint8:
    case DataType::Uint16:
    case DataType::Uint32:
    case DataType::Uint64:
    case DataType::Float32:
    case DataType::Float64:
        return true;
    default:
        return false;
    }
}

/**
 * Check one range of instructions and recurse into each container body with
 * the range of that body.
 *
 * @param steps Every instruction of the plan.
 * @param begin Index of the first instruction of the range.
 * @param end Index one past the last instruction of the range; a container body inside it must end by here.
 * @param nested Whether the range is a container body, where a read is not allowed.
 * @param types Type of every column, filled by the read into it; Undefined marks a column not read yet.
 */
// NOLINTNEXTLINE(misc-no-recursion)
static void check_steps(
    std::vector<PlanStep> const & steps, size_t begin, size_t end, bool nested, std::vector<DataType> & types)
{
    size_t it = begin;
    while (it < end)
    {
        PlanStep const & step = steps[it];
        ++it;
        switch (step.op)
        {
        case PlanOp::Align:
            // The cursor rounds the offset with a mask.
            if (!is_power_of_two(step.operand))
            {
                throw std::invalid_argument("the MCAP decode plan aligns to a boundary that is not a power of two");
            }
            break;
        case PlanOp::SkipSequence:
            // The cursor aligns the elements to their own width, which CDR
            // states only for a primitive.
            if (!is_power_of_two(step.operand) || 8 < step.operand)
            {
                throw std::invalid_argument("the MCAP decode plan skips a sequence of no CDR primitive width");
            }
            break;
        case PlanOp::SkipSequenceBody:
        case PlanOp::SkipArrayBody:
        {
            size_t const body_end = it + body_length(step);
            if (body_end > end)
            {
                throw std::invalid_argument("the MCAP decode plan states a container body that runs past its container");
            }
            check_steps(steps, it, body_end, true, types); // recursive here
            it = body_end;
            break;
        }
        case PlanOp::Read:
        {
            // A read inside a container appends one element per container
            // element, not one per message.
            if (nested)
            {
                throw std::invalid_argument("the MCAP decode plan reads a field inside a container");
            }
            DataType const type(static_cast<DataType::enum_type>(step.operand));
            // The enum is a byte, so the cast alone would let a wider operand wrap onto a valid type.
            if (step.operand != static_cast<uint32_t>(type.type()) || !is_cdr_primitive(type))
            {
                throw std::invalid_argument("the MCAP decode plan names a type CDR has no primitive for");
            }
            if (step.extra >= types.size() || DataType::Undefined != types[step.extra])
            {
                throw std::invalid_argument("the MCAP decode plan does not read every column exactly once");
            }
            types[step.extra] = type;
            break;
        }
        default:
            break;
        }
    }
}

DecodePlan::DecodePlan(std::vector<PlanStep> steps, size_t column_count)
    : m_steps(std::move(steps))
    , m_types(column_count, DataType::Undefined)
{
    if (0 == column_count)
    {
        throw std::invalid_argument("the MCAP decode plan reads no field");
    }

    check_steps(m_steps, 0, m_steps.size(), false, m_types);
    if (std::ranges::any_of(m_types, [](DataType type)
                            { return DataType::Undefined == type; }))
    {
        throw std::invalid_argument("the MCAP decode plan does not read every column exactly once");
    }
}

} /* end namespace mcap */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
