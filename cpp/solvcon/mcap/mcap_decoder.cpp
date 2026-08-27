/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mcap/mcap_decoder.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include <solvcon/buffer/SimpleCollector.hpp>
#include <solvcon/mcap/mcap_cursor.hpp>

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

namespace
{

/// Representation identifier and options of the encapsulation header; every offset a plan states counts from the byte after it.
constexpr size_t ENCAPSULATION_SIZE = 4;

/**
 * @internal
 * Cursor over the body of one CDR message.  Unlike the cursor over a record,
 * it offers align(), because a CDR primitive starts at the next multiple of
 * its width.  The multiple counts from the start of the body.
 */
class CdrCursor
    : public detail::ByteCursor
{

public:

    explicit CdrCursor(std::string_view payload)
        : ByteCursor(body_of(payload), "the MCAP message payload is too short for the decode plan")
    {
    }

    /// Advance to the next multiple of a boundary, which is a power of two.
    void align(size_t boundary)
    {
        size_t const remainder = m_pos & (boundary - 1);
        if (0 != remainder)
        {
            skip(boundary - remainder);
        }
    }

    template <typename T>
    T read()
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            // A CDR boolean is one byte, and a byte other than zero or one
            // would make an invalid bool out of a plain copy.
            return 0 != ByteCursor::read<uint8_t>();
        }
        else
        {
            return ByteCursor::read<T>();
        }
    }

private:

    static std::string_view body_of(std::string_view payload);
}; /* end class CdrCursor */

using ColumnCollector = std::variant<
    SimpleCollector<bool>,
    SimpleCollector<int8_t>,
    SimpleCollector<int16_t>,
    SimpleCollector<int32_t>,
    SimpleCollector<int64_t>,
    SimpleCollector<uint8_t>,
    SimpleCollector<uint16_t>,
    SimpleCollector<uint32_t>,
    SimpleCollector<uint64_t>,
    SimpleCollector<float>,
    SimpleCollector<double>>;

/**
 * @internal
 * Walk of one plan over the messages of a topic.  It holds the columns the
 * reads append to, and takes the cursor of each message.
 */
class Executor
{

public:

    Executor(DecodePlan const & plan, std::vector<ColumnCollector> & columns)
        : m_steps(plan.steps())
        , m_columns(columns)
    {
    }

    void run(CdrCursor & cursor) { run(cursor, 0, m_steps.size()); }

private:

    void run(CdrCursor & cursor, size_t begin, size_t end);
    size_t run_body(CdrCursor & cursor, size_t begin, size_t length, uint32_t count);

    std::vector<PlanStep> const & m_steps; ///< Instructions of the plan, in walk order.
    std::vector<ColumnCollector> & m_columns; ///< Columns the reads append to, one per read of the plan.
}; /* end class Executor */

} /* end namespace */

std::string_view CdrCursor::body_of(std::string_view payload)
{
    if (payload.size() < ENCAPSULATION_SIZE)
    {
        throw std::runtime_error("the MCAP message carries no CDR encapsulation header");
    }
    // 0x0001 is plain CDR, little-endian, which is what a ROS 2 recording
    // writes.  A big-endian payload would need every field byte-swapped.
    if (0x00 != static_cast<uint8_t>(payload[0]) || 0x01 != static_cast<uint8_t>(payload[1]))
    {
        throw std::runtime_error("unsupported CDR encapsulation in the MCAP message");
    }

    return payload.substr(ENCAPSULATION_SIZE);
}

// NOLINTNEXTLINE(misc-no-recursion)
void Executor::run(CdrCursor & cursor, size_t begin, size_t end)
{
    size_t it = begin;
    while (it < end)
    {
        PlanStep const & step = m_steps[it];
        ++it;
        switch (step.op)
        {
        case PlanOp::Align:
            cursor.align(step.operand);
            break;
        case PlanOp::Skip:
            cursor.skip(step.operand);
            break;
        case PlanOp::SkipString:
            cursor.skip(cursor.read<uint32_t>());
            break;
        case PlanOp::SkipSequence:
        {
            auto const count = cursor.read<uint32_t>();
            // CDR pads between the count and the elements, and an empty
            // sequence carries no padding because it has nothing to align.
            if (0 != count)
            {
                cursor.align(step.operand);
                cursor.skip(static_cast<size_t>(count) * step.operand);
            }
            break;
        }
        case PlanOp::SkipSequenceBody:
        {
            auto const count = cursor.read<uint32_t>();
            it = run_body(cursor, it, body_length(step), count); // recursive here
            break;
        }
        case PlanOp::SkipArrayBody:
            it = run_body(cursor, it, body_length(step), step.operand); // recursive here
            break;
        case PlanOp::Read:
            std::visit(
                [&cursor](auto & collector)
                {
                    using value_type = typename std::remove_reference_t<decltype(collector)>::value_type;
                    collector.push_back(cursor.read<value_type>());
                },
                m_columns[step.extra]);
            break;
        }
    }
}

/// Run the instructions of one container body once per element.
// NOLINTNEXTLINE(misc-no-recursion)
size_t Executor::run_body(CdrCursor & cursor, size_t begin, size_t length, uint32_t count)
{
    for (uint32_t remaining = count; 0 != remaining; --remaining)
    {
        size_t const before = cursor.position();
        run(cursor, begin, begin + length); // recursive here
        // An element that consumed no byte leaves the cursor where it is, and
        // so does every element after it; a large count would then spin for
        // nothing.  A body reads no column, so the walk skips the rest.
        if (before == cursor.position())
        {
            break;
        }
    }

    return begin + length;
}

/// Collector of the type a column holds; the plan check let only a CDR primitive type through.
static ColumnCollector make_collector(DataType type)
{
    switch (type)
    {
    case DataType::Bool:
        return SimpleCollector<bool>();
    case DataType::Int8:
        return SimpleCollector<int8_t>();
    case DataType::Int16:
        return SimpleCollector<int16_t>();
    case DataType::Int32:
        return SimpleCollector<int32_t>();
    case DataType::Int64:
        return SimpleCollector<int64_t>();
    case DataType::Uint8:
        return SimpleCollector<uint8_t>();
    case DataType::Uint16:
        return SimpleCollector<uint16_t>();
    case DataType::Uint32:
        return SimpleCollector<uint32_t>();
    case DataType::Uint64:
        return SimpleCollector<uint64_t>();
    case DataType::Float32:
        return SimpleCollector<float>();
    case DataType::Float64:
        return SimpleCollector<double>();
    default:
        throw std::logic_error("the MCAP decode plan holds a column type the check does not admit");
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

ColumnSet extract(Reader & reader, std::string const & topic, DecodePlan const & plan)
{
    if (!reader.channels_share_cdr_schema(topic))
    {
        throw std::runtime_error("the MCAP topic carries a channel of another schema or of no CDR encoding: " + topic);
    }

    std::vector<ColumnCollector> collectors;
    for (DataType const type : plan.types())
    {
        collectors.push_back(make_collector(type));
    }

    SimpleCollector<uint64_t> time;
    Executor executor(plan, collectors);
    MessageIterator iterator = reader.messages(topic);
    uint64_t log_time = 0;
    std::string_view payload;
    while (iterator.next(log_time, payload))
    {
        CdrCursor cursor(payload);
        executor.run(cursor);
        time.push_back(log_time);
    }

    ColumnSet out{.time = time.as_array(), .columns = {}};
    for (ColumnCollector & collector : collectors)
    {
        out.columns.push_back(std::visit([](auto & one)
                                         { return Column(one.as_array()); },
                                         collector));
    }

    return out;
}

} /* end namespace mcap */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
