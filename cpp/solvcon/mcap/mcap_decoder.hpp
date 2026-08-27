#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The flat CDR decode plan and its executor.
 *
 * @ingroup group_inout
 */

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/mcap/mcap_reader.hpp>

namespace solvcon
{

namespace mcap
{

/**
 * Kind of an instruction of a decode plan.  The compiler in
 * solvcon/mcap/_decode_plan.py emits each one as a snake_case name and
 * documents what it walks over.  plan_step() in pymod/wrap_McapDecodePlan.cpp
 * maps the name onto this enum.
 *
 * @ingroup group_inout
 */
enum class PlanOp : uint8_t
{
    Align = 0,
    Skip,
    SkipString,
    SkipSequence,
    SkipSequenceBody,
    SkipArrayBody,
    Read,
}; /* end enum class PlanOp */

/**
 * One instruction of a decode plan.  The operand and the extra hold what the
 * tuple carries after the name, in the same order:
 *
 * - Align: the boundary.
 * - Skip: the length.
 * - SkipSequence: the element width.
 * - SkipSequenceBody: the body length.
 * - SkipArrayBody: the count, then the body length.
 * - Read: the DataType value of the type the tuple names, then the column.
 *
 * @ingroup group_inout
 */
struct PlanStep
{
    PlanOp op = PlanOp::Align;
    uint32_t operand = 0;
    uint32_t extra = 0;
}; /* end struct PlanStep */

/**
 * A column of one of the scalar types a CDR primitive decodes into.  IDL
 * declares no half-precision and no complex type, so a column is never a
 * SimpleArray of Float16, Complex64, or Complex128.
 *
 * @ingroup group_inout
 */
using Column = std::variant<
    SimpleArray<bool>,
    SimpleArray<int8_t>,
    SimpleArray<int16_t>,
    SimpleArray<int32_t>,
    SimpleArray<int64_t>,
    SimpleArray<uint8_t>,
    SimpleArray<uint16_t>,
    SimpleArray<uint32_t>,
    SimpleArray<uint64_t>,
    SimpleArray<float>,
    SimpleArray<double>>;

/**
 * Decode plan over one CDR message.  Construction rejects a plan that reads a
 * column inside a container, more than once, or not at all, so the columns
 * share a row index.  Construction also rejects a read of a type no CDR
 * primitive decodes into, a container body that runs past the range holding
 * it, and an operand the cursor cannot align or skip by.
 *
 * @ingroup group_inout
 */
class DecodePlan
{

public:

    /**
     * Check and hold the instructions of a plan.
     *
     * @param steps Instructions in walk order.
     * @param column_count Number of columns the reads fill; the reads must
     *                     name each column from 0 to this count exactly once.
     */
    DecodePlan(std::vector<PlanStep> steps, size_t column_count);

    std::vector<PlanStep> const & steps() const { return m_steps; }
    /// Type of every column, as the read into it states.
    std::vector<DataType> const & types() const { return m_types; }

private:

    // TODO: Once the buffer subsystem offers a collector of records, replace
    // the STL containers below.  SimpleCollector is for fundamental types, and
    // these hold instructions and column types (issue #1286).
    std::vector<PlanStep> m_steps;
    std::vector<DataType> m_types;
}; /* end class DecodePlan */

/**
 * Columns extracted from the messages of one topic: the log time of every
 * message in nanoseconds, and one column per read of the plan.  Every column
 * holds one element per message, so a row index names the same message in
 * all of them.
 *
 * @ingroup group_inout
 */
struct ColumnSet
{
    SimpleArrayUint64 time;
    // TODO: The same replacement as in DecodePlan applies here (issue #1286).
    std::vector<Column> columns;
}; /* end struct ColumnSet */

/**
 * Run a decode plan over every message of a topic.
 *
 * The walk visits each payload once and appends what it reads to a collector
 * per column.  The cost therefore follows the instructions the plan runs,
 * once per container element inside a body, not the fields the schema
 * declares.
 *
 * @param reader Reader of the open file.
 * @param topic Topic whose messages the plan walks.
 * @param plan Decode plan to run over each message.
 * @return The log time column and one column per read of the plan.
 *
 * @ingroup group_inout
 */
ColumnSet extract(Reader & reader, std::string const & topic, DecodePlan const & plan);

} /* end namespace mcap */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
