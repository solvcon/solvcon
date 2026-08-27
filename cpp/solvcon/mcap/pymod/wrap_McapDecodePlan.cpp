/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mcap/pymod/mcap_pymod.hpp>
#include <solvcon/solvcon.hpp>

#include <unordered_map>

namespace solvcon
{

namespace python
{

/**
 * Translate one instruction tuple that solvcon.mcap.DecodePlan compiled.  The
 * Python side owns the schema grammar, so the instruction names it emits are
 * the contract between the two halves.
 */
static mcap::PlanStep plan_step(pybind11::sequence const & item)
{
    static std::unordered_map<std::string, mcap::PlanOp> const ops = {
        {"align", mcap::PlanOp::Align},
        {"skip", mcap::PlanOp::Skip},
        {"skip_string", mcap::PlanOp::SkipString},
        {"skip_sequence", mcap::PlanOp::SkipSequence},
        {"skip_sequence_body", mcap::PlanOp::SkipSequenceBody},
        {"skip_array_body", mcap::PlanOp::SkipArrayBody},
        {"read", mcap::PlanOp::Read},
    };

    auto const name = item[0].cast<std::string>();
    auto const found = ops.find(name);
    if (ops.end() == found)
    {
        throw std::invalid_argument("the MCAP decode plan states an unknown instruction: " + name);
    }

    mcap::PlanStep step{.op = found->second, .operand = 0, .extra = 0};
    // A read names its type; every other operand is a number.
    if (mcap::PlanOp::Read == step.op)
    {
        step.operand = static_cast<uint32_t>(DataType(item[1].cast<std::string>()).type());
        step.extra = item[2].cast<uint32_t>();
    }
    else
    {
        step.operand = item.size() > 1 ? item[1].cast<uint32_t>() : 0;
        step.extra = item.size() > 2 ? item[2].cast<uint32_t>() : 0;
    }

    return step;
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapMcapDecodePlan
    : public WrapBase<WrapMcapDecodePlan, mcap::DecodePlan>
{

public:

    using base_type = WrapBase<WrapMcapDecodePlan, mcap::DecodePlan>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapMcapDecodePlan(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(
                py::init(
                    [](py::iterable const & instructions, size_t column_count)
                    {
                        std::vector<mcap::PlanStep> steps;
                        for (py::handle const item : instructions)
                        {
                            steps.push_back(plan_step(item.cast<py::sequence>()));
                        }
                        return wrapped_type(std::move(steps), column_count);
                    }),
                py::arg("instructions"),
                py::arg("column_count"))
            //
            ;
    }

}; /* end class WrapMcapDecodePlan */

void wrap_McapDecodePlan(pybind11::module & mod)
{
    WrapMcapDecodePlan::commit(mod, "McapDecodePlan", "McapDecodePlan");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
