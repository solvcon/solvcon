/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/matmul.hpp>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace solvcon
{

namespace detail
{

TEST(MatmulPacking, checks_streamed_thresholds)
{
    MatmulTuning::StreamedPacking const tuning{
        .require_square = true,
        .minimum_dimension = 16,
        .minimum_batch_size = 2,
        .lhs_only = true,
        .rhs_only = true,
        .both_operands = true,
    };

    EXPECT_EQ(PackingSchedule::Complete,
              select_packing_schedule(tuning, {}, true, true, 16, 2));
    EXPECT_EQ(PackingSchedule::Complete,
              select_packing_schedule(
                  tuning, {.lhs = true}, false, true, 16, 2));
    EXPECT_EQ(PackingSchedule::Complete,
              select_packing_schedule(
                  tuning, {.lhs = true}, true, false, 16, 2));
    EXPECT_EQ(PackingSchedule::Complete,
              select_packing_schedule(
                  tuning, {.lhs = true}, true, true, 15, 2));
    EXPECT_EQ(PackingSchedule::Complete,
              select_packing_schedule(
                  tuning, {.lhs = true}, true, true, 16, 1));

    EXPECT_EQ(PackingSchedule::Streamed,
              select_packing_schedule(
                  tuning, {.lhs = true}, true, true, 16, 2));
    EXPECT_EQ(PackingSchedule::Streamed,
              select_packing_schedule(
                  tuning, {.rhs = true}, true, true, 16, 2));
    EXPECT_EQ(PackingSchedule::Streamed,
              select_packing_schedule(
                  tuning, {.lhs = true, .rhs = true}, true, true, 16, 2));
}

TEST(MatmulPacking, keeps_disabled_roles_complete)
{
    MatmulTuning::StreamedPacking const tuning{
        .minimum_dimension = 1,
        .minimum_batch_size = 1,
        .both_operands = true,
    };
    EXPECT_EQ(PackingSchedule::Complete,
              select_packing_schedule(
                  tuning, {.lhs = true}, true, true, 1, 1));
    EXPECT_EQ(PackingSchedule::Complete,
              select_packing_schedule(
                  tuning, {.rhs = true}, true, true, 1, 1));
    EXPECT_EQ(PackingSchedule::Streamed,
              select_packing_schedule(
                  tuning, {.lhs = true, .rhs = true}, true, true, 1, 1));
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
