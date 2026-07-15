#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Encode rectangle and polygon geometry as OASIS format byte records.
 *
 * @ingroup group_inout
 */

#include <solvcon/base.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace solvcon
{

class OasisRecordPoly;
class OasisRecordRect;

/**
 * OASIS device converts coordinates information to OASIS format.
 *
 * OasisDevice class store rectangles or polygons as OASIS format,
 * the implementation based on OASIS specification, convert coordinates
 * information (xy) to OASIS geometry format. The format are represented as
 * byte-continuations. LSB present that the next bytes is belonging the
 * group or not.
 *
 * @ingroup group_inout
 */
class OasisDevice
{
public:
    static void append_signed_integer(std::vector<uint8_t> & segment, int value);
    static void append_unsigned_integer(std::vector<uint8_t> & segment, int value);

    OasisDevice() = default;
    OasisDevice(OasisDevice const &) = default;
    OasisDevice(OasisDevice &&) = default;
    OasisDevice & operator=(OasisDevice const &) = default;
    OasisDevice & operator=(OasisDevice &&) = default;
    ~OasisDevice() = default;

    void add_poly_record(const OasisRecordPoly & record);
    void add_rect_record(const OasisRecordRect & record);

    std::vector<uint8_t> to_bytes();

private:
    static void append_magic_bytes(std::vector<uint8_t> & segment);
    static void append_start_record_bytes(std::vector<uint8_t> & segment);
    static void append_cell_and_cell_name_record_byte(std::vector<uint8_t> & segment);
    static void append_end_record_byte(std::vector<uint8_t> & segment);

    template <typename T>
    static void append_record_bytes(std::vector<uint8_t> & bytes, const T & record);

    std::vector<OasisRecordPoly> m_polygon_records;
    std::vector<OasisRecordRect> m_rect_records;
}; /* end class OasisDevice */

/**
 * Convert vertex information to OASIS polygon record bytes.
 *
 * @ingroup group_inout
 */
class OasisRecordPoly
{

public:
    explicit OasisRecordPoly(std::vector<std::pair<int, int>> vertices)
        : m_vertices(std::move(vertices)) {};
    OasisRecordPoly() = delete;
    OasisRecordPoly(OasisRecordPoly const &) = default;
    OasisRecordPoly(OasisRecordPoly &&) = default;
    OasisRecordPoly & operator=(OasisRecordPoly const &) = default;
    OasisRecordPoly & operator=(OasisRecordPoly &&) = default;
    ~OasisRecordPoly() = default;

    std::vector<uint8_t> to_bytes() const;

private:
    std::vector<std::pair<int, int>> m_vertices;
}; /* end class OasisRecordPoly */

/**
 * Convert rectangle information to OASIS rectangle record bytes.
 *
 * @ingroup group_inout
 */
class OasisRecordRect
{
public:
    OasisRecordRect(int left, int lower, int w, int h)
        : m_left(left)
        , m_lower(lower)
        , m_w(w)
        , m_h(h)
    {
    }
    OasisRecordRect() = delete;
    OasisRecordRect(OasisRecordRect const &) = default;
    OasisRecordRect(OasisRecordRect &&) = default;
    OasisRecordRect & operator=(OasisRecordRect const &) = default;
    OasisRecordRect & operator=(OasisRecordRect &&) = default;
    ~OasisRecordRect() = default;

    std::vector<uint8_t> to_bytes() const;

private:
    int m_left, m_lower, m_w, m_h;
}; /* end class OasisRecordRect */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: