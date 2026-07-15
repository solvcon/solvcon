/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/oasis/oasis_device.hpp>
#include <cstdint>
#include <utility>

namespace solvcon
{

/*
 * Convert value to OASIS unsigned integer bytes and append to segment array.
 * Encodes payload as a base-128 variable-length sequence, emitting 7 data
 * bits per byte and using the MSB as a continuation flag.
 * Refer OASIS Draft 7.2.1 for more information.
 */
void OasisDevice::append_unsigned_integer(std::vector<uint8_t> & segment, int value)
{
    int payload = value;

    if (payload == 0)
    {
        segment.push_back(0x00);
    }

    while (payload >= 1)
    {
        int const first_bit = payload >= 128 ? 1 : 0;
        segment.push_back((first_bit << 7) | (payload % 128));
        payload /= 128;
    }
}

/*
 * Convert value to OASIS signed integer bytes and append to segment array.
 * 1. DIR_BIT will be 1 if value is negative, otherwise, DIRBIT will be 0.
 * 2. The payload will left-shift 1 bit on value and do OR with DIR_BIT.
 * 3. Encodes payload as a base-128 variable-length sequence, emitting 7 data
 *    bits per byte and using the MSB as a continuation flag. This operation is
 *    as same as OASIS unsigned interger bytes.
 * Refer OASIS Draft 7.2.2 for more information.
 */
void OasisDevice::append_signed_integer(std::vector<uint8_t> & segment, int value)
{
    const int DIR_BIT = value < 0 ? 1 : 0;
    int const delta_codec = abs(value) << 1 | DIR_BIT;

    int payload = delta_codec;

    if (payload == 0)
    {
        segment.push_back(0x02);
    }

    while (payload >= 1)
    {
        int const first_bit = payload >= 128 ? 1 : 0;
        segment.push_back((first_bit << 7) | (payload % 128));
        payload /= 128;
    }
}

std::vector<uint8_t> OasisRecordRect::to_bytes() const
{
    // RECTANGLE record should be
    // '20' info-bytes [layer] [datatype] [w] [h] [x] [y] [repetition]
    // Please refer OASIS Draft section 25.
    std::vector<uint8_t> segment;

    segment.push_back(0x14);

    const int S = (m_w == m_h); // Is square? (1 if yes, 0 if no)
    const int W = 1; // Have width? (1 if yes, 0 if no)
    const int H = (m_w != m_h); // Have height? (1 if yes, 0 if no, must be 0 if S = 1)
    const int X = 1;
    const int Y = 1;
    const int R = 0;
    const int D = 1; // Have Datatype? (1 if yes, 0 if no)
    const int L = 1; // Have Layer? (1 if yes, 0 if no)

    const int INFO = (S << 7) | (W << 6) | (H << 5) |
                     (X << 4) | (Y << 3) | (R << 2) | (D << 1) | L;
    segment.push_back(INFO);

    // Layer-num and datatype-num (0 in default).
    segment.push_back(0x00);
    segment.push_back(0x00);

    OasisDevice::append_unsigned_integer(segment, m_w);
    OasisDevice::append_unsigned_integer(segment, m_h);
    OasisDevice::append_signed_integer(segment, m_left);
    OasisDevice::append_signed_integer(segment, m_lower);

    return segment;
}

std::vector<uint8_t> OasisRecordPoly::to_bytes() const
{
    // Polygon record should be
    // '21' 00PXYRDL [layer-num] [datatype-num] [point-list] [x] [y] [rep]
    // Please refer OASIS Draft section 26.
    std::vector<uint8_t> segment;

    segment.push_back(0x15);

    // Info bytes:
    // - Have point-list, X and Y.
    // - Have datatype and layer (Use 0)
    const int P = 1; // Have Polygon-List? (1 if yes, 0 if no)
    const int X = 1;
    const int Y = 1;
    const int R = 0;
    const int D = 1; // Have Datatype? (1 if yes, 0 if no)
    const int L = 1; // Have Layer? (1 if yes, 0 if no)
    const int INFO = (P << 5) | (X << 4) | (Y << 3) | (R << 2) | (D << 1) | L;
    segment.push_back(INFO);

    // Layer-num and datatype-num (0 in default).
    segment.push_back(0x00);
    segment.push_back(0x00);

    // The 1-Delta have two different type:
    //  - Type 0: Start with horizontal
    //  - Type 1: Start with vertical
    if (m_vertices[0].second == m_vertices[1].second)
    {
        segment.push_back(0x00);
    }
    else if (m_vertices[0].first == m_vertices[0].first)
    {
        segment.push_back(0x01);
    }

    // The vertex count shoud be (vertex - 1)
    segment.push_back(m_vertices.size() - 1);

    // In this implementation, point-list only support 1-delta format.
    // Please refer Point-list in OASIS draft 7.7.
    std::vector<uint8_t> point_list;

    for (int i = 0; i < m_vertices.size() - 1; i++)
    {
        std::pair<int, int> const curr_v = m_vertices[i];
        std::pair<int, int> const next_v = m_vertices[i + 1];

        // Convert delta value to OASIS signed interegr bytes.
        OasisDevice::append_signed_integer(
            point_list,
            (next_v.first - curr_v.first) + (next_v.second - curr_v.second));
    }

    segment.insert(segment.end(), point_list.begin(), point_list.end());

    // X value
    OasisDevice::append_signed_integer(segment, m_vertices[0].first);

    // Y value
    OasisDevice::append_signed_integer(segment, m_vertices[0].second);

    return segment;
}

std::vector<uint8_t> OasisDevice::to_bytes()
{
    // The simple OASIS byte format should be:
    // <magic-byte> <START>
    // <CELLNAME> <CELL>
    // <GEOMETRY_RECORD_1> <GEOMETRY_RECORD_2> ... <GEOMETRY_RECORD_N>
    // <END>
    std::vector<uint8_t> result;

    append_magic_bytes(result);
    append_start_record_bytes(result);
    append_cell_and_cell_name_record_byte(result);

    for (const OasisRecordRect & record : m_rect_records)
    {
        append_record_bytes(result, record);
    }

    for (const OasisRecordPoly & record : m_polygon_records)
    {
        append_record_bytes(result, record);
    }

    append_end_record_byte(result);

    return result;
}

void OasisDevice::add_poly_record(const OasisRecordPoly & record)
{
    m_polygon_records.push_back(record);
}

void OasisDevice::add_rect_record(const OasisRecordRect & record)
{
    m_rect_records.push_back(record);
}

void OasisDevice::append_magic_bytes(std::vector<uint8_t> & segment)
{
    // Magic byte should be %SEMI-OASIS<CR><NL>, which <CR><NL> is 0x0D 0x0A.
    // Please refer OASIS Draft section 6.4.

    std::string magic_byte = "%SEMI-OASIS\x0D\x0A";

    segment.insert(segment.end(), magic_byte.begin(), magic_byte.end());
}

void OasisDevice::append_start_record_bytes(std::vector<uint8_t> & segment)
{
    // START record should be
    // '1' version-string unit offset-flag [ table-offsets ]
    // Please refer OASIS Draft section 13.

    // The first byte should 1.
    segment.push_back(0x01);

    // The version string will be [LENGTH][STRING-ASCII], we
    // fixed it as "1.0".
    segment.push_back(0x03);

    std::string version = "1.0";
    segment.insert(segment.end(), version.begin(), version.end());

    // The unit we using 0.001 as default.
    std::vector<uint8_t> unit = {0x00, 0xE8, 0x07};
    segment.insert(segment.end(), unit.begin(), unit.end());

    // The offset-flag we using 0x00 as default.
    // The table-offset will store in END record if offset-flag is 0x01.
    // We put 6 pairs of 0x00 to describe that we don't have any tables.
    // Therefore, it insert 13 0x00 in total.
    segment.insert(segment.end(), 13, 0x00);
}

void OasisDevice::append_cell_and_cell_name_record_byte(std::vector<uint8_t> & segment)
{
    // Append CELLNAME record, the default CELLNAME is TOP.
    // TODO: This record is fixed. It can be modify by user in the future.
    std::vector<uint8_t> cellname = {0x03, 03, 0x54, 0x4F, 0x50};
    segment.insert(segment.end(), cellname.begin(), cellname.end());

    // Append CELL record.
    // TODO: This record is fixed. It can be modify by user in the future.
    std::vector<uint8_t> cell = {0x0D, 0x00};
    segment.insert(segment.end(), cell.begin(), cell.end());
}

void OasisDevice::append_end_record_byte(std::vector<uint8_t> & segment)
{
    // END record should be
    // '2' [table-offsets] padding validation-scheme [validation-signature]
    // Please refer OASIS Draft section 14.

    // The first byte should 2.
    segment.push_back(0x02);

    // Write padding. The padding should be 256 bytes.
    int const padding_length = 254;
    segment.insert(segment.end(), padding_length, 0x00);

    // Validation-scheme: No validation.
    segment.push_back(0x00);
}

template <typename T>
void OasisDevice::append_record_bytes(std::vector<uint8_t> & bytes, const T & rec)
{
    std::vector<uint8_t> rec_bytes = rec.to_bytes();
    bytes.insert(bytes.end(), rec_bytes.begin(), rec_bytes.end());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: