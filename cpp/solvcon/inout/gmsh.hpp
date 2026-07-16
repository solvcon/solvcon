#pragma once

/**
 * @file
 * Reader for Gmsh MSH text mesh files and the element-type table it uses.
 *
 * @ingroup group_inout
 */

#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <functional>

#include <solvcon/base.hpp>
#include <solvcon/mesh/mesh.hpp>
#include <solvcon/buffer/buffer.hpp>
#include <solvcon/inout/inout_util.hpp>

namespace solvcon
{
namespace inout
{

/**
 * Definition of one Gmsh element type mapped to the solvcon mesh
 * representation.
 *
 * It carries the spatial dimension, the node count, the solvcon cell type
 * number, and the node ordering (mmcl) that rearranges Gmsh nodes into the
 * solvcon cell order. by_id() returns the definition for a given Gmsh element
 * type id.
 *
 * @ingroup group_inout
 */
struct GmshElementDef
{
    static GmshElementDef by_id(uint16_t id);

    GmshElementDef(uint8_t ndim, uint16_t nnds, uint8_t mmtpn, small_vector<uint8_t> const & mmcl)
        : m_ndim(ndim)
        , m_nnds(nnds)
        , m_mmtpn(mmtpn)
        , m_mmcl(mmcl)
    {
    }

    ~GmshElementDef() = default;

    GmshElementDef() = default;
    GmshElementDef(const GmshElementDef & other) = delete;
    GmshElementDef(GmshElementDef && other) = delete;
    GmshElementDef & operator=(const GmshElementDef & other) = delete;
    GmshElementDef & operator=(GmshElementDef && other) = delete;

    uint8_t ndim() const { return m_ndim; }
    uint16_t nnds() const { return m_nnds; }
    uint8_t mmtpn() const { return m_mmtpn; }
    small_vector<uint8_t> mmcl() const { return m_mmcl; }

private:
    uint8_t m_ndim = 0; /* Number of dimension */
    uint16_t m_nnds = 0; /* Number of nodes     */
    uint8_t m_mmtpn = 0; /* solvcon cell type   */
    small_vector<uint8_t> m_mmcl; /* solvcon cell order  */
}; /* end struct GmshElementDef */

/**
 * Reader that parses a Gmsh MSH text mesh into a solvcon StaticMesh.
 *
 * Only the ASCII MSH format version 2.2 is supported. Construct it with the
 * file contents as a string, then call to_block() to build the mesh. A finite
 * state machine walks the MeshFormat, PhysicalNames, Nodes, and Elements
 * sections in order. Node and element indices are converted from the Gmsh
 * 1-based numbering to the solvcon 0-based numbering.
 *
 * @ingroup group_inout
 */
class Gmsh
    : public NumberBase<int32_t, double>
{
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:
    explicit Gmsh(const std::string & data);

    ~Gmsh() = default;

    Gmsh() = delete;
    Gmsh(Gmsh const & other) = delete;
    Gmsh(Gmsh && other) = delete;
    Gmsh & operator=(Gmsh const & other) = delete;
    Gmsh & operator=(Gmsh && other) = delete;

    std::shared_ptr<StaticMesh> to_block();

private:
    enum class FormatState : uint8_t
    {
        BEGIN,
        META_END,
        PYHSICAL_NAME_END,
        NODE_END,
        ELEMENT_END
    };

    // Check the finite state machine state transition is valid or not to check msh file format is correct
    bool is_valid_transition(std::string const & s)
    {
        if (last_fmt_state == FormatState::BEGIN)
        {
            return s == "$MeshFormat";
        }
        else if (last_fmt_state == FormatState::META_END || last_fmt_state == FormatState::PYHSICAL_NAME_END)
        {
            return s == "$PhysicalNames" || s == "$Nodes";
        }
        else if (last_fmt_state == FormatState::NODE_END)
        {
            return s == "$Elements";
        }

        return false;
    }

    void load_meta()
    {
        std::string line;

        while (std::getline(stream, line) && line.find('$') == std::string::npos)
        {
            auto tokens = tokenize(line, ' ');
            msh_ver = std::stod(tokens[0]);

            // The parse only support ver 2.2 msh file.
            if (msh_ver != 2.2)
            {
                throw std::invalid_argument(std::format("solvcon does not support msh file ver {}.", msh_ver));
            }

            msh_file_type = std::stoi(tokens[1]);
            msh_data_size = std::stoi(tokens[2]);
        }

        if (!line.compare("$EndMeshFormat")) // FIXME: NOLINT(readability-string-compare)
        {
            last_fmt_state = FormatState::META_END;
        }
    }

    // TODO: PhysicalNames section parsing logic not complete yet, but without PhysicalNames section
    //       solvcon mesh viewer still working, therefore we can finish this later.
    void load_physical()
    {
        std::string line;

        while (std::getline(stream, line) && line.find('$') == std::string::npos) {}

        if (!line.compare("$EndPhysicalNames")) // FIXME: NOLINT(readability-string-compare)
        {
            last_fmt_state = FormatState::PYHSICAL_NAME_END;
        }
    }

    void load_nodes()
    {
        std::string line;
        std::getline(stream, line);
        auto nnode = std::stoul(line);

        m_nds.remake(small_vector<ssize_t>{static_cast<ssize_t>(nnode), 3}, 0);

        while (std::getline(stream, line) && line.find('$') == std::string::npos)
        {
            auto tokens = tokenize(line, ' ');
            // gmsh node index is 1-based index
            m_nds(std::stoul(tokens[0]) - 1, 0) = std::stod(tokens[1]);
            m_nds(std::stoul(tokens[0]) - 1, 1) = std::stod(tokens[2]);
            m_nds(std::stoul(tokens[0]) - 1, 2) = std::stod(tokens[3]);
        }

        if (!line.compare("$EndNodes")) // FIXME: NOLINT(readability-string-compare)
        {
            last_fmt_state = FormatState::NODE_END;
        }
    }

    void load_elements()
    {
        std::string line;
        std::getline(stream, line);
        auto nelement = std::stoul(line);
        std::vector<uint_type> usnds;

        m_cltpn.remake(small_vector<ssize_t>{static_cast<ssize_t>(nelement)}, 0);
        m_elgrp.remake(small_vector<ssize_t>{static_cast<ssize_t>(nelement)}, 0);
        m_elgeo.remake(small_vector<ssize_t>{static_cast<ssize_t>(nelement)}, 0);
        m_eldim.remake(small_vector<ssize_t>{static_cast<ssize_t>(nelement)}, 0);

        uint_type idx = 0;

        while (std::getline(stream, line) && line.find('$') == std::string::npos && idx < nelement)
        {
            auto tokens = tokenize(line, ' ');

            // parse element type
            auto tpn = std::stoul(tokens[1]);
            auto eldef = GmshElementDef::by_id(tpn);
            // parse element tag
            std::vector<uint_type> tag;
            for (size_t i = 0; i < std::stoul(tokens[2]); ++i)
            {
                // FIXME: NOLINTNEXTLINE(performance-inefficient-vector-operation)
                tag.push_back(std::stoul(tokens[3 + i]));
            }

            // parse node number list
            std::vector<uint_type> nds;
            for (size_t i = 3 + std::stoul(tokens[2]); i < tokens.size(); ++i)
            {
                nds.push_back(std::stoul(tokens[i]));
            }

            m_cltpn[idx] = eldef.mmtpn();
            m_elgrp[idx] = tag[0];
            m_elgeo[idx] = tag[1];
            m_eldim[idx] = eldef.ndim();

            small_vector<uint_type> nds_temp(nds.size() + 1, 0);
            auto mmcl = eldef.mmcl();

            for (size_t i = 0; i < mmcl.size(); ++i)
            {
                nds_temp[mmcl[i] + 1] = nds[i] - 1;
            }
            usnds.insert(usnds.end(), nds_temp.begin() + 1, nds_temp.end());
            nds_temp[0] = mmcl.size();
            m_elems.insert(std::pair{idx, nds_temp});
            idx++;
        }

        if (!line.compare("$EndElements")) // FIXME: NOLINT(readability-string-compare)
        {
            last_fmt_state = FormatState::ELEMENT_END;
        }

        // sorting used node and remove duplicate node id
        std::sort(usnds.begin(), usnds.end());
        auto remove = std::unique(usnds.begin(), usnds.end());
        usnds.erase(remove, usnds.end());

        // put used node id to m_ndmap
        m_ndmap.remake(small_vector<ssize_t>{static_cast<ssize_t>(usnds.size())}, -1);
        for (size_t i = 0; i < usnds.size(); ++i)
        {
            m_ndmap(usnds[i]) = i;
        }
    }

    void build_interior(
        const std::shared_ptr<StaticMesh> & blk, std::vector<uint_type> const & cellmap);

    std::stringstream stream;
    FormatState last_fmt_state = FormatState::BEGIN;

    real_type msh_ver = 0.0;

    uint_type msh_file_type = 0;
    uint_type msh_data_size = 0;
    uint_type nnodes = 0;

    SimpleArray<int_type> m_cltpn;

    SimpleArray<real_type> m_nds;

    SimpleArray<uint_type> m_elgrp;
    SimpleArray<uint_type> m_elgeo;
    SimpleArray<uint_type> m_eldim;
    SimpleArray<uint_type> m_usnds;
    SimpleArray<uint_type> m_ndmap;

    std::unordered_map<uint_type, small_vector<uint_type>> m_elems;
}; /* end class Gmsh */

inline GmshElementDef GmshElementDef::by_id(uint16_t id)
{
#define VEC(...) __VA_ARGS__
#define MM_DECL_SWITCH_ELEMENT_TYPE(ID, NDIM, NNDS, MMTPN, MMCL) \
    case ID: return GmshElementDef(NDIM, NNDS, MMTPN, small_vector<uint8_t>{MMCL}); break;
    switch (id)
    {
        // clang-format off
        // NOLINTBEGIN(bugprone-branch-clone,-warnings-as-error)
        //                          id, dim, nnodes, cell type, cell order
        MM_DECL_SWITCH_ELEMENT_TYPE( 1,   1,      2,         2, VEC(0, 1))                       // 2-node line
        MM_DECL_SWITCH_ELEMENT_TYPE( 2,   2,      3,         4, VEC(0, 1, 2))                    // 3-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE( 3,   2,      4,         3, VEC(0, 1, 2, 3))                 // 4-node quadrangle
        MM_DECL_SWITCH_ELEMENT_TYPE( 4,   3,      4,         6, VEC(0, 1, 2, 3))                 // 4-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE( 5,   3,      8,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 8-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE( 6,   3,      6,         7, VEC(0, 2, 1, 3, 5, 4))           // 6-node prism
        MM_DECL_SWITCH_ELEMENT_TYPE( 7,   3,      5,         8, VEC(0, 1, 2, 3, 4))              // 5-node pyramid
        MM_DECL_SWITCH_ELEMENT_TYPE( 8,   1,      3,         2, VEC(0, 1))                       // 3-node line
        MM_DECL_SWITCH_ELEMENT_TYPE( 9,   2,      6,         4, VEC(0, 1, 2))                    // 6-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(10,   2,      9,         3, VEC(0, 1, 2, 3))                 // 9-node quadrangle
        MM_DECL_SWITCH_ELEMENT_TYPE(11,   3,     10,         6, VEC(0, 1, 2, 3))                 // 10-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(12,   3,     27,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 27-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(13,   3,     18,         7, VEC(0, 2, 1, 3, 5, 4))           // 18-node prism
        MM_DECL_SWITCH_ELEMENT_TYPE(14,   3,     14,         8, VEC(0, 1, 2, 3, 4))              // 14-node pyramid
        MM_DECL_SWITCH_ELEMENT_TYPE(15,   0,      1,         1, VEC(0))                          // 1-node point
        MM_DECL_SWITCH_ELEMENT_TYPE(16,   2,      8,         3, VEC(0, 1, 2, 3))                 // 8-node quadrangle
        MM_DECL_SWITCH_ELEMENT_TYPE(17,   3,     20,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 20-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(18,   3,     15,         7, VEC(0, 2, 1, 3, 5, 4))           // 15-node prism
        MM_DECL_SWITCH_ELEMENT_TYPE(19,   3,     13,         8, VEC(0, 1, 2, 3, 4))              // 13-node pyramid
        MM_DECL_SWITCH_ELEMENT_TYPE(20,   2,      9,         4, VEC(0, 1, 2))                    // 9-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(21,   2,     10,         4, VEC(0, 1, 2))                    // 10-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(22,   2,     12,         4, VEC(0, 1, 2))                    // 12-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(23,   2,     15,         4, VEC(0, 1, 2))                    // 15-node triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(24,   2,     15,         4, VEC(0, 1, 2))                    // 15-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(25,   2,     21,         4, VEC(0, 1, 2))                    // 21-node incomplete triangle
        MM_DECL_SWITCH_ELEMENT_TYPE(26,   1,      4,         2, VEC(0, 1))                       // 4-node edge
        MM_DECL_SWITCH_ELEMENT_TYPE(27,   1,      5,         2, VEC(0, 1))                       // 5-node edge
        MM_DECL_SWITCH_ELEMENT_TYPE(28,   1,      6,         2, VEC(0, 1))                       // 6-node edge
        MM_DECL_SWITCH_ELEMENT_TYPE(29,   3,     20,         6, VEC(0, 1, 2, 3))                 // 20-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(30,   3,     35,         6, VEC(0, 1, 2 ,3))                 // 35-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(31,   3,     56,         6, VEC(0, 1, 2, 3))                 // 56-node tetrahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(92,   3,     64,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 64-node hexahedron
        MM_DECL_SWITCH_ELEMENT_TYPE(93,   3,    125,         5, VEC(0, 1, 2, 3, 4, 5, 6, 7))     // 125-node hexahedron
        default: return GmshElementDef{}; break;
        // NOLINTEND(bugprone-branch-clone,-warnings-as-error)
        // clang-format on
    }
#undef MM_DECL_SWITCH_ELEMENT_TYPE
#undef VEC
}
} /* end namespace inout */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
