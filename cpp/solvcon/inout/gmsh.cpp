#include <solvcon/inout/gmsh.hpp>
#include <vector>
#ifdef _WIN32
#include <algorithm>
#endif // _WIN32
namespace solvcon
{
namespace inout
{
Gmsh::Gmsh(const std::string & data)
{
    bool meta_enter = false;
    bool node_enter = false;
    bool element_enter = false;
    // clang-format off
    std::unordered_map<std::string, std::function<void()>> keyword_handler = {
        {"$MeshFormat", [this, &meta_enter]() { load_meta(); meta_enter = true; }},
        {"$Nodes", [this, &node_enter]() { load_nodes(); node_enter = true; }},
        {"$Elements", [this, &element_enter]() { load_elements(); element_enter = true; }},
        {"$PhysicalNames", [this]() { load_physical(); }}};
    // clang-format on

    // String stream on windows need to remove \r for keyword comparison.
    // DOS file newline character is CRLF, std::getline default using LF as delimeter
    // therefore string seperated by std::getline will contain \r, it will cause
    // keyword compare failed.
    // TODO: Keyword comparison can use regular expression.
#ifdef _WIN32
    std::string data_copy = data;
    data_copy.erase(std::remove(data_copy.begin(), data_copy.end(), '\r'), data_copy.end());
    stream << data_copy;
#else // _WIN32
    stream << data;
#endif // _WIN32

    std::string line;
    while (std::getline(stream, line))
    {
        // Using a finite state machine to check the input msh file format is valid or not.
        // $ is a keyword to trigger state transition.
        if (line.find('$') != std::string::npos)
        {
            auto it = keyword_handler.find(line);
            if (it != keyword_handler.end())
            {
                if (is_valid_transition(it->first))
                {
                    it->second();
                }
                else
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
    }

    // MeshFormat, Nodes and Elements section is mandatory in gmsh msh file,
    // therefore need to check these sections are exist or not otherwise
    // solvcon may crash during mesh processing.
    if (!(meta_enter && node_enter && element_enter))
    {
        throw std::invalid_argument("Incorrect msh file format.");
    }
}

std::shared_ptr<StaticMesh> Gmsh::to_block()
{
    // Only the top-dimension elements are cells; a file's lower-dimension
    // elements (points, boundary lines) merely tag physical groups. Taking
    // them as cells builds degenerate cells that later read out of bounds.
    uint_type const topdim = m_eldim.max();
    std::vector<uint_type> cellmap;
    cellmap.reserve(m_elems.size());
    for (size_t iel = 0; iel < m_eldim.size(); ++iel)
    {
        if (m_eldim[iel] == topdim)
        {
            cellmap.push_back(static_cast<uint_type>(iel));
        }
    }

    std::shared_ptr<StaticMesh> block = StaticMesh::construct(
        topdim,
        static_cast<StaticMesh::uint_type>(m_nds.shape(0)),
        0,
        static_cast<StaticMesh::uint_type>(cellmap.size()));
    build_interior(block, cellmap);
    return block;
}

void Gmsh::build_interior(
    const std::shared_ptr<StaticMesh> & blk, std::vector<uint_type> const & cellmap)
{
    blk->ndcrd().swap(m_nds);
    for (size_t icl = 0; icl < cellmap.size(); ++icl)
    {
        uint_type const iel = cellmap[icl];
        blk->cltpn()(icl) = m_cltpn[iel];
        small_vector<uint_type> const & nds = m_elems.at(iel);
        blk->clnds()(icl, 0) = nds[0];
        for (size_t j = 1; j <= nds[0]; ++j)
        {
            blk->clnds()(icl, j) = nds[j];
        }
    }
    blk->build_interior(true);
    blk->build_boundary();
    blk->build_ghost();
}
} /* end namespace inout */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
