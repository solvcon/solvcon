#include <solvcon/device/metal/metal.hpp>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace solvcon
{

namespace device
{

TEST(MetalManager, repeated_lifecycle_calls_preserve_state)
{
    MetalManager & manager = MetalManager::instance();
    bool const available = manager.started();

    manager.startup();
    EXPECT_EQ(available, manager.started());

    manager.shutdown();
    EXPECT_FALSE(manager.started());
    manager.shutdown();
    EXPECT_FALSE(manager.started());

    manager.startup();
    EXPECT_EQ(available, manager.started());
}

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
