#include <solvcon/buffer/buffer.hpp>

#include <gtest/gtest.h>

#include <random>
#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(ConcreteBuffer, iterator)
{
    using namespace solvcon;

    auto buffer = ConcreteBuffer::construct(10);
    int8_t i = 0;
    for (auto & it : *buffer)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : *buffer)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(SimpleArray, construction)
{
    namespace mm = solvcon;
    mm::SimpleArray<double> arr_double(10);
    EXPECT_EQ(arr_double.nbody(), 10);
    mm::SimpleArray<int> arr_int(17);
    EXPECT_EQ(arr_int.nbody(), 17);
}

TEST(SimpleArray, minmaxsum)
{
    using namespace solvcon;

    SimpleArray<double> arr_double(small_vector<ssize_t>{10}, 0);
    EXPECT_EQ(arr_double.sum(), 0);
    EXPECT_EQ(arr_double.min(), 0);
    EXPECT_EQ(arr_double.max(), 0);
    arr_double.fill(3.14);
    EXPECT_EQ(arr_double.sum(), 3.14 * 10);
    EXPECT_EQ(arr_double.min(), 3.14);
    EXPECT_EQ(arr_double.max(), 3.14);
    arr_double(2) = -2.9;
    arr_double(4) = 12.7;
    EXPECT_EQ(arr_double.min(), -2.9);
    EXPECT_EQ(arr_double.max(), 12.7);

    SimpleArray<int> arr_int(small_vector<ssize_t>{3, 4}, -2);
    EXPECT_EQ(arr_int.sum(), -2 * 3 * 4);
    EXPECT_EQ(arr_int.min(), -2);
    EXPECT_EQ(arr_int.max(), -2);
    arr_int.fill(7);
    EXPECT_EQ(arr_int.sum(), 7 * 3 * 4);
    EXPECT_EQ(arr_int.min(), 7);
    EXPECT_EQ(arr_int.max(), 7);
    arr_int(1, 2) = -8;
    arr_int(2, 0) = 9;
    EXPECT_EQ(arr_int.min(), -8);
    EXPECT_EQ(arr_int.max(), 9);
}

TEST(SimpleArray, abs)
{
    using namespace solvcon;

    SimpleArray<double> arr(small_vector<ssize_t>{10}, -1.0);
    EXPECT_EQ(arr.sum(), -10.0);

    SimpleArray<double> brr = arr.abs();
    EXPECT_EQ(brr.sum(), 10.0);
}

TEST(SimpleArray, iterator)
{
    using namespace solvcon;

    SimpleArray<double> arr(10);
    int8_t i = 0;
    for (auto & it : arr)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : arr)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(SimpleArray_DataType, from_type)
{
    solvcon::DataType dt_double = solvcon::DataType::from<double>();
    EXPECT_EQ(dt_double.type(), solvcon::DataType::Float64);

    solvcon::DataType dt_int = solvcon::DataType::from<int>();
    EXPECT_EQ(dt_int.type(), solvcon::DataType::Int32);
}

TEST(SimpleArray_DataType, from_string)
{
    solvcon::DataType dt_double = solvcon::DataType("float64");
    EXPECT_EQ(dt_double.type(), solvcon::DataType::Float64);

    solvcon::DataType dt_bool = solvcon::DataType("bool");
    EXPECT_EQ(dt_bool.type(), solvcon::DataType::Bool);

    EXPECT_THROW(solvcon::DataType("float16"), std::invalid_argument); // float16 does not exist
    EXPECT_THROW(solvcon::DataType("bool8"), std::invalid_argument); // bool8 does not exist
}

TEST(BufferExpander, iterator)
{
    using namespace solvcon;

    auto buffer = BufferExpander::construct(10);
    int8_t i = 0;
    for (auto & it : *buffer)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : *buffer)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(BufferExpander, pop_size)
{
    using namespace solvcon;

    auto buffer = BufferExpander::construct(4);
    EXPECT_FALSE(buffer->empty());
    EXPECT_EQ(buffer->size(), 4);
    buffer->pop_size(3);
    EXPECT_EQ(buffer->size(), 1);
    // Capacity is left untouched when the size is pulled down.
    EXPECT_EQ(buffer->capacity(), 4);
    buffer->pop_size(1);
    EXPECT_TRUE(buffer->empty());
    EXPECT_THROW(buffer->pop_size(1), std::out_of_range);
}

TEST(SimpleCollector, pop_back)
{
    using namespace solvcon;

    SimpleCollector<int32_t> coll;
    EXPECT_TRUE(coll.empty());
    coll.push_back(10);
    coll.push_back(20);
    coll.push_back(30);
    EXPECT_EQ(coll.size(), 3);
    EXPECT_EQ(coll.front(), 10);
    EXPECT_EQ(coll.back(), 30);

    coll.pop_back();
    EXPECT_EQ(coll.size(), 2);
    EXPECT_EQ(coll.back(), 20);

    // pop_back keeps the surviving elements intact and reusable.
    coll.push_back(40);
    EXPECT_EQ(coll.front(), 10);
    EXPECT_EQ(coll.back(), 40);
    EXPECT_EQ(coll[0], 10);

    coll.pop_back();
    coll.pop_back();
    coll.pop_back();
    EXPECT_TRUE(coll.empty());
    EXPECT_THROW(coll.pop_back(), std::out_of_range);
}

TEST(small_vector, select_kth)
{
    const size_t n = 1024;
    std::vector<int> scrambled(n);
    // gcd(31, 1024) = 1,
    // so we can get all numbers from 0 to 1023.
    for (size_t i = 0; i < n; ++i)
    {
        scrambled[i] = static_cast<int>((i * 31) % n);
    }

    for (size_t k = 0; k < n; ++k)
    {
        solvcon::small_vector<int> sv(scrambled);
        int result = sv.select_kth(k);
        EXPECT_EQ(result, static_cast<int>(k));
    }
}

TEST(small_vector, select_kth_random)
{
    size_t n = 1024;
    std::vector<int> vec(n);
    std::iota(vec.begin(), vec.end(), 0);

    solvcon::small_vector<int> sv(vec);
    for (size_t i = 0; i < n; ++i)
    {
        auto rng = std::default_random_engine{};
        std::shuffle(sv.begin(), sv.end(), rng);
        int it = sv.select_kth(i);
        EXPECT_EQ(it, i);
    }
}

TEST(TakeAlongAxisSimd, basic_int32)
{
    using namespace solvcon;

    // Create a simple array with values [10, 20, 30, 40, 50]
    SimpleArray<int32_t> data(small_vector<ssize_t>{5});
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;

    // Create indices [2, 0, 4, 1]
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{4});
    indices[0] = 2;
    indices[1] = 0;
    indices[2] = 4;
    indices[3] = 1;

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 30);
    EXPECT_EQ(result[1], 10);
    EXPECT_EQ(result[2], 50);
    EXPECT_EQ(result[3], 20);
}

TEST(TakeAlongAxisSimd, basic_float64)
{
    using namespace solvcon;

    // Create a simple array with float values
    SimpleArray<double> data(small_vector<ssize_t>{6});
    data[0] = 1.5;
    data[1] = 2.5;
    data[2] = 3.5;
    data[3] = 4.5;
    data[4] = 5.5;
    data[5] = 6.5;

    // Create indices [5, 2, 0, 3]
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{4});
    indices[0] = 5;
    indices[1] = 2;
    indices[2] = 0;
    indices[3] = 3;

    // Call take_along_axis_simd
    SimpleArray<double> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), 4);
    EXPECT_DOUBLE_EQ(result[0], 6.5);
    EXPECT_DOUBLE_EQ(result[1], 3.5);
    EXPECT_DOUBLE_EQ(result[2], 1.5);
    EXPECT_DOUBLE_EQ(result[3], 4.5);
}

TEST(TakeAlongAxisSimd, large_array)
{
    using namespace solvcon;

    // Create a larger array
    const size_t data_size = 1000;
    SimpleArray<int64_t> data(small_vector<ssize_t>{static_cast<ssize_t>(data_size)});
    for (size_t i = 0; i < data_size; ++i)
    {
        data[i] = static_cast<int64_t>(i * 10);
    }

    // Create indices that sample from the array
    const size_t indices_size = 100;
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{static_cast<ssize_t>(indices_size)});
    for (size_t i = 0; i < indices_size; ++i)
    {
        indices[i] = i * 10; // Sample every 10th element
    }

    // Call take_along_axis_simd
    SimpleArray<int64_t> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), indices_size);
    for (size_t i = 0; i < indices_size; ++i)
    {
        EXPECT_EQ(result[i], static_cast<int64_t>(i * 10 * 10));
    }
}

TEST(TakeAlongAxisSimd, out_of_range)
{
    using namespace solvcon;

    // Create a simple array
    SimpleArray<int32_t> data(small_vector<ssize_t>{5});
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;

    // Create indices with out-of-range value
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{3});
    indices[0] = 2;
    indices[1] = 10; // Out of range
    indices[2] = 1;

    // Should throw an exception
    EXPECT_THROW(data.take_along_axis_simd(indices), std::out_of_range);
}

TEST(TakeAlongAxisSimd, empty_indices)
{
    using namespace solvcon;

    // Create a simple array
    SimpleArray<int32_t> data(small_vector<ssize_t>{5});
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;

    // Create empty indices
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{0});

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Verify the result is empty
    EXPECT_EQ(result.size(), 0);
}

TEST(TakeAlongAxisSimd, sequential_indices)
{
    using namespace solvcon;

    // Create array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    const size_t size = 10;
    SimpleArray<int32_t> data(small_vector<ssize_t>{static_cast<ssize_t>(size)});
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int32_t>(i);
    }

    // Create sequential indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{static_cast<ssize_t>(size)});
    for (size_t i = 0; i < size; ++i)
    {
        indices[i] = i;
    }

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Result should be identical to input
    EXPECT_EQ(result.size(), size);
    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(result[i], data[i]);
    }
}

TEST(TakeAlongAxisSimd, single_index_element)
{
    using namespace solvcon;

    // Create a data array with multiple elements
    SimpleArray<int32_t> data(small_vector<ssize_t>{10});
    for (size_t i = 0; i < 10; ++i)
    {
        data[i] = static_cast<int32_t>(i * 10);
    }

    // Create indices array with ONLY 1 ELEMENT (smaller than N_lane=2 on ARM NEON)
    // This should trigger the bug without the fix!
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{1});
    indices[0] = 3;

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 30);
}
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
