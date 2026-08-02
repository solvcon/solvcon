# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import functools
import itertools
import statistics
import numpy as np
import solvcon


def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = solvcon.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result

    return wrapper


def make_container(data):
    if np.issubdtype(data.dtype, np.float32):
        return solvcon.SimpleArrayFloat32(array=data)
    elif np.issubdtype(data.dtype, np.float64):
        return solvcon.SimpleArrayFloat64(array=data)
    raise ValueError(f"Unsupported dtype: {data.dtype}")


@profile_function
def profile_matmul_np(lhs, rhs):
    return np.matmul(lhs, rhs)


@profile_function
def profile_matmul_naive_sa(lhs, rhs):
    return lhs.matmul(rhs)


@profile_function
def profile_matmul_blas_sa(lhs, rhs):
    return lhs.matmul_blas(rhs)


@profile_function
def profile_matmul_planned_sa(lhs, rhs):
    return lhs.matmul_planned(rhs)


def profile_matmul_fast_sa(lhs, rhs, tile_x, tile_y, tile_z):
    name = f"profile_matmul_fast_sa_{tile_x}_{tile_y}_{tile_z}"
    _ = solvcon.CallProfilerProbe(name)
    return lhs.matmul_fast(rhs, tile_x=tile_x, tile_y=tile_y, tile_z=tile_z)


def make_data(dtype, shape):
    return np.random.rand(*shape).astype(dtype)


def iter_stride_cases(lhs, rhs):
    rhs_inner_axis = -1 if rhs.ndim == 1 else -2
    lhs_cases = (
        ("c_contiguous", lhs),
        ("negative_inner_stride", np.flip(lhs, axis=-1)),
    )
    rhs_cases = (
        ("c_contiguous", rhs),
        ("negative_inner_stride", np.flip(rhs, axis=rhs_inner_axis)),
    )
    for lhs_case, rhs_case in itertools.product(lhs_cases, rhs_cases):
        lhs_name, case_lhs = lhs_case
        rhs_name, case_rhs = rhs_case
        name = f"lhs_{lhs_name}_rhs_{rhs_name}"
        yield name, case_lhs, case_rhs


def element_strides(data):
    return tuple(stride // data.itemsize for stride in data.strides)


def print_profile_row(*columns):
    print(str.format(
        "| {:20s} | {:15s} | {:15s} |", *(columns[0:3])))


def profile_one_call(func, *args):
    solvcon.call_profiler.reset()
    func(*args)
    result = solvcon.call_profiler.result()["children"]
    if len(result) != 1 or result[0]["count"] != 1:
        raise RuntimeError("Expected exactly one profiled call")
    return result[0]["total_time"]


def profile_unbatched_gemm(dtype, sides, samples=10):
    tile_configs = (
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
    )
    for side in sides:
        lhs = make_data(dtype, (side, side))
        rhs = make_data(dtype, (side, side))
        for case_name, case_lhs, case_rhs in iter_stride_cases(lhs, rhs):
            lhs_sa = make_container(case_lhs)
            rhs_sa = make_container(case_rhs)
            solvcon.call_profiler.reset()
            for _ in range(samples):
                profile_matmul_np(case_lhs, case_rhs)
                profile_matmul_naive_sa(lhs_sa, rhs_sa)
                profile_matmul_blas_sa(lhs_sa, rhs_sa)
                profile_matmul_planned_sa(lhs_sa, rhs_sa)
                for tile_x, tile_y, tile_z in tile_configs:
                    profile_matmul_fast_sa(
                        lhs_sa, rhs_sa, tile_x, tile_y, tile_z)

            result = solvcon.call_profiler.result()["children"]
            timings = {}
            for item in result:
                name = item["name"].replace("profile_matmul_", "")
                timings[name] = item["total_time"] / item["count"]

            print(f"## 2D x 2D strides: `{case_name}` "
                  f"dtype: `{np.dtype(dtype)}`\n")
            print(f"- lhs shape: `{case_lhs.shape}`, element strides: "
                  f"`{element_strides(case_lhs)}`")
            print(f"- rhs shape: `{case_rhs.shape}`, element strides: "
                  f"`{element_strides(case_rhs)}`\n")

            print_profile_row("func", "per call (ms)", "cmp to np")
            print_profile_row("-" * 20, "-" * 15, "-" * 15)
            numpy_time = timings["np"]
            methods = ["np", "naive_sa", "blas_sa", "planned_sa"]
            methods += [
                f"fast_sa_{tile_x}_{tile_y}_{tile_z}"
                for tile_x, tile_y, tile_z in tile_configs
            ]
            for method in methods:
                value = timings[method]
                print_profile_row(
                    f"{method:8s}", f"{value:.3E}",
                    f"{value / numpy_time:.3f}")
            print()


def profile_planned_case(
        title, dtype, case_name, lhs, rhs, warmups, samples, rounds):
    lhs_sa = make_container(lhs)
    rhs_sa = make_container(rhs)
    timings = {"np": [], "planned_sa": []}
    for _ in range(rounds):
        for _ in range(warmups):
            np.matmul(lhs, rhs)
            lhs_sa.matmul_planned(rhs_sa)

        for _ in range(samples):
            timings["np"].append(profile_one_call(
                profile_matmul_np, lhs, rhs))
            timings["planned_sa"].append(profile_one_call(
                profile_matmul_planned_sa, lhs_sa, rhs_sa))

    methods = ("np", "planned_sa")
    timings = {
        method: statistics.median(timings[method])
        for method in methods
    }

    print(f"## {title}: `{case_name}`, dtype: `{np.dtype(dtype)}`")
    print(f"- lhs shape: `{lhs.shape}`, element strides: "
          f"`{element_strides(lhs)}`")
    print(f"- rhs shape: `{rhs.shape}`, element strides: "
          f"`{element_strides(rhs)}`\n")

    print_profile_row("func", "median (ms)", "cmp to np")
    print_profile_row("-" * 20, "-" * 15, "-" * 15)
    numpy_time = timings["np"]
    for method in methods:
        value = timings[method]
        print_profile_row(
            f"{method:8s}", f"{value:.3E}",
            f"{value / numpy_time:.3f}")
    print()


def iter_planned_cases(dtype):
    small_sides = (4, 9, 16, 27, 64, 81)
    vector_sides = (*small_sides, 256, 1024)
    dot_sizes = (*vector_sides, 16_384, 1_048_576)

    for size in dot_sizes:
        yield (
            "DOT",
            make_data(dtype, (size,)),
            make_data(dtype, (size,)),
            15,
        )

    for side in vector_sides:
        vector = make_data(dtype, (side,))
        matrix = make_data(dtype, (side, side))
        batch_matrix = make_data(dtype, (2, 5, side, side))
        yield ("GEVM", vector, matrix, 15)
        yield ("GEMV", matrix, vector, 15)
        yield ("Batched GEVM", vector, batch_matrix, 15)
        yield ("Batched GEMV", batch_matrix, vector, 15)

    for side in (*small_sides, 256):
        batch_lhs = make_data(dtype, (10, side, side))
        batch_rhs = make_data(dtype, (10, side, side))
        cross_lhs = make_data(dtype, (2, 1, side, side))
        cross_rhs = make_data(dtype, (1, 5, side, side))
        yield ("Equal-batch GEMM", batch_lhs, batch_rhs, 15)
        yield ("Cross-broadcast GEMM", cross_lhs, cross_rhs, 15)


def profile_planned_suite(dtype, warmups, rounds):
    for title, lhs, rhs, samples in iter_planned_cases(dtype):
        for case_name, case_lhs, case_rhs in iter_stride_cases(lhs, rhs):
            profile_planned_case(
                title, dtype, case_name, case_lhs, case_rhs,
                warmups, samples, rounds)


def main():
    gemm_sides = (4, 9, 16, 27, 64, 81, 243, 256, 729, 1024)
    for dtype in (np.float32, np.float64):
        profile_unbatched_gemm(dtype, gemm_sides)

    for dtype in (np.float32, np.float64):
        profile_planned_suite(dtype, warmups=2, rounds=5)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
