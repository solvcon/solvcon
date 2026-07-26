# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import functools
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


def make_layout_cases(lhs, rhs):
    return (
        ("c_contiguous", lhs, rhs),
        ("non_contiguous",
         np.flip(lhs, axis=1),
         np.flip(rhs, axis=0)),
    )


def element_strides(data):
    return tuple(stride // data.itemsize for stride in data.strides)


def print_profile_row(*columns):
    print(str.format(
        "| {:20s} | {:15s} | {:15s} |", *(columns[0:3])))


def profile_matmul_operation(dtype, shapes, it=10):
    tile_configs = (
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
    )
    for m in shapes:
        lhs = make_data(dtype, (m, m))
        rhs = make_data(dtype, (m, m))
        for layout, case_lhs, case_rhs in make_layout_cases(lhs, rhs):
            lhs_sa = make_container(case_lhs)
            rhs_sa = make_container(case_rhs)
            solvcon.call_profiler.reset()
            for _ in range(it):
                profile_matmul_np(case_lhs, case_rhs)
                profile_matmul_naive_sa(lhs_sa, rhs_sa)
                profile_matmul_blas_sa(lhs_sa, rhs_sa)
                profile_matmul_planned_sa(lhs_sa, rhs_sa)
                for tile_x, tile_y, tile_z in tile_configs:
                    profile_matmul_fast_sa(
                        lhs_sa, rhs_sa, tile_x, tile_y, tile_z)

            res = solvcon.call_profiler.result()["children"]
            out = {}
            for r in res:
                name = r["name"].replace("profile_matmul_", "")
                out[name] = r["total_time"] / r["count"]

            print(f"## 2D x 2D layout: `{layout}` "
                  f"dtype: `{np.dtype(dtype)}`\n")
            print(f"- lhs shape: `{case_lhs.shape}`, element strides: "
                  f"`{element_strides(case_lhs)}`")
            print(f"- rhs shape: `{case_rhs.shape}`, element strides: "
                  f"`{element_strides(case_rhs)}`\n")

            print_profile_row("func", "per call (ms)", "cmp to np")
            print_profile_row("-" * 20, "-" * 15, "-" * 15)
            npbase = out["np"]
            keys = ["np", "naive_sa", "blas_sa", "planned_sa"]
            keys += [
                f"fast_sa_{tile_x}_{tile_y}_{tile_z}"
                for tile_x, tile_y, tile_z in tile_configs
            ]
            for key in keys:
                value = out[key]
                print_profile_row(
                    f"{key:8s}", f"{value:.3E}",
                    f"{value / npbase:.3f}")
            print()


def main():
    shapes = [4, 16, 64, 256, 1024]

    for dtype in (np.float32, np.float64):
        profile_matmul_operation(dtype, shapes)

    shapes = [9, 27, 81, 243, 729]

    for dtype in (np.float32, np.float64):
        profile_matmul_operation(dtype, shapes)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
