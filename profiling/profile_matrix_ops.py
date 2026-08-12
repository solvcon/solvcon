# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
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


@profile_function
def profile_matmul_packing_sa(lhs, rhs, streamed):
    return lhs._matmul_planned_with_packing(rhs, streamed=streamed)


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


def make_strided_view(data, axis, step):
    storage_shape = list(data.shape)
    storage_shape[axis] *= abs(step)
    storage = np.empty(storage_shape, dtype=data.dtype.name)
    selection = [slice(None)] * data.ndim
    selection[axis] = slice(None, None, step)
    view = storage[tuple(selection)]
    view[...] = data
    return view


def make_incompatible_matrix_view(data, operand, layout):
    if layout == "negative":
        axis = -1 if operand == "lhs" else -2
        return make_strided_view(data, axis, -1)
    if layout == "step_two":
        return make_strided_view(data, -1, 2)
    raise ValueError(f"Unsupported matrix layout: {layout}")


def make_zero_batch_stride(data, batch_size):
    # The NumPy caster requests writable storage, but matmul only reads it.
    return np.lib.stride_tricks.as_strided(
        data,
        shape=(batch_size, *data.shape),
        strides=(0, *data.strides),
        writeable=True,
    )


def make_packing_operands(
        dtype, topology, packing_role, layout, batch_size, side):
    lhs_requires_packing = packing_role in ("lhs", "both")
    rhs_requires_packing = packing_role in ("rhs", "both")
    batch_shape = (batch_size, side, side)

    if topology == "unique":
        lhs = make_data(dtype, batch_shape)
        rhs = make_data(dtype, batch_shape)
    elif topology == "zero_stride_reuse":
        lhs_shape = (side, side) if lhs_requires_packing else batch_shape
        rhs_shape = (side, side) if rhs_requires_packing else batch_shape
        lhs = make_data(dtype, lhs_shape)
        rhs = make_data(dtype, rhs_shape)
    elif topology == "cross_broadcast":
        lhs = make_data(dtype, (2, 1, side, side))
        rhs = make_data(dtype, (1, 5, side, side))
    else:
        raise ValueError(f"Unsupported packing topology: {topology}")

    if lhs_requires_packing:
        lhs = make_incompatible_matrix_view(lhs, "lhs", layout)
    if rhs_requires_packing:
        rhs = make_incompatible_matrix_view(rhs, "rhs", layout)

    if topology == "zero_stride_reuse":
        if lhs_requires_packing:
            lhs = make_zero_batch_stride(lhs, batch_size)
        if rhs_requires_packing:
            rhs = make_zero_batch_stride(rhs, batch_size)
    return lhs, rhs


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


def profile_unbatched_gemm(dtype, sides, samples=1):
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


def profile_packing_case(
        title, dtype, case_name, lhs, rhs, warmups, samples, rounds):
    lhs_sa = make_container(lhs)
    rhs_sa = make_container(rhs)
    routes = (("complete", False), ("streamed", True), ("planned", None))
    timings = {name: [] for name, _ in routes}
    raw_samples = []

    expected = np.matmul(lhs, rhs)
    tolerance = 64 * np.finfo(np.dtype(dtype)).eps
    for _, streamed in routes:
        if streamed is None:
            result = lhs_sa.matmul_planned(rhs_sa)
        else:
            result = lhs_sa._matmul_planned_with_packing(
                rhs_sa, streamed=streamed)
        np.testing.assert_allclose(
            result.ndarray, expected, rtol=tolerance, atol=tolerance)

    for round_index in range(rounds):
        offset = round_index % len(routes)
        round_routes = routes[offset:] + routes[:offset]
        for _ in range(warmups):
            for _, streamed in round_routes:
                if streamed is None:
                    lhs_sa.matmul_planned(rhs_sa)
                else:
                    lhs_sa._matmul_planned_with_packing(
                        rhs_sa, streamed=streamed)

        for sample_index in range(samples):
            offset = sample_index % len(routes)
            sample_routes = round_routes[offset:] + round_routes[:offset]
            sample = {}
            for name, streamed in sample_routes:
                if streamed is None:
                    value = profile_one_call(
                        profile_matmul_planned_sa, lhs_sa, rhs_sa)
                else:
                    value = profile_one_call(
                        profile_matmul_packing_sa,
                        lhs_sa, rhs_sa, streamed)
                timings[name].append(value)
                sample[name] = value
            order = "/".join(name for name, _ in sample_routes)
            raw_samples.append((order, sample))

    medians = {
        name: statistics.median(timings[name])
        for name, _ in routes
    }

    print(f"## Packing schedule: `{title}`, case: `{case_name}`, "
          f"dtype: `{np.dtype(dtype)}`")
    print(f"- lhs shape: `{lhs.shape}`, element strides: "
          f"`{element_strides(lhs)}`")
    print(f"- rhs shape: `{rhs.shape}`, element strides: "
          f"`{element_strides(rhs)}`\n")

    print_profile_row("schedule", "median (ms)", "cmp to complete")
    print_profile_row("-" * 20, "-" * 15, "-" * 15)
    complete_time = medians["complete"]
    for name, _ in routes:
        value = medians[name]
        print_profile_row(name, f"{value:.3E}",
                          f"{value / complete_time:.3f}")
    print()
    for sample_index, (order, sample) in enumerate(raw_samples, start=1):
        streamed_ratio = sample["streamed"] / sample["complete"]
        planned_ratio = sample["planned"] / sample["complete"]
        print(f"- sample {sample_index} raw (ms), order={order}: "
              f"complete={sample['complete']:.6E}, "
              f"streamed={sample['streamed']:.6E}, "
              f"planned={sample['planned']:.6E}, "
              f"streamed/complete={streamed_ratio:.3f}, "
              f"planned/complete={planned_ratio:.3f}")
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
        )

    for side in vector_sides:
        vector = make_data(dtype, (side,))
        matrix = make_data(dtype, (side, side))
        batch_matrix = make_data(dtype, (2, 5, side, side))
        yield ("GEVM", vector, matrix)
        yield ("GEMV", matrix, vector)
        yield ("Batched GEVM", vector, batch_matrix)
        yield ("Batched GEMV", batch_matrix, vector)

    for side in (*small_sides, 256):
        batch_lhs = make_data(dtype, (10, side, side))
        batch_rhs = make_data(dtype, (10, side, side))
        cross_lhs = make_data(dtype, (2, 1, side, side))
        cross_rhs = make_data(dtype, (1, 5, side, side))
        yield ("Equal-batch GEMM", batch_lhs, batch_rhs)
        yield ("Cross-broadcast GEMM", cross_lhs, cross_rhs)


def iter_batched_vector_threshold_cases(dtype):
    cases = ((24, 4), (24, 8), (32, 2), (32, 4), (64, 2))
    for side, batch_size in cases:
        vector = make_data(dtype, (side,))
        matrix = make_data(dtype, (batch_size, side, side))
        yield (f"Batched GEVM B={batch_size}", vector, matrix)
        yield (f"Batched GEMV B={batch_size}", matrix, vector)


def iter_packing_cases(dtype):
    dtype_name = np.dtype(dtype).name
    lower_side = {"float32": 24, "float64": 16}[dtype_name]
    packing_roles = ("lhs", "rhs", "both")

    for topology, packing_role, layout in itertools.product(
            ("unique", "zero_stride_reuse", "cross_broadcast"),
            packing_roles,
            ("negative", "step_two")):
        lhs, rhs = make_packing_operands(
            dtype, topology, packing_role, layout,
            batch_size=10, side=64)
        yield (
            f"{topology} B=10, S=64",
            f"{packing_role}_{layout}", lhs, rhs)

    for side in (lower_side, 256):
        for packing_role in packing_roles:
            lhs, rhs = make_packing_operands(
                dtype, "unique", packing_role, "negative",
                batch_size=10, side=side)
            yield (
                f"unique B=10, S={side}",
                f"{packing_role}_negative", lhs, rhs)

    for batch_size, side in ((2, lower_side), (32, lower_side), (10, 32)):
        lhs, rhs = make_packing_operands(
            dtype, "unique", "both", "negative",
            batch_size=batch_size, side=side)
        yield (
            f"unique B={batch_size}, S={side}",
            "both_negative", lhs, rhs)


def profile_planned_suite(dtype, warmups=1, samples=1, rounds=3):
    cases = itertools.chain(
        iter_planned_cases(dtype),
        iter_batched_vector_threshold_cases(dtype),
    )
    for title, lhs, rhs in cases:
        for case_name, case_lhs, case_rhs in iter_stride_cases(lhs, rhs):
            profile_planned_case(
                title, dtype, case_name, case_lhs, case_rhs,
                warmups, samples, rounds)


def profile_packing_suite(dtype, warmups=1, samples=1, rounds=3):
    for title, case_name, lhs, rhs in iter_packing_cases(dtype):
        profile_packing_case(
            title, dtype, case_name, lhs, rhs,
            warmups, samples, rounds)


def profile_winograd_boundary(dtype, side, rng):
    dtype = np.dtype(dtype)
    shape = (side, side)
    lhs = rng.random(shape, dtype=dtype.name)
    rhs = rng.random(shape, dtype=dtype.name)
    lhs_sa = make_container(lhs)
    rhs_sa = make_container(rhs)
    routes = (
        ("blas_sa", profile_matmul_blas_sa),
        ("planned_sa", profile_matmul_planned_sa),
    )

    for _, func in routes:
        func(lhs_sa, rhs_sa)

    timings = {
        name: profile_one_call(func, lhs_sa, rhs_sa)
        for name, func in routes
    }

    print(f"## Winograd boundary: `{side} x {side} x {side}`, "
          f"dtype: `{dtype.name}`\n")
    print_profile_row("func", "per call (ms)", "cmp to BLAS")
    print_profile_row("-" * 20, "-" * 15, "-" * 15)
    blas_time = timings["blas_sa"]
    for name, value in timings.items():
        ratio = value / blas_time
        print_profile_row(name, f"{value:.3E}", f"{ratio:.3f}")
    print()
    for name, value in timings.items():
        print(f"- {name} raw (ms): {value:.6E}")
    print()


def parse_positive_count(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile matrix multiplication operations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--warmups", type=parse_positive_count, default=1,
        help="untimed planned calls before each round")
    parser.add_argument(
        "--samples", type=parse_positive_count, default=1,
        help="timed calls per method in each round")
    parser.add_argument(
        "--rounds", type=parse_positive_count, default=3,
        help="planned profiling rounds; use 5 or more for stable results")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    gemm_sides = (4, 9, 16, 27, 64, 81, 243, 256, 729, 1024)
    for dtype in (np.float32, np.float64):
        profile_unbatched_gemm(dtype, gemm_sides, samples=args.samples)

    for dtype in (np.float32, np.float64):
        profile_planned_suite(
            dtype, warmups=args.warmups, samples=args.samples,
            rounds=args.rounds)

    for dtype in (np.float32, np.float64):
        profile_packing_suite(
            dtype, warmups=args.warmups, samples=args.samples,
            rounds=args.rounds)

    rng = np.random.default_rng(20260812)
    for dtype in (np.float32, np.float64):
        profile_winograd_boundary(dtype, side=16_384, rng=rng)


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
