# modified from https://github.com/ACEsuit/mace/blob/d39cc6b5f0f416dbc5eb3462f67544592130076e/tests/test_benchmark.py
import json
from pathlib import Path
from typing import List

import ase.build
import equinox as eqx
import jax
import jraph
import pandas as pd
import pytest

from nequix.data import atomic_numbers_to_indices, dict_to_graphstuple, preprocess_graph
from nequix.model import load_model


def load_nequix_mp_1():
    model, config = load_model("models/nequix-mp-1.nqx")
    return eqx.filter_jit(model), config


@pytest.mark.skipif(not jax.default_backend() == "gpu", reason="gpu not available")
@pytest.mark.benchmark(warmup=True, warmup_iterations=4, min_rounds=8)
@pytest.mark.parametrize("size", (1, 2, 3, 4, 5, 6, 7))
# @pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dtype", ["float32"])
def test_inference(benchmark, size: int, dtype: str):
    model, config = load_nequix_mp_1()
    batch = create_batch(size, config)
    log_bench_info(benchmark, dtype, batch, model)

    def run_benchmark():
        def func():
            energy, forces, stress = model(batch)
            energy.block_until_ready()
            forces.block_until_ready()
            stress.block_until_ready()

        benchmark(func)

    if dtype == "float64":
        with jax.experimental.enable_x64():
            run_benchmark()
    else:
        run_benchmark()


def create_batch(size: int, config: dict) -> jraph.GraphsTuple:
    atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))

    # TODO: extract from model
    atomic_indices = atomic_numbers_to_indices(config["atomic_numbers"])
    cutoff = config["cutoff"]

    graph = preprocess_graph(atoms, atomic_indices, cutoff, targets=False)
    graph = dict_to_graphstuple(graph)
    batch = jraph.pad_with_graphs(graph, n_node=graph.n_node + 1, n_edge=graph.n_edge)
    return batch


def log_bench_info(benchmark, dtype, batch, model):
    benchmark.extra_info["num_atoms"] = int(batch.n_node[0])
    benchmark.extra_info["num_edges"] = int(batch.n_edge[0])
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["device_name"] = jax.devices("gpu")[0].device_kind
    benchmark.extra_info["param_count"] = sum(
        p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0]
    )


def process_benchmark_file(bench_file: Path) -> pd.DataFrame:
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    records = []
    for bench in bench_data["benchmarks"]:
        record = {**bench["extra_info"], **bench["stats"]}
        records.append(record)

    result_df = pd.DataFrame(records)
    result_df["ns/day (1 fs/step)"] = 0.086400 / result_df["median"]
    result_df["Steps per day"] = result_df["ops"] * 86400
    columns = [
        "param_count",
        "num_atoms",
        "num_edges",
        "dtype",
        "device_name",
        "median",
        "Steps per day",
        "ns/day (1 fs/step)",
    ]
    return result_df[columns]


def read_bench_results(result_files: List[str]) -> pd.DataFrame:
    return pd.concat([process_benchmark_file(Path(f)) for f in result_files])


if __name__ == "__main__":
    # Print to stdout a csv of the benchmark metrics
    import subprocess

    result = subprocess.run(
        ["pytest-benchmark", "list"], capture_output=True, text=True, check=True
    )

    bench_files = result.stdout.strip().split("\n")
    bench_results = read_bench_results(bench_files)
    print(bench_results.to_csv(index=False))
