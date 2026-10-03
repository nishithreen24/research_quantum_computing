# qanneal

Research-grade simulated quantum annealing toolkit (CPU-first, CUDA-ready).

## Build (C++ core)

```bash
cmake -S qanneal -B qanneal/build
cmake --build qanneal/build
ctest --test-dir qanneal/build
```

## Install as a pip package

From a clone:

```bash
python -m pip install -U pip
python -m pip install -e qanneal
```

Build a wheel locally:

```bash
python -m pip install -U build
python -m build qanneal
```

The wheel/sdist will be in `qanneal/dist`.

If you want to install directly from Git:

```bash
python -m pip install "git+https://your-repo-url.git#subdirectory=qanneal"
```

### CMake presets (CPU-only)

Requires CMake 3.19+ for presets.

```bash
cmake --preset cpu-only
cmake --build --preset cpu-only
ctest --preset cpu-only
```

### CPU + MPI preset (OpenMPI recommended)

```bash
cmake --preset cpu-mpi
cmake --build --preset cpu-mpi
```

If CMake cannot find OpenMPI, set `QANNEAL_MPI_HOME` or `MPI_HOME`:

```bash
cmake -S qanneal -B qanneal/build -DQANNEAL_ENABLE_MPI=ON -DQANNEAL_MPI_HOME=/path/to/openmpi
```

## Python examples

```bash
python -m pip install -e qanneal
python qanneal/examples/python/sa_multi.py
python qanneal/examples/python/sqa_basic.py
python qanneal/examples/python/metrics_plot.py
python qanneal/examples/python/parallel_tempering.py
```

### Optional MPI build

```bash
cmake -S qanneal -B qanneal/build -DQANNEAL_ENABLE_MPI=ON
cmake --build qanneal/build
```

Run MPI example:

```bash
mpirun -n 4 qanneal/build/qanneal_mpi_example
```

### SLURM examples (OpenMPI)

Use either launcher style depending on your cluster policy:

- `qanneal/scripts/slurm/run_sa_mpi_srun.sh` (srun)
- `qanneal/scripts/slurm/run_sa_mpi_mpirun.sh` (mpirun)

The original `qanneal/scripts/slurm/run_sa_mpi.sh` remains as a simple srun starter.

## Roadmap

- Core Ising/QUBO models
- Classical and SQA annealers
- Observer and metrics API
- CUDA backend (optional)
- Python bindings (pybind11)
- MPI / SLURM examples

## License

Apache-2.0 (see `LICENSE`). Portions derived from the `sqaod` project with attribution in `NOTICE`.
