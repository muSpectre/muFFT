# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**µFFT** is a unified C++ library (with Python bindings) that provides a consistent interface to serial and MPI-parallel FFT libraries. It abstracts away backend differences between PocketFFT, FFTW, FFTW-MPI, and PFFT, allowing users to write once and switch implementations based on available hardware and dependencies. The library is designed for scientific computing workflows involving field operations and partial differential equation solving, built on top of the sister library **µGrid**.

**Key characteristics:**
- C++17 header library with pybind11 Python bindings
- Strategy pattern: pluggable FFT engine backends
- Domain decomposition support via MPI
- Spectral and finite-difference derivative operators
- Version: 0.95.0 (development), previous stable: 0.94.0 (26 Jul 2025)

## Build and Development Commands

### Build Setup
```bash
# Configure Meson build (auto-detects MPI, FFTW, PFFT)
meson setup buildDir

# Build and install
meson compile -C buildDir
meson install -C buildDir

# For Python development (editable install)
pip install -e . --no-build-isolation
# Or with compilation from source
pip install -v --no-binary muFFT muFFT
```

### Build Configuration
During setup, Meson reports detected dependencies:
```
MPI        : *** YES *** (if MPI found)
pocketfft  : *** YES *** (always available)
FFTW3      : *** YES *** (optional, faster than PocketFFT)
FFTW3 MPI  : *** YES *** (if MPI + FFTW3-MPI found)
PFFT       : *** YES *** (if MPI + PFFT found)
```

### Testing

**Python tests (recommended for development):**
```bash
# All tests
pytest tests/

# Specific test file
pytest tests/python_fft_tests.py -v

# Single test function
pytest tests/python_fft_tests.py::TestFFTEngine::test_serial_fft_engine -v

# With verbose output and immediate printing
pytest tests/ -v -s

# Run tests with MPI (2-8 processes as configured in meson.build)
mpirun -n 2 pytest tests/
```

**C++ unit tests (if Boost Test Framework available):**
```bash
# Via Meson
meson test -C buildDir

# Direct execution
./buildDir/mufft_main_test_suite
mpirun -n 2 ./buildDir/mpi_mufft_main_test_suite
```

**Full test suite via Meson:**
```bash
meson test -C buildDir --verbose
```

### Python Bindings Development
```bash
# Rebuild Python bindings after C++ changes
meson compile -C buildDir mufft_python

# Install in development mode
pip install -e . --no-build-isolation --force-reinstall
```

## Architecture Overview

### Core Design: FFT Engine Strategy Pattern

All FFT engines inherit from `FFTEngineBase` (src/libmufft/fft_engine_base.hh:cc), which defines:
- `forward()` / `reverse()` - FFT transforms
- `register_real_space_field()` / `register_fourier_space_field()` - field lifecycle
- Field collection getters: `get_field_collection_*()` for real/fourier/half-complex spaces
- Planning flags: `FFT_PlanFlags::estimate|measure|patient` (trade planning vs execution speed)

**Available engines:**
- `PocketFFT` - Pure C fallback (always available, subproject in `/src/libmufft/subproject/`)
- `FFTW` - Serial only (requires libfftw3)
- `FFTWMPI` - Parallel (requires libfftw3-mpi + MPI)
- `PFFT` - Alternative parallel (requires libpfft + MPI)

**Engine selection:** Factory function `muFFT.FFT(nb_grid_pts, engine='serial')` (src/language_bindings/python/muFFT/__init__.py) auto-detects available backends.

### Memory Management

Each engine maintains **three field collections** for memory efficiency:
1. **Real-space fields** - Full array of real values
2. **Fourier-space fields** - Complex-valued FFT output
3. **Half-complex fields** - Optimized storage (r2c/c2r transforms only need ~N/2 complex values due to Hermitian symmetry)

Field registration creates persistent buffers, avoiding repeated allocations. `allow_temporary_buffer` and `allow_destroy_input` flags control buffer handling when formats mismatch.

### Domain Decomposition (MPI)

- Inherits from `muGrid::CartesianDecomposition`
- Each MPI process holds a local subdomain
- **Ghost buffers (0.95.0 feature)** - Halo cells for stencil operations (src/tests/python_mpi_ghost_tests.py)
- Communicator wrapper for MPI operations

### Derivative Operators

**Two derivative types (src/libmufft/derivative.hh:cc):**

1. **FourierDerivative** - Spectral derivatives via multiplication in Fourier space (high accuracy, needs complete domain)
2. **DiscreteDerivative** - Finite-difference stencils (local, configurable order). Stencil presets in `muFFT.Stencils{1D,2D,3D}`

### Python Interface Layers

1. **Low-level (C++ via pybind11):** `_muFFT.FFTW`, `_muFFT.FFTWMPI`, etc. (language_bindings/python/bind_*.cc)
2. **Mid-level factory:** `muFFT.FFT()` resolves engine selection
3. **High-level convenience:** NumPy-like arrays, pre-defined stencil sets, automatic field management

## Key Files and Responsibilities

| File/Directory | Purpose |
|---|---|
| `src/libmufft/fft_engine_base.hh/cc` | Abstract FFT engine interface; all backends implement this |
| `src/libmufft/fftw_engine.hh/cc` | Serial FFTW backend |
| `src/libmufft/fftwmpi_engine.hh/cc` | Parallel FFTW-MPI backend |
| `src/libmufft/pfft_engine.hh/cc` | Alternative parallel PFFT backend |
| `src/libmufft/pocketfft_engine.hh/cc` | Fallback PocketFFT backend |
| `src/libmufft/derivative.hh/cc` | Fourier & discrete derivative operators |
| `src/libmufft/mufft_common.hh` | Type definitions, enums (FFTDirection, FFT_PlanFlags) |
| `language_bindings/python/muFFT/__init__.py` | Python factory functions, engine discovery |
| `language_bindings/python/bind_py_fftengine.cc` | pybind11 FFT engine bindings |
| `tests/python_fft_tests.py` | Core FFT functionality tests (serial + MPI) |
| `tests/python_derivative_tests.py` | Derivative operator tests |
| `tests/python_mpi_ghost_tests.py` | MPI ghost buffer tests |
| `tests/python_netcdf_tests.py` | I/O (NetCDF serialization) tests |

## Adding New Functionality

### Adding a New FFT Engine Backend

1. Create `src/libmufft/new_backend_engine.hh` and `new_backend_engine.cc`
2. Inherit from `FFTEngineBase` and implement:
   - Constructor taking `muGrid::CommunicationMap` for domain decomposition
   - `forward()` and `reverse()` transforms
   - Field collection registration methods
   - Pure virtual methods from base class
3. Update `src/libmufft/meson.build` to conditionally compile (check for dependency)
4. Add compiler flag (e.g., `-DWITH_NEWBACKEND`) in root `meson.build`
5. Update Python bindings in `language_bindings/python/bind_py_fftengine.cc`
6. Add tests in `tests/python_fft_tests.py`

### Extending Python API

1. Edit pybind11 bindings in `language_bindings/python/bind_py_*.cc`
2. Rebuild: `meson compile -C buildDir mufft_python`
3. Test changes with `pytest tests/`

### Adding Tests

- **Python tests:** Add to appropriate file in `/tests/` (python_fft_tests.py, python_derivative_tests.py, etc.)
- **C++ tests:** Add to test_serial_fft_engines.cc (or mpi_test_fft_engine.cc for MPI)
- Use existing test patterns as template (compare against NumPy reference implementations)
- Run with `pytest tests/` or `meson test -C buildDir`

## Dependencies and Version Management

**Required:**
- Eigen3 ≥3.4.0 (linear algebra)
- muGrid (field abstractions, domain decomposition)
- Python 3.9+ (for bindings)
- C++17 compiler

**Optional (auto-detected by Meson):**
- FFTW3 (faster serial FFT)
- MPI (distributed computing)
- FFTW3-MPI (parallel FFT with FFTW)
- PFFT (alternative parallel FFT)
- Boost Test Framework (C++ unit tests)

**Version discovery:** Git-based via `discover_version.py`:
- `git describe --tags` → semantic version
- Detects dirty state (uncommitted changes)
- PEP 440 compatible for wheel distribution

## Common Workflows

### Running Examples
```bash
# After building, examples in src/examples/ demonstrate:
# - Basic FFT: transform.py
# - Gradients: gradient.py
# - Spectral derivatives: fourier_derivative.py
# - Finite-difference derivatives: discrete_derivative.py
# - MPI parallel: parallel.py

python src/examples/transform.py
```

### Debugging FFT Operations
- Check engine selection: `muFFT.mangle_engine_identifier(engine)` resolves name → backend
- Use `FFT_PlanFlags.estimate` for fast turnaround during development
- Verify field collections: engine exposes `get_field_collection_*()` after `register_*()` calls
- Compare against NumPy FFT for correctness: test pattern in python_fft_tests.py

### Performance Profiling
- FFTW planning flags affect performance: `measure` and `patient` are slower to plan but faster to execute
- MPI communication overhead: test strong vs weak scaling with mpirun
- Ghost buffer overhead: py/mpi_ghost_tests.py shows communication patterns

## Documentation References

- **Full API docs:** https://muspectre.github.io/muFFT/
- **DeepWiki (LLM-generated):** https://deepwiki.com/muSpectre/muFFT
- **Examples:** `src/examples/` directory
- **Tests as documentation:** python_fft_tests.py shows practical usage patterns
