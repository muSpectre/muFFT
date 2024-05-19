project(
    'muFFT', # Project name
    'c', 'cpp', # Project type. We need a C and C++ compiler.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for std::optional
    version: run_command('python3', 'discover_version.py', check: true).stdout().strip()
)

pymod = import('python')
python = pymod.find_installation('python3',
    required: true,
)

version = run_command(python, 'discover_version.py', '--full', check: true).stdout().strip().split()
version_dirty = version[0]
version_str = version[1]
version_hash = version[2]

fftw3 = dependency('fftw3', required: false)
if fftw3.found()
    message('muFFT FFTW: *** YES ***')
    mu_with_fftw3 = true
    add_project_arguments('-DWITH_FFTW', language : ['c', 'cpp'])
else
    message('muFFT FFTW: no')
    mu_with_fftw3 = false
endif

# Those are requirements on *some* systems, hence optional
cc = meson.get_compiler('c')
#dl = cc.find_library('dl', required: false)
#execinfo = cc.find_library('execinfo', required: false)

#muspectre_dependencies = [dl, execinfo, fftw3]

# This produces lots of Warning from Eigen3. Disabling for now.
# add_project_arguments('-Weffc++', language: 'cpp')

mufft_dependencies = []

mpi = dependency('mpi', language: 'cpp', required: false)
mu_with_mpi = false
if mpi.found()
    fftw3mpi = cc.find_library(
        'fftw3_mpi',
        dirs: ['/usr/lib', '/usr/lib/x86_64-linux-gnu'],
        required: false)
    if fftw3mpi.found() and cc.has_header('fftw3-mpi.h')
        message('muFFT FFTW3 MPI: *** YES ***')
        mu_with_fftw3mpi = true
        add_project_arguments('-DWITH_FFTWMPI', language : ['c', 'cpp'])
        mufft_dependencies += [fftw3mpi]
    else
        message('muFFT FFTW3 MPI: no')
        mu_with_fftw3mpi = false
    endif
    pfft = cc.find_library('pfft', required: false)
    if pfft.found() and cc.has_header('pfft.h')
        message('muFFT PFFT: *** YES ***')
        mu_with_pfft = true
        add_project_arguments('-DWITH_PFFT', language : ['c', 'cpp'])
        mufft_dependencies += [pfft]
    else
        message('muFFT PFFT: no')
        mu_with_pfft = false
    endif
    if mu_with_fftw3mpi or mu_with_pfft
        mu_with_mpi = true  # We found MPI and a parallel FFT library
        mpi_processes = ['1', '2']  # MPI processes to use for parallel tests
    endif

    message('muFFT MPI: *** YES ***')
    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp'])
    mufft_dependencies += [mpi]
else
    message('muFFT MPI: no')
    mu_with_fftw3mpi = false
    mu_with_pfft = false
endif

eigen3 = dependency('eigen3', required: true)
mugrid = dependency('mugrid', required: true)
mufft_dependencies += [eigen3, mugrid]

subdir('src')
subdir('language_bindings')

test_timeout = 300  # timeout in seconds for long running tests

subdir('tests')
