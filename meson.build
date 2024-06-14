project(
    'muFFT', # Project name
    'c', 'cpp', # Project type. We need a C and C++ compiler.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for std::optional
    version : run_command('python3', 'discover_version.py', check : true).stdout().strip()
)

pymod = import('python')
python = pymod.find_installation('python3',
                                 required : true,
)

version = run_command(python, 'discover_version.py', '--full', check : true).stdout().strip().split()
version_dirty = version[0]
version_str = version[1]
version_hash = version[2]

mufft_dependencies = []

fftw3 = dependency('fftw3', required : false)
if fftw3.found()
    add_project_arguments('-DWITH_FFTW', language : ['c', 'cpp'])
    mufft_dependencies += [fftw3]
endif

# Those are requirements on *some* systems, hence optional
cc = meson.get_compiler('cpp')
#dl = cc.find_library('dl', required: false)
#execinfo = cc.find_library('execinfo', required: false)

#muspectre_dependencies = [dl, execinfo, fftw3]

# This produces lots of Warning from Eigen3. Disabling for now.
# add_project_arguments('-Weffc++', language: 'cpp')

mpi = dependency('mpi', language : 'cpp', required : false)
mu_with_mpi = false
if mpi.found()
    fftw3mpi = cc.find_library(
        'fftw3_mpi',
        has_headers : ['fftw3-mpi.h'],
        required : false)
    if fftw3mpi.found()
        add_project_arguments('-DWITH_FFTWMPI', language : ['c', 'cpp'])
        mufft_dependencies += [fftw3mpi]
    endif
    pfft = dependency('pfft', required : false)
    if pfft.found()
        add_project_arguments('-DWITH_PFFT', language : ['c', 'cpp'])
        mufft_dependencies += [pfft]
    endif
    if fftw3mpi.found() or pfft.found()
        mu_with_mpi = true  # We found MPI and a parallel FFT library
        mpi_processes = ['1', '2']  # MPI processes to use for parallel tests
    endif

    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp'])
    mufft_dependencies += [mpi]
endif

eigen3 = dependency('eigen3', version : '>=3.4.0', required : true)
mugrid = dependency('mugrid', required : true)
mufft_dependencies += [eigen3, mugrid]

message('  -------------------')
message('  muFFT configuration')
message('    MPI      : ' + (mpi.found() ? '*** YES ***' : 'no'))
message('    pocketfft: *** YES ***')
message('    FFTW3    : ' + (fftw3.found() ? '*** YES ***' : 'no'))
if mpi.found()
    message('    FFTW3 MPI: ' + (fftw3mpi.found() ? '*** YES ***' : 'no'))
    message('    PFFT     : ' + (pfft.found() ? '*** YES ***' : 'no'))
endif
message('  -------------------')

subdir('src')
subdir('language_bindings')

test_timeout = 300  # timeout in seconds for long running tests

subdir('tests')
subdir('examples')
