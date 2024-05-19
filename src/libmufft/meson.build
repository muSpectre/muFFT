pocketfft_proj = subproject('pocketfft')
pocketfft = pocketfft_proj.get_variable('pocketfft')

mufft_sources = [
    'version.cc',
    'derivative.cc',
    'fft_engine_base.cc',
    'fft_utils.cc',
    'pocketfft_engine.cc'
]

if mu_with_fftw3
    mufft_sources += ['fftw_engine.cc']
endif

if mu_with_fftw3mpi
    mufft_sources += ['fftwmpi_engine.cc']
endif

if mu_with_pfft
    mufft_sources += ['pfft_engine.cc']
endif

libmufft = shared_library(
    'muFFT',
    mufft_sources,
    dependencies: [mufft_dependencies, pocketfft],
    install: true  # needed, otherwise it is not copied to wheel
)
