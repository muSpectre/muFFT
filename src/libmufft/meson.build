pocketfft_proj = subproject('pocketfft')
pocketfft = pocketfft_proj.get_variable('pocketfft')

mufft_sources = [
    'version.cc',
    'derivative.cc',
    'fft_engine_base.cc',
    'fft_utils.cc',
    'pocketfft_engine.cc'
]

if fftw3.found()
    mufft_sources += ['fftw_engine.cc']
endif

if mpi.found()
    if fftw3mpi.found()
        mufft_sources += ['fftwmpi_engine.cc']
    endif

    if pfft.found()
        mufft_sources += ['pfft_engine.cc']
    endif
endif

libmufft = shared_library(
    'muFFT',
    mufft_sources,
    dependencies : [mufft_dependencies, pocketfft],
    install : true  # needed, otherwise it is not copied to wheel
)
