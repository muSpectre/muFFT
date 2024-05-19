mufft_python_sources = [
    'bind_py_module.cc',
    'bind_py_common_mufft.cc',
    'bind_py_derivatives.cc',
    'bind_py_fftengine.cc'
]

pybind11 = dependency('pybind11', required: true)

mufft_python = python.extension_module(
    '_muFFT', # Name of the module
    mufft_python_sources,
    install: true,  # Install it
    dependencies: [mugrid, mufft, pybind11]
)

subdir('muFFT')
