#
# C++ unit tests (through Boost's framework)
#

boost_test = dependency('boost', modules: ['unit_test_framework'],
                        required: false)

if boost_test.found()
    main_test_suite = executable('mufft_main_test_suite',
        'main_test_suite.cc',
        'test_fft_utils.cc',
        'test_serial_fft_engines.cc',
        dependencies: [mufft, mugrid, boost_test])

    test('mufft_main_test_suite',
        main_test_suite)
endif

#
# Test environment
#

envdata = environment()
python_paths = [
    join_paths(meson.current_build_dir(), '../language_bindings/python'),
    join_paths(meson.current_source_dir(), '../language_bindings/python'),
    join_paths(meson.current_build_dir(), '../subprojects/mugrid/language_bindings/python'),
    join_paths(meson.current_source_dir(), '../subprojects/mugrid/language_bindings/python')
]
envdata.append('PYTHONPATH', python_paths)
envdata.append('TESTS_BUILDDIR', meson.current_build_dir())

#
# Python unit tests
#

test('mufft_python_binding_tests',
    python,
    args: [files('python_binding_tests.py'), '-v', '-f'],
    env: envdata,
    depends: [mufft_python],
    timeout: test_timeout)

#
# If we have MPI, we also want to run the MPI tests
#

if mu_with_mpi
    if boost_test.found()
        mpi_main_test_suite = executable('mpi_mufft_main_test_suite',
            'mpi_main_test_suite.cc',
            'mpi_test_fft_engine.cc',
            dependencies: [mufft, mugrid, boost_test])
    endif

    mpiexec = find_program('mpiexec', required: false)

    if mpiexec.found()
        foreach n : mpi_processes
            if boost_test.found()
                test('mpi_mufft_main_test_suite-np_' + n,
                     mpiexec,
                     args: ['-n', n, '--oversubscribe', mpi_main_test_suite],
                     env: envdata,
                     is_parallel: false)  # Don't run any other test in parallel
            endif

            test('mpi_mufft_python_binding_tests',
                 mpiexec,
                 args: ['-n', n, '--oversubscribe', python.path(),
                        files('python_mpi_binding_tests.py'), '-v', '-f'],
                   depends: [mufft_python],
                   env: envdata,
                   timeout: test_timeout,  # Test can take a long time
                   is_parallel: false)  # Don't run any other test in parallel
        endforeach
    endif
endif
