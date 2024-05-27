.. _Summary:

Summary
-------

*µ*\FFT is a part of the project *µ*\Spectre, which provides an open-source platform for efficient FFT-based continuum mesoscale modelling. Its development is funded by the `Swiss National Science Foundation <https://snf.ch>`_ within an Ambizione Project and by the `European Research Council <https://erc.europa.eu>`_ within `Starting Grant 757343 <https://cordis.europa.eu/project/id/757343>`_.

*µ*\FFT is a unified C++ interface to serial and MPI-parallel FFT libraries. It is designed to be used in combination with *µ*\Spectre, but can also be used as a standalone library. The library provides a set of classes for (real-to-complex) FFT transforms of real fields, which are implemented using the abstraction layer provided by `µGrid <https://github.com/muSpectre/muGrid>`_ The library is designed to be efficient and scalable, and to be used in combination with modern C++ features, such as template metaprogramming and parallelism.

*µ*\FFT has language bindings for Python.