#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_test_imports.py

@author Till Junge <till.junge@altermail.ch>

@date   18 Jan 2018

@brief  prepares sys.path to load muSpectre

Copyright © 2018 Till Junge

µFFT is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µFFT is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µFFT; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import sys
import os

project_home = os.path.join(os.getcwd(), '../..')

# Default path of the library
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/python"))
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/libmugrid/python"))
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/libmufft/python"))

# Path of the library when compiling with Xcode
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/python/Debug"))
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/libmugrid/python/Debug"))
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/libmufft/python/Debug"))

import muFFT
import muGrid
