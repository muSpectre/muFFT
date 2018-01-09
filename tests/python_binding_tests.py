#!/usr/bin/env python3
"""
file   python_binding_tests.py

@author Till Junge <till.junge@epfl.ch>

@date   09 Jan 2018

@brief  Unit tests for python bindings

@section LICENCE

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import unittest
import sys

sys.path.append("src/language_bindings/python")

try:
    import common
    import system
    import solvers
    import material
except ImportError as err:
    print(err)
    sys.exit(-1)

class SystemCheck(unittest.TestCase):
    def test_Construction(self):
        """
        Simple check for system constructors
        """
        print(dir(system))
        print(help(system.SystemFactory2d))

class SolverCheck(unittest.TestCase):
    pass




if __name__ == '__main__':
    unittest.main()
