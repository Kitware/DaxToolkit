##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

set(unit_tests
  UnitTestWorkletAssert.cxx
  UnitTestWorkletCellGradient.cxx
  UnitTestWorkletCosine.cxx
  UnitTestWorkletElevation.cxx
  UnitTestWorkletMagnitude.cxx
  UnitTestWorkletMapCellError.cxx
  UnitTestWorkletMapFieldError.cxx
  UnitTestWorkletMarchingCubes.cxx
  UnitTestWorkletSine.cxx
  UnitTestWorkletSquare.cxx
  UnitTestWorkletThreshold.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
