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

set(worklets
  AssertWorklet.h
  CellMapError.h
  FieldMapError.h
  )

dax_declare_worklets(${worklets})

set(unit_tests
  UnitTestWorkletCellAverage.cxx
  UnitTestWorkletCellDataToPointData.cxx
  UnitTestWorkletCellGradient.cxx
  UnitTestWorkletCosine.cxx
  UnitTestWorkletElevation.cxx
  UnitTestWorkletMagnitude.cxx
  UnitTestWorkletMarchingCubes.cxx
  UnitTestWorkletMarchingTetrahedra.cxx
  UnitTestWorkletPointDataToCellData.cxx
  UnitTestWorkletSine.cxx
  UnitTestWorkletSlice.cxx
  UnitTestWorkletSquare.cxx
  UnitTestWorkletTetrahedralize.cxx
  UnitTestWorkletThreshold.cxx
  UnitTestWorkletAssert.cxx
  UnitTestWorkletMapCellError.cxx
  UnitTestWorkletMapFieldError.cxx
  )

dax_save_worklet_unit_tests( ${unit_tests} )

