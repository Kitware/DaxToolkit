#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  CellGradient.h
  Cosine.h
  Elevation.h
  Sine.h
  Square.h

  testing/Assert.h
  testing/CellMapError.h
  testing/FieldMapError.h
  )

dax_declare_headers(${headers})

set(unit_tests
  UnitTestWorkletCellGradient.cxx
  UnitTestWorkletCosine.cxx
  UnitTestWorkletElevation.cxx
  UnitTestWorkletSine.cxx
  UnitTestWorkletSquare.cxx

  UnitTestWorkletAssert.cxx
  UnitTestWorkletMapCellError.cxx
  UnitTestWorkletMapFieldError.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
