#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Cell.h
  Derivative.h
  Field.h
  Interpolate.h
  WorkMapCell.h
  WorkMapField.h
)

dax_declare_headers(${headers})

add_subdirectory(internal)

set(unit_tests
  UnitTestCells.cxx
  UnitTestWorkMapCell.cxx
  UnitTestWorkMapField.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
