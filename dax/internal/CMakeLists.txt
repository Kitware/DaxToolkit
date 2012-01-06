#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
set(headers
  DataArray.h
  DataSet.h
  ExportMacros.h
  GridStructures.h
  )

dax_declare_headers(${headers})

set(unit_tests
  UnitTestDataArray.cxx
  UnitTestGridStructures.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
