#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Elevation.h
  )

dax_declare_headers(CUDA ${headers})

set(unit_tests
  )
#dax_unit_tests(CUDA SOURCES ${unit_tests} LIBRARIES daxCont)
