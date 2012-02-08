#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Exp.h
  Trig.h
  )

dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(unit_tests
  UnitTestMathExp.cxx
  UnitTestMathTrig.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
