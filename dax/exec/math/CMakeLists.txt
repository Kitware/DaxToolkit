#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Exp.h
  Precision.h
  Trig.h
  )
# TODO: Sign.h (abs, signbit, copysign), Compare.h (min, max)

dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(unit_tests
  UnitTestMathExp.cxx
  UnitTestMathPrecision.cxx
  UnitTestMathTrig.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
