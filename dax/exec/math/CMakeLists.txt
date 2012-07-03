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

set(headers
  Compare.h
  Exp.h
  Matrix.h
  Numerical.h
  Precision.h
  Sign.h
  Trig.h
  VectorAnalysis.h
  )

dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(unit_tests
  UnitTestMathCompare.cxx
  UnitTestMathExp.cxx
  UnitTestMathMatrix.cxx
  UnitTestMathNumerical.cxx
  UnitTestMathPrecision.cxx
  UnitTestMathSign.cxx
  UnitTestMathTrig.cxx
  UnitTestMathVectorAnalysis.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
