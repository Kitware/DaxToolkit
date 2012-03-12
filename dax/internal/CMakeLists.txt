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
  DataArray.h
  ExportMacros.h
  GridTopologys.h
  Testing.h
  TopologyStructured.h
  TopologyUnstructured.h
  )

dax_declare_headers(${headers})

set(unit_tests
  UnitTestDataArray.cxx
  UnitTestGridTopologys.cxx
  UnitTestTesting.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
